# Step-by-Step Runbook: Test 07 GPU Snapshot & Restore Without CUDA Configuration

This test demonstrates the **two strict architectural barriers** of gVisor GPU Save/Restore on `stg-nc-workload1-worker-3` and `stg-nc-workload1-master-1`:
1. **Barrier 1 (At Checkpoint Time):** Attempting to take a snapshot containing only filesystem and CPU memory without the CUDA checkpoint flag (`--cuda-checkpoint-path`) on an active GPU workload triggers an immediate **runtime panic in `nvproxy.beforeSave()`**.
2. **Barrier 2 (At Restore Time):** Restoring a valid GPU snapshot into a container **without** CUDA configuration (stripping `resourceClaims` and `/opt/cuda-checkpoint`) fails immediately with **`spec "Mounts" does not match`**.

---

## Step 0: Clean Up Previous Test State & Recreate GPU Claim

Executed on `root@stg-nc-workload1-master-1`:

```bash
kubectl delete pod -n gpu-gvisor-poc airuntime-gpu-workload airuntime-gpu-scaled-up airuntime-gpu-scaled-down airuntime-gpu-cross-template neev-gpu-ceph-workload neev-gpu-ceph-restore-clone neev-gpu-ceph-restore-paused neev-gpu-upgrade-source neev-gpu-upgrade-restore neev-gpu-card1-source neev-gpu-card0-restore gpu-source-workload gpu-restore-no-cuda gpu-restore-with-cuda --force --grace-period=0 2>/dev/null || true
kubectl delete snapshotjob -n gpu-gvisor-poc sj-cuda-config-test sj-airuntime-gpu-01 2>/dev/null || true
kubectl delete resourceclaim -n gpu-gvisor-poc gpu-claim 2>/dev/null || true

cat << 'EOF' | kubectl apply -f -
apiVersion: resource.k8s.io/v1
kind: ResourceClaim
metadata:
  name: gpu-claim
  namespace: gpu-gvisor-poc
spec:
  devices:
    requests:
    - name: gpu
      exactly:
        allocationMode: ExactCount
        count: 1
        deviceClassName: gpu.nvidia.com
        selectors:
        - cel:
            expression: "has(device.attributes['gpu.nvidia.com'].uuid) && device.attributes['gpu.nvidia.com'].uuid == 'GPU-36df8b09-2435-5c42-7324-850c168ecc94'"
EOF
```

Verify the claim exists:
```bash
kubectl get resourceclaim -n gpu-gvisor-poc gpu-claim
```

---

## Step 1: Deploy Source GPU Workload Pod

Executed on `root@stg-nc-workload1-master-1`:

Deploy the GPU source pod `gpu-source-workload` allocating 256MB of tensor memory on the GPU:

```bash
cat << 'EOF' | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: gpu-source-workload
  namespace: gpu-gvisor-poc
  annotations:
    dev.gvisor.spec.mount.workspace.type: bind
    dev.gvisor.spec.mount.workspace.share: pod
    dev.gvisor.spec.mount.workspace.options: size=1073741824
spec:
  runtimeClassName: runsc-root-self
  nodeSelector:
    kubernetes.io/hostname: stg-nc-workload1-worker-3
  restartPolicy: Never
  resourceClaims:
  - name: gpu
    resourceClaimName: gpu-claim
  volumes:
  - name: workspace
    emptyDir: {}
  - name: cuda-checkpoint
    hostPath:
      path: /opt/cuda-checkpoint
      type: Directory
  - name: ckpt-gate
    csi:
      driver: checkpoint.csi.neev.io
  containers:
  - name: cuda
    image: harbor.neev.work/test-template/pytorch-v2.8.0-ubuntu-24.04:0.1.2
    command:
    - /bin/bash
    - -c
    - |
      export PATH=/opt/venv/bin:$PATH
      cat << 'PYEOF' > /workspace/airuntime_gpu_workload.py
      import os, sys, time, torch

      log_path = "/workspace/airuntime_gpu.log"
      token_path = "/workspace/runtime-token.txt"

      with open(token_path, "w") as f:
          f.write("airuntime-gpu-session-active\n")

      print(f"=== AIRuntime GPU Workload Starting (PID={os.getpid()}) ===", flush=True)
      dev = torch.device("cuda:0")
      state = torch.zeros(64 * 1024 * 1024, dtype=torch.float32, device=dev)

      tick = 0
      with open(log_path, "a") as f:
          f.write(f"=== Session Initialized: PID={os.getpid()} ===\n")

      while True:
          tick += 1
          state += 1.0
          torch.cuda.synchronize()
          sample = float(state[0].item())
          allocated_mb = torch.cuda.memory_allocated(dev) / (1024 * 1024)
          msg = f"[AIRUNTIME-GPU] tick={tick} sample={sample:.1f} VRAM={allocated_mb:.1f}MB time={time.time()}\n"
          print(msg, end="", flush=True)
          with open(log_path, "a") as f:
              f.write(msg)
              f.flush()
          time.sleep(1)
      PYEOF

      exec python3 -u /workspace/airuntime_gpu_workload.py
    env:
    - name: NVIDIA_VISIBLE_DEVICES
      value: void
    volumeMounts:
    - name: workspace
      mountPath: /workspace
    - name: cuda-checkpoint
      mountPath: /opt/cuda-checkpoint
      readOnly: true
    - name: ckpt-gate
      mountPath: /var/run/neev-checkpoint-gate
      readOnly: true
    resources:
      requests:
        cpu: "2"
        memory: "4Gi"
      limits:
        cpu: "2"
        memory: "4Gi"
      claims:
      - name: gpu
EOF
```

Wait until the pod reaches `Running`:
```bash
kubectl get pod -n gpu-gvisor-poc gpu-source-workload -w
```

Check the live GPU log:
```bash
kubectl logs -n gpu-gvisor-poc gpu-source-workload -c cuda --tail=10
```

---

## Step 2: [BARRIER 1] Test Checkpointing WITHOUT CUDA Checkpoint Flag

SSH into `stg-nc-workload1-worker-3`:
```bash
ssh root@stg-nc-workload1-worker-3
```

Attempt to take a snapshot of the active GPU container containing **only filesystem and CPU memory**, intentionally omitting `--cuda-checkpoint-path`:

```bash
CID=$(crictl ps --name cuda --state Running -q | head -n 1)
echo "CUDA Container ID: $CID"

mkdir -p /var/lib/neev-checkpoint-scratch/gpu-gvisor-poc/sj-no-cuda-flag/checkpoint
rm -rf /var/lib/neev-checkpoint-scratch/gpu-gvisor-poc/sj-no-cuda-flag/checkpoint/*

/usr/local/bin/runsc-20260905 --root /run/containerd/runsc/k8s.io checkpoint \
  --image-path=/var/lib/neev-checkpoint-scratch/gpu-gvisor-poc/sj-no-cuda-flag/checkpoint \
  --exclude-committed-zero-pages \
  --leave-running \
  "$CID"
```

### Observed Result: ❌ IMMEDIATE RUNTIME CRASH (Panic)

gVisor's `nvproxy` driver halts execution with a panic because active NVIDIA driver handles cannot be serialized without the CUDA checkpoint utility:

```text
checkpoint failed: checkpointing container "6486b966a20cc1cbdae13810411820abe1947603815faa7a235f8f20826a94c8": encoding error: nvproxy: object handle 0xc1d06bc7:0x5c00007f (class 0x00000079, type *nvproxy.miscObject) is not restorable:
goroutine 1158 [running]:
gvisor.dev/gvisor/pkg/state.safely.func1()
        pkg/state/state.go:309 +0x19e
panic({0x12fc9a0?, 0x5115e2ecf90?})
        GOROOT/src/runtime/panic.go:860 +0x13a
gvisor.dev/gvisor/pkg/sentry/devices/nvproxy.(*nvproxy).beforeSave(0x421b5f?)
        pkg/sentry/devices/nvproxy/save_restore.go:182 +0x1f9
gvisor.dev/gvisor/pkg/sentry/devices/nvproxy.(*nvproxy).StateSave(0x5115b73b208, {{0x5115dafe388?, 0x5115e2f2528?}})
        bazel-out/k8-fastbuild/bin/pkg/sentry/devices/nvproxy/nvproxy_state_autogen.go:230 +0x30
```

Exit back to `stg-nc-workload1-master-1`:
```bash
exit
```

---

## Step 3: Capture Full Snapshot via SnapshotJob

Executed on `root@stg-nc-workload1-master-1`:

Create the `SnapshotJob` `sj-cuda-config-test`:

```bash
cat << 'EOF' | kubectl apply -f -
apiVersion: sandbox.neevcloud.com/v1alpha1
kind: SnapshotJob
metadata:
  name: sj-cuda-config-test
  namespace: gpu-gvisor-poc
spec:
  sandboxRef: gpu-source-workload
  nodeName: stg-nc-workload1-worker-3
  mode: snapshot
  snapshotType: full
  runtimeClass: runsc-root-self
EOF
```

Monitor until `PHASE` is `Ready`:
```bash
kubectl get snapshotjob -n gpu-gvisor-poc sj-cuda-config-test -w
```

Retrieve the SHA256 checksum into the shell variable `$SHA`:
```bash
SHA=$(kubectl get snapshotjob -n gpu-gvisor-poc sj-cuda-config-test -o jsonpath='{.status.sha256}')
echo "Snapshot SHA256: $SHA"
```

Delete the source pod to free the GPU claim:
```bash
kubectl delete pod -n gpu-gvisor-poc gpu-source-workload --force --grace-period=0
```

---

## Step 4: [BARRIER 2 - NEGATIVE TEST] Restore WITHOUT CUDA Configuration

Executed on `root@stg-nc-workload1-master-1`:

Attempt to restore the snapshot into `gpu-restore-no-cuda`, which **strips**:
1. The `resourceClaims` and container `claims` (no GPU CDI devices).
2. The `/opt/cuda-checkpoint` volume and volume mount.

```bash
cat << RESTORE_EOF | sed "s/PASTE_SHA_HERE/$SHA/g" | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: gpu-restore-no-cuda
  namespace: gpu-gvisor-poc
  annotations:
    dev.gvisor.internal.restore.host-image-path: /var/lib/neev-checkpoint-scratch/gpu-gvisor-poc/sj-cuda-config-test/checkpoint
    dev.gvisor.spec.mount.workspace.type: bind
    dev.gvisor.spec.mount.workspace.share: pod
    dev.gvisor.spec.mount.workspace.options: size=1073741824
spec:
  runtimeClassName: runsc-root-self
  nodeSelector:
    kubernetes.io/hostname: stg-nc-workload1-worker-3
  restartPolicy: Never
  volumes:
  - name: workspace
    emptyDir: {}
  - name: ckpt-gate
    csi:
      driver: checkpoint.csi.neev.io
      volumeAttributes:
        restoreId: sj-cuda-config-test
        objectRef: s3://sandbox-snapshots/gpu-gvisor-poc/sj-cuda-config-test.tar.zst
        sha256: "PASTE_SHA_HERE"
  containers:
  - name: cuda
    image: harbor.neev.work/test-template/pytorch-v2.8.0-ubuntu-24.04:0.1.2
    command:
    - /bin/bash
    - -c
    - |
      export PATH=/opt/venv/bin:$PATH
      cat << 'PYEOF' > /workspace/airuntime_gpu_workload.py
      import os, sys, time, torch

      log_path = "/workspace/airuntime_gpu.log"
      token_path = "/workspace/runtime-token.txt"

      with open(token_path, "w") as f:
          f.write("airuntime-gpu-session-active\n")

      print(f"=== AIRuntime GPU Workload Starting (PID={os.getpid()}) ===", flush=True)
      dev = torch.device("cuda:0")
      state = torch.zeros(64 * 1024 * 1024, dtype=torch.float32, device=dev)

      tick = 0
      with open(log_path, "a") as f:
          f.write(f"=== Session Initialized: PID={os.getpid()} ===\n")

      while True:
          tick += 1
          state += 1.0
          torch.cuda.synchronize()
          sample = float(state[0].item())
          allocated_mb = torch.cuda.memory_allocated(dev) / (1024 * 1024)
          msg = f"[AIRUNTIME-GPU] tick={tick} sample={sample:.1f} VRAM={allocated_mb:.1f}MB time={time.time()}\n"
          print(msg, end="", flush=True)
          with open(log_path, "a") as f:
              f.write(msg)
              f.flush()
          time.sleep(1)
      PYEOF

      exec python3 -u /workspace/airuntime_gpu_workload.py
    resources:
      requests:
        cpu: "2"
        memory: "4Gi"
      limits:
        cpu: "2"
        memory: "4Gi"
    env:
    - name: NVIDIA_VISIBLE_DEVICES
      value: void
    volumeMounts:
    - name: workspace
      mountPath: /workspace
    - name: ckpt-gate
      mountPath: /var/run/neev-checkpoint-gate
      readOnly: true
RESTORE_EOF
```

Check the status:
```bash
kubectl get pod -n gpu-gvisor-poc gpu-restore-no-cuda
```

It immediately fails with **`StartError`** (exit code 128). Inspect the exact failure reason:
```bash
kubectl describe pod -n gpu-gvisor-poc gpu-restore-no-cuda | grep -A 5 "failed to create shim task"
```

**Observed Result:**
```text
failed to create shim task: OCI runtime create failed: ... cannot restore: spec "Mounts" does not match: ...: unknown
```

Clean up the failed pod:
```bash
kubectl delete pod -n gpu-gvisor-poc gpu-restore-no-cuda --force --grace-period=0
```

---

## Step 5: [BARRIER 2 - POSITIVE CONTROL] Restore WITH CUDA Configuration

Executed on `root@stg-nc-workload1-master-1`:

Restore the **exact same snapshot** into `gpu-restore-with-cuda`, providing the required CUDA configuration (`resourceClaims`, container claims, and `/opt/cuda-checkpoint` volume mounts):

```bash
cat << RESTORE_EOF | sed "s/PASTE_SHA_HERE/$SHA/g" | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: gpu-restore-with-cuda
  namespace: gpu-gvisor-poc
  annotations:
    dev.gvisor.internal.restore.host-image-path: /var/lib/neev-checkpoint-scratch/gpu-gvisor-poc/sj-cuda-config-test/checkpoint
    dev.gvisor.spec.mount.workspace.type: bind
    dev.gvisor.spec.mount.workspace.share: pod
    dev.gvisor.spec.mount.workspace.options: size=1073741824
spec:
  runtimeClassName: runsc-root-self
  nodeSelector:
    kubernetes.io/hostname: stg-nc-workload1-worker-3
  restartPolicy: Never
  resourceClaims:
  - name: gpu
    resourceClaimName: gpu-claim
  volumes:
  - name: workspace
    emptyDir: {}
  - name: cuda-checkpoint
    hostPath:
      path: /opt/cuda-checkpoint
      type: Directory
  - name: ckpt-gate
    csi:
      driver: checkpoint.csi.neev.io
      volumeAttributes:
        restoreId: sj-cuda-config-test
        objectRef: s3://sandbox-snapshots/gpu-gvisor-poc/sj-cuda-config-test.tar.zst
        sha256: "PASTE_SHA_HERE"
  containers:
  - name: cuda
    image: harbor.neev.work/test-template/pytorch-v2.8.0-ubuntu-24.04:0.1.2
    command:
    - /bin/bash
    - -c
    - |
      export PATH=/opt/venv/bin:$PATH
      cat << 'PYEOF' > /workspace/airuntime_gpu_workload.py
      import os, sys, time, torch

      log_path = "/workspace/airuntime_gpu.log"
      token_path = "/workspace/runtime-token.txt"

      with open(token_path, "w") as f:
          f.write("airuntime-gpu-session-active\n")

      print(f"=== AIRuntime GPU Workload Starting (PID={os.getpid()}) ===", flush=True)
      dev = torch.device("cuda:0")
      state = torch.zeros(64 * 1024 * 1024, dtype=torch.float32, device=dev)

      tick = 0
      with open(log_path, "a") as f:
          f.write(f"=== Session Initialized: PID={os.getpid()} ===\n")

      while True:
          tick += 1
          state += 1.0
          torch.cuda.synchronize()
          sample = float(state[0].item())
          allocated_mb = torch.cuda.memory_allocated(dev) / (1024 * 1024)
          msg = f"[AIRUNTIME-GPU] tick={tick} sample={sample:.1f} VRAM={allocated_mb:.1f}MB time={time.time()}\n"
          print(msg, end="", flush=True)
          with open(log_path, "a") as f:
              f.write(msg)
              f.flush()
          time.sleep(1)
      PYEOF

      exec python3 -u /workspace/airuntime_gpu_workload.py
    resources:
      requests:
        cpu: "2"
        memory: "4Gi"
      limits:
        cpu: "2"
        memory: "4Gi"
      claims:
      - name: gpu
    env:
    - name: NVIDIA_VISIBLE_DEVICES
      value: void
    volumeMounts:
    - name: workspace
      mountPath: /workspace
    - name: cuda-checkpoint
      mountPath: /opt/cuda-checkpoint
      readOnly: true
    - name: ckpt-gate
      mountPath: /var/run/neev-checkpoint-gate
      readOnly: true
RESTORE_EOF
```

Monitor the pod status:
```bash
kubectl get pod -n gpu-gvisor-poc gpu-restore-with-cuda -w
```

The pod reaches **`Running`**! Check the live container logs to verify GPU continuity:
```bash
kubectl logs -n gpu-gvisor-poc gpu-restore-with-cuda -c cuda --tail=15 -f
```

**Observed Result:**
The workload immediately resumes from the checkpointed tick with exact VRAM intact (256MB) and tensor math continuing seamlessly!

---

## Step 6: Clean Up Test 07 State

Executed on `root@stg-nc-workload1-master-1`:

```bash
kubectl delete pod -n gpu-gvisor-poc gpu-restore-with-cuda --force --grace-period=0
kubectl delete snapshotjob -n gpu-gvisor-poc sj-cuda-config-test
```
