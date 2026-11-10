# Step-by-Step Runbook: Test 07 Restoring Workload Snapshot Without CUDA Configuration

This runbook demonstrates **Barrier 2** of gVisor GPU save/restore on `root@stg-nc-workload1-master-1`:
> **"Can a GPU workload snapshot be restored into a container without passing the CUDA restore configuration?"**

It restores directly from the baseline snapshot captured via `SnapshotJob` `sj-airuntime-gpu-01`.

---

## Step 0: Ensure GPU Claim is Free & Retrieve Snapshot SHA

```bash
# 1. Clean up pods from previous tests
kubectl delete pod -n gpu-gvisor-poc airuntime-gpu-workload airuntime-gpu-scaled-up airuntime-gpu-scaled-down airuntime-gpu-cross-template neev-gpu-ceph-workload neev-gpu-ceph-restore-clone neev-gpu-ceph-restore-paused neev-gpu-upgrade-source neev-gpu-upgrade-restore neev-gpu-card1-source neev-gpu-card0-restore gpu-restore-skipped-cuda --force --grace-period=0 2>/dev/null || true

# 2. Confirm SnapshotJob sj-airuntime-gpu-01 exists and is Ready
kubectl get snapshotjob -n gpu-gvisor-poc sj-airuntime-gpu-01

# 3. Retrieve SHA256 checksum into shell variable
SHA=$(kubectl get snapshotjob -n gpu-gvisor-poc sj-airuntime-gpu-01 -o jsonpath='{.status.sha256}')
echo "Snapshot SHA256: $SHA"
```

---

## Step 1: Deploy Restore Pod Without CUDA Configuration

Deploy the restored pod `gpu-restore-skipped-cuda`, which **completely strips**:
1. The GPU `resourceClaims` (omitting CDI device mounts `/dev/nvidia*`).
2. The `/opt/cuda-checkpoint` volume and volume mount.

```bash
cat << 'RESTORE_EOF' | sed "s/PASTE_SHA_HERE/$SHA/g" | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: gpu-restore-skipped-cuda
  namespace: gpu-gvisor-poc
  annotations:
    dev.gvisor.internal.restore.host-image-path: /var/lib/neev-checkpoint-scratch/gpu-gvisor-poc/sj-airuntime-gpu-01/checkpoint
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
        restoreId: sj-airuntime-gpu-01
        objectRef: s3://sandbox-snapshots/gpu-gvisor-poc/sj-airuntime-gpu-01.tar.zst
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
    env:
    - name: NVIDIA_VISIBLE_DEVICES
      value: void
    resources:
      requests:
        cpu: "2"
        memory: "4Gi"
      limits:
        cpu: "2"
        memory: "4Gi"
    volumeMounts:
    - name: workspace
      mountPath: /workspace
    - name: ckpt-gate
      mountPath: /var/run/neev-checkpoint-gate
      readOnly: true
RESTORE_EOF
```

---

## Step 2: Inspect Pod Status & Rejection Reason

Check pod status:
```bash
kubectl get pod -n gpu-gvisor-poc gpu-restore-skipped-cuda
```

*Expected Status: `StartError` (Exit Code 128).*

Inspect Kubelet failure events:
```bash
kubectl describe pod -n gpu-gvisor-poc gpu-restore-skipped-cuda | grep -E "Warning|Failed|Error"
```

**Observed Rejection Output:**
```text
Status:              Failed
      Reason:   StartError
  Warning  Failed   4s   kubelet  spec.containers{cuda}: Error: failed to start containerd task "cuda": 
OCI runtime restore failed: starting container: starting sub-container [/bin/bash]: 
failed to handle restore spec validation: failed to validate restore spec: 
failed to validate spec for container "cuda": 
"Mounts" does not match across checkpoint restore for container: "agent"
```

---

## Step 3: Why This Failed (The Mount Invariant)

1. At checkpoint time, the running container had CDI NVIDIA device mounts and the `/opt/cuda-checkpoint` volume mount recorded in `checkpoint.img`.
2. When restoring without the CUDA configuration, gVisor validates the original mounts against the new pod spec:
   $$\text{Checkpoint Spec (with CUDA mounts)} \quad \neq \quad \text{Restore Spec (CUDA mounts omitted)}$$
3. Because the required GPU device files and restore volumes were missing, gVisor aborted restoration before starting the container process.

---

## Step 4: Clean Up Pod

```bash
kubectl delete pod -n gpu-gvisor-poc gpu-restore-skipped-cuda --force --grace-period=0
```

