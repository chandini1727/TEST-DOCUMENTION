# Step-by-Step Runbook: Test 03 Cross-Template Compatibility

This runbook contains all exact commands to execute and verify the cross-template restore test scenario on `root@stg-nc-workload1-master-1`.

### Objective:
Verify whether a snapshot captured from the PyTorch GPU template (`pytorch-v2.8.0-ubuntu-24.04:0.1.2`) can be restored into a pod provisioned with a different base image (`ubuntu-24.04:0.1.3`, which lacks CUDA and PyTorch virtual environments).

This test restores directly from the snapshot captured via `SnapshotJob` `sj-airuntime-gpu-01`.

---

## Step 0: Ensure GPU Claim is Free & Retrieve Snapshot SHA

Confirm any previous test pods are cleaned up (so `gpu-claim` is unbound) and retrieve the snapshot SHA256 checksum:

```bash
# 1. Clean up any leftover test pods
kubectl delete pod -n gpu-gvisor-poc airuntime-gpu-workload airuntime-gpu-scaled-up airuntime-gpu-scaled-down airuntime-gpu-cross-template --force --grace-period=0 2>/dev/null || true

# 2. Confirm SnapshotJob sj-airuntime-gpu-01 exists and is Ready
kubectl get snapshotjob -n gpu-gvisor-poc sj-airuntime-gpu-01

# 3. Retrieve SHA256 checksum into shell variable
SHA=$(kubectl get snapshotjob -n gpu-gvisor-poc sj-airuntime-gpu-01 -o jsonpath='{.status.sha256}')
echo "Snapshot SHA256: $SHA"
```

*Expected output: A 64-character hex string (e.g., `499ec4fac7fe6e78...`)*

---

## Step 1: Deploy Cross-Template Restore Pod

Deploy the restored pod pointing to the target image `harbor.neev.work/test-template/ubuntu-24.04:0.1.3`:

```bash
cat << 'EOF' | sed "s/PASTE_SHA_HERE/$SHA/g" | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: airuntime-gpu-cross-template
  namespace: gpu-gvisor-poc
  annotations:
    dev.gvisor.spec.mount.workspace.type: bind
    dev.gvisor.spec.mount.workspace.share: pod
    dev.gvisor.spec.mount.workspace.options: size=1073741824
    dev.gvisor.internal.restore.host-image-path: /var/lib/neev-checkpoint-scratch/gpu-gvisor-poc/sj-airuntime-gpu-01/checkpoint
spec:
  runtimeClassName: runsc-root-self
  restartPolicy: Never
  nodeSelector:
    kubernetes.io/hostname: stg-nc-workload1-worker-3
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
        restoreId: sj-airuntime-gpu-01
        objectRef: s3://sandbox-snapshots/gpu-gvisor-poc/sj-airuntime-gpu-01.tar.zst
        sha256: "PASTE_SHA_HERE"
  containers:
  - name: cuda
    # NEEVCLOUD BASE TEMPLATE: Lacks PyTorch virtualenv (/opt/venv)
    image: harbor.neev.work/test-template/ubuntu-24.04:0.1.3
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

---

## Step 2: Inspect Pod Status & Rejection Events

Check the pod status:
```bash
kubectl get pod -n gpu-gvisor-poc airuntime-gpu-cross-template
```

*Expected Status: `StartError` or `CrashLoopBackOff`.*

Inspect Kubelet failure events:
```bash
kubectl describe pod -n gpu-gvisor-poc airuntime-gpu-cross-template | grep -E "Warning|Failed|Error"
```

**Observed Rejection Output:**
```text
Status:              Failed
      Reason:   StartError
  Warning  Failed   8s   kubelet  spec.containers{cuda}: Error: failed to start containerd task "cuda": 
OCI runtime restore failed: starting container: starting sub-container: 
failed to handle restore spec validation: failed to validate restore spec: 
failed to validate spec for container "cuda": 
"Mounts" does not match across checkpoint restore for container: "agent"
```

---

## Step 3: Why This Failed (The Invariant)

1. When `runsc checkpoint` runs, the container's entire rootfs overlay structure, paths, and mounts are saved into `checkpoint.img`.
2. Upon restore, gVisor's validator verifies that the new container template matches the recorded image filesystem and mount topology.
3. Because `ubuntu-24.04:0.1.3` has different rootfs layers, directory inodes, and lacks the CUDA/PyTorch paths present in `pytorch:0.1.2`, gVisor flags `"Mounts" does not match` and halts restoration before running any code.

---

## Step 4: Clean Up Pod

Delete the failed cross-template pod:

```bash
kubectl delete pod -n gpu-gvisor-poc airuntime-gpu-cross-template --force --grace-period=0
```

Verify clean state:
```bash
kubectl get pod -n gpu-gvisor-poc airuntime-gpu-cross-template
```
*Expected output: `NotFound`*

