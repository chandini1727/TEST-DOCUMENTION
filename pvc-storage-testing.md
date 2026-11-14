# Test 04: Ceph Persistent Volume (PVC) Checkpoint and Restore

This runbook evaluates Ceph persistent storage synchronization across two distinct scenarios:
1. **Case 1: Live Checkpoint Without Pausing (Storage Drift Demonstrated)** — Checkpointing with `--leave-running` while disk writes continue, demonstrating storage drift.
2. **Case 2: Coordinated Checkpoint With Pause (Zero Drift Verified)** — Pausing container execution via `runsc pause` before snapshotting and cloning storage, verifying zero storage drift.

---

## Step 0: Clean Up Previous Resources & Release the GPU

Run this first to clean up any lingering pods, snapshot jobs, and PVCs:

```bash
# 1. Clean up pods from previous tests
kubectl delete pod -n gpu-gvisor-poc airuntime-gpu-workload airuntime-gpu-scaled-up airuntime-gpu-scaled-down airuntime-gpu-cross-template neev-gpu-ceph-workload neev-gpu-ceph-restore-clone neev-gpu-ceph-restore-paused --force --grace-period=0 2>/dev/null || true

# 2. Delete existing snapshotjobs
kubectl delete snapshotjob -n gpu-gvisor-poc sj-ceph-live-snapshot sj-ceph-paused-snapshot 2>/dev/null || true

# 3. Delete all PVCs (including source and clones)
kubectl delete pvc -n gpu-gvisor-poc gpu-ceph-pvc-source gpu-ceph-pvc-clone-1 gpu-ceph-pvc-clone-2 --force --grace-period=0 2>/dev/null || true

kubectl get pods,pvc,snapshotjob -n gpu-gvisor-poc
```

---

## Step 1: Create Source PVC & Start Ceph GPU Workload

### 1.1 Create the Source Ceph PVC
```bash
cat << 'EOF' | kubectl apply -f -
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: gpu-ceph-pvc-source
  namespace: gpu-gvisor-poc
spec:
  storageClassName: neevai-ceph-rbd
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 5Gi
EOF
```

Verify PVC is `Bound`:
```bash
kubectl get pvc -n gpu-gvisor-poc gpu-ceph-pvc-source
```

---

### 1.2 Start the Ceph GPU Workload Pod
```bash
cat << 'EOF' | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: neev-gpu-ceph-workload
  namespace: gpu-gvisor-poc
spec:
  runtimeClassName: runsc-root-self
  restartPolicy: Never
  nodeSelector:
    kubernetes.io/hostname: stg-nc-workload1-worker-3
  resourceClaims:
  - name: gpu
    resourceClaimName: gpu-claim
  volumes:
  - name: ceph-workspace
    persistentVolumeClaim:
      claimName: gpu-ceph-pvc-source
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
      cat << 'PYEOF' > /workspace/ceph_workload.py
      import os, sys, time, torch

      log_path = "/workspace/workload.log"
      token_path = "/workspace/ceph-token.txt"

      with open(token_path, "w") as f:
          f.write("ceph-rbd-token-999888\n")

      print("=== PyTorch Ceph RBD Workload Starting ===", flush=True)
      dev = torch.device("cuda:0")
      state = torch.zeros(64 * 1024 * 1024, dtype=torch.float32, device=dev)
      tick = 0

      with open(log_path, "a") as f:
          f.write(f"=== Session Started: PID={os.getpid()} ===\n")

      while True:
          tick += 1
          state += 1.0
          torch.cuda.synchronize()
          sample = float(state[0].item())
          allocated_mb = torch.cuda.memory_allocated(dev) / (1024 * 1024)
          msg = f"[CEPH-GPU-WORKLOAD] tick={tick} sample={sample:.1f} VRAM={allocated_mb:.1f}MB\n"
          print(msg, end="", flush=True)
          with open(log_path, "a") as f:
              f.write(msg)
              f.flush()
          time.sleep(1)
      PYEOF

      exec python3 -u /workspace/ceph_workload.py
    env:
    - name: NVIDIA_VISIBLE_DEVICES
      value: void
    volumeMounts:
    - name: ceph-workspace
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

Stream logs to confirm it is running and writing ticks to Ceph storage:
```bash
kubectl logs -f -n gpu-gvisor-poc neev-gpu-ceph-workload
```
*(Press `Ctrl+C` after seeing 3–5 ticks).*

---

## CASE 1: Live Checkpoint (Demonstrating Storage Drift)

### 2.1 Trigger Live Snapshot
```bash
cat << 'EOF' | kubectl apply -f -
apiVersion: sandbox.neevcloud.com/v1alpha1
kind: SnapshotJob
metadata:
  name: sj-ceph-live-snapshot
  namespace: gpu-gvisor-poc
spec:
  sandboxRef: neev-gpu-ceph-workload
  nodeName: stg-nc-workload1-worker-3
  mode: snapshot
  snapshotType: full
  runtimeClass: runsc-root-self
EOF
```

Wait until `Ready`:
```bash
kubectl get snapshotjob -n gpu-gvisor-poc sj-ceph-live-snapshot -w
```
*(Press `Ctrl+C` once `Ready`).*

Retrieve SHA:
```bash
SHA_LIVE=$(kubectl get snapshotjob -n gpu-gvisor-poc sj-ceph-live-snapshot -o jsonpath='{.status.sha256}')
echo "Live Snapshot SHA: $SHA_LIVE"
```

---

### 2.2 Clone Ceph PVC While Pod Continues Writing
```bash
cat << 'EOF' | kubectl apply -f -
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: gpu-ceph-pvc-clone-1
  namespace: gpu-gvisor-poc
spec:
  storageClassName: neevai-ceph-rbd
  dataSource:
    name: gpu-ceph-pvc-source
    kind: PersistentVolumeClaim
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 5Gi
EOF
```

Wait until `Bound`:
```bash
kubectl get pvc -n gpu-gvisor-poc gpu-ceph-pvc-clone-1 -w
```
*(Press `Ctrl+C` once `Bound`).*

---

### 2.3 Delete Live Pod to Free GPU
```bash
kubectl delete pod -n gpu-gvisor-poc neev-gpu-ceph-workload --force --grace-period=0
```

---

### 2.4 Deploy Restored Pod on Cloned PVC
```bash
cat << 'RESTORE_EOF' | sed "s/PASTE_SHA_HERE/$SHA_LIVE/g" | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: neev-gpu-ceph-restore-clone
  namespace: gpu-gvisor-poc
  annotations:
    dev.gvisor.internal.restore.host-image-path: /var/lib/neev-checkpoint-scratch/gpu-gvisor-poc/sj-ceph-live-snapshot/checkpoint
spec:
  runtimeClassName: runsc-root-self
  restartPolicy: Never
  nodeSelector:
    kubernetes.io/hostname: stg-nc-workload1-worker-3
  resourceClaims:
  - name: gpu
    resourceClaimName: gpu-claim
  volumes:
  - name: ceph-workspace
    persistentVolumeClaim:
      claimName: gpu-ceph-pvc-clone-1
  - name: cuda-checkpoint
    hostPath:
      path: /opt/cuda-checkpoint
      type: Directory
  - name: ckpt-gate
    csi:
      driver: checkpoint.csi.neev.io
      volumeAttributes:
        restoreId: sj-ceph-live-snapshot
        objectRef: s3://sandbox-snapshots/gpu-gvisor-poc/sj-ceph-live-snapshot.tar.zst
        sha256: "PASTE_SHA_HERE"
  containers:
  - name: cuda
    image: harbor.neev.work/test-template/pytorch-v2.8.0-ubuntu-24.04:0.1.2
    command:
    - /bin/bash
    - -c
    - |
      export PATH=/opt/venv/bin:$PATH
      cat << 'PYEOF' > /workspace/ceph_workload.py
      import os, sys, time, torch

      log_path = "/workspace/workload.log"
      token_path = "/workspace/ceph-token.txt"

      with open(token_path, "w") as f:
          f.write("ceph-rbd-token-999888\n")

      print("=== PyTorch Ceph RBD Workload Starting ===", flush=True)
      dev = torch.device("cuda:0")
      state = torch.zeros(64 * 1024 * 1024, dtype=torch.float32, device=dev)
      tick = 0

      with open(log_path, "a") as f:
          f.write(f"=== Session Started: PID={os.getpid()} ===\n")

      while True:
          tick += 1
          state += 1.0
          torch.cuda.synchronize()
          sample = float(state[0].item())
          allocated_mb = torch.cuda.memory_allocated(dev) / (1024 * 1024)
          msg = f"[CEPH-GPU-WORKLOAD] tick={tick} sample={sample:.1f} VRAM={allocated_mb:.1f}MB\n"
          print(msg, end="", flush=True)
          with open(log_path, "a") as f:
              f.write(msg)
              f.flush()
          time.sleep(1)
      PYEOF

      exec python3 -u /workspace/ceph_workload.py
    env:
    - name: NVIDIA_VISIBLE_DEVICES
      value: void
    volumeMounts:
    - name: ceph-workspace
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
RESTORE_EOF
```

Wait until Running:
```bash
kubectl get pod -n gpu-gvisor-poc neev-gpu-ceph-restore-clone -w
```
*(Press `Ctrl+C` once `Running`).*

---

### 2.5 Observe Storage Drift
Check the restored disk log file:
```bash
kubectl exec -n gpu-gvisor-poc neev-gpu-ceph-restore-clone -- tail -n 25 /workspace/workload.log
```

**Observed Drift Phenomenon:**
The disk file will show ticks advancing (e.g., up to `tick=46`), followed by a jump backwards to the snapshot's in-memory tick (e.g., `tick=23`), because the disk kept advancing while memory was captured earlier:
```text
[CEPH-GPU-WORKLOAD] tick=45 sample=45.0 VRAM=256.0MB
[CEPH-GPU-WORKLOAD] tick=46 sample=46.0 VRAM=256.0MB
[CEPH-GPU-WORKLOAD] tick=23 sample=23.0 VRAM=256.0MB   <-- JUMPED BACKWARDS (STORAGE DRIFT!)
[CEPH-GPU-WORKLOAD] tick=24 sample=24.0 VRAM=256.0MB
```

Clean up Case 1 restore pod:
```bash
kubectl delete pod -n gpu-gvisor-poc neev-gpu-ceph-restore-clone --force --grace-period=0
```

---

## CASE 2: Coordinated Checkpoint with Pause (Zero Drift Verified)

### 3.1 Clean Up Case 1 Artifacts & Reset Source PVC
Delete previous pods, snapshot jobs, and PVCs so Case 2 starts with a fresh empty volume:

```bash
kubectl delete pod -n gpu-gvisor-poc neev-gpu-ceph-restore-clone --force --grace-period=0 2>/dev/null || true
kubectl delete snapshotjob -n gpu-gvisor-poc sj-ceph-live-snapshot 2>/dev/null || true
kubectl delete pvc -n gpu-gvisor-poc gpu-ceph-pvc-clone-1 gpu-ceph-pvc-source --force --grace-period=0 2>/dev/null || true
kubectl get pods,pvc,snapshotjob -n gpu-gvisor-poc
```

---

### 3.2 Create Fresh Source Ceph PVC for Case 2
```bash
cat << 'EOF' | kubectl apply -f -
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: gpu-ceph-pvc-source
  namespace: gpu-gvisor-poc
spec:
  storageClassName: neevai-ceph-rbd
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 5Gi
EOF
```

Wait until `Bound`:
```bash
kubectl get pvc -n gpu-gvisor-poc gpu-ceph-pvc-source -w
```
*(Press `Ctrl+C` once `Bound`).*

---

### 3.3 Start Fresh Ceph GPU Workload Pod (Ticks Start from 1)
```bash
cat << 'EOF' | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: neev-gpu-ceph-workload
  namespace: gpu-gvisor-poc
spec:
  runtimeClassName: runsc-root-self
  restartPolicy: Never
  nodeSelector:
    kubernetes.io/hostname: stg-nc-workload1-worker-3
  resourceClaims:
  - name: gpu
    resourceClaimName: gpu-claim
  volumes:
  - name: ceph-workspace
    persistentVolumeClaim:
      claimName: gpu-ceph-pvc-source
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
      cat << 'PYEOF' > /workspace/ceph_workload.py
      import os, sys, time, torch

      log_path = "/workspace/workload.log"
      token_path = "/workspace/ceph-token.txt"

      with open(token_path, "w") as f:
          f.write("ceph-rbd-token-999888\n")

      print("=== PyTorch Ceph RBD Workload Starting ===", flush=True)
      dev = torch.device("cuda:0")
      state = torch.zeros(64 * 1024 * 1024, dtype=torch.float32, device=dev)
      tick = 0

      with open(log_path, "a") as f:
          f.write(f"=== Session Started: PID={os.getpid()} ===\n")

      while True:
          tick += 1
          state += 1.0
          torch.cuda.synchronize()
          sample = float(state[0].item())
          allocated_mb = torch.cuda.memory_allocated(dev) / (1024 * 1024)
          msg = f"[CEPH-GPU-WORKLOAD] tick={tick} sample={sample:.1f} VRAM={allocated_mb:.1f}MB\n"
          print(msg, end="", flush=True)
          with open(log_path, "a") as f:
              f.write(msg)
              f.flush()
          time.sleep(1)
      PYEOF

      exec python3 -u /workspace/ceph_workload.py
    env:
    - name: NVIDIA_VISIBLE_DEVICES
      value: void
    volumeMounts:
    - name: ceph-workspace
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

Wait until it is running and writing ticks:
```bash
kubectl logs -f -n gpu-gvisor-poc neev-gpu-ceph-workload
```
*(Press `Ctrl+C` after seeing 3–5 ticks).*

---

### 3.4 Pause Container via `runsc pause` on Worker-3 to Freeze Disk Writes

SSH into worker-3 and pause the running container directly:

```bash
# 1. SSH into worker-3 from master-1
ssh root@stg-nc-workload1-worker-3

# 2. Find the container ID and pause it using runsc
CID=$(crictl ps --name cuda --state Running -q | head -n 1)
/usr/local/bin/runsc-20260905 --root /run/containerd/runsc/k8s.io pause $CID

# 3. Exit back to master-1
exit
```

*The container execution threads are now frozen. Disk writes to the Ceph volume stop immediately.*

---

### 3.5 Trigger Snapshot While Paused
```bash
cat << 'EOF' | kubectl apply -f -
apiVersion: sandbox.neevcloud.com/v1alpha1
kind: SnapshotJob
metadata:
  name: sj-ceph-paused-snapshot
  namespace: gpu-gvisor-poc
spec:
  sandboxRef: neev-gpu-ceph-workload
  nodeName: stg-nc-workload1-worker-3
  mode: snapshot
  snapshotType: full
  runtimeClass: runsc-root-self
EOF
```

Wait until `Ready`:
```bash
kubectl get snapshotjob -n gpu-gvisor-poc sj-ceph-paused-snapshot -w
```
*(Press `Ctrl+C` once `Ready`).*

Retrieve the paused snapshot SHA:
```bash
SHA_PAUSED=$(kubectl get snapshotjob -n gpu-gvisor-poc sj-ceph-paused-snapshot -o jsonpath='{.status.sha256}')
echo "Paused Snapshot SHA: $SHA_PAUSED"
```

---

### 3.6 Clone Ceph PVC While Paused
```bash
cat << 'EOF' | kubectl apply -f -
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: gpu-ceph-pvc-clone-2
  namespace: gpu-gvisor-poc
spec:
  storageClassName: neevai-ceph-rbd
  dataSource:
    name: gpu-ceph-pvc-source
    kind: PersistentVolumeClaim
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 5Gi
EOF
```

Wait until `Bound`:
```bash
kubectl get pvc -n gpu-gvisor-poc gpu-ceph-pvc-clone-2 -w
```
*(Press `Ctrl+C` once `Bound`).*

---

### 3.7 Delete the Source Workload Pod to Free GPU
```bash
kubectl delete pod -n gpu-gvisor-poc neev-gpu-ceph-workload --force --grace-period=0
```

---

### 3.8 Deploy Restored Pod from Paused Snapshot & Cloned PVC
```bash
cat << 'RESTORE_EOF' | sed "s/PASTE_SHA_HERE/$SHA_PAUSED/g" | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: neev-gpu-ceph-restore-paused
  namespace: gpu-gvisor-poc
  annotations:
    dev.gvisor.internal.restore.host-image-path: /var/lib/neev-checkpoint-scratch/gpu-gvisor-poc/sj-ceph-paused-snapshot/checkpoint
spec:
  runtimeClassName: runsc-root-self
  restartPolicy: Never
  nodeSelector:
    kubernetes.io/hostname: stg-nc-workload1-worker-3
  resourceClaims:
  - name: gpu
    resourceClaimName: gpu-claim
  volumes:
  - name: ceph-workspace
    persistentVolumeClaim:
      claimName: gpu-ceph-pvc-clone-2
  - name: cuda-checkpoint
    hostPath:
      path: /opt/cuda-checkpoint
      type: Directory
  - name: ckpt-gate
    csi:
      driver: checkpoint.csi.neev.io
      volumeAttributes:
        restoreId: sj-ceph-paused-snapshot
        objectRef: s3://sandbox-snapshots/gpu-gvisor-poc/sj-ceph-paused-snapshot.tar.zst
        sha256: "PASTE_SHA_HERE"
  containers:
  - name: cuda
    image: harbor.neev.work/test-template/pytorch-v2.8.0-ubuntu-24.04:0.1.2
    command:
    - /bin/bash
    - -c
    - |
      export PATH=/opt/venv/bin:$PATH
      cat << 'PYEOF' > /workspace/ceph_workload.py
      import os, sys, time, torch

      log_path = "/workspace/workload.log"
      token_path = "/workspace/ceph-token.txt"

      with open(token_path, "w") as f:
          f.write("ceph-rbd-token-999888\n")

      print("=== PyTorch Ceph RBD Workload Starting ===", flush=True)
      dev = torch.device("cuda:0")
      state = torch.zeros(64 * 1024 * 1024, dtype=torch.float32, device=dev)
      tick = 0

      with open(log_path, "a") as f:
          f.write(f"=== Session Started: PID={os.getpid()} ===\n")

      while True:
          tick += 1
          state += 1.0
          torch.cuda.synchronize()
          sample = float(state[0].item())
          allocated_mb = torch.cuda.memory_allocated(dev) / (1024 * 1024)
          msg = f"[CEPH-GPU-WORKLOAD] tick={tick} sample={sample:.1f} VRAM={allocated_mb:.1f}MB\n"
          print(msg, end="", flush=True)
          with open(log_path, "a") as f:
              f.write(msg)
              f.flush()
          time.sleep(1)
      PYEOF

      exec python3 -u /workspace/ceph_workload.py
    env:
    - name: NVIDIA_VISIBLE_DEVICES
      value: void
    volumeMounts:
    - name: ceph-workspace
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
RESTORE_EOF
```

Wait until Running:
```bash
kubectl get pod -n gpu-gvisor-poc neev-gpu-ceph-restore-paused -w
```
*(Press `Ctrl+C` once `Running`).*

---

### 3.9 Verify Zero Storage Drift
Inspect the disk log on the restored pod:

```bash
kubectl exec -n gpu-gvisor-poc neev-gpu-ceph-restore-paused -- tail -n 25 /workspace/workload.log
```

**Observed Zero Drift Output:**
```text
[CEPH-GPU-WORKLOAD] tick=28 sample=28.0 VRAM=256.0MB
[CEPH-GPU-WORKLOAD] tick=29 sample=29.0 VRAM=256.0MB
[CEPH-GPU-WORKLOAD] tick=30 sample=30.0 VRAM=256.0MB
[CEPH-GPU-WORKLOAD] tick=31 sample=31.0 VRAM=256.0MB   <-- EXACT CONTINUITY (ZERO DRIFT, ZERO GAPS!)
[CEPH-GPU-WORKLOAD] tick=32 sample=32.0 VRAM=256.0MB
[CEPH-GPU-WORKLOAD] tick=33 sample=33.0 VRAM=256.0MB
```

---

## Step 4: Final Cleanup

```bash
kubectl delete pod -n gpu-gvisor-poc neev-gpu-ceph-restore-paused --force --grace-period=0 2>/dev/null || true
kubectl delete pvc -n gpu-gvisor-poc gpu-ceph-pvc-clone-1 gpu-ceph-pvc-clone-2 gpu-ceph-pvc-source 2>/dev/null || true
kubectl delete snapshotjob -n gpu-gvisor-poc sj-ceph-live-snapshot sj-ceph-paused-snapshot 2>/dev/null || true
```

