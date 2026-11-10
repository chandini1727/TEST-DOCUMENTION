# Step-by-Step Runbook: Test 05 Template Version Upgrade (v0.1.1 ➡️ v0.1.2)

This runbook contains all exact commands to execute and verify that a workload snapshot captured on an earlier container template (`pytorch:0.1.1`) can be restored onto an upgraded patch version (`pytorch:0.1.2`) on `root@stg-nc-workload1-master-1`.

---

## Step 0: Clean Up Previous Test State

```bash
# 1. Clean up pods from previous tests
kubectl delete pod -n gpu-gvisor-poc airuntime-gpu-workload airuntime-gpu-scaled-up airuntime-gpu-scaled-down airuntime-gpu-cross-template neev-gpu-ceph-workload neev-gpu-ceph-restore-clone neev-gpu-ceph-restore-paused neev-gpu-upgrade-source neev-gpu-upgrade-restore --force --grace-period=0 2>/dev/null || true

# 2. Delete any old upgrade snapshotjobs
kubectl delete snapshotjob -n gpu-gvisor-poc sj-version-upgrade 2>/dev/null || true
```

---

## Step 1: Deploy Source Pod on Template v0.1.1

Deploy the source pod using image `harbor.neev.work/test-template/pytorch-v2.8.0-ubuntu-24.04:0.1.1`:

```bash
cat << 'EOF' | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: neev-gpu-upgrade-source
  namespace: gpu-gvisor-poc
  annotations:
    dev.gvisor.spec.mount.workspace.type: bind
    dev.gvisor.spec.mount.workspace.share: pod
    dev.gvisor.spec.mount.workspace.options: size=1073741824
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
  containers:
  - name: cuda
    image: harbor.neev.work/test-template/pytorch-v2.8.0-ubuntu-24.04:0.1.1
    command:
    - /bin/bash
    - -c
    - |
      export PATH=/opt/venv/bin:$PATH
      cat << 'PYEOF' > /workspace/gpu_workload.py
      import os, sys, time, torch

      print("=== PyTorch Workload (Source: v0.1.1) Starting ===", flush=True)
      dev = torch.device("cuda:0")
      state = torch.zeros(64 * 1024 * 1024, dtype=torch.float32, device=dev)
      tick = 0

      while True:
          tick += 1
          state += 1.0
          torch.cuda.synchronize()
          sample = float(state[0].item())
          allocated_mb = torch.cuda.memory_allocated(dev) / (1024 * 1024)
          print(f"[TEMPLATE-V0.1.1] tick={tick} sample={sample:.1f} VRAM={allocated_mb:.1f}MB", flush=True)
          time.sleep(1)
      PYEOF

      exec python3 -u /workspace/gpu_workload.py
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

Monitor logs to confirm the source workload is actively computing on the GPU:
```bash
kubectl logs -f -n gpu-gvisor-poc neev-gpu-upgrade-source
```
*(Press `Ctrl+C` after seeing 5–10 ticks)*.

---

## Step 2: Trigger Snapshot via SnapshotJob Reconciler

Trigger the snapshot targeting `neev-gpu-upgrade-source`:

```bash
cat << 'EOF' | kubectl apply -f -
apiVersion: sandbox.neevcloud.com/v1alpha1
kind: SnapshotJob
metadata:
  name: sj-version-upgrade
  namespace: gpu-gvisor-poc
spec:
  sandboxRef: neev-gpu-upgrade-source
  nodeName: stg-nc-workload1-worker-3
  mode: snapshot
  snapshotType: full
  runtimeClass: runsc-root-self
EOF
```

Wait until `Ready`:
```bash
kubectl get snapshotjob -n gpu-gvisor-poc sj-version-upgrade -w
```
*(Takes ~5–8 seconds, press `Ctrl+C` once Phase shows `Ready`)*.

Retrieve the snapshot SHA256 checksum:
```bash
SHA=$(kubectl get snapshotjob -n gpu-gvisor-poc sj-version-upgrade -o jsonpath='{.status.sha256}')
echo "Snapshot SHA256: $SHA"
```

---

## Step 3: Delete Source Pod to Release GPU Claim

```bash
kubectl delete pod -n gpu-gvisor-poc neev-gpu-upgrade-source --force --grace-period=0
```

Wait until terminated:
```bash
kubectl get pod -n gpu-gvisor-poc neev-gpu-upgrade-source
```
*Expected: `NotFound`*

---

## Step 4: Deploy Restore Pod on Upgraded Template v0.1.2

Deploy the restore pod referencing the upgraded template image `harbor.neev.work/test-template/pytorch-v2.8.0-ubuntu-24.04:0.1.2`:

```bash
cat << 'RESTORE_EOF' | sed "s/PASTE_SHA_HERE/$SHA/g" | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: neev-gpu-upgrade-restore
  namespace: gpu-gvisor-poc
  annotations:
    dev.gvisor.internal.restore.host-image-path: /var/lib/neev-checkpoint-scratch/gpu-gvisor-poc/sj-version-upgrade/checkpoint
    dev.gvisor.spec.mount.workspace.type: bind
    dev.gvisor.spec.mount.workspace.share: pod
    dev.gvisor.spec.mount.workspace.options: size=1073741824
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
        restoreId: sj-version-upgrade
        objectRef: s3://sandbox-snapshots/gpu-gvisor-poc/sj-version-upgrade.tar.zst
        sha256: "PASTE_SHA_HERE"
  containers:
  - name: cuda
    image: harbor.neev.work/test-template/pytorch-v2.8.0-ubuntu-24.04:0.1.2
    command:
    - /bin/bash
    - -c
    - |
      export PATH=/opt/venv/bin:$PATH
      cat << 'PYEOF' > /workspace/gpu_workload.py
      import os, sys, time, torch

      print("=== PyTorch Workload (Source: v0.1.1) Starting ===", flush=True)
      dev = torch.device("cuda:0")
      state = torch.zeros(64 * 1024 * 1024, dtype=torch.float32, device=dev)
      tick = 0

      while True:
          tick += 1
          state += 1.0
          torch.cuda.synchronize()
          sample = float(state[0].item())
          allocated_mb = torch.cuda.memory_allocated(dev) / (1024 * 1024)
          print(f"[TEMPLATE-V0.1.1] tick={tick} sample={sample:.1f} VRAM={allocated_mb:.1f}MB", flush=True)
          time.sleep(1)
      PYEOF

      exec python3 -u /workspace/gpu_workload.py
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
RESTORE_EOF
```

---

## Step 5: Verify Restored Workload Execution

Stream logs from the restored pod:

```bash
kubectl logs -f -n gpu-gvisor-poc neev-gpu-upgrade-restore
```

**Expected Restored Stream:**
```text
[TEMPLATE-V0.1.1] tick=35 sample=35.0 VRAM=256.0MB
[TEMPLATE-V0.1.1] tick=36 sample=36.0 VRAM=256.0MB
[TEMPLATE-V0.1.1] tick=37 sample=37.0 VRAM=256.0MB
[TEMPLATE-V0.1.1] tick=38 sample=38.0 VRAM=256.0MB
```

**Invariants Verified:**
1. The workload resumes inside the new image (`v0.1.2`) without restarting from `tick=1`.
2. PyTorch memory (256MB VRAM) and process state remain fully intact.
3. Patch/minor version template upgrades do not break gVisor checkpoint restore compatibility.

---

## Step 6: Clean Up Test 05

```bash
kubectl delete pod -n gpu-gvisor-poc neev-gpu-upgrade-restore --force --grace-period=0
kubectl delete snapshotjob -n gpu-gvisor-poc sj-version-upgrade
```

