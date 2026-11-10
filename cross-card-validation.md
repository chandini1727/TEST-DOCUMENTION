# Step-by-Step Runbook: Test 06 Cross-Card and DRA Claim Invariant

This runbook tests whether a GPU snapshot captured on physical Card 1 (`0000:04:00.0`) can be restored onto a different physical GPU card (Card 0: `0000:01:00.0`) on `root@stg-nc-workload1-master-1`.

---

## Step 0: Clean Up Previous Test State

```bash
# 1. Clean up pods from previous tests
kubectl delete pod -n gpu-gvisor-poc airuntime-gpu-workload airuntime-gpu-scaled-up airuntime-gpu-scaled-down airuntime-gpu-cross-template neev-gpu-ceph-workload neev-gpu-ceph-restore-clone neev-gpu-ceph-restore-paused neev-gpu-upgrade-source neev-gpu-upgrade-restore neev-gpu-card1-source neev-gpu-card0-restore --force --grace-period=0 2>/dev/null || true

# 2. Delete existing snapshotjobs
kubectl delete snapshotjob -n gpu-gvisor-poc sj-cross-card-01 2>/dev/null || true

# 3. Delete card0 resourceclaim if any
kubectl delete resourceclaim -n gpu-gvisor-poc gpu-claim-card0 2>/dev/null || true
```

---

## Step 1: Create DRA Claim for Card 0

Create the DRA `ResourceClaim` specifically requesting Card 0 (`0000:01:00.0`):

```bash
cat << 'EOF' | kubectl apply -f -
apiVersion: resource.k8s.io/v1
kind: ResourceClaim
metadata:
  name: gpu-claim-card0
  namespace: gpu-gvisor-poc
spec:
  devices:
    requests:
    - name: gpu
      deviceClassName: gpu.nvidia.com
      selectors:
      - cel:
          expression: "device.attributes['gpu.nvidia.com']['resource.kubernetes.io/pciBusID'] == '0000:01:00.0' || device.attributes['gpu.nvidia.com'].pciAddress == '0000:01:00.0'"
EOF
```

Verify the claim is bound:
```bash
kubectl get resourceclaim -n gpu-gvisor-poc gpu-claim-card0
```

---

## Step 2: Deploy Workload on Card 1 (Source Pod)

Deploy the source workload claiming Card 1 via default claim `gpu-claim`:

```bash
cat << 'EOF' | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: neev-gpu-card1-source
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
    image: harbor.neev.work/test-template/pytorch-v2.8.0-ubuntu-24.04:0.1.2
    command:
    - /bin/bash
    - -c
    - |
      export PATH=/opt/venv/bin:$PATH
      cat << 'PYEOF' > /workspace/gpu_workload.py
      import os, sys, time, torch

      print("=== PyTorch Workload on Card 1 Starting ===", flush=True)
      dev = torch.device("cuda:0")
      state = torch.zeros(64 * 1024 * 1024, dtype=torch.float32, device=dev)
      tick = 0

      while True:
          tick += 1
          state += 1.0
          torch.cuda.synchronize()
          sample = float(state[0].item())
          allocated_mb = torch.cuda.memory_allocated(dev) / (1024 * 1024)
          print(f"[CARD-1] tick={tick} sample={sample:.1f} VRAM={allocated_mb:.1f}MB", flush=True)
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

Monitor logs to confirm PyTorch is actively executing on Card 1:
```bash
kubectl logs -f -n gpu-gvisor-poc neev-gpu-card1-source
```
*(Press `Ctrl+C` after 3–5 ticks)*.

---

## Step 3: Trigger Snapshot via SnapshotJob Reconciler

```bash
cat << 'EOF' | kubectl apply -f -
apiVersion: sandbox.neevcloud.com/v1alpha1
kind: SnapshotJob
metadata:
  name: sj-cross-card-01
  namespace: gpu-gvisor-poc
spec:
  sandboxRef: neev-gpu-card1-source
  nodeName: stg-nc-workload1-worker-3
  mode: snapshot
  snapshotType: full
  runtimeClass: runsc-root-self
EOF
```

Wait until `Ready`:
```bash
kubectl get snapshotjob -n gpu-gvisor-poc sj-cross-card-01 -w
```
*(Press `Ctrl+C` once Phase shows `Ready`)*.

Retrieve the snapshot SHA:
```bash
SHA=$(kubectl get snapshotjob -n gpu-gvisor-poc sj-cross-card-01 -o jsonpath='{.status.sha256}')
echo "Snapshot SHA256: $SHA"
```

---

## Step 4: Delete Source Pod

Delete the source pod to free Card 1:

```bash
kubectl delete pod -n gpu-gvisor-poc neev-gpu-card1-source --force --grace-period=0
```

---

## Step 5: Deploy Restore Pod Targeted at Card 0

Deploy the restored pod claiming Card 0 (`resourceClaimName: gpu-claim-card0`):

```bash
cat << 'RESTORE_EOF' | sed "s/PASTE_SHA_HERE/$SHA/g" | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: neev-gpu-card0-restore
  namespace: gpu-gvisor-poc
  annotations:
    dev.gvisor.internal.restore.host-image-path: /var/lib/neev-checkpoint-scratch/gpu-gvisor-poc/sj-cross-card-01/checkpoint
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
    resourceClaimName: gpu-claim-card0
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
        restoreId: sj-cross-card-01
        objectRef: s3://sandbox-snapshots/gpu-gvisor-poc/sj-cross-card-01.tar.zst
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

      print("=== PyTorch Workload on Card 1 Starting ===", flush=True)
      dev = torch.device("cuda:0")
      state = torch.zeros(64 * 1024 * 1024, dtype=torch.float32, device=dev)
      tick = 0

      while True:
          tick += 1
          state += 1.0
          torch.cuda.synchronize()
          sample = float(state[0].item())
          allocated_mb = torch.cuda.memory_allocated(dev) / (1024 * 1024)
          print(f"[CARD-1] tick={tick} sample={sample:.1f} VRAM={allocated_mb:.1f}MB", flush=True)
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

## Step 6: Inspect Pod Status & Hardware Affinity Rejection

Check pod status:
```bash
kubectl get pod -n gpu-gvisor-poc neev-gpu-card0-restore
```

*Expected Status: `StartError` or `CrashLoopBackOff`.*

Inspect Kubelet failure events:
```bash
kubectl describe pod -n gpu-gvisor-poc neev-gpu-card0-restore | grep -E "Warning|Failed|Error"
```

**Why This Failed (Hardware Affinity Invariant):**
gVisor `nvproxy` serializes open NVIDIA driver device file handles (`/dev/nvidia1`, PCI bus `0000:04:00.0`). When restored onto Card 0 (`/dev/nvidia0`, PCI bus `0000:01:00.0`), the CDI device nodes, minor device numbers, and physical hardware addresses mismatch, preventing the restored driver state from attaching to a different physical card.

---

## Step 7: Clean Up Test 06

```bash
kubectl delete pod -n gpu-gvisor-poc neev-gpu-card0-restore --force --grace-period=0
kubectl delete resourceclaim -n gpu-gvisor-poc gpu-claim-card0
kubectl delete snapshotjob -n gpu-gvisor-poc sj-cross-card-01
```

