# Test 06: Cross-Card GPU Snapshot and Restore Validation (Card 1 ➡️ Card 0)

This test evaluates whether a GPU workload checkpoint captured on physical Card 1 (`GPU-36df8b09-2435-5c42-7324-850c168ecc94`) can be restored onto a different physical GPU (Card 0: `GPU-860e5920-e4bc-f9fd-18c3-365260cc3a01`) using Kubernetes Dynamic Resource Allocation (DRA) and gVisor `nvproxy`.

---

## Step 0: Clean Up Previous Resources & Free the GPUs

Run this first to clean up any lingering pods, snapshot jobs, and claims:

```bash
# 1. Clean up pods from previous tests
kubectl delete pod -n gpu-gvisor-poc airuntime-gpu-workload airuntime-gpu-scaled-up airuntime-gpu-scaled-down airuntime-gpu-cross-template neev-gpu-ceph-workload neev-gpu-ceph-restore-clone neev-gpu-ceph-restore-paused neev-gpu-upgrade-source neev-gpu-upgrade-restore neev-gpu-card1-source neev-gpu-card0-restore --force --grace-period=0 2>/dev/null || true

# 2. Delete existing snapshot jobs
kubectl delete snapshotjob -n gpu-gvisor-poc sj-cross-card-01 --force --grace-period=0 2>/dev/null || true

# 3. Delete existing resource claims
kubectl delete resourceclaim -n gpu-gvisor-poc gpu-claim gpu-claim-card0 2>/dev/null || true

kubectl get pods,snapshotjob,resourceclaims -n gpu-gvisor-poc
```

---

## Step 1: Create DRA Claim for Source GPU (Card 1)

Create the DRA `ResourceClaim` targeting physical GPU Card 1:

```bash
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

Verify the claim is created:
```bash
kubectl get resourceclaim -n gpu-gvisor-poc gpu-claim
```

---

## Step 2: Deploy Workload on Card 1 (Source Pod)

Deploy the source workload claiming Card 1:

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
*(Press `Ctrl+C` after seeing 5–10 ticks).*

---

## Step 3: Trigger Snapshot via SnapshotJob Reconciler

Trigger the snapshot targeting `neev-gpu-card1-source`:

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
*(Press `Ctrl+C` once Phase shows `Ready`).*

Retrieve the snapshot SHA256 checksum:
```bash
SHA=$(kubectl get snapshotjob -n gpu-gvisor-poc sj-cross-card-01 -o jsonpath='{.status.sha256}')
echo "Snapshot SHA256: $SHA"
```

---

## Step 4: Delete Source Pod & Release Card 1

Delete the source pod and its associated claim to free physical Card 1:

```bash
kubectl delete pod -n gpu-gvisor-poc neev-gpu-card1-source --force --grace-period=0
kubectl delete resourceclaim -n gpu-gvisor-poc gpu-claim
```

---

## Step 5: Create DRA Claim for Target GPU (Card 0)

Create a new ResourceClaim targeting physical GPU Card 0:

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
      exactly:
        allocationMode: ExactCount
        count: 1
        deviceClassName: gpu.nvidia.com
        selectors:
        - cel:
            expression: "has(device.attributes['gpu.nvidia.com'].uuid) && device.attributes['gpu.nvidia.com'].uuid == 'GPU-860e5920-e4bc-f9fd-18c3-365260cc3a01'"
EOF
```

Verify Card 0 claim is created:
```bash
kubectl get resourceclaim -n gpu-gvisor-poc gpu-claim-card0
```

---

## Step 6: Deploy Restore Pod Targeted at Card 0

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

## Step 7: Inspect Pod Status & Hardware Affinity Rejection

Check pod status:
```bash
kubectl get pod -n gpu-gvisor-poc neev-gpu-card0-restore
```
*Observed Status: `StartError`*

Inspect Kubelet failure events:
```bash
kubectl describe pod -n gpu-gvisor-poc neev-gpu-card0-restore | tail -n 25
```

### Observed Error Event:
```text
Warning  Failed  kubelet  spec.containers{cuda}: Error: failed to start containerd task "cuda": 
OCI runtime restore failed: starting container: starting sub-container: 
failed to get nvproxy device IDs: NV0000_CTRL_CMD_GPU_GET_ID_INFO for minor=0 gpuId=0x100 failed: status=0x1f
```

### Technical Root Cause (Hardware Affinity Invariant):
* **Why This Failed:** During the checkpoint on Card 1, gVisor's `nvproxy` serialized the exact hardware minor device number, PCI address, and GPU ID info of Card 1.
* When restoring onto Card 0, `nvproxy` queries the kernel for the serialized GPU ID and receives `status=0x1f` (Device Not Found / ID Mismatch) because the physical device handles point to Card 1, not Card 0.
* **Key Finding:** GPU snapshots have **strict physical hardware affinity**. A workload snapshotted on Card 1 must be restored onto Card 1; cross-card restoration across different physical GPUs is strictly rejected by the driver runtime.

---

## Step 8: Final Cleanup

```bash
kubectl delete pod -n gpu-gvisor-poc neev-gpu-card0-restore --force --grace-period=0 2>/dev/null || true
kubectl delete resourceclaim -n gpu-gvisor-poc gpu-claim-card0 2>/dev/null || true
kubectl delete snapshotjob -n gpu-gvisor-poc sj-cross-card-01 2>/dev/null || true
```

