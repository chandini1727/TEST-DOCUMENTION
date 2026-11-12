# Test 09: Change GPU Count on Restore (1 GPU -> 2 GPUs)

This scenario evaluates whether a container checkpoint created with **1 GPU** can be restored with a different DRA claim requesting **2 GPUs** (dynamic GPU scaling on restore).

---

## Step 0: Clean Up Previous Resources

Run on Master Node (`stg-nc-workload1-master-1`):

```bash
kubectl delete pod -n gpu-gvisor-poc airuntime-gpu-scale-source airuntime-gpu-scale-restored-2gpu --force --grace-period=0 2>/dev/null || true
kubectl delete snapshotjob -n gpu-gvisor-poc sj-scale-gpu-01 --force --grace-period=0 2>/dev/null || true
```

---

## Step 1: Create 1-GPU and 2-GPU DRA ResourceClaims

```bash
cat << 'EOF' | kubectl apply -f -
apiVersion: resource.k8s.io/v1
kind: ResourceClaim
metadata:
  name: gpu-claim-scale-1gpu
  namespace: gpu-gvisor-poc
spec:
  devices:
    requests:
    - name: gpu
      exactly:
        allocationMode: ExactCount
        count: 1
        deviceClassName: gpu.nvidia.com
---
apiVersion: resource.k8s.io/v1
kind: ResourceClaim
metadata:
  name: gpu-claim-scale-2gpu
  namespace: gpu-gvisor-poc
spec:
  devices:
    requests:
    - name: gpu
      exactly:
        allocationMode: ExactCount
        count: 2
        deviceClassName: gpu.nvidia.com
EOF

# Verify both claims are created
kubectl get resourceclaim -n gpu-gvisor-poc gpu-claim-scale-1gpu gpu-claim-scale-2gpu
```

---

## Step 2: Deploy Source Pod with 1 GPU

```bash
kubectl apply -f poc/gvisor-gpu/tests/09-change-gpu-count-restore/pod-1gpu.yaml
```

**Wait for pod to reach `Running`:**
```bash
kubectl get pod -n gpu-gvisor-poc airuntime-gpu-scale-source -w
```
*(Once Running, press `Ctrl+C`)*

**Verify 1-GPU workload:**
```bash
kubectl logs -f -n gpu-gvisor-poc airuntime-gpu-scale-source
```
*(Confirm output lines like: `[SCALE-TEST] Initial GPU count: 1`, `[SCALE-TEST] tick=... sample=... VRAM=256.0MB`. Then press `Ctrl+C`)*

---

## Step 3: Capture Snapshot via SnapshotJob

```bash
cat << 'EOF' | kubectl apply -f -
apiVersion: sandbox.neevcloud.com/v1alpha1
kind: SnapshotJob
metadata:
  name: sj-gpu-scale-01
  namespace: gpu-gvisor-poc
spec:
  sandboxRef: airuntime-gpu-scale-source
  nodeName: stg-nc-workload1-worker-3
  mode: snapshot
  snapshotType: full
  runtimeClass: runsc-root-self
EOF
```

**Watch SnapshotJob until `Ready`:**
```bash
kubectl get snapshotjob -n gpu-gvisor-poc sj-scale-gpu-01 -w
```
*(Once Ready, press `Ctrl+C`)*

---

## Step 4: Delete Source Pod

Release the 1-GPU claim:
```bash
kubectl delete pod -n gpu-gvisor-poc airuntime-gpu-scale-source --force --grace-period=0
```

---

## Step 5: Restore with 2-GPU Claim

Fetch the snapshot SHA and apply the restore pod attached to `gpu-claim-scale-2gpu`:

```bash
SHA=$(kubectl get snapshotjob -n gpu-gvisor-poc sj-scale-gpu-01 -o jsonpath='{.status.sha256}')
echo "Restoring from Snapshot SHA256: $SHA"

sed "s/PASTE_SHA_HERE/$SHA/g" poc/gvisor-gpu/tests/09-change-gpu-count-restore/restore-scale-up-2gpu.yaml | kubectl apply -f -
```

**Watch the restored pod status:**
```bash
kubectl get pod -n gpu-gvisor-poc airuntime-gpu-scale-restored-2gpu -w
```

---

## Step 6: Inspect Result and Failure Boundary

Check pod events and logs:
```bash
kubectl describe pod -n gpu-gvisor-poc airuntime-gpu-scale-restored-2gpu | tail -n 25
kubectl logs -n gpu-gvisor-poc airuntime-gpu-scale-restored-2gpu
```

**Expected Boundary Analysis:**
- If rejected by gVisor / nvproxy: `runsc restore` fails during CDI / device reconciliation because the restored container environment has different GPU device nodes (`/dev/nvidia1` introduced) than the checkpoint image.
- If allowed to start: Observe whether PyTorch resumes with or without recognizing the second GPU.

---

## Step 7: Teardown & Post-Test Cleanup

```bash
kubectl delete pod -n gpu-gvisor-poc airuntime-gpu-scale-restored-2gpu --force --grace-period=0 2>/dev/null || true
kubectl delete snapshotjob -n gpu-gvisor-poc sj-scale-gpu-01 --force --grace-period=0 2>/dev/null || true
kubectl delete resourceclaim -n gpu-gvisor-poc gpu-claim-scale-1gpu gpu-claim-scale-2gpu 2>/dev/null || true
```
