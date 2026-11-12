# Test 08: Multi-GPU Checkpoint & Restore (DRA Count > 1)

This scenario tests checkpointing and restoring a container attached to **two or more GPUs** via Kubernetes DRA (`count: 2`).

---

## Step 0: Clean Up Previous Resources

Run on Master Node (`stg-nc-workload1-master-1`):

```bash
kubectl delete pod -n gpu-gvisor-poc airuntime-gpu-multi-source airuntime-gpu-multi-restored --force --grace-period=0 2>/dev/null || true
kubectl delete snapshotjob -n gpu-gvisor-poc sj-multi-gpu-01 --force --grace-period=0 2>/dev/null || true
```

---

## Step 1: Create 2-GPU DRA ResourceClaim

```bash
cat << 'EOF' | kubectl apply -f -
apiVersion: resource.k8s.io/v1
kind: ResourceClaim
metadata:
  name: gpu-claim-2gpu
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
```

Verify claim is created:
```bash
kubectl get resourceclaim -n gpu-gvisor-poc gpu-claim-2gpu
```

---

## Step 2: Deploy Source Multi-GPU Pod

```bash
kubectl apply -f poc/gvisor-gpu/tests/08-multi-gpu-dra/pod.yaml
```

**Wait for pod to reach `Running`:**
```bash
kubectl get pod -n gpu-gvisor-poc airuntime-gpu-multi-source -w
```
*(Once Running, press `Ctrl+C`)*

**Verify PyTorch recognizes both GPUs and computes on both:**
```bash
kubectl logs -f -n gpu-gvisor-poc airuntime-gpu-multi-source
```
*(Confirm output lines like:*
`[MULTI-GPU] Detected 2 CUDA device(s)`
`[MULTI-GPU] GPU0: tick=... val=... VRAM=256.0MB | GPU1: tick=... val=... VRAM=256.0MB`
*Then press `Ctrl+C`)*

---

## Step 3: Capture Snapshot via SnapshotJob

```bash
cat << 'EOF' | kubectl apply -f -
apiVersion: sandbox.neevcloud.com/v1alpha1
kind: SnapshotJob
metadata:
  name: sj-multi-gpu-01
  namespace: gpu-gvisor-poc
spec:
  sandboxRef: airuntime-gpu-multi-source
  nodeName: stg-nc-workload1-worker-3
  mode: snapshot
  snapshotType: full
  runtimeClass: runsc-root-self
EOF
```

**Watch SnapshotJob until `Ready`:**
```bash
kubectl get snapshotjob -n gpu-gvisor-poc sj-multi-gpu-01 -w
```
*(Once Ready, press `Ctrl+C`)*

---

## Step 4: Delete Source Pod

Release the allocated GPUs:
```bash
kubectl delete pod -n gpu-gvisor-poc airuntime-gpu-multi-source --force --grace-period=0
```

---

## Step 5: Restore Multi-GPU Pod

```bash
SHA=$(kubectl get snapshotjob -n gpu-gvisor-poc sj-multi-gpu-01 -o jsonpath='{.status.sha256}')
echo "Restoring from Snapshot SHA256: $SHA"

sed "s/PASTE_SHA_HERE/$SHA/g" poc/gvisor-gpu/tests/08-multi-gpu-dra/restore.yaml | kubectl apply -f -
```

**Watch the restored pod reach `Running`:**
```bash
kubectl get pod -n gpu-gvisor-poc airuntime-gpu-multi-restored -w
```

---

## Step 6: Verify Restored State & Continuity

Stream logs from the restored pod:
```bash
kubectl logs -f -n gpu-gvisor-poc airuntime-gpu-multi-restored
```

**Check for:**
1. State resumption from the exact `tick` and `val` where the snapshot occurred.
2. Both GPU0 and GPU1 active and advancing without CUDA error.

---

## Step 7: Teardown & Post-Test Cleanup

```bash
kubectl delete pod -n gpu-gvisor-poc airuntime-gpu-multi-restored --force --grace-period=0 2>/dev/null || true
kubectl delete snapshotjob -n gpu-gvisor-poc sj-multi-gpu-01 --force --grace-period=0 2>/dev/null || true
kubectl delete resourceclaim -n gpu-gvisor-poc gpu-claim-2gpu 2>/dev/null || true
```
