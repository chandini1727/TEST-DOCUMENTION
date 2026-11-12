# Test 06: Cross-Card Snapshot and Restore (Card 1 -> Card 0)

Step-by-step instructions to reproduce the cross-card hardware invariant test.

---

## Step 0: Clean Up Previous Resources

```bash
kubectl delete pod -n gpu-gvisor-poc neev-gpu-card1-source neev-gpu-card0-restore --force --grace-period=0 2>/dev/null || true
kubectl delete snapshotjob -n gpu-gvisor-poc sj-cross-card-01 --force --grace-period=0 2>/dev/null || true
kubectl patch resourceclaim -n gpu-gvisor-poc gpu-claim -p '{"metadata":{"finalizers":[]}}' --type=merge 2>/dev/null || true
kubectl delete resourceclaim -n gpu-gvisor-poc gpu-claim --force --grace-period=0 2>/dev/null || true
```

---

## Step 1: Create DRA Claim for Card 1 (`0000:04:00.0`)

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

---

## Step 2: Deploy Source Pod on Card 1

```bash
kubectl apply -f poc/gvisor-gpu/tests/06-cross-card-dra/pod-card1.yaml
```

Wait until `Running`:
```bash
kubectl get pod -n gpu-gvisor-poc neev-gpu-card1-source -w
```

Verify logs:
```bash
kubectl logs -f -n gpu-gvisor-poc neev-gpu-card1-source
```
*(Confirm `[CARD-1] tick=... sample=... VRAM=256.0MB`)*

---

## Step 3: Trigger Snapshot via SnapshotJob

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

Wait until `PHASE` is `Ready`:
```bash
kubectl get snapshotjob -n gpu-gvisor-poc sj-cross-card-01 -w
```

---

## Step 4: Delete Source Pod and Release Claim

```bash
kubectl delete pod -n gpu-gvisor-poc neev-gpu-card1-source
kubectl delete resourceclaim -n gpu-gvisor-poc gpu-claim
```

---

## Step 5: Create Claim for Card 0 (`0000:01:00.0`)

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
            expression: "has(device.attributes['gpu.nvidia.com'].uuid) && device.attributes['gpu.nvidia.com'].uuid == 'GPU-860e5920-e4bc-f9fd-18c3-365260cc3a01'"
EOF
```

---

## Step 6: Deploy Restore Pod onto Card 0

```bash
SHA=$(kubectl get snapshotjob -n gpu-gvisor-poc sj-cross-card-01 -o jsonpath='{.status.sha256}')
echo "Restoring from SHA: $SHA"

sed "s/PASTE_SHA_HERE/$SHA/g" poc/gvisor-gpu/tests/06-cross-card-dra/restore-card0.yaml | kubectl apply -f -
```

Watch restored pod status:
```bash
kubectl get pod -n gpu-gvisor-poc neev-gpu-card0-restore -w
```

---

## Step 7: Observe Hardware Invariant in Pod Events

```bash
kubectl describe pod -n gpu-gvisor-poc neev-gpu-card0-restore | tail -n 25
```

**Expected Error Output:**
```text
failed to start containerd task "cuda": OCI runtime restore failed: starting container: starting sub-container: 
failed to get nvproxy device IDs: NV0000_CTRL_CMD_GPU_GET_ID_INFO for minor=0 gpuId=0x100 failed: status=0x1f
```

---

## Step 8: Post-Test Cleanup

```bash
kubectl delete pod -n gpu-gvisor-poc neev-gpu-card0-restore --force --grace-period=0 2>/dev/null || true
kubectl delete snapshotjob -n gpu-gvisor-poc sj-cross-card-01 --force --grace-period=0 2>/dev/null || true
kubectl patch resourceclaim -n gpu-gvisor-poc gpu-claim -p '{"metadata":{"finalizers":[]}}' --type=merge 2>/dev/null || true
kubectl delete resourceclaim -n gpu-gvisor-poc gpu-claim --force --grace-period=0 2>/dev/null || true
```
