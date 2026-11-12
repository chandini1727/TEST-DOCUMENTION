# Execution Steps: Test 12 - ComfyUI Checkpoint & Restore (with Exact Latency Timing & RAM Watermark)

Run these steps in your master node terminal (`stg-nc-workload1-master-1`). Every step is self-contained with inlined manifests written to `/tmp/*.yaml` to prevent SSH paste truncation.

---

## Step 0: Clean Up Previous Resources

```bash
# 1. Delete all pods in gpu-gvisor-poc
kubectl delete pods -n gpu-gvisor-poc --all --force --grace-period=0

# 2. Delete all snapshot jobs in gpu-gvisor-poc
kubectl delete snapshotjobs -n gpu-gvisor-poc --all --force --grace-period=0

# 3. Verify namespace is clean
kubectl get pods,snapshotjobs -n gpu-gvisor-poc
```

---

## Step 0.5: Create DRA ResourceClaim for GPU (`gpu-claim`)

```bash
cat << 'EOF' > /tmp/comfy-claim.yaml
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
kubectl apply -f /tmp/comfy-claim.yaml
```

---

## Step 1: Deploy Source ComfyUI Pod on RTX 5090

```bash
cat << 'EOF' > /tmp/comfy-pod.yaml
apiVersion: v1
kind: Pod
metadata:
  name: airuntime-gpu-comfyui
  namespace: gpu-gvisor-poc
  labels:
    app: comfyui
    platform.neev.ai/sandbox-id: comfyui-test-01
spec:
  runtimeClassName: runsc-root-self
  restartPolicy: Never
  nodeSelector:
    kubernetes.io/hostname: stg-nc-workload1-worker-3
  tolerations:
  - key: nvidia.com/gpu
    operator: Exists
    effect: NoSchedule
  resourceClaims:
  - name: gpu
    resourceClaimName: gpu-claim
  volumes:
  - name: cuda-checkpoint
    hostPath:
      path: /opt/cuda-checkpoint
      type: Directory
  - name: ckpt-gate
    csi:
      driver: checkpoint.csi.neev.io
  containers:
  - name: comfyui
    image: harbor.neev.work/gpu/comfyui:0.1.5
    env:
    - name: NVIDIA_VISIBLE_DEVICES
      value: void
    - name: COMFYUI_ENABLE_MANAGER
      value: "true"
    - name: JUPYTER_PASSWORD
      value: "neevcloud"
    ports:
    - containerPort: 3000
      name: comfyui
    - containerPort: 8888
      name: jupyter
    - containerPort: 22
      name: ssh
    volumeMounts:
    - name: cuda-checkpoint
      mountPath: /opt/cuda-checkpoint
      readOnly: true
    - name: ckpt-gate
      mountPath: /var/run/neev-checkpoint-gate
      readOnly: true
    resources:
      requests:
        cpu: "4"
        memory: "16Gi"
      limits:
        cpu: "8"
        memory: "32Gi"
      claims:
      - name: gpu
EOF
kubectl apply -f /tmp/comfy-pod.yaml
```

**Wait for the pod to reach `Running`:**
```bash
kubectl get pod -n gpu-gvisor-poc airuntime-gpu-comfyui -w
```
*(Once Running, press `Ctrl+C`).*

---

## Step 2: Verify Health & Inject RAM Watermark Proof

```bash
POD_IP=$(kubectl get pod -n gpu-gvisor-poc airuntime-gpu-comfyui -o jsonpath='{.status.podIP}')
echo "ComfyUI Pod IP: $POD_IP"

# 1. Health check ComfyUI HTTP Web server
curl -s -o /dev/null -w "HTTP Response: %{http_code}\n" "http://$POD_IP:3000/"
# Expected: HTTP Response: 200

# 2. Query GPU System Stats
curl -s "http://$POD_IP:3000/system_stats" | jq .devices

# 3. Write an in-memory watermark into RAM (/dev/shm)
kubectl exec -n gpu-gvisor-poc airuntime-gpu-comfyui -c comfyui -- sh -c "echo 'CONFIRMED_RESTORE_$(date +%s)' > /dev/shm/proof.txt"

# Confirm the file is currently in RAM:
kubectl exec -n gpu-gvisor-poc airuntime-gpu-comfyui -c comfyui -- cat /dev/shm/proof.txt
```

---

## Step 3: Trigger Snapshot & Measure Exact Snapshot Time

This block automatically creates the `SnapshotJob`, polls until it reaches `Ready`, and prints the **exact snapshot capture duration in seconds and milliseconds**:

```bash
echo "=== Starting Snapshot Timing ==="
START_SNAP=$(date +%s%N)

cat << 'EOF' > /tmp/sj-comfyui.yaml
apiVersion: sandbox.neevcloud.com/v1alpha1
kind: SnapshotJob
metadata:
  name: sj-comfyui-02
  namespace: gpu-gvisor-poc
spec:
  sandboxRef: airuntime-gpu-comfyui
  nodeName: stg-nc-workload1-worker-3
  mode: snapshot
  snapshotType: full
  runtimeClass: runsc-root-self
EOF
kubectl apply -f /tmp/sj-comfyui.yaml

echo "Waiting for SnapshotJob sj-comfyui-02 to reach Ready..."
while true; do
  PHASE=$(kubectl get snapshotjob -n gpu-gvisor-poc sj-comfyui-02 -o jsonpath='{.status.phase}' 2>/dev/null)
  if [ "$PHASE" == "Ready" ]; then
    END_SNAP=$(date +%s%N)
    DURATION_MS=$(( (END_SNAP - START_SNAP) / 1000000 ))
    DURATION_S=$(awk "BEGIN {printf \"%.2f\", $DURATION_MS / 1000}")
    echo "=========================================================="
    echo ">> Snapshot Status: READY"
    echo ">> Actual Real Snapshot Duration: ${DURATION_S}s (${DURATION_MS} ms)"
    echo "=========================================================="
    break
  elif [ "$PHASE" == "Failed" ]; then
    echo "Snapshot Failed!"
    kubectl get snapshotjob -n gpu-gvisor-poc sj-comfyui-02 -o yaml
    break
  fi
  sleep 0.5
done
```

---

## Step 4: Delete Pod and Release the Physical GPU

```bash
kubectl delete pod -n gpu-gvisor-poc airuntime-gpu-comfyui --force --grace-period=0
```

---

## Step 5: Restore ComfyUI & Measure Exact Restore Latency

This block applies the restored pod, measures **Time to Running** (S3 fetch + `runsc restore`), and measures **Time to First HTTP 200 response**:

```bash
SHA=$(kubectl get snapshotjob -n gpu-gvisor-poc sj-comfyui-02 -o jsonpath='{.status.sha256}')
echo "Restoring from Snapshot SHA256: $SHA"

echo "=== Starting Restore Timing ==="
START_RESTORE=$(date +%s%N)

cat << EOF > /tmp/comfy-restore-pod.yaml
apiVersion: v1
kind: Pod
metadata:
  name: airuntime-gpu-comfyui-restored
  namespace: gpu-gvisor-poc
  labels:
    app: comfyui
    platform.neev.ai/sandbox-id: comfyui-test-01
  annotations:
    dev.gvisor.internal.restore.host-image-path: /var/lib/neev-checkpoint-scratch/gpu-gvisor-poc/sj-comfyui-02/checkpoint
spec:
  runtimeClassName: runsc-root-self
  restartPolicy: Never
  nodeSelector:
    kubernetes.io/hostname: stg-nc-workload1-worker-3
  tolerations:
  - key: nvidia.com/gpu
    operator: Exists
    effect: NoSchedule
  resourceClaims:
  - name: gpu
    resourceClaimName: gpu-claim
  volumes:
  - name: cuda-checkpoint
    hostPath:
      path: /opt/cuda-checkpoint
      type: Directory
  - name: ckpt-gate
    csi:
      driver: checkpoint.csi.neev.io
      volumeAttributes:
        restoreId: sj-comfyui-02
        objectRef: s3://sandbox-snapshots/gpu-gvisor-poc/sj-comfyui-02.tar.zst
        sha256: "${SHA}"
  containers:
  - name: comfyui
    image: harbor.neev.work/gpu/comfyui:0.1.5
    env:
    - name: NVIDIA_VISIBLE_DEVICES
      value: void
    - name: COMFYUI_ENABLE_MANAGER
      value: "true"
    - name: JUPYTER_PASSWORD
      value: "neevcloud"
    ports:
    - containerPort: 3000
      name: comfyui
    - containerPort: 8888
      name: jupyter
    - containerPort: 22
      name: ssh
    volumeMounts:
    - name: cuda-checkpoint
      mountPath: /opt/cuda-checkpoint
      readOnly: true
    - name: ckpt-gate
      mountPath: /var/run/neev-checkpoint-gate
      readOnly: true
    resources:
      requests:
        cpu: "4"
        memory: "16Gi"
      limits:
        cpu: "8"
        memory: "32Gi"
      claims:
      - name: gpu
EOF
kubectl apply -f /tmp/comfy-restore-pod.yaml

echo "Waiting for airuntime-gpu-comfyui-restored to reach Running..."
while true; do
  STATUS=$(kubectl get pod -n gpu-gvisor-poc airuntime-gpu-comfyui-restored -o jsonpath='{.status.phase}' 2>/dev/null)
  if [ "$STATUS" == "Running" ]; then
    END_RUNNING=$(date +%s%N)
    RUNNING_MS=$(( (END_RUNNING - START_RESTORE) / 1000000 ))
    RUNNING_S=$(awk "BEGIN {printf \"%.2f\", $RUNNING_MS / 1000}")
    echo "=========================================================="
    echo ">> Pod State: RUNNING"
    echo ">> Time to Running (S3 Download + Restore): ${RUNNING_S}s (${RUNNING_MS} ms)"
    echo "=========================================================="
    break
  fi
  sleep 0.2
done

# Measure Time to First HTTP 200 response
RESTORED_IP=$(kubectl get pod -n gpu-gvisor-poc airuntime-gpu-comfyui-restored -o jsonpath='{.status.podIP}')
echo "Waiting for HTTP Web Service readiness at http://$RESTORED_IP:3000/ ..."
while true; do
  CODE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 1 "http://$RESTORED_IP:3000/" 2>/dev/null || true)
  if [ "$CODE" == "200" ]; then
    END_HTTP=$(date +%s%N)
    HTTP_MS=$(( (END_HTTP - START_RESTORE) / 1000000 ))
    HTTP_S=$(awk "BEGIN {printf \"%.2f\", $HTTP_MS / 1000}")
    echo "=========================================================="
    echo ">> ComfyUI Web Server READY (HTTP 200)"
    echo ">> Total Time to HTTP Service Available: ${HTTP_S}s (${HTTP_MS} ms)"
    echo "=========================================================="
    break
  fi
  sleep 0.2
done
```

---

## Step 6: Verify Service & Prove RAM Restoration

```bash
RESTORED_IP=$(kubectl get pod -n gpu-gvisor-poc airuntime-gpu-comfyui-restored -o jsonpath='{.status.podIP}')
echo "Restored Pod IP: $RESTORED_IP"

# 1. Health check HTTP Web server (immediate 200)
curl -s -o /dev/null -w "HTTP Response: %{http_code}\n" "http://$RESTORED_IP:3000/"

# 2. Verify RTX 5090 is attached
curl -s "http://$RESTORED_IP:3000/system_stats" | jq .devices

# 3. PROOF: Read the RAM watermark file from /dev/shm
kubectl exec -n gpu-gvisor-poc airuntime-gpu-comfyui-restored -c comfyui -- cat /dev/shm/proof.txt

# 4. Check restored logs (shows no reboot logs)
kubectl logs -n gpu-gvisor-poc airuntime-gpu-comfyui-restored
```
