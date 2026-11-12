# Execution Steps: Test 11 - vLLM Inference & KV Cache State

Run these commands directly in your cluster master node terminal (`stg-nc-workload1-master-1`).

---

## Step 0: Clean Up Old Test Resources

```bash
# 1. Delete all pods in gpu-gvisor-poc
kubectl delete pods -n gpu-gvisor-poc --all --force --grace-period=0

# 2. Delete all snapshot jobs in gpu-gvisor-poc
kubectl delete snapshotjobs -n gpu-gvisor-poc --all --force --grace-period=0

# 3. Verify namespace is clean
kubectl get pods,snapshotjobs -n gpu-gvisor-poc
```

---

## Step 1: Deploy Single-Process vLLM Pod (GPT-2 on RTX 5090)

```bash
cat << 'EOF' | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: airuntime-gpu-vllm
  namespace: gpu-gvisor-poc
  labels:
    app: airuntime-gpu-vllm
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
  - name: vllm
    image: harbor.neev.work/test-template/vllm:0.1.5
    env:
    - name: VLLM_ENABLE_V1_MULTIPROCESSING
      value: "0"
    - name: NVIDIA_VISIBLE_DEVICES
      value: void
    command:
    - python3
    - -m
    - vllm.entrypoints.openai.api_server
    - --model
    - gpt2
    - --enforce-eager
    - --distributed-executor-backend
    - uni
    - --gpu-memory-utilization
    - "0.15"
    - --disable-custom-all-reduce
    - --port
    - "8000"
    - --host
    - "0.0.0.0"
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
        cpu: "4"
        memory: "16Gi"
      limits:
        cpu: "8"
        memory: "32Gi"
      claims:
      - name: gpu
EOF

echo "Waiting for vLLM pod to be Ready..."
kubectl wait --namespace gpu-gvisor-poc --for=condition=Ready pod/airuntime-gpu-vllm --timeout=180s
```

---

## Step 2: Warm the KV Cache with a Completion Request

```bash
POD_IP=$(kubectl get pod -n gpu-gvisor-poc airuntime-gpu-vllm -o jsonpath='{.status.podIP}')

curl -s "http://$POD_IP:8000/v1/completions" \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt2", "prompt": "NeevCloud AI Platform is accelerating AI workloads by", "max_tokens": 30}'
```

---

## Step 3: Trigger the SnapshotJob (`sj-vllm-01`)

```bash
START_TIME=$(date +%s)

cat << 'EOF' | kubectl apply -f -
apiVersion: sandbox.neevcloud.com/v1alpha1
kind: SnapshotJob
metadata:
  name: sj-vllm-01
  namespace: gpu-gvisor-poc
spec:
  sandboxRef: airuntime-gpu-vllm
  nodeName: stg-nc-workload1-worker-3
  mode: snapshot
  snapshotType: full
  runtimeClass: runsc-root-self
EOF

echo "Taking snapshot of vLLM engine..."
while true; do
  PHASE=$(kubectl get snapshotjob -n gpu-gvisor-poc sj-vllm-01 -o jsonpath='{.status.phase}' 2>/dev/null)
  if [ "$PHASE" = "Ready" ]; then
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    echo "=========================================="
    echo ">>> vLLM Snapshot Completed in: ${DURATION}s! <<<"
    echo "=========================================="
    break
  elif [ "$PHASE" = "Failed" ]; then
    echo ">>> Snapshot Failed! <<<"
    break
  fi
  sleep 1
done
```

---

## Step 4: Delete the Original Pod to Release the GPU

```bash
echo "Releasing RTX 5090 GPU hardware..."
kubectl delete pod -n gpu-gvisor-poc airuntime-gpu-vllm --grace-period=0 --force
```

---

## Step 5: Restore vLLM from Snapshot

```bash
RESTORE_SHA=$(kubectl get snapshotjob -n gpu-gvisor-poc sj-vllm-01 -o jsonpath='{.status.sha256}')
RESTORE_START=$(date +%s)

cat << EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: airuntime-gpu-vllm-restored
  namespace: gpu-gvisor-poc
  labels:
    app: airuntime-gpu-vllm
  annotations:
    dev.gvisor.spec.mount.workspace.type: bind
    dev.gvisor.spec.mount.workspace.share: pod
    dev.gvisor.spec.mount.workspace.options: size=1073741824
    dev.gvisor.internal.restore.host-image-path: /var/lib/neev-checkpoint-scratch/gpu-gvisor-poc/sj-vllm-01/checkpoint
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
        restoreId: sj-vllm-01
        objectRef: s3://sandbox-snapshots/gpu-gvisor-poc/sj-vllm-01.tar.zst
        sha256: "${RESTORE_SHA}"
  containers:
  - name: vllm
    image: harbor.neev.work/test-template/vllm:0.1.5
    env:
    - name: VLLM_ENABLE_V1_MULTIPROCESSING
      value: "0"
    - name: NVIDIA_VISIBLE_DEVICES
      value: void
    command:
    - python3
    - -m
    - vllm.entrypoints.openai.api_server
    - --model
    - gpt2
    - --enforce-eager
    - --distributed-executor-backend
    - uni
    - --gpu-memory-utilization
    - "0.15"
    - --disable-custom-all-reduce
    - --port
    - "8000"
    - --host
    - "0.0.0.0"
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
        cpu: "4"
        memory: "16Gi"
      limits:
        cpu: "8"
        memory: "32Gi"
      claims:
      - name: gpu
EOF

echo "Waiting for restored vLLM pod to be Ready..."
kubectl wait --namespace gpu-gvisor-poc --for=condition=Ready pod/airuntime-gpu-vllm-restored --timeout=120s
RESTORE_END=$(date +%s)
echo "=========================================="
echo ">>> vLLM Restored in: $((RESTORE_END - RESTORE_START))s! <<<"
echo "=========================================="
```

---

## Step 6: Verify Immediate Inference Serving

```bash
RESTORED_IP=$(kubectl get pod -n gpu-gvisor-poc airuntime-gpu-vllm-restored -o jsonpath='{.status.podIP}')

curl -i "http://$RESTORED_IP:8000/v1/completions" \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt2", "prompt": "NeevCloud AI snapshot test:", "max_tokens": 15}'
```
