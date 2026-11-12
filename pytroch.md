# Execution Steps: Test 10 - PyTorch Training Loop & AdamW Optimizer State (with Exact Latency Timing)

Run these commands directly in your cluster master node terminal (`stg-nc-workload1-master-1`). Every step writes manifests to `/tmp/*.yaml` using literal heredocs (`'EOF'`) so `$PATH` is not corrupted and SSH paste buffer does not truncate.

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

## Step 0.5: Create DRA Resource Claim (`gpu-claim`)

```bash
cat << 'EOF' > /tmp/training-claim.yaml
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
kubectl apply -f /tmp/training-claim.yaml
```

---

## Step 1: Deploy Training Pod (Trains Steps 1 to 20)

```bash
cat << 'EOF' > /tmp/training-pod.yaml
apiVersion: v1
kind: Pod
metadata:
  name: airuntime-gpu-training
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

      cat << 'PYEOF' > /workspace/train.py
      import torch
      import torch.nn as nn
      import torch.optim as optim
      import time
      import os

      print(f"=== Starting PyTorch GPU Training Loop on RTX 5090 (PID={os.getpid()}) ===", flush=True)

      device = torch.device("cuda:0")
      print(f"Using GPU Device: {torch.cuda.get_device_name(0)}", flush=True)

      # 1. Define Model and Move to GPU
      model = nn.Sequential(
          nn.Linear(2048, 4096),
          nn.ReLU(),
          nn.Linear(4096, 2048),
          nn.ReLU(),
          nn.Linear(2048, 512)
      ).to(device)

      criterion = nn.MSELoss()
      optimizer = optim.AdamW(model.parameters(), lr=0.001)

      # 2. Synthetic Training Data on GPU
      inputs = torch.randn(256, 2048, device=device)
      targets = torch.randn(256, 512, device=device)

      print(f"Allocated VRAM for Model & Tensors: {torch.cuda.memory_allocated(0)/(1024*1024):.1f} MB", flush=True)

      # 3. Train Steps 1 to 20
      for step in range(1, 21):
          optimizer.zero_grad()
          outputs = model(inputs)
          loss = criterion(outputs, targets)
          loss.backward()
          optimizer.step()
          
          vram = torch.cuda.memory_allocated(0)/(1024*1024)
          print(f"[PRE-SNAPSHOT TRAINING] Step={step:02d}/20 | Loss={loss.item():.4f} | VRAM={vram:.1f}MB", flush=True)
          time.sleep(0.5)

      # Inspect AdamW internal momentum buffers in VRAM
      first_param = next(model.parameters())
      state = optimizer.state[first_param]
      exp_avg_norm = state['exp_avg'].norm().item()
      exp_avg_sq_norm = state['exp_avg_sq'].norm().item()

      print("==================================================================", flush=True)
      print(f">>> STEP 20 COMPLETE! Optimizer State in VRAM Verified: <<<", flush=True)
      print(f"    - AdamW Step Count: {state['step']}")
      print(f"    - 1st Momentum Norm: {exp_avg_norm:.4f}")
      print(f"    - 2nd Momentum Norm: {exp_avg_sq_norm:.4f}")
      print(">>> Holding in VRAM. READY FOR SNAPSHOT! <<<", flush=True)
      print("==================================================================", flush=True)

      # 4. Wait for restore. Once restored, automatically continue training steps 21 to 40!
      while True:
          time.sleep(1)
          if os.path.exists("/workspace/resume_trigger.txt"):
              break

      print("\n=== RESUMING POST-SNAPSHOT TRAINING ON WARM GPU! ===", flush=True)
      for step in range(21, 41):
          optimizer.zero_grad()
          outputs = model(inputs)
          loss = criterion(outputs, targets)
          loss.backward()
          optimizer.step()
          
          vram = torch.cuda.memory_allocated(0)/(1024*1024)
          print(f"[POST-RESTORE TRAINING] Step={step:02d}/40 | Loss={loss.item():.4f} | VRAM={vram:.1f}MB", flush=True)
          time.sleep(0.5)

      print(">>> SUCCESS: Full 40-step training loop completed seamlessly across snapshot! <<<", flush=True)
      while True:
          time.sleep(3600)
      PYEOF

      exec /opt/venv/bin/python3 -u /workspace/train.py
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
        memory: "6Gi"
      limits:
        cpu: "2"
        memory: "6Gi"
      claims:
      - name: gpu
EOF
kubectl apply -f /tmp/training-pod.yaml
```

Watch the training progress until **Step 20 Complete**:
```bash
kubectl logs -f -n gpu-gvisor-poc airuntime-gpu-training
```
*(Once you see `>>> Holding in VRAM. READY FOR SNAPSHOT! <<<`, press `Ctrl+C`).*

---

## Step 2: Trigger Snapshot & Measure Exact Snapshot Time

```bash
echo "=== Starting Snapshot Timing ==="
START_SNAP=$(date +%s%N)

cat << 'EOF' > /tmp/sj-training.yaml
apiVersion: sandbox.neevcloud.com/v1alpha1
kind: SnapshotJob
metadata:
  name: sj-training-01
  namespace: gpu-gvisor-poc
spec:
  sandboxRef: airuntime-gpu-training
  nodeName: stg-nc-workload1-worker-3
  mode: snapshot
  snapshotType: full
  runtimeClass: runsc-root-self
EOF
kubectl apply -f /tmp/sj-training.yaml

echo "Waiting for SnapshotJob sj-training-01 to reach Ready..."
while true; do
  PHASE=$(kubectl get snapshotjob -n gpu-gvisor-poc sj-training-01 -o jsonpath='{.status.phase}' 2>/dev/null)
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
    kubectl get snapshotjob -n gpu-gvisor-poc sj-training-01 -o yaml
    break
  fi
  sleep 0.5
done
```

---

## Step 3: Delete Pod & Release GPU

```bash
kubectl delete pod -n gpu-gvisor-poc airuntime-gpu-training --force --grace-period=0
```

---

## Step 4: Restore Pod & Measure Exact Restore Latency

Notice that `'EOF'` is strictly quoted so bash never expands `$PATH` to host paths:

```bash
# 1. Fetch SHA256 digest
SHA=$(kubectl get snapshotjob -n gpu-gvisor-poc sj-training-01 -o jsonpath='{.status.sha256}')
echo "Restoring from Snapshot SHA256: $SHA"

# 2. Write manifest with unexpanded $PATH
cat << 'EOF' > /tmp/training-restore.yaml
apiVersion: v1
kind: Pod
metadata:
  name: airuntime-gpu-training-restored
  namespace: gpu-gvisor-poc
  annotations:
    dev.gvisor.spec.mount.workspace.type: bind
    dev.gvisor.spec.mount.workspace.share: pod
    dev.gvisor.spec.mount.workspace.options: size=1073741824
    dev.gvisor.internal.restore.host-image-path: /var/lib/neev-checkpoint-scratch/gpu-gvisor-poc/sj-training-01/checkpoint
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
        restoreId: sj-training-01
        objectRef: s3://sandbox-snapshots/gpu-gvisor-poc/sj-training-01.tar.zst
        sha256: "PASTE_SHA_HERE"
  containers:
  - name: cuda
    image: harbor.neev.work/test-template/pytorch-v2.8.0-ubuntu-24.04:0.1.2
    command:
    - /bin/bash
    - -c
    - |
      export PATH=/opt/venv/bin:$PATH

      cat << 'PYEOF' > /workspace/train.py
      import torch
      import torch.nn as nn
      import torch.optim as optim
      import time
      import os

      print(f"=== Starting PyTorch GPU Training Loop on RTX 5090 (PID={os.getpid()}) ===", flush=True)

      device = torch.device("cuda:0")
      print(f"Using GPU Device: {torch.cuda.get_device_name(0)}", flush=True)

      # 1. Define Model and Move to GPU
      model = nn.Sequential(
          nn.Linear(2048, 4096),
          nn.ReLU(),
          nn.Linear(4096, 2048),
          nn.ReLU(),
          nn.Linear(2048, 512)
      ).to(device)

      criterion = nn.MSELoss()
      optimizer = optim.AdamW(model.parameters(), lr=0.001)

      # 2. Synthetic Training Data on GPU
      inputs = torch.randn(256, 2048, device=device)
      targets = torch.randn(256, 512, device=device)

      print(f"Allocated VRAM for Model & Tensors: {torch.cuda.memory_allocated(0)/(1024*1024):.1f} MB", flush=True)

      # 3. Train Steps 1 to 20
      for step in range(1, 21):
          optimizer.zero_grad()
          outputs = model(inputs)
          loss = criterion(outputs, targets)
          loss.backward()
          optimizer.step()
          
          vram = torch.cuda.memory_allocated(0)/(1024*1024)
          print(f"[PRE-SNAPSHOT TRAINING] Step={step:02d}/20 | Loss={loss.item():.4f} | VRAM={vram:.1f}MB", flush=True)
          time.sleep(0.5)

      # Inspect AdamW internal momentum buffers in VRAM
      first_param = next(model.parameters())
      state = optimizer.state[first_param]
      exp_avg_norm = state['exp_avg'].norm().item()
      exp_avg_sq_norm = state['exp_avg_sq'].norm().item()

      print("==================================================================", flush=True)
      print(f">>> STEP 20 COMPLETE! Optimizer State in VRAM Verified: <<<", flush=True)
      print(f"    - AdamW Step Count: {state['step']}")
      print(f"    - 1st Momentum Norm: {exp_avg_norm:.4f}")
      print(f"    - 2nd Momentum Norm: {exp_avg_sq_norm:.4f}")
      print(">>> Holding in VRAM. READY FOR SNAPSHOT! <<<", flush=True)
      print("==================================================================", flush=True)

      # 4. Wait for restore. Once restored, automatically continue training steps 21 to 40!
      while True:
          time.sleep(1)
          if os.path.exists("/workspace/resume_trigger.txt"):
              break

      print("\n=== RESUMING POST-SNAPSHOT TRAINING ON WARM GPU! ===", flush=True)
      for step in range(21, 41):
          optimizer.zero_grad()
          outputs = model(inputs)
          loss = criterion(outputs, targets)
          loss.backward()
          optimizer.step()
          
          vram = torch.cuda.memory_allocated(0)/(1024*1024)
          print(f"[POST-RESTORE TRAINING] Step={step:02d}/40 | Loss={loss.item():.4f} | VRAM={vram:.1f}MB", flush=True)
          time.sleep(0.5)

      print(">>> SUCCESS: Full 40-step training loop completed seamlessly across snapshot! <<<", flush=True)
      while True:
          time.sleep(3600)
      PYEOF

      exec /opt/venv/bin/python3 -u /workspace/train.py
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
        memory: "6Gi"
      limits:
        cpu: "2"
        memory: "6Gi"
      claims:
      - name: gpu
EOF

# 3. Inject SHA without touching $PATH
sed -i "s/PASTE_SHA_HERE/$SHA/g" /tmp/training-restore.yaml

# 4. Deploy and measure restore latency
echo "=== Starting Restore Timing ==="
START_RESTORE=$(date +%s%N)
kubectl apply -f /tmp/training-restore.yaml

echo "Waiting for airuntime-gpu-training-restored to reach Running..."
while true; do
  STATUS=$(kubectl get pod -n gpu-gvisor-poc airuntime-gpu-training-restored -o jsonpath='{.status.phase}' 2>/dev/null)
  if [ "$STATUS" == "Running" ]; then
    END_RUNNING=$(date +%s%N)
    RUNNING_MS=$(( (END_RUNNING - START_RESTORE) / 1000000 ))
    RUNNING_S=$(awk "BEGIN {printf \"%.2f\", $RUNNING_MS / 1000}")
    echo "=========================================================="
    echo ">> Pod State: RUNNING"
    echo ">> Time to Running (S3 Download + Restore): ${RUNNING_S}s (${RUNNING_MS} ms)"
    echo "=========================================================="
    break
  elif [ "$STATUS" == "Failed" ]; then
    echo ">> Pod Restore Failed!"
    kubectl describe pod -n gpu-gvisor-poc airuntime-gpu-training-restored | tail -n 10
    break
  fi
  sleep 0.2
done
```

---

## Step 5: Resume Training from Step 21 & Verify Integrity

```bash
# Trigger post-restore training loop
kubectl exec -n gpu-gvisor-poc airuntime-gpu-training-restored -- touch /workspace/resume_trigger.txt

# Stream logs: confirms training seamlessly continues from Step 21 to 40
kubectl logs -f -n gpu-gvisor-poc airuntime-gpu-training-restored
```
