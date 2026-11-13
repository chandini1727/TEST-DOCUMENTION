# Test 10: Goal-Driven PyTorch Model Training & State Continuity

## Step 0: Clean Up All Previous Resources

Run this first to delete any lingering pods, jobs, and claims:

```bash
kubectl delete pods,snapshotjobs,resourceclaims -n gpu-gvisor-poc --all --force --grace-period=0 2>/dev/null
rm -f /tmp/training-*.yaml /tmp/sj-training.yaml
kubectl get pods,snapshotjobs,resourceclaims -n gpu-gvisor-poc
```

---

## Step 1: Create DRA GPU Claim (`gpu-claim`)

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

## Step 2: Deploy Goal-Driven Training Pod

This runs a neural network training loop that trains continuously with AdamW until convergence (`Loss < 0.005`):

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
  - name: agent
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

      print(f"=== Starting Goal-Driven Neural Network Training (PID={os.getpid()}) ===", flush=True)

      device = torch.device("cuda:0")
      print(f"Using GPU Device: {torch.cuda.get_device_name(0)}", flush=True)

      # 1. Define Model Architecture and Move to GPU
      model = nn.Sequential(
          nn.Linear(2048, 4096),
          nn.ReLU(),
          nn.Linear(4096, 2048),
          nn.ReLU(),
          nn.Linear(2048, 512)
      ).to(device)

      criterion = nn.MSELoss()
      optimizer = optim.AdamW(model.parameters(), lr=0.001)

      # 2. Input and Target Training Data on GPU
      inputs = torch.randn(256, 2048, device=device)
      targets = torch.randn(256, 512, device=device)

      print(f"Allocated VRAM: {torch.cuda.memory_allocated(0)/(1024*1024):.1f} MB", flush=True)

      # 3. Train until expected target output is achieved (Loss < 0.005)
      target_loss = 0.005
      step = 0
      print(f"=== Training Until Model Converges (Target: Loss < {target_loss}) ===", flush=True)

      while True:
          step += 1
          optimizer.zero_grad()
          outputs = model(inputs)
          loss = criterion(outputs, targets)
          loss.backward()
          optimizer.step()

          current_loss = loss.item()
          vram = torch.cuda.memory_allocated(0)/(1024*1024)
          print(f"[ACTIVE TRAINING] Step={step:03d} | Loss={current_loss:.4f} (Target < {target_loss}) | VRAM={vram:.1f}MB", flush=True)

          # Stop when model achieves the expected output quality
          if current_loss < target_loss:
              print(f"\n=======================================================", flush=True)
              print(f">>> EXPECTED OUTPUT REACHED! Model Converged at Step {step} with Loss={current_loss:.4f} <<<", flush=True)
              break

          time.sleep(2)

      # 4. Generate Output Predictions on Test Data
      print("\n=== Generating Predictions on Test Inputs ===", flush=True)
      with torch.no_grad():
          test_input = torch.randn(1, 2048, device=device)
          prediction = model(test_input)
          print(f"Test Input evaluated successfully! Output shape: {prediction.shape}")
          sample_outputs = [round(float(x), 4) for x in prediction[0, :5].cpu().tolist()]
          print(f"Final Model Predictions (sample 5 elements): {sample_outputs}")
          print("=== Training & Convergence End-to-End SUCCESS ===", flush=True)
      print("=======================================================", flush=True)

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

**Stream logs to confirm it starts training:**
```bash
kubectl logs -f -n gpu-gvisor-poc airuntime-gpu-training
```
*(Once you see it actively training around **Step 10–15**, hit `Ctrl+C` to snapshot it in mid-training).*

---

## Step 3: Trigger Snapshot in Mid-Training

```bash
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
```

**Wait for the SnapshotJob to reach `Ready`:**
```bash
kubectl get snapshotjob -n gpu-gvisor-poc sj-training-01 -w
```
*(Once Phase shows `Ready`, press `Ctrl+C`).*

---

## Step 4: Delete the Running Pod (Free the Physical GPU)

```bash
kubectl delete pod -n gpu-gvisor-poc airuntime-gpu-training --force --grace-period=0
```

---

## Step 5: Restore the Training Workload

```bash
# 1. Fetch SHA256 digest
SHA=$(kubectl get snapshotjob -n gpu-gvisor-poc sj-training-01 -o jsonpath='{.status.sha256}')
echo "Restoring from Snapshot SHA256: $SHA"

# 2. Write restore manifest
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
  - name: agent
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

      print(f"=== Starting Goal-Driven Neural Network Training (PID={os.getpid()}) ===", flush=True)

      device = torch.device("cuda:0")
      print(f"Using GPU Device: {torch.cuda.get_device_name(0)}", flush=True)

      # 1. Define Model Architecture and Move to GPU
      model = nn.Sequential(
          nn.Linear(2048, 4096),
          nn.ReLU(),
          nn.Linear(4096, 2048),
          nn.ReLU(),
          nn.Linear(2048, 512)
      ).to(device)

      criterion = nn.MSELoss()
      optimizer = optim.AdamW(model.parameters(), lr=0.001)

      # 2. Input and Target Training Data on GPU
      inputs = torch.randn(256, 2048, device=device)
      targets = torch.randn(256, 512, device=device)

      print(f"Allocated VRAM: {torch.cuda.memory_allocated(0)/(1024*1024):.1f} MB", flush=True)

      # 3. Train until expected target output is achieved (Loss < 0.005)
      target_loss = 0.005
      step = 0
      print(f"=== Training Until Model Converges (Target: Loss < {target_loss}) ===", flush=True)

      while True:
          step += 1
          optimizer.zero_grad()
          outputs = model(inputs)
          loss = criterion(outputs, targets)
          loss.backward()
          optimizer.step()

          current_loss = loss.item()
          vram = torch.cuda.memory_allocated(0)/(1024*1024)
          print(f"[ACTIVE TRAINING] Step={step:03d} | Loss={current_loss:.4f} (Target < {target_loss}) | VRAM={vram:.1f}MB", flush=True)

          # Stop when model achieves the expected output quality
          if current_loss < target_loss:
              print(f"\n=======================================================", flush=True)
              print(f">>> EXPECTED OUTPUT REACHED! Model Converged at Step {step} with Loss={current_loss:.4f} <<<", flush=True)
              break

          time.sleep(2)

      # 4. Generate Output Predictions on Test Data
      print("\n=== Generating Predictions on Test Inputs ===", flush=True)
      with torch.no_grad():
          test_input = torch.randn(1, 2048, device=device)
          prediction = model(test_input)
          print(f"Test Input evaluated successfully! Output shape: {prediction.shape}")
          sample_outputs = [round(float(x), 4) for x in prediction[0, :5].cpu().tolist()]
          print(f"Final Model Predictions (sample 5 elements): {sample_outputs}")
          print("=== Training & Convergence End-to-End SUCCESS ===", flush=True)
      print("=======================================================", flush=True)

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

# 3. Inject actual SHA digest
sed -i "s/PASTE_SHA_HERE/$SHA/g" /tmp/training-restore.yaml

# 4. Deploy restored workload
kubectl apply -f /tmp/training-restore.yaml
```

**Wait for restored pod to reach `Running`:**
```bash
kubectl get pod -n gpu-gvisor-poc airuntime-gpu-training-restored -w
```
*(Once status reaches `Running`, press `Ctrl+C`).*

---

## Step 6: Watch It Automatically Resume Until It Reaches the Expected Output

Stream the logs directly:

```bash
kubectl logs -f -n gpu-gvisor-poc airuntime-gpu-training-restored
```

