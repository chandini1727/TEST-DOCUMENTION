# End-to-End AIRuntime GPU Snapshot & Restore Execution Steps

This document provides self-contained, copy-paste commands to execute the end-to-end GPU snapshot and restore test directly inside the Kubernetes cluster.

---

## Step 0: Clean Up Old Test Resources & Scratch State

Run this on the master node to clear previous pods, snapshot jobs, and worker scratch files:

```bash
kubectl delete pod -n gpu-gvisor-poc airuntime-real-llm airuntime-gpu-workload --force --grace-period=0 2>/dev/null || true
kubectl delete snapshotjob -n gpu-gvisor-poc sj-real-llm sj-airuntime-gpu-01 --force --grace-period=0 2>/dev/null || true
ssh root@stg-nc-workload1-worker-3 "rm -rf /var/lib/neev-checkpoint-scratch/gpu-gvisor-poc/*" 2>/dev/null || true
```

---

## Step 1: Deploy Source GPU Workload Pod

Deploy the GPT-2 LLM server running on the RTX 5090 GPU inside gVisor:

```bash
cat << 'EOF' | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: airuntime-gpu-workload
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
      /opt/venv/bin/pip install --quiet transformers

      cat << 'PYEOF' > /workspace/real_llm_server.py
      import http.server
      import urllib.parse
      import json
      import time
      import os
      import torch
      from transformers import pipeline

      print(f"=== Starting Real GPT-2 LLM Server on RTX 5090 (PID={os.getpid()}) ===", flush=True)

      print("Loading GPT-2 model into RTX 5090 VRAM...", flush=True)
      generator = pipeline("text-generation", model="gpt2", device="cuda:0")
      print("GPT-2 Loaded and Ready on GPU!", flush=True)

      request_counter = 0

      class LLMHandler(http.server.BaseHTTPRequestHandler):
          def do_GET(self):
              global request_counter
              parsed = urllib.parse.urlparse(self.path)

              if parsed.path == "/health":
                  self.send_response(200)
                  self.send_header("Content-Type", "application/json")
                  self.end_headers()
                  data = {
                      "status": "healthy",
                      "model": "GPT-2 (Real Pretrained LLM)",
                      "gpu": torch.cuda.get_device_name(0),
                      "vram_allocated_mb": round(torch.cuda.memory_allocated(0) / (1024 * 1024), 1),
                      "requests_served": request_counter
                  }
                  self.wfile.write(json.dumps(data, indent=2).encode())
                  return

              if parsed.path == "/generate":
                  request_counter += 1
                  params = urllib.parse.parse_qs(parsed.query)
                  prompt = params.get("prompt", ["The future of computing is"])[0]

                  t0 = time.time()
                  # Generate real text using the GPU LLM!
                  result = generator(prompt, max_length=45, do_sample=True, temperature=0.7)
                  latency_ms = (time.time() - t0) * 1000

                  generated_text = result[0]["generated_text"]

                  response_data = {
                      "request_id": request_counter,
                      "input_prompt": prompt,
                      "gpu_latency_ms": round(latency_ms, 2),
                      "ai_generated_text": generated_text
                  }

                  self.send_response(200)
                  self.send_header("Content-Type", "application/json")
                  self.end_headers()
                  self.wfile.write(json.dumps(response_data, indent=2).encode())
                  print(f"[LLM-SERVER] Request #{request_counter} served in {latency_ms:.1f}ms: '{prompt}'", flush=True)
                  return

              self.send_response(404)
              self.end_headers()

      server = http.server.ThreadingHTTPServer(("0.0.0.0", 8000), LLMHandler)
      print(">>> Real LLM Server listening on port 8000! <<<", flush=True)
      server.serve_forever()
      PYEOF

      exec /opt/venv/bin/python3 -u /workspace/real_llm_server.py
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
```

Stream the logs until the server is ready:
```bash
kubectl logs -f -n gpu-gvisor-poc airuntime-gpu-workload
```
*(Wait until `>>> Real LLM Server listening on port 8000! <<<`, then press `Ctrl+C`).*

---

## Step 2: Test Pod Health and Query Model (Pre-Snapshot State)

Send an initial request to warm up the GPU and increment the in-memory counter `requests_served`:

```bash
POD_IP=$(kubectl get pod -n gpu-gvisor-poc airuntime-gpu-workload -o jsonpath='{.status.podIP}')

# Health check
curl -s "http://$POD_IP:8000/health"

# Query 1
curl -s "http://$POD_IP:8000/generate?prompt=NeevCloud%20AI%20platform%20is"
```

---

## Step 3: Trigger Snapshot and Measure Duration

Create the `SnapshotJob` and measure the time until phase becomes `Ready`:

```bash
START_TIME=$(date +%s)

cat << 'EOF' | kubectl apply -f -
apiVersion: sandbox.neevcloud.com/v1alpha1
kind: SnapshotJob
metadata:
  name: sj-airuntime-gpu-01
  namespace: gpu-gvisor-poc
spec:
  sandboxRef: airuntime-gpu-workload
  nodeName: stg-nc-workload1-worker-3
  mode: snapshot
  snapshotType: full
  runtimeClass: runsc-root-self
EOF

echo "Taking snapshot and measuring time..."
while true; do
  PHASE=$(kubectl get snapshotjob -n gpu-gvisor-poc sj-airuntime-gpu-01 -o jsonpath='{.status.phase}' 2>/dev/null)
  if [ "$PHASE" = "Ready" ]; then
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    echo "=========================================="
    echo ">>> Snapshot completed in: ${DURATION} seconds! <<<"
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

## Step 4: Delete Pod and Release the RTX 5090 GPU

Delete the pod to release the physical RTX 5090 GPU:

```bash
kubectl delete pod -n gpu-gvisor-poc airuntime-gpu-workload --force --grace-period=0
```

Verify deletion:
```bash
kubectl get pod -n gpu-gvisor-poc airuntime-gpu-workload
```

---

## Step 5: Restore Pod and Measure Duration

Prepare the restore manifest with the exact snapshot SHA and measure the time until the pod reaches `Running`:

```bash
SHA=$(kubectl get snapshotjob -n gpu-gvisor-poc sj-airuntime-gpu-01 -o jsonpath='{.status.sha256}')
echo "Snapshot SHA256: $SHA"

cat << 'EOF' > /tmp/airuntime-restore.yaml
apiVersion: v1
kind: Pod
metadata:
  name: airuntime-gpu-workload
  namespace: gpu-gvisor-poc
  annotations:
    dev.gvisor.spec.mount.workspace.type: bind
    dev.gvisor.spec.mount.workspace.share: pod
    dev.gvisor.spec.mount.workspace.options: size=1073741824
    dev.gvisor.internal.restore.host-image-path: /var/lib/neev-checkpoint-scratch/gpu-gvisor-poc/sj-airuntime-gpu-01/checkpoint
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
        restoreId: sj-airuntime-gpu-01
        objectRef: s3://sandbox-snapshots/gpu-gvisor-poc/sj-airuntime-gpu-01.tar.zst
        sha256: "PASTE_SHA_HERE"
  containers:
  - name: cuda
    image: harbor.neev.work/test-template/pytorch-v2.8.0-ubuntu-24.04:0.1.2
    command:
    - /bin/bash
    - -c
    - |
      export PATH=/opt/venv/bin:$PATH
      /opt/venv/bin/pip install --quiet transformers

      cat << 'PYEOF' > /workspace/real_llm_server.py
      import http.server
      import urllib.parse
      import json
      import time
      import os
      import torch
      from transformers import pipeline

      print(f"=== Starting Real GPT-2 LLM Server on RTX 5090 (PID={os.getpid()}) ===", flush=True)

      print("Loading GPT-2 model into RTX 5090 VRAM...", flush=True)
      generator = pipeline("text-generation", model="gpt2", device="cuda:0")
      print("GPT-2 Loaded and Ready on GPU!", flush=True)

      request_counter = 0

      class LLMHandler(http.server.BaseHTTPRequestHandler):
          def do_GET(self):
              global request_counter
              parsed = urllib.parse.urlparse(self.path)

              if parsed.path == "/health":
                  self.send_response(200)
                  self.send_header("Content-Type", "application/json")
                  self.end_headers()
                  data = {
                      "status": "healthy",
                      "model": "GPT-2 (Real Pretrained LLM)",
                      "gpu": torch.cuda.get_device_name(0),
                      "vram_allocated_mb": round(torch.cuda.memory_allocated(0) / (1024 * 1024), 1),
                      "requests_served": request_counter
                  }
                  self.wfile.write(json.dumps(data, indent=2).encode())
                  return

              if parsed.path == "/generate":
                  request_counter += 1
                  params = urllib.parse.parse_qs(parsed.query)
                  prompt = params.get("prompt", ["The future of computing is"])[0]

                  t0 = time.time()
                  # Generate real text using the GPU LLM!
                  result = generator(prompt, max_length=45, do_sample=True, temperature=0.7)
                  latency_ms = (time.time() - t0) * 1000

                  generated_text = result[0]["generated_text"]

                  response_data = {
                      "request_id": request_counter,
                      "input_prompt": prompt,
                      "gpu_latency_ms": round(latency_ms, 2),
                      "ai_generated_text": generated_text
                  }

                  self.send_response(200)
                  self.send_header("Content-Type", "application/json")
                  self.end_headers()
                  self.wfile.write(json.dumps(response_data, indent=2).encode())
                  print(f"[LLM-SERVER] Request #{request_counter} served in {latency_ms:.1f}ms: '{prompt}'", flush=True)
                  return

              self.send_response(404)
              self.end_headers()

      server = http.server.ThreadingHTTPServer(("0.0.0.0", 8000), LLMHandler)
      print(">>> Real LLM Server listening on port 8000! <<<", flush=True)
      server.serve_forever()
      PYEOF

      exec /opt/venv/bin/python3 -u /workspace/real_llm_server.py
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

sed -i "s/PASTE_SHA_HERE/$SHA/g" /tmp/airuntime-restore.yaml

START_TIME=$(date +%s)
kubectl apply -f /tmp/airuntime-restore.yaml

echo "Waiting for restored pod to reach Running..."
while true; do
  PHASE=$(kubectl get pod -n gpu-gvisor-poc airuntime-gpu-workload -o jsonpath='{.status.phase}' 2>/dev/null)
  if [ "$PHASE" = "Running" ]; then
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    echo "=========================================="
    echo ">>> Restore completed in: ${DURATION} seconds! <<<"
    echo "=========================================="
    break
  fi
  sleep 1
done
```

---

## Step 6: Verify In-Memory State Continuity & Fast Warm Inference

Query the restored pod to verify zero cold-start and in-memory state preservation:

```bash
POD_IP=$(kubectl get pod -n gpu-gvisor-poc airuntime-gpu-workload -o jsonpath='{.status.podIP}')

# Health check (should show requests_served: 1 preserved in RAM/VRAM)
curl -s "http://$POD_IP:8000/health"

# Query 2 (fast warm GPU inference, ~100ms)
curl -s "http://$POD_IP:8000/generate?prompt=After%20restoring%20from%20snapshot%20the%20AI"
```

