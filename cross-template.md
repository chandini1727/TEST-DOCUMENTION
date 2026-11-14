# Test 03: Cross-Template Compatibility on Snapshot Restore

This test evaluates whether a GPU workload snapshot captured from a PyTorch container (`pytorch-v2.8.0-ubuntu-24.04:0.1.2`) can be restored into a pod provisioned with a different container template: the plain Ubuntu base image (`ubuntu-24.04:0.1.3`), which lacks the `/opt/venv` virtual environment and CUDA libraries.

---

## Step 0: Clean Up Lingering Resources & Free the GPU

Run this first to delete any existing pods, jobs, and claims:

```bash
kubectl delete pods,snapshotjobs,resourceclaims -n gpu-gvisor-poc --all --force --grace-period=0 2>/dev/null || true
rm -f /tmp/airuntime-*.yaml /tmp/sj-*.yaml /tmp/gpu-claim.yaml
kubectl get pods,snapshotjobs,resourceclaims -n gpu-gvisor-poc
```

---

## Step 1: Create DRA GPU Claim (`gpu-claim`)

```bash
cat << 'EOF' > /tmp/gpu-claim.yaml
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
kubectl apply -f /tmp/gpu-claim.yaml
```

---

## Step 2: Deploy Baseline Workload Pod (`airuntime-gpu-workload`)

Deploy the PyTorch GPU workload pod:

```bash
cat << 'EOF' > /tmp/airuntime-workload.yaml
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
kubectl apply -f /tmp/airuntime-workload.yaml
```

**Stream logs to confirm workload is loaded on RTX 5090:**
```bash
kubectl logs -f -n gpu-gvisor-poc airuntime-gpu-workload
```
*(Once you see `>>> Real LLM Server listening on port 8000! <<<`, press `Ctrl+C`).*

---

## Step 3: Trigger Snapshot via SnapshotJob

Create the `SnapshotJob` to capture the baseline workload:

```bash
cat << 'EOF' > /tmp/sj-airuntime-gpu-01.yaml
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
kubectl apply -f /tmp/sj-airuntime-gpu-01.yaml
```

**Wait for the SnapshotJob to reach `Ready`:**
```bash
kubectl get snapshotjob -n gpu-gvisor-poc sj-airuntime-gpu-01 -w
```
*(Once Phase shows `Ready`, press `Ctrl+C`).*

---

## Step 4: Delete Baseline Pod (Free the Physical GPU)

Delete the original workload pod so the physical RTX 5090 GPU is released:

```bash
kubectl delete pod -n gpu-gvisor-poc airuntime-gpu-workload --force --grace-period=0
```

---

## Step 5: Deploy Cross-Template Restore Pod

Deploy the restored pod with the container image changed to the base Ubuntu image (`harbor.neev.work/test-template/ubuntu-24.04:0.1.3`):

```bash
SHA=$(kubectl get snapshotjob -n gpu-gvisor-poc sj-airuntime-gpu-01 -o jsonpath='{.status.sha256}')
echo "Using Snapshot SHA256: $SHA"

cat << 'EOF' > /tmp/airuntime-cross-template.yaml
apiVersion: v1
kind: Pod
metadata:
  name: airuntime-gpu-cross-template
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
    # INCOMPATIBLE BASE IMAGE: Lacks PyTorch virtualenv (/opt/venv)
    image: harbor.neev.work/test-template/ubuntu-24.04:0.1.3
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

sed -i "s/PASTE_SHA_HERE/$SHA/g" /tmp/airuntime-cross-template.yaml
kubectl apply -f /tmp/airuntime-cross-template.yaml
```

---

## Step 6: Observe Failure & Incompatible Rootfs Rejection

Check the pod status and inspect the failure event:

```bash
kubectl get pod -n gpu-gvisor-poc airuntime-gpu-cross-template
kubectl describe pod -n gpu-gvisor-poc airuntime-gpu-cross-template | tail -n 25
```

### Observed Output & Error Event:
```text
NAME                           READY   STATUS       RESTARTS   AGE
airuntime-gpu-cross-template   0/1     StartError   0          12s

Events:
  Type     Reason    Age   From     Message
  ----     ------    ----  ----     -------
  Warning  Failed    8s    kubelet  spec.containers{cuda}: Error: failed to start containerd task "cuda": 
  OCI runtime restore failed: starting container: starting sub-container: 
  failed to load kernel: vfs.CompleteRestore() failed: failed to complete restore for filesystem type "9p": 
  failed to walk "opt/venv" of type 040000 in mount "cuda:/": no such file or directory
```

### Technical Explanation:
* **The Root Cause:** The source snapshot was taken when the running Python process had open files, shared libraries, and dependencies residing in `/opt/venv`.
* When restoring into the plain base Ubuntu image (`ubuntu-24.04:0.1.3`), gVisor's virtual file system (`vfs.CompleteRestore()`) attempted to remap the active Python virtualenv directory.
* Because the target template does not contain `/opt/venv`, gVisor refused to start a corrupted container state and aborted with **`StartError`**.
* **Key Finding:** Cross-template snapshot restore is **incompatible** unless the target image preserves the exact root filesystem structure and binary dependencies of the source container.

---

## Step 7: Clean Up Cross-Template Pod

Clean up the failed pod to free the GPU for future tests:

```bash
kubectl delete pod -n gpu-gvisor-poc airuntime-gpu-cross-template --force --grace-period=0
```
