# Execution Steps: Test 03 - Cross-Template Compatibility on Restore

Run these commands directly in your cluster master node terminal.

---

## Prerequisites: Verify Snapshot & Release GPU

The snapshot checkpoint is already stored on worker-3 at:
`/var/lib/neev-checkpoint-scratch/gpu-gvisor-poc/sj-airuntime-gpu-01/checkpoint/`

Delete previous pods to free the RTX 5090 GPU:

```bash
kubectl delete pod -n gpu-gvisor-poc airuntime-gpu-workload airuntime-gpu-cross-template --force --grace-period=0 2>/dev/null || true
```

---

## Step 1: Deploy Cross-Template Restore Pod

Deploy the restored pod pointing to the base Ubuntu 24.04 image (`harbor.neev.work/test-template/ubuntu-24.04:0.1.3`) instead of the original PyTorch image:

```bash
SHA="f8d06885f488a78d7f425da21ec613e3a5330604d7783b9b22c9926fe72dca9e"
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
    # NEEVCLOUD BASE TEMPLATE: Lacks PyTorch virtualenv (/opt/venv)
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

sed -i "s/PASTE_SHA_HERE/$SHA/g" /tmp/airuntime-cross-template.yaml
kubectl apply -f /tmp/airuntime-cross-template.yaml
```

---

## Step 2: Observe Failure & Incompatible Rootfs Rejection

Watch the pod status. It enters `StartError` because gVisor's kernel verifies that the container rootfs matches the open file descriptors of the checkpointed process:

```bash
kubectl get pod -n gpu-gvisor-poc airuntime-gpu-cross-template
kubectl describe pod -n gpu-gvisor-poc airuntime-gpu-cross-template | tail -n 25
```

**Observed Output & Event:**
```text
NAME                           READY   STATUS       RESTARTS   AGE
airuntime-gpu-cross-template   0/1     StartError   0          12s

Events:
  Warning  Failed  kubelet  spec.containers{cuda}: Error: failed to start containerd task "cuda": OCI runtime restore failed: starting container: starting sub-container: failed to load kernel: vfs.CompleteRestore() failed: failed to complete restore for filesystem type "9p": failed to walk "opt/venv" of type 040000 in mount "cuda:/": no such file or directory
```

> **Why It Failed:** The snapshot was created in an environment where `/opt/venv` existed on the root filesystem. When restoring into the base Ubuntu image (`ubuntu-24.04:0.1.3`), gVisor's virtual file system (`vfs.CompleteRestore()`) attempted to remap the active Python virtualenv directory. Because `ubuntu-24.04:0.1.3` does not contain `/opt/venv`, gVisor refused to start a corrupted container state.

---

## Step 3: Clean Up Cross-Template Pod

```bash
kubectl delete pod -n gpu-gvisor-poc airuntime-gpu-cross-template --force --grace-period=0
```

