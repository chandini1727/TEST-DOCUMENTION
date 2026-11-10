# Step-by-Step Runbook: Test 02 CPU & RAM Resource Scaling

This runbook contains all exact commands to execute and verify the two resource scaling test scenarios on `root@stg-nc-workload1-master-1`:
1. **Scenario 1: Scale-Up (`4 CPU / 8Gi RAM`)** — Expanding resources beyond the original `2 CPU / 4Gi RAM` baseline.
2. **Scenario 2: Scale-Down (`1 CPU / 100Mi RAM`)** — Constraining memory below the active 644 MiB snapshot footprint.

Both tests restore directly from the snapshot captured via `SnapshotJob` `sj-airuntime-gpu-01`.

---

## Step 0: Ensure GPU Claim is Free & Retrieve Snapshot SHA

Before restoring either scenario, confirm the source pod is deleted (so `gpu-claim` is unbound) and retrieve the snapshot SHA256 checksum:

```bash
# 1. Ensure any previous test pods are cleaned up
kubectl delete pod -n gpu-gvisor-poc airuntime-gpu-workload airuntime-gpu-scaled-up airuntime-gpu-scaled-down --force --grace-period=0 2>/dev/null || true

# 2. Verify SnapshotJob sj-airuntime-gpu-01 is Ready
kubectl get snapshotjob -n gpu-gvisor-poc sj-airuntime-gpu-01

# 3. Retrieve SHA256 checksum into shell variable
SHA=$(kubectl get snapshotjob -n gpu-gvisor-poc sj-airuntime-gpu-01 -o jsonpath='{.status.sha256}')
echo "Snapshot SHA256: $SHA"
```

*Expected output: A 64-character hex string (e.g., `499ec4fac7fe6e78...`)*

> [!NOTE]
> If `sj-airuntime-gpu-01` has not been taken yet, execute Steps 1–3 in `poc/airuntime-snapshot-reconciler/STEPS.md` first.

---

## Scenario 1: Scale-Up Test (`4 CPU / 8Gi RAM`)

### 1.1 Deploy Scaled-Up Pod

Deploy the restored pod with upgraded resources (`requests.cpu: 4`, `requests.memory: 8Gi`):

```bash
cat << 'EOF' | sed "s/PASTE_SHA_HERE/$SHA/g" | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: airuntime-gpu-scaled-up
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
      cat << 'PYEOF' > /workspace/airuntime_gpu_workload.py
      import os, sys, time, torch

      log_path = "/workspace/airuntime_gpu.log"
      token_path = "/workspace/runtime-token.txt"

      with open(token_path, "w") as f:
          f.write("airuntime-gpu-session-active\n")

      print(f"=== AIRuntime GPU Workload Starting (PID={os.getpid()}) ===", flush=True)
      dev = torch.device("cuda:0")
      state = torch.zeros(64 * 1024 * 1024, dtype=torch.float32, device=dev)

      tick = 0
      with open(log_path, "a") as f:
          f.write(f"=== Session Initialized: PID={os.getpid()} ===\n")

      while True:
          tick += 1
          state += 1.0
          torch.cuda.synchronize()
          sample = float(state[0].item())
          allocated_mb = torch.cuda.memory_allocated(dev) / (1024 * 1024)
          msg = f"[AIRUNTIME-GPU] tick={tick} sample={sample:.1f} VRAM={allocated_mb:.1f}MB time={time.time()}\n"
          print(msg, end="", flush=True)
          with open(log_path, "a") as f:
              f.write(msg)
              f.flush()
          time.sleep(1)
      PYEOF

      exec python3 -u /workspace/airuntime_gpu_workload.py
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
        cpu: "4"
        memory: "8Gi"
      limits:
        cpu: "4"
        memory: "8Gi"
      claims:
      - name: gpu
EOF
```

---

### 1.2 Verify Scale-Up Resumption

Stream logs to confirm execution resumes from the exact tick where it was paused:

```bash
kubectl logs -f -n gpu-gvisor-poc airuntime-gpu-scaled-up
```

**Expected Log Stream:**
```text
[AIRUNTIME-GPU] tick=15 sample=15.0 VRAM=256.0MB time=1794305283.000729
[AIRUNTIME-GPU] tick=16 sample=16.0 VRAM=256.0MB time=1794305284.0035625
[AIRUNTIME-GPU] tick=17 sample=17.0 VRAM=256.0MB time=1794305285.0057042
[AIRUNTIME-GPU] tick=18 sample=18.0 VRAM=256.0MB time=1794305286.0079675
```

**Confirm Allocated Resources via Pod Spec:**
```bash
kubectl get pod -n gpu-gvisor-poc airuntime-gpu-scaled-up -o jsonpath='{.spec.containers[0].resources}' | jq .
```

*Expected limits: `{"cpu": "4", "memory": "8Gi"}`*

---

### 1.3 Clean Up Scale-Up Pod

Delete the scale-up pod to release the RTX 5090 GPU for the scale-down test:

```bash
kubectl delete pod -n gpu-gvisor-poc airuntime-gpu-scaled-up --force --grace-period=0
```

Wait until terminated:
```bash
kubectl get pod -n gpu-gvisor-poc airuntime-gpu-scaled-up
```
*Expected output: `NotFound`*

---

## Scenario 2: Scale-Down Test (`1 CPU / 100Mi RAM`)

### 2.1 Deploy Scaled-Down Pod

Deploy the restored pod with constrained memory below the 644 MiB checkpoint footprint (`requests.memory: 100Mi`):

```bash
cat << 'EOF' | sed "s/PASTE_SHA_HERE/$SHA/g" | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: airuntime-gpu-scaled-down
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
      cat << 'PYEOF' > /workspace/airuntime_gpu_workload.py
      import os, sys, time, torch

      log_path = "/workspace/airuntime_gpu.log"
      token_path = "/workspace/runtime-token.txt"

      with open(token_path, "w") as f:
          f.write("airuntime-gpu-session-active\n")

      print(f"=== AIRuntime GPU Workload Starting (PID={os.getpid()}) ===", flush=True)
      dev = torch.device("cuda:0")
      state = torch.zeros(64 * 1024 * 1024, dtype=torch.float32, device=dev)

      tick = 0
      with open(log_path, "a") as f:
          f.write(f"=== Session Initialized: PID={os.getpid()} ===\n")

      while True:
          tick += 1
          state += 1.0
          torch.cuda.synchronize()
          sample = float(state[0].item())
          allocated_mb = torch.cuda.memory_allocated(dev) / (1024 * 1024)
          msg = f"[AIRUNTIME-GPU] tick={tick} sample={sample:.1f} VRAM={allocated_mb:.1f}MB time={time.time()}\n"
          print(msg, end="", flush=True)
          with open(log_path, "a") as f:
              f.write(msg)
              f.flush()
          time.sleep(1)
      PYEOF

      exec python3 -u /workspace/airuntime_gpu_workload.py
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
        cpu: "1"
        memory: "100Mi"
      limits:
        cpu: "1"
        memory: "100Mi"
      claims:
      - name: gpu
EOF
```

---

### 2.2 Inspect Pod Status & Failure Reason

Check pod status:
```bash
kubectl get pod -n gpu-gvisor-poc airuntime-gpu-scaled-down
```

*Status will be `StartError` or `CrashLoopBackOff`.*

Inspect Kubelet failure events:
```bash
kubectl describe pod -n gpu-gvisor-poc airuntime-gpu-scaled-down | tail -n 25
```

**Observed Rejection Output:**
```text
Events:
  Type     Reason     Age   From     Message
  ----     ------     ----  ----     -------
  Warning  Failed     10s   kubelet  Error: failed to start containerd task "cuda": OCI runtime restore failed: starting container: systemd error: Unit cri-containerd-...scope failed to load properly, please adjust/correct and reload service manager: No such file or directory
```

**Why this failed:**
The saved snapshot holds a 644 MiB memory footprint. When containerd/gVisor configures cgroups for the new container with a 100 MiB limit, the memory allocator rejects loading the 644 MiB memory state, terminating the container immediately.

---

### 2.3 Clean Up Scale-Down Pod

```bash
kubectl delete pod -n gpu-gvisor-poc airuntime-gpu-scaled-down --force --grace-period=0
```

---

## Summary of Results

| Scenario | Limits Tested | Outcome | Observation |
| :--- | :--- | :---: | :--- |
| **Scale-Up** | 4 CPU / 8Gi RAM | ✅ **PASSED** | Workload seamlessly resumed at `tick=15` with expanded resources; 256MB VRAM was intact. |
| **Scale-Down** | 1 CPU / 100Mi RAM | ❌ **FAILED** | Container refused to load; memory limit (100Mi) was lower than active snapshot memory (644Mi). |

