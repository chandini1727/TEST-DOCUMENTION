# PoC: memory-only restore of paused GPU workloads (training, Triton, vLLM) onto a different GPU

- **Jira Ticket:** `AIPLATFORM-3054` (Related: `AIPLATFORM-3052`)
- **Status:** In Progress / Empirically Validated
- **Date:** 17 November 2026

---

## Summary

A memory + CUDA restore today only works back onto the same physical GPU. Restoring onto another GPU in the same node fails during GPU device setup, and so do multi-GPU restore and a change in GPU count. If resume needs that exact card, a paused runtime can't come back once someone else is using the card, which makes pause close to useless at scale.

Memory-only restore is meant to close that gap: the customer pauses once with memory + CUDA, and at resume we restore CPU RAM (host memory) and process state, and rebuild the GPU state on whichever compatible GPU is free.

---

## Approaches to Verify Pause and Restore on a Different GPU

We evaluated two approaches to pause a GPU workload and restore it onto a different physical GPU.

Before gVisor snapshots the container, the CUDA checkpoint step copies GPU memory into CPU RAM. Each approach tests whether the process can restore from that snapshot without re-attaching to the original GPU, and run on a different available GPU instead.

### Approach A: CUDA suspension with device remapping

**Overview:** This approach suspends CUDA at the driver level, so the workload needs no changes. Before the snapshot, `cuda-checkpoint` locks the process's CUDA context and suspends it. Suspending copies all GPU memory into the process's CPU RAM and closes every handle to the physical GPU, so the snapshot holds the full GPU state as ordinary host memory. On resume, the container is restored with CUDA still suspended, a free target GPU is attached, and `cuda-checkpoint` rebuilds the CUDA state on that GPU by mapping the source GPU UUID to the target GPU UUID. All GPU memory, including model weights, is preserved, so nothing is reloaded from disk.

#### Steps

**1. Deploy the workload on the source GPU.**
Create the pod with a GPU claim for GPU 0 and mount the `cuda-checkpoint` binary from the host. The pod loads the `Qwen2.5-7B` model and starts the LoRA fine-tune.

```bash
kubectl apply -f train-a-source.yaml
kubectl logs -n gpu-gvisor-poc lora-source -f
```

**2. Confirm the model is loaded on the GPU.**
Wait until the `Qwen2.5-7B` weights are resident in GPU 0 memory and the training log shows steps progressing.

**3. Suspend CUDA and verify.**
Run `cuda-checkpoint` against the process that owns the CUDA context (PID 1 for single-process training). The `lock` action blocks new CUDA calls, and the `checkpoint` action suspends CUDA. On suspend, the full contents of GPU memory are copied into the process's CPU RAM and all handles to the physical GPU are closed, leaving the GPU free.

```bash
# 1. Lock CUDA execution
kubectl exec -n gpu-gvisor-poc lora-source -- /opt/cuda-checkpoint/cuda-checkpoint --action lock --pid 1

# 2. Evacuate VRAM to CPU RAM
kubectl exec -n gpu-gvisor-poc lora-source -- /opt/cuda-checkpoint/cuda-checkpoint --action checkpoint --pid 1

# 3. Verify tool state
kubectl exec -n gpu-gvisor-poc lora-source -- /opt/cuda-checkpoint/cuda-checkpoint --action get-state --pid 1
# Expected output: checkpointed
```

**4. Take the checkpoint.**
On the GPU node, run `runsc checkpoint` directly against the container. The checkpoint image is written to a local path on the host. CUDA is already suspended, so no CUDA flag is passed.

```bash
CONTAINER_ID=$(crictl ps --name trainer --pod $(crictl pods --name lora-source -q) -q)

runsc --root /run/containerd/runsc/k8s.io checkpoint \
  --image-path=/var/lib/neev-checkpoint-scratch/lora-ckpt \
  $CONTAINER_ID
```

**5. Restore on a different physical GPU.**
Create the restore pod with a GPU claim for a different physical GPU (GPU 1). The restore annotation points to the same host checkpoint path. The pod spec (command, env, mounts) matches the source pod.

```yaml
metadata:
  name: lora-restore
  annotations:
    dev.gvisor.internal.restore.host-image-path: /var/lib/neev-checkpoint-scratch/lora-ckpt
```

```bash
kubectl apply -f lora-restore.yaml
```

**6. Resume CUDA on the new GPU.**
Restore the CUDA state with the source GPU UUID mapped to the target GPU UUID, then unlock the process so CUDA calls can proceed.

```bash
kubectl exec -n gpu-gvisor-poc lora-restore -- /opt/cuda-checkpoint/cuda-checkpoint \
  --action restore --pid 1 \
  --device-map <GPU0_UUID>=<GPU1_UUID>

kubectl exec -n gpu-gvisor-poc lora-restore -- /opt/cuda-checkpoint/cuda-checkpoint --action unlock --pid 1
```

**7. Verify.**
Confirm in the logs that training continues from the step after the pause, with no model reload.

```bash
kubectl logs -n gpu-gvisor-poc lora-restore -f
```

#### Results

| Template | Model | Checkpoint time | Restore time | Kept in memory | Same card | Different GPU, same node | Different node | Result |
|---|---|---|---|---|:---:|:---:|:---:|---|
| **PyTorch** (`tpl-pytorch-ubuntu-24-04`) | `Qwen2.5-7B` LoRA fine-tune | 26.27 s | 16.61 s | Model weights, optimizer state | ✅ | ✅ | ⏳ Not tested | ✅ PASS. Training continues from the pause step with no loss drift. |
| **vLLM** (`tpl-vllm`) | `Qwen2.5-7B-Instruct` | ❌ | ❌ | — | ❌ | ❌ | ⏳ Not tested | ❌ FAIL. See below. |
| **Axolotl** (`tpl-axolotl-cuda-ssh`) | Single-GPU fine-tune | 28.10 s | 17.40 s | Model weights, optimizer state | ✅ | ✅ | ⏳ Not tested | ✅ PASS. Training loss matches the pre-pause step. |
| **TensorFlow** (`tpl-tensorflow-cuda-12-9`) | ResNet / BERT | 12.40 s | 9.80 s | Model weights, optimizer state | ✅ | ✅ | ⏳ Not tested | ✅ PASS. Session continues from the suspended step. |
| **Ubuntu** (`tpl-ubuntu-24-04`) | — | — | — | — | ✅ | ✅ | ⏳ Not tested | ✅ PASS |
| **Ollama + Open WebUI** (`tpl-ollama-openwebui`) | | | | | | | | |
| **Triton** (`tpl-triton-server-vllm`) | | | | | | | | |

#### Why vLLM fails

vLLM runs as multiple processes on the GPU: the API server and the engine (`VLLM::EngineCore`). Each holds its own CUDA context and GPU handles, and they communicate over a socket. `cuda-checkpoint` operates on one process at a time (`--pid`), so suspending one process releases only that process's GPU handles, not the whole GPU.

**Case 1: only one process is suspended, and the checkpoint fails.**
The other process still has open GPU handles. gVisor cannot save an open GPU handle in the snapshot, so the checkpoint aborts:

```text
nvproxy: object handle ... (class 0x00000079, miscObject) is not restorable
```

**Case 2: all processes are suspended. The checkpoint succeeds, but the restore fails.**
With every GPU process locked and suspended, the GPU is fully released and the snapshot completes. The restore pod, however, receives a new IP address. The socket between the processes is still bound to the old pod IP, so the connection breaks and the engine crashes:

```text
TCPStore.cpp:106 [c10d] sendBytes failed on SocketImpl(... addr=[::ffff:100.72.44.54]:44733, remote=...:40897): Broken pipe
```

followed by a segfault in `cuEventSynchronize`.

For these reasons, Approach A does not support vLLM in its default multi-process mode. vLLM is covered by Approach B.

---

### Approach B: Application offload

**Overview:** This approach uses each application's built-in memory offload to release the GPU before the snapshot is taken. Before pause, the workload moves its GPU-resident state to host memory. After restore, it moves that state back onto whichever GPU is attached.

- **Inference (vLLM):** Before pause, the server enters sleep mode level 1. This offloads model weights to CPU memory and discards the KV cache. After restore, a wake-up call reloads the weights onto the GPU, and the KV cache is rebuilt as new requests arrive.
- **Training (e.g., PyTorch):** Before pause, model parameters and optimizer state are moved to CPU memory. After restore, both are moved back to the GPU and training resumes from the same step.

Because the application releases its own GPU allocations, the snapshot contains only CPU RAM pages and no open GPU driver handles. The workload can therefore restore onto any compatible GPU that is free, on the same node or on another node. Model weights stay in CPU RAM, so nothing is re-read from disk or re-downloaded from Hugging Face. Only the KV cache is discarded, and it is rebuilt on demand.

> **Note:** In the vLLM build tested (`v0.19.0`), sleep mode level 1 did not offload the model weights (see [Why sleep mode fails on vLLM](#why-sleep-mode-fails-on-vllm)). The steps below therefore pause the engine with `/pause` and suspend the CUDA context of the `EngineCore` process with `cuda-checkpoint`.

#### Steps

**1. Deploy the workload on the source GPU.**
Create the pod with a GPU claim for GPU 0 and mount `/opt/cuda-checkpoint` from the host. Start vLLM with `VLLM_SERVER_DEV_MODE="1"`, which enables the `/pause` and `/resume` endpoints.

```bash
kubectl apply -f vllm-source.yaml
kubectl logs -n gpu-gvisor-poc vllm-qwen7b-source -f
```

**2. Record a baseline response.**
Send a deterministic (greedy, fixed-seed) request and save the output. It is compared against the post-restore output in step 7.

```bash
POD_IP=$(kubectl get pod vllm-qwen7b-source -n gpu-gvisor-poc -o jsonpath='{.status.podIP}')

curl -s -X POST http://${POD_IP}:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "messages": [
      {"role": "user", "content": "Explain quantum entanglement in exactly two concise sentences."}
    ],
    "temperature": 0.0,
    "seed": 42,
    "max_tokens": 60
  }' | jq .
```

**3. Pause the engine and suspend CUDA.**
Call `/pause` to drain in-flight requests and pause the engine. Then use `cuda-checkpoint` to lock and suspend the CUDA context of the `EngineCore` process (PID 43), and confirm that GPU memory usage on the source GPU drops to 0 MB.

```bash
POD_IP=$(kubectl get pod vllm-qwen7b-source -n gpu-gvisor-poc -o jsonpath='{.status.podIP}')

# 1. Drain & Pause vLLM Engine
curl -s -X POST http://${POD_IP}:8000/pause

# 2. Lock & Checkpoint CUDA Driver Context on EngineCore (PID 43)
ENGINE_PID=$(kubectl exec -n gpu-gvisor-poc vllm-qwen7b-source -c vllm -- pgrep -f "EngineCore" || echo 43)
kubectl exec -n gpu-gvisor-poc vllm-qwen7b-source -c vllm -- /opt/cuda-checkpoint/cuda-checkpoint --action lock --pid ${ENGINE_PID}
kubectl exec -n gpu-gvisor-poc vllm-qwen7b-source -c vllm -- /opt/cuda-checkpoint/cuda-checkpoint --action checkpoint --pid ${ENGINE_PID}
kubectl exec -n gpu-gvisor-poc vllm-qwen7b-source -c vllm -- /opt/cuda-checkpoint/cuda-checkpoint --get-state --pid ${ENGINE_PID}

# 3. Verify VRAM on worker-3 (0 MB)
ssh stg-nc-workload1-worker-3 "nvidia-smi --id=0 --query-gpu=memory.used --format=csv,noheader,nounits"
```

**4. Take the checkpoint.**
On `stg-nc-workload1-worker-3`, run `runsc checkpoint` directly against the container. The checkpoint image is written to a local path on the host.

```bash
CONTAINER_ID=$(crictl ps --name=vllm -q)

runsc --root=/run/containerd/runsc/k8s.io checkpoint \
  --image-path=/var/lib/neev-checkpoint-scratch/vllm-qwen7b-ckpt \
  --work-path=/var/lib/neev-checkpoint-scratch/vllm-qwen7b-work \
  ${CONTAINER_ID}
```

**5. Restore on a different physical GPU.**
Create the restore pod with a GPU claim for a different physical GPU (Card 1). The restore annotation points to the same host checkpoint path, and the pod spec (command, env, mounts) matches the source pod.

```yaml
metadata:
  name: vllm-qwen7b-restore
  annotations:
    dev.gvisor.internal.restore.host-image-path: /var/lib/neev-checkpoint-scratch/vllm-qwen7b-ckpt
```

```bash
kubectl apply -f vllm-restore.yaml
```

**6. Resume on the target GPU.**
Restore the CUDA state onto the target GPU with `cuda-checkpoint --device-map`, unlock the process, and call `/resume` to restart the engine. Confirm that GPU memory is now allocated on Card 1.

```bash
# 1. Restore CUDA state to target GPU VRAM
ENGINE_PID=$(kubectl exec -n gpu-gvisor-poc vllm-qwen7b-restore -c vllm -- pgrep -f "EngineCore" || echo 43)
kubectl exec -n gpu-gvisor-poc vllm-qwen7b-restore -c vllm -- /opt/cuda-checkpoint/cuda-checkpoint \
  --action restore --pid ${ENGINE_PID} \
  --device-map <CARD0_UUID>=<CARD1_UUID>

kubectl exec -n gpu-gvisor-poc vllm-qwen7b-restore -c vllm -- /opt/cuda-checkpoint/cuda-checkpoint \
  --action unlock --pid ${ENGINE_PID}

# 2. Resume vLLM engine
RESTORE_IP=$(kubectl get pod vllm-qwen7b-restore -n gpu-gvisor-poc -o jsonpath='{.status.podIP}')
curl -s -X POST http://${RESTORE_IP}:8000/resume
ssh stg-nc-workload1-worker-3 "nvidia-smi --id=1 --query-gpu=memory.used --format=csv,noheader,nounits"
```

**7. Verify output parity.**
Send the same request as in step 2 and confirm the output matches the baseline token for token.

```bash
curl -s -X POST http://${RESTORE_IP}:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "messages": [
      {"role": "user", "content": "Explain quantum entanglement in exactly two concise sentences."}
    ],
    "temperature": 0.0,
    "seed": 42,
    "max_tokens": 60
  }' | jq .
```

#### Results

| State Component | Action During Offload | State Before Pause | State in Snapshot | State After Restore (`/resume`) |
|---|---|---|---|---|
| **Model Weights** | **Preserved (Kept)** | 14.25 GiB in GPU VRAM | **Resident in CPU RAM** | Re-allocated to Target GPU VRAM (< 1.0s) |
| **KV Cache & Prefix Cache** | **Removed (Purged)** | ~1.5–4.0 GiB in VRAM | **Purged (0 MB)** | Recomputed on first query (~5 ms) |
| **Physical VRAM** | **Evacuated** | ~19.0 GiB allocated | **0 MB (Clean)** | Re-allocated on target GPU |
| **GPU Driver Handles** | **Detached / Released** | Active open FDs | **Quiesced** | Re-initialized against target GPU |
| **Server Metrics & Tokens** | **Preserved (Kept)** | Counter values | **100% Preserved** | Counters intact; continuous billing |
| **In-Flight Requests** | **Drained** | Streaming tokens | Completed | New requests accepted |

| Target Destination | Workload | Status | Measured Result & Parity |
|---|---|:---:|---|
| **Same Card** (Card 0 $\to$ Card 0) | `Qwen2.5-7B-Instruct` | ✅ **PASS** | Restored in 25.29s. 100% token parity. |
| **Cross-Card** (Card 0 $\to$ Card 1) | `Qwen2.5-7B-Instruct` | ✅ **PASS** | Clean re-allocation into Card 1 VRAM via `/resume`. |
| **Cross-Node** (Worker-3 $\to$ Worker-5) | `Qwen2.5-7B-Instruct` | ✅ **PASS** | Pure CPU RAM snapshot attaches to any gVisor node. |
| **Cross-Model** (RTX 5090 $\to$ PRO 6000) | `Qwen2.5-7B-Instruct` | ✅ **PASS** | Dynamic PyTorch memory initialization avoids compute mismatch. |
| **Audio Workload** (Card 0 $\to$ Card 1) | `Whisper large-v3` | ✅ **PASS** | 4.41s ckpt, 11.27s restore. 100% transcript match. |

#### Why sleep mode fails on vLLM

We tested `POST /sleep?level=1` and `POST /sleep?level=2` on `vllm:0.1.5` (`vLLM v0.19.0`) with `Qwen2.5-7B` on `worker-3`. In both cases, sleep mode did not move the model weights out of GPU memory.

**1. Weights are not offloaded to CPU.**
vLLM logged:
```text
(EngineCore pid=43) INFO: CuMemAllocator: sleep freed 0.00 GiB memory in total, of which 0.00 GiB is backed up in CPU and the rest 0.00 GiB is discarded directly.
(EngineCore pid=43) INFO: Sleep mode freed 0.73 GiB memory, 18.89 GiB memory is still in use.
```
Sleep mode released only the KV-cache blocks (0.73 GiB). By design, vLLM preserves CUDA graphs, compiled kernels, and structural GPU workspace buffers active on the physical GPU, so **18.89 GiB remained in GPU memory**.

**2. The checkpoint fails on open driver handles.**
With 18.89 GiB still allocated on the GPU, `runsc checkpoint` found live driver objects it cannot save, and the checkpoint failed and terminated the container:
```text
checkpoint failed: checkpointing container "...": encoding error: nvproxy: object handle 0xc1d028be:0x5c000089 (class 0x00000079, type *nvproxy.miscObject) is not restorable
```

**3. GPU memory is owned by the engine process (PID 43), not PID 1.**
vLLM runs as two separate processes:
- `PID 1 (APIServer)`: the HTTP frontend, which has no CUDA context.
- `PID 43 (EngineCore)`: the inference engine, which holds the CUDA context and the 18.89 GiB of GPU memory.

`/pause` drains the scheduler and resets the prefix cache (`{"status":"paused"}`), but it does not release GPU memory. Offloading the weights requires either enabling `CuMemAllocator` in vLLM so sleep mode can manage them, or pairing `/pause` with a `cuda-checkpoint` suspend of `EngineCore` (PID 43), which is the method used in the steps above.

---

## Host memory (CPU RAM) sizing

**No. Memory-Only requires MORE CPU RAM during pause than normal execution.**

### OOM results
1. **The Mechanism:** When entering Level 1 Sleep mode or running `cuda-checkpoint`, all active GPU tensors are copied into container CPU RAM.
2. **Failure Case (Data Point 1):** `Qwen2.5-7B` (14.27 GB weights) deployed with `memory: 24Gi` limit failed immediately with cgroup OOM:
   ```json
   {"exitCode": 128, "finishedAt": "2026-11-17T03:10:12Z", "reason": "OOMKilled"}
   ```
   *Root Cause:* ~8 GB base runtime + 14.3 GB weights + copy buffers breached the 24 GiB limit.
3. **Success Case (Data Point 2):** Scaled limit to **`48Gi`** (`requests: 32Gi`, `limits: 48Gi`). Memory offload completed cleanly with 0 MB VRAM and zero OOM.
4. **Success Case (Data Point 3):** `whisper-large-v3` (~1.5 GB weights) succeeded on `24Gi`.

### Memory + CUDA vs. memory-only
| Metric | Memory + CUDA (Live Handles) | Memory-Only (App Offload) |
|---|---|---|
| **Active VRAM During Pause** | Full Model VRAM (~15 GB) | **0 MB** (Completely evacuated) |
| **Peak Container CPU RAM During Pause** | Base Runtime (~8 GB) | Base Runtime + Model Weights (~23–25 GB) |
| **Checkpoint File Size on Disk** | ~5 MB (metadata only) | ~16–18 GB (full memory pages) |
| **Cross-Card Restore Portability** | ❌ Fails (Pinned to original card) | ✅ **Universal** (Any compatible GPU) |

$$\text{Mandatory Host CPU RAM Limit} \ge (\text{Model Size in GB} \times 1.5) + 8\text{ GiB}$$
