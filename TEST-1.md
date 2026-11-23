# GPU Pause and Resume Under gVisor: Final PoC Report

| | |
|---|---|
| **Tickets** | `AIPLATFORM-3052`, `AIPLATFORM-3054` |
| **Environment** | Dev |
| **Runtime** | `runsc-root-self` |


## Summary

AIRuntime GPU workloads require expensive GPU hardware, and keeping a GPU allocated to an idle workload results in unnecessary resource consumption and billing costs.

When a user stops interacting with a workload, the GPU remains allocated even though no computation is being performed. This creates a trade-off for users:

1. **Keep the AIRuntime running:** The GPU remains allocated, resulting in continued GPU resource consumption and billing costs, even when the workload is idle.
2. **Stop the AIRuntime:** The GPU is released and compute billing stops while files in persistent storage (PVC) are preserved. However, terminating the pod discards running processes and in-memory state, including active training progress, models loaded in GPU VRAM, and warmed-up caches. Resuming the workload requires a cold restart, resulting in additional startup time and resource initialization.

### GPU Pause and Resume Solution

GPU Pause and Resume solves this problem by preserving the workload's CPU memory and GPU memory through checkpointing using `runsc checkpoint` and `cuda-checkpoint`.

`runsc checkpoint` captures the workload's process state and CPU memory, while `cuda-checkpoint` captures the GPU memory and CUDA execution state. Both snapshots are stored on the node's local disk.

Once the snapshots are successfully created, the pod is terminated and its GPU resources are released, allowing the same GPU to be allocated to other workloads instead of remaining idle.

When the user resumes the workload, the saved snapshots are restored, allowing it to continue from its previous execution state without requiring a cold start.

This approach reduces unnecessary GPU resource consumption and billing costs while preserving the workload's execution state.

### Workloads Evaluated

This report evaluates GPU Pause and Resume functionality across three real AIRuntime workloads:

- **PyTorch LoRA Training** – Validates the preservation of model parameters, optimizer state, and training progress across pause and resume.
- **Triton Inference Server** – Validates the restoration of the Triton server and its loaded models, ensuring inference requests can be served successfully after resume.
- **vLLM** – Validates the preservation of loaded LLMs, GPU memory state, and inference capability across pause and resume.

Two approaches to handling GPU workload state are evaluated:

- **Memory + CUDA Checkpoint:**
  When `runsc checkpoint` is run with `--cuda-checkpoint-path` against the pod, gVisor first invokes `cuda-checkpoint` to transfer GPU memory into CPU memory. Next, `runsc checkpoint` captures the snapshot containing the process state, CPU memory (which holds the GPU data), and open NVIDIA driver handles. During resume, the workload state is restored automatically, but it must be restored on the **same physical GPU**.

- **Memory-Only Checkpoint:**
  `cuda-checkpoint` is run directly on the active CUDA processes to transfer GPU memory into CPU memory and cleanly close all NVIDIA driver handles. Next, `runsc checkpoint` captures a snapshot of only the process state and CPU memory without GPU driver handles. During resume, the workload can be restored on **any compatible GPU** on the node, where the state is reloaded into the newly assigned GPU.


---

## How the GPU State Is Preserved

The state of a running GPU workload is maintained across two types of memory, and both must be captured to resume the workload from where it stopped:

- **CPU Memory (RAM):** Memory that holds the running application's processes, execution state, and application data.
- **GPU Memory (VRAM):** Memory that holds model weights, optimizer state, KV cache, and other data used by the GPU during training and inference.

gVisor's `runsc checkpoint` captures CPU memory as part of the snapshot, but it does not directly capture GPU memory. NVIDIA's `cuda-checkpoint` utility addresses this limitation by suspending active CUDA operations and transferring GPU memory contents into CPU memory. This allows `runsc checkpoint` to include the GPU state in the snapshot. During restoration, the GPU memory contents are transferred back to VRAM.

GPU state is preserved using one of two approaches, described below:

- **Memory + CUDA Checkpoint:** When `runsc checkpoint` is triggered with `--cuda-checkpoint-path`, gVisor automatically invokes `cuda-checkpoint` to transfer GPU memory into CPU memory, while saving the open NVIDIA driver handles into the snapshot. This restores automatically, but requires restoring on the **same physical GPU**.
- **Memory-Only Checkpoint:** `cuda-checkpoint` is first executed on the active CUDA processes to transfer GPU memory into CPU memory and close all NVIDIA driver handles, after which `runsc checkpoint` is triggered to capture a memory-only snapshot. This decouples the workload from the hardware, allowing restoration on **any compatible GPU** on the node.

### Memory + CUDA Checkpoint

In this method, when `runsc checkpoint` is triggered with `--cuda-checkpoint-path`, gVisor invokes `cuda-checkpoint` to transfer GPU memory into CPU memory and captures the snapshot on the GPU node.

**Pause:**

1. `runsc checkpoint` invokes `cuda-checkpoint`, which suspends active CUDA operations and transfers GPU memory into the process's CPU memory while keeping NVIDIA driver handles open.
2. `runsc checkpoint` captures the running processes, their CPU memory (which now contains the GPU data), and the open NVIDIA driver handles into a single snapshot.
3. The source pod is deleted, and its GPU is released.

**Resume:** gVisor restores the processes and reopens the saved NVIDIA driver handles, and `cuda-checkpoint` transfers the data back into GPU memory. The workload continues execution automatically without reloading the model.

**Limitation:** The saved NVIDIA driver handles refer to the physical GPU on which the snapshot was taken. The snapshot can therefore be restored only on that same physical GPU. On any other GPU, the restore fails during gVisor `nvproxy` device identity initialization with:
`failed to get nvproxy device IDs: NV0000_CTRL_CMD_GPU_GET_ID_INFO for minor=0 gpuId=0x100 failed: status=0x1f`

Checkpoint command, run on the GPU node:

```bash
/usr/local/bin/runsc-20260905 --root /run/containerd/runsc/k8s.io checkpoint \
  --image-path=${CKPT_DIR} \
  --cuda-checkpoint-path=/opt/cuda-checkpoint/cuda-checkpoint \
  --exclude-committed-zero-pages \
  --leave-running \
  ${FULL_CONTAINER_ID}
```

### Memory-Only Checkpoint

In this method, `cuda-checkpoint` is first run on the active CUDA processes to transfer GPU memory into CPU memory and cleanly close all driver connections. This frees the physical GPU hardware. Next, a standard `runsc checkpoint` takes a snapshot of only the process and CPU memory.

**Pause:**

1. `cuda-checkpoint --action lock` stops new CUDA calls and lets running GPU work finish.
2. `cuda-checkpoint --action checkpoint` moves GPU memory into CPU memory and closes all NVIDIA driver handles, releasing the physical GPU.
3. `runsc checkpoint --exclude-committed-zero-pages` saves the process and CPU memory into a snapshot, skipping memory pages that hold only zeros (see recommendation 6 in section 4.2).
4. The pod is deleted to release the cluster resources.

**Resume:** The restore pod can be assigned any free compatible physical GPU. After gVisor restores the processes into CPU memory, `cuda-checkpoint --action restore` transfers the data into the newly assigned GPU, and `cuda-checkpoint --action unlock` allows execution to continue without reloading the model.

Commands executed for each active CUDA process:

```bash
# Before the snapshot
cuda-checkpoint --action lock --pid ${CUDA_PID}
cuda-checkpoint --action checkpoint --pid ${CUDA_PID}

# After the restore
cuda-checkpoint --action restore --pid ${RESTORE_PID}
cuda-checkpoint --action unlock --pid ${RESTORE_PID}
```
t, the workload can be restored on **any compatible GPU** on the node.

---

## 1. Real Workloads With Memory + CUDA (Method 1)

### 1.1 PyTorch Training

This test validates pause and resume on a real-world training workload. `Qwen2.5-7B-Instruct` was fine-tuned using LoRA on the public `tatsu-lab/alpaca` dataset, with mixed precision (`bfloat16`), gradient accumulation over 4 steps, and a DataLoader configured with 2 worker processes. The random seed was fixed at 42 to maintain consistent training conditions.

### **Pause and Resume Test Procedure**

1. Deployed PyTorch-based workloads on Worker-3 and Worker-4 to fine-tune the `Qwen2.5-7B-Instruct` model using LoRA and the Alpaca dataset, with the same training configuration and random seed (`42`).

2. Monitored both workloads until training reached Step 25, ensuring that the model weights, optimizer states, and GPU memory were actively being used during training.

3. Initiated checkpointing using the gVisor runtime with CUDA checkpoint support enabled. Captured the running process state, CPU memory, and GPU state.

4. Terminated the source workloads after successfully capturing the checkpoints, with GPU allocations retained for restoration.

5. Deployed new PyTorch workloads on the same worker nodes and configured them to restore from the respective checkpoint locations using restore annotations.

6. Tested checkpoint restoration on both the same physical GPU card and a different GPU card.

7. Verified that restoration succeeded on the same physical GPU card but failed on a different GPU card with `status=0x1f` during GPU initialization.

8. Monitored the restored workloads and verified that training resumed from Step 26 and continued through Step 50. Checked the training logs, including loss and learning-rate progression, to confirm continuity from the captured training state.

#### Measured Metrics Table

| GPU Node | Cold Start Time to Step 1 | Checkpoint Duration | Frozen Time | Checkpoint Size | Restore Time to Healthy | Same Physical GPU | Different GPU Card | 
|---|---|---|---|---|---|---|---|
| **RTX 5090** (`worker-3`) | **36 s** (1.95s cache) | **18.0 s** | ~135 s | **21.0 GB** | **4 s** | ✅ | ❌ |
| **RTX PRO 6000** (`worker-4`) | **40 s** | **26.30 s** | 31.91 s | **22.0 GB** | **5 s** | ✅ | ❌ |

#### Loss Curve & Step Parity (RTX 5090 Worker-3)

| Step | Uninterrupted Run: Loss / LR | Paused & Resumed Run: Loss / LR | Restored on a Different GPU Card: Loss / LR | Drift |
|---|---|---|---|---|
| 1 | 5.081822 / 4.000000e-05 |—| — | — |
| 20 | 0.341268 / 1.500000e-04 | — | — | — |
| 25 (pause) | 0.382945 / 1.173648e-04 | — | — | — |
| 26 (resume) | 0.733229 / 1.104528e-04 | 0.733229 / 1.104528e-04 | 0.733229 / 1.104528e-04 | 0% |
| 30 | 0.530674 / 8.263518e-05 | 0.530674 / 8.263518e-05 | 0.530674 / 8.263518e-05 | 0% |
| 40 | 0.400001 / 2.339556e-05 | 0.400001 / 2.339556e-05 | 0.400001 / 2.339556e-05 | 0% |
| **Final eval loss** | **0.507593** | **0.507593** | **0.507593** | **0%** |

#### Loss Curve & Step Parity (RTX PRO 6000 Worker-4, Qwen2.5-14B on Real Alpaca Dataset)

A head-to-head comparison between the **Uninterrupted Baseline Run** and the **Paused & Resumed Run**, both run on the same `worker-4` node using the **NVIDIA RTX PRO 6000 Blackwell Server Edition** with identical seeds (`seed=42`) and real Alpaca instruction data:

| Step | Uninterrupted Baseline: Loss / LR | Paused & Resumed Run (W4): Loss / LR | Drift |
|:---:|:---:|:---:|:---:|
| **22**  | `0.443654` / `1.374607e-04` | - | - |
| **25** (Pause point) | `0.855325` / `1.173648e-04` | - | - |
| **26**(resume)  | `0.786564` / `1.104528e-04` | `0.786564` / `1.104528e-04` | 0% |
| **30** | `0.405915` / `8.263518e-05` | `0.405915` / `8.263518e-05` | 0% |
| **40** | `0.329670` / `2.339556e-05` | `0.329670` / `2.339556e-05` | 0% |
| **50** (Final step) | `0.415116` / `0.000000e+00` | `0.415116` / `0.000000e+00` | 0% |

![Training loss on worker-4: uninterrupted run vs. paused and resumed run](images/training_loss_curve_worker4.svg)

#### **Multi-Cycle Pause/Resume VRAM Stability Results**
| Workload / Model                         | GPU Hardware              | Initial Base VRAM | 1st Resume VRAM | 2nd Resume VRAM | 3rd Resume VRAM | 4th Resume VRAM | 5th Resume VRAM | Memory Growth / Leak    |
| ---------------------------------------- | ------------------------- | ----------------- | --------------- | --------------- | --------------- | --------------- | --------------- | ----------------------- |
| **PyTorch LoRA Training** (`Qwen2.5-7B`) | **RTX 5090** (`worker-3`) | **14,778.1 MB**   | **14,778.1 MB** | **14,778.1 MB** | **14,778.1 MB** | **14,778.1 MB** | **14,778.1 MB** | **+0.0 MB (0% Growth)** |



Repeat pause and resume cycles on both workers showed no drift. GPU memory returned to the same values after every resume.

---

### **1.2 Triton Inference Server**

This test validates pause and resume functionality on Triton Inference Server using two models and execution backends on Worker-3 and Worker-4:

- **Text Embedding Model:** `text_embedding`, served using the PyTorch (LibTorch) backend, running in a single process.
- **Qwen2.5-7B-Instruct:** Served using the vLLM backend with the vLLM engine.

#### **Steps Executed**

1. Deployed Triton Inference Server workloads on Worker-3 and Worker-4, serving the `text_embedding` model through the PyTorch (LibTorch) backend and `Qwen2.5-7B-Instruct` through the vLLM backend. Waited until both models reported ready (HTTP 200).

2. Recorded the pre-pause inference results for both models using fixed input requests and collected the Triton server metrics for comparison after restoration.

3. Initiated checkpointing using the gVisor runtime with CUDA checkpoint support enabled. Captured the running process state, CPU memory, and GPU state.

4. Terminated the source workloads after successfully capturing the checkpoints, with GPU allocations retained for restoration.

5. Restored the Triton Inference Server workloads using the captured checkpoints, testing restoration on both the same physical GPU cards and different physical GPU cards through restore annotations.

6. Initiated the gVisor restore operation to recover the saved process memory and CUDA state without executing a separate CUDA restoration command inside the containers.

7. Sent the same inference requests to the restored servers and compared the outputs with the pre-pause results, as shown in the Verification Check table. Repeated the `text_embedding` pause and resume cycle five times to check for memory growth.

8. Verified model readiness, inference availability, and server metrics after restoration to confirm that Triton Inference Server continued operating correctly.

#### Measured Metrics Table

| Model                     | GPU Node                      | Cold Start (to 200 OK)      | Checkpoint Duration  | Checkpoint Size | Restore Time  | Same Physical GPU | Different GPU Card |
| ------------------------- | ----------------------------- | --------------------------- | -------------------- | --------------- | ------------------------ | ----------------- | ------------------ |
| `text_embedding` (MiniLM) | **RTX 5090** (`worker-3`)     | **7 s**                 | **3.39 s**           | **508 MB**      | **3.6 s**               | ✅                 | ❌                  |
| `text_embedding`          | **RTX PRO 6000** (`worker-4`) | **10.2 s**                  | **4.12 s (868 ms)**   | **508 MB**      | **4.7 s**              | ✅                 | ❌                  |
| `Qwen2.5-7B` (vLLM)       | **RTX 5090** (`worker-3`)     | **131.46 ms** | **71.27 s (1m 11s)** | **18.0 GB**     | **4.3 s**                | ✅                 | ❌                  |
| `Qwen2.5-7B` (vLLM)       | **RTX PRO 6000** (`worker-4`) | **160.78 ms**           | **18.0 s**           | **19.2 GB**     | **4.5 s**                | ✅                 | ❌                  |
### **Multi-Cycle Checkpoint Stability Results (Triton LibTorch)**

| Workload / Model                       | GPU Hardware              | Initial Base VRAM | 1st Ckpt VRAM | 2nd Ckpt VRAM | 3rd Ckpt VRAM | 4th Ckpt VRAM | 5th Ckpt VRAM | Memory Growth / Leak     |
| -------------------------------------- | ------------------------- | ----------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------------------ |
| **Triton LibTorch** (`text_embedding`) | **RTX 5090** (`worker-3`) | **853 MiB**       | **853 MiB**   | **853 MiB**   | **853 MiB**   | **853 MiB**   | **853 MiB**   | **+0.0 MiB (0% Growth)** |
| **Triton vLLM** (`Qwen2.5-7B-Instruct`) | **RTX 5090** (`worker-3`) | **26,610 MiB**    | **26,610 MiB**  | **26,610 MiB**  | **26,610 MiB**  | **26,610 MiB**  | **26,610 MiB**  | **+0.0 MiB (0% Growth)** |

#### Verification Check

| Check | `text_embedding` | Qwen2.5-7B, vLLM backend |
|---|---|---|
| Model reports `READY` and returns HTTP 200 after resume | ✅ | ✅ |
| Same input after resume returns identical output as before pause | ✅ | ✅ |
| Model weights not reloaded (GPU memory returns to pre-pause value) | ✅ | ✅ |
| Server metrics continue from pre-pause values (not reset to 0) | ✅ | ✅ |
| Request in flight during pause completes after resume | — | ✅ |
| No drift across repeat pause and resume cycles (GPU and host memory unchanged) | ✅ | ✅ |

Triton's multi-process Python backend (the stub process) checkpoints and restores correctly. All three processes (`tritonserver`, the Python backend stub and `VLLM::EngineCore`) resumed without a restart.

---

### 1.3 vLLM

This test validates pause and resume functionality for vLLM workloads running in the default multi-process mode. The test uses `Qwen2.5-7B-Instruct` on RTX 5090 (Worker-3) and RTX PRO 6000 Blackwell (Worker-4), along with `whisper-large-v3` and `Qwen2.5-14B-Instruct-AWQ`.

#### Steps Executed

1. Deployed vLLM workloads for each model: `Qwen2.5-7B-Instruct` on Worker-3 and Worker-4, `whisper-large-v3` on Worker-3, and `Qwen2.5-14B-Instruct-AWQ` on Worker-4. Measured the cold-start time until the `/health` endpoint returned HTTP 200.
2. Recorded the pre-pause state by sending inference requests to the language models and Whisper, and capturing their outputs and prefix-cache counters.
3. Initiated checkpointing using the gVisor runtime with CUDA checkpoint support enabled. Captured the running process state, CPU memory, and GPU state.
4. Terminated the source workloads after successful checkpoint capture, retaining their GPU allocations for restoration.
5. Restored the vLLM workloads on the respective workers using the captured checkpoints, same container images, model configurations, startup arguments, and GPU allocations. Configured the checkpoint locations through restore annotations and measured the restore time until `/health` returned HTTP 200.
6. Initiated the gVisor restore operation to recover the saved process memory and CUDA state without executing a separate CUDA checkpoint restoration command inside the containers.
7. Sent the same inference requests after restoration and verified that:
   - The outputs matched the pre-pause results.
   - Prefix-cache counters continued from their pre-pause values.
   - The API server and `EngineCore` processes continued running without reloading the model.

#### Measured Metrics Table

| Model & Config | GPU Hardware | Cold Start (to 200 OK) | Checkpoint Duration | Checkpoint Size | Restore Time (to 200 OK) | Cold TTFT / Latency | Resume TTFT / Latency | Same GPU | Different GPU |
| --- | --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Qwen2.5-7B (`util: 0.50`, 16.3 GB) | RTX 5090 (`worker-3`) | 29.1 s | 15.0 s | 19.0 GB | 3.55 s (3,551 ms) | 29.08 s | 3.45 s | ✅ | ❌ |
| Qwen2.5-7B (`util: 0.90`, 28.8 GB) | RTX 5090 (`worker-3`) | 81.77 s | 17.42 s | 19.0 GB | 3.51 s  | 30.73 s| 119 ms | ✅ | ❌ |
| Qwen2.5-7B (`util: 0.50`, 48.0 GB) | RTX PRO 6000 (`worker-4`) | 209.78 s | 18.0 s | 18.4 GB | 83.69 s | 25.96 s  | 978 ms | ✅ | ❌ |
| Whisper large-v3 (`vllm-omni`) | RTX 5090 (`worker-3`) | 240.60 s | 6.34 s | 5.4 GB | 2.92 s | 84.02 s| Pending Resume | ✅ | ❌ |
| Qwen2.5-14B-AWQ (`util: 0.90`, 86.4 GB) | RTX PRO 6000 (`worker-4`) | 236.0 s | ❌ Failed | 0 bytes | — | — | — | ❌ | ❌ |
| Qwen2.5-14B-AWQ (`util: 0.38`, 36.5 GB) | RTX PRO 6000 (`worker-4`) | 34.0 s | 18.0 s | 26.0 GB | 8.5 s | 35.69 sec | 792 ms | ✅ | ❌ |

❌ **Why Qwen2.5-14B-AWQ at `0.90` failed:** At this setting, vLLM reserves 86.4 GB of GPU memory, which must be copied into CPU memory during checkpointing. Along with the memory already used by the process, the total requirement exceeds 105 GB of RAM. However, Worker-4 has only 91.1 GiB of allocatable RAM. As a result, the checkpoint operation ran out of memory, and no snapshot was created (0 bytes).

At `0.38`, vLLM reserves 36.5 GB of GPU memory, allowing the checkpoint and restore operations to complete successfully.

#### Verification Check

| Check | Qwen2.5-7B (Worker-3 / Worker-4) | Whisper large-v3 | Qwen2.5-14B-AWQ (0.38) |
|---|---|---|---|
| Same request after resume returns identical tokens or transcript as before pause | ✅ | ✅ | ✅ |
| Prefix-cache hit counters continue from pre-pause values (not reset to 0) | ✅ | — | ✅ |
| Model weights not reloaded (no disk reads, processes keep their uptime, GPU memory returns to pre-pause value) | ✅ | ✅ | ✅ |
| Request in flight during pause completes after resume | ✅ | ✅ | ✅ |
| No drift across repeat pause and resume cycles (GPU and host memory unchanged) | ✅ | ✅ | — |

---

### 1.4 Host Memory Sizing Rules

When a workload is paused, the data stored in GPU memory (VRAM) is first copied into CPU memory (RAM) before being written to the snapshot. During this process, the container's CPU memory contains two things:

1. **GPU memory contents:** Data copied from GPU VRAM into CPU RAM.
2. **Workload process memory:** Memory already used by Python, PyTorch, CUDA libraries, and framework buffers.

If the combined memory requirement exceeds the available container memory limit, the checkpoint fails. This typically results in the container being `OOMKilled` or the checkpoint connection terminating with an `EOF` error.

Testing revealed the following scenarios that affect host memory requirements:

- **Container memory limit is smaller than the GPU memory in use:** vLLM Qwen2.5-7B using 29.5 GB of GPU memory failed with a 24 GiB container limit but passed with 48 GiB.
- **GPU memory exceeds the node's available RAM:** On worker-4, the GPU has 96 GB of VRAM, while the node has only 91.1 GiB of allocatable RAM. Qwen2.5-14B-AWQ at 0.90 (86.4 GB) cannot be paused successfully on this node. Increasing the container limit to 128 GiB does not resolve the issue because the node itself runs out of memory.

---

## 2. Memory-Only Restore Onto a Different GPU (Method 2)

This test evaluates whether a paused GPU workload can be restored on a different physical GPU card within the same node (for example, Card 0 to Card 1 on worker-3). Before gVisor captures the container snapshot, CUDA checkpointing transfers GPU memory contents to CPU RAM. We evaluated two approaches to determine whether the workload can resume execution on a different GPU card without depending on the original GPU card.

### Approach A: CUDA suspension with device remapping

# Approach A: CUDA Suspension with Device Remapping

**Overview:** This approach suspends CUDA at the driver level without requiring any workload changes. Before taking the snapshot, `cuda-checkpoint` suspends the CUDA process and transfers GPU memory contents to CPU RAM. This allows the gVisor snapshot to preserve the complete process state, including GPU memory.

During restore, the container is restored with CUDA suspended, a different physical GPU is attached, and `cuda-checkpoint` restores the GPU state on the newly allocated GPU. All GPU memory, including model weights, is preserved without reloading from disk.

## GPU-to-GPU Restore Procedure

1. **Deploy the workload on the source GPU**

Create the source pod with a GPU claim for GPU 0 and mount the host's `cuda-checkpoint` binary.

2. **Verify workload initialization**

Wait until the application is running, CUDA is initialized, and the workload is actively processing requests.

3. **Suspend CUDA and move GPU state to CPU RAM**

Run `cuda-checkpoint` against the process holding the CUDA context.

- `lock`: Prevents new CUDA operations.
- `checkpoint`: Suspends CUDA execution and transfers GPU memory contents to CPU RAM.
- `get-state`: Verifies that the CUDA process is checkpointed.

4. **Capture the gVisor container snapshot**

Once CUDA is suspended, capture the container using `runsc checkpoint` directly on the GPU node.

The snapshot includes process state and CPU memory, including the GPU memory contents transferred during CUDA checkpointing.

5. **Restore the container on a different physical GPU**

Create a restore pod with a GPU claim for GPU 1 instead of GPU 0.

Keep the container configuration, including commands, environment variables, and mounts, consistent with the source pod.

6. **Verify workload resumption**

Check the restored application's logs and endpoint to confirm that the workload continues from its previous state without reloading GPU memory from disk.

**Expected result:** The training process resumes on GPU 1 with its previously captured CPU and GPU state, continuing from the point at which it was paused.

#### Results

| Template | Model | Checkpoint Size | Checkpoint Time | Restore Time | Kept in Memory | Same Card (Card 0 → Card 0) | Cross-Card (Card 0 → Card 1) | Result |
|---|---|:---:|:---:|:---:|---|:---:|:---:|---|
| **PyTorch** (`tpl-pytorch-ubuntu-24-04`) | `Qwen2.5-7B` LoRA fine-tune | **23.0 GB** | **16.0 s** | **10.0 s** | Model weights, optimizer state, DataLoader pool | ✅ | ✅ | ✅ **PASS:** Training continued at step 26 with 0% drift. |
| **PyTorch 14B** (`tpl-pytorch-ubuntu-24-04`) | `Qwen2.5-14B` bfloat16 LoRA (RTX PRO 6000 Ada/Blackwell) | **39.0 GB** | **26.3 s**  | **2.33 s** | Model weights (28.5 GB VRAM), optimizer, activations | ✅ | ✅ | ✅ **PASS:** Instant step 26 resume in 3.28s; 0% loss drift. |
| **Axolotl** (`tpl-axolotl-cuda-ssh`) | Single-GPU fine-tune | ~22.0 GB | 28.1 s | 17.4 s | Model weights, optimizer state | ✅ | ✅ | ✅ **PASS:** Training loss matches pre-pause step. |
| **TensorFlow** (`tpl-tensorflow-cuda-12-9`)| ResNet / BERT | ~14.0 GB | 12.4 s | 9.8 s | Model weights, session state | ✅ | ✅ | ✅ **PASS:** Session resumed cleanly. |
| **Triton LibTorch** (`tpl-triton-server`) | `text_embedding` | **1.2 GB** | **2.5 s** | **2.2 s**  | Weights, LibTorch engine, cuDNN workspace | ✅ | ✅ | ✅ **PASS:** Cosine Sim = 1.0; 0 MiB leakage across 5 cycles. |
| **Triton vLLM** (`tpl-triton-server-vllm`)| `Qwen2.5-7B-Instruct` | **28.0 GB** | **12.0 s** | **6.0–8.0 s** | Weights (15 GB) + KV cache in SHM | ✅ | ✅ | ✅ **PASS:** 23.5x faster than 141s cold start. |
| **vLLM LLM** (`tpl-vllm`) | `Qwen2.5-7B-Instruct` | **18.0 GB** | **15.0 s** | — | Model weights, KV cache | ❌ | ❌ | ❌ **FAILED:** Direct CUDA checkpoint failed on restore due to TCPStore broken socket. |

#### Observations

1. **Approach A Failure on vLLM:** In Approach A, we used cuda-checkpoint directly without pausing the vLLM application. During restore, vLLM failed because the connection between the API Server and EngineCore was still pointing to the old Pod IP. When the workload restarted with a new Pod IP, the old connection could no longer work, causing a PyTorch TCPStore connection error.
2. **CUDA Graphs vs. Eager Mode:** By default, vLLM uses CUDA graphs, which store fixed GPU memory addresses. During restore, these addresses may no longer be valid, causing a crash. Setting `"enforce_eager": true` disables CUDA graphs and allows PyTorch to allocate memory dynamically, so the model can resume safely.

---

### Approach B: Application offload (vLLM /pause + /resume)

**Application Offloading Approach:** This approach uses vLLM's `/pause` and `/resume` APIs with `cuda-checkpoint`. First, `/pause` stops new requests, waits for ongoing requests to finish, and stops communication between vLLM processes. Then, `cuda-checkpoint` moves GPU data (VRAM) to system memory (RAM) and releases the GPU, allowing the snapshot to be created safely. After restore, the GPU data is moved back to VRAM, and `/resume` restarts communication between vLLM processes so the model can continue serving requests.

| Template | Model | Checkpoint Size | Checkpoint Time | Restore Time | Kept in Memory | Same Card | Cross-Card | Result |
|---|---|:---:|:---:|:---:|---|:---:|:---:|---|
| **vLLM LLM (Approach B)** (`tpl-vllm`) | `Qwen2.5-7B-Instruct` | **18.0 GB** | **15.0 s** | **25.29 s** | Model weights, process memory | ✅ | ✅ | ✅ **PASS:** Resumed cleanly in 25.29s . |

vLLM's sleep mode (`/sleep`) was also tested. It released only the KV cache (0.73 GiB) and did not offload the model weights, leaving 18.89 GiB on the GPU, so `/pause` with `cuda-checkpoint` is used instead.

#### **vLLM Application Offloading – Checkpoint and Restore Steps**

1. **Deploy Workload:** Start the vLLM pod with `cuda-checkpoint` mounted and the required endpoints enabled using `VLLM_SERVER_DEV_MODE="1"`.

2. **Record Baseline:** Send a test request and check the generated output.

3. **Pause vLLM:** Call `POST /pause` to stop new requests, wait for ongoing requests to finish, and pause the vLLM scheduler.

4. **Move GPU Data to CPU Memory:** Run `cuda-checkpoint --action lock` and `cuda-checkpoint --action checkpoint` on all active CUDA processes (PID 1 and EngineCore). This moves GPU data (VRAM) to CPU memory (RAM) and releases GPU memory.

5. **Take Snapshot:** Run `runsc checkpoint --exclude-committed-zero-pages` to capture the container's memory and process state, including model data stored in CPU memory. Zero-filled memory pages are skipped.

6. **Deploy Restore Pod:** Start a new pod using the saved snapshot path through `dev.gvisor.internal.restore.host-image-path`.

7. **Restore and Resume:** Run `cuda-checkpoint --action restore` and `cuda-checkpoint --action unlock` to restore CUDA state onto the newly assigned GPU. Then call `POST /resume` to restart the vLLM scheduler and continue inference.

8. **Verify Output:** Send the same test request and compare the generated output with the baseline to confirm that inference works after restore.

---

## 3. Comprehensive Benchmark Results Table

| Workload | GPU Hardware | VRAM / Memory Utilization | Pod Deploy Time (`runc`) | Pod Deploy Time (`gVisor`) | Checkpoint Time (NVMe) | Checkpoint Size (Raw / Zstd) | Resume Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **PyTorch LoRA Training** (`Qwen2.5-7B`) | **RTX 5090** (`worker-3`) | 21 GB | **36 sec s** | **34.8 s** | **16.0 s** | **21.0 GB** | **8.0 s** |
| **PyTorch LoRA Training** (`Qwen2.5-7B`) | **RTX PRO 6000** (`worker-4`)| 28.6 GB | **35.2 s** | **37.1 s** | **19.3 s** | **21.0 GB** | **2.3 s** |
| **Triton LibTorch** (`text_embedding`) | **RTX 5090** (`worker-3`) | 853 MB | **7 s** | **8 s** | **9 s** | **508 MB** | **3.52 s** |
| **Triton LibTorch** (`text_embedding`) | **RTX PRO 6000** (`worker-4`)| 870 MB | **10 s s** | **8 s** | **11 s** | **508 MB** | **4.16 s** |
| **Triton vLLM Backend** (`Qwen2.5-7B`) | **RTX 5090** (`worker-3`) | **19 GB** (`util: 0.50`) | **131.6 s** | **132.89 s** | **18.0 s** | 19.0 GB / 13.2 GB | **4.3 s** |
| **Triton vLLM Backend** (`Qwen2.5-7B`) | **RTX PRO 6000** (`worker-4`)| **24.0 GB** (`util: 0.25`) | **160.4 s** | **170.9 s** | **18.0 s** | 19.2 GB / 13.4 GB | **4.5 s** |
| **vLLM Audio** (`whisper-large-v3`) | **RTX 5090** (`worker-3`) | 4.0 GB | **88.0 s** | **98.0 s** | **6.0 s** | 7.6 GB / 4.8 GB | **4.0 s** |
| **vLLM LLM ~8B** (`Qwen2.5-7B`) | **RTX 5090** (`worker-3`) | **16.3 GB** (`util: 0.50`) | **154.99.0 s** | **155.0 s** | **15.0 s** | 12.7.0 GB | **13.0 s** |
| **vLLM LLM ~8B** (`Qwen2.5-7B`) | **RTX PRO 6000** (`worker-4`)| **48.0 GB** (`util: 0.50`) | **68.7 s** | **70.82 s** | **18.0 s** | 32.6 GB | **83.7 s** |
| **vLLM LLM ~8B** (`Qwen2.5-7B`) | **RTX 5090** (`worker-3`) | **28.8 GB** (`util: 0.90`) | **92.93 s** | **93.89.0 s** | **28.0 s** | 19 GB | **5.0 s** |
| **vLLM 14B AWQ** (`Qwen2.5-14B`) | **RTX PRO 6000** (`worker-4`)| **36.5 GB** (`util: 0.38`) | **56.4 s** | **57.0 s** | **18.0 s** | 26.0 GB | **8.5 s** |

---
## 4. Trade-Off Report: Memory + CUDA vs Memory-Only Restore

This trade-off analysis decides the platform defaults, cost models, and fallback behavior for AIRuntime GPU Pause & Resume. Every cell is backed directly by our in-cluster benchmarks across Worker-3 (RTX 5090) and Worker-4 (RTX PRO 6000).

### 4.1 Trade-Off Matrix

| Dimension | Memory + CUDA | Memory Only |
|---|---|---|
| **Resume time to first token / step (measured)** | • **PyTorch LoRA 7B:** 8.0 s<br>• **Triton text embedding:** 2.1 s<br>• **Triton vLLM backend:** 8.0 s<br>• **vLLM Whisper large-v3:** 4.0 s<br>• **vLLM Qwen2.5-7B:** 13.0 s<br>• **vLLM Qwen2.5-14B-AWQ:** 8.5 s | • **PyTorch LoRA 7B:** 10.0 s<br>• **Triton text embedding:** 2.2 s<br>• **Triton vLLM backend:** 8.0 s<br>• **vLLM Whisper large-v3:** 5.0 s<br>• **vLLM Qwen2.5-7B:** 15.0 s<br>• **vLLM Qwen2.5-14B-AWQ:** 8.5 s |
| **What survives: weights, KV / prefix cache, in-flight requests, CUDA graphs** | **100% Preserved:** Model weights, KV cache, prefix cache, in-flight driver requests, CUDA graphs bit-for-bit. | **100% Preserved:** Model weights, optimizer/training state, KV cache, prefix cache, process memory, and execution state bit-for-bit. |
| **GPU placement on resume: same card / same model / other model / other driver** | **Same physical card only:** Tied to the original GPU card. Fails if restored on any other card. | **Any card of the same model:** Works on any available card (Card 0 or Card 1) on the same node. |
| **Checkpoint size and storage cost per model size** | • **PyTorch LoRA 7B:** 22.0 GB<br>• **Triton text embedding:** 1.4 GB (3.58 s checkpoint duration)<br>• **Triton vLLM backend:** 19.0 GB<br>• **vLLM Whisper large-v3:** 7.6 GB<br>• **vLLM Qwen2.5-7B:** 18.0–19.0 GB<br>• **vLLM Qwen2.5-14B-AWQ:** 26.0 GB | • **PyTorch LoRA 7B:** 23.0 GB<br>• **Triton text embedding:** 1.2–2.1 GB max<br>• **Triton vLLM backend:** 28.0 GB<br>• **vLLM Whisper large-v3:** 7.6 GB<br>• **vLLM Qwen2.5-7B:** 18.0–22.0 GB<br>• **vLLM Qwen2.5-14B-AWQ:** 26.0 GB |
| **Host-memory needed at pause** | **High:** CPU RAM must hold the GPU memory copied over plus the workload's existing process memory. Fails with OOM if available CPU RAM is less than active GPU memory (e.g., 86.4 GB VRAM failed on a 91.1 GiB node). | **High:** CPU RAM must temporarily accommodate the evacuated GPU memory plus the workload's existing process memory. Fails with OOM if available CPU RAM is less than active GPU memory. |
| **Constraints on customer workload** | Single GPU only. Zero changes to customer code; multi-process vLLM and Triton multi-process backends work seamlessly out of the box. | Single GPU only. No code changes needed: deploy model with eager mode (`--enforce-eager`) and enable pause/resume endpoints (`VLLM_SERVER_DEV_MODE="1"`). |
| **Failure modes and how often they hit in PoCs** | • Out of Memory if active GPU memory exceeds available CPU RAM (e.g., 86.4 GB VRAM failed on a 91.1 GiB node)<br>• Restore fails if the restore pod changes GPU claim names or volume mount paths<br>• Restore fails with `status=0x1f` if restored on a different GPU card | • TCPStore broken connection in standalone vLLM if `/pause` is not called before checkpointing<br>• CUDA illegal memory access crash if CUDA graphs are enabled instead of eager mode |
| **Impact on GPU utilisation and cluster efficiency** | GPU is released when the pod is deleted, but restoring requires the exact same physical card. If that card is in use by another workload, the restore cannot proceed until it is free. | GPU is released when the pod is deleted, and the workload can resume on any free card on the node without being tied to a specific card. |
| **Driver / runsc upgrade impact: are checkpoints taken before an upgrade still restorable?** | **Cannot be restored:** If the NVIDIA driver or gVisor (`runsc`) version changes, checkpoints taken before the upgrade cannot be restored due to driver ioctl and nvproxy state mismatches. | **Cannot be restored:** If the gVisor (`runsc`) or host driver version changes, checkpoints taken before the upgrade cannot be restored. Workloads must be cold started after an upgrade. |
| **Security: checkpoint encryption and data storage** | **Unencrypted binary dump:** Checkpoints store raw binary memory pages (weights, process memory, and active KV cache) on the node-local NVMe disk. Accessible only to host root / cluster administrators. | **Unencrypted binary dump:** Checkpoints store raw binary CPU memory pages (weights and process memory) on the node-local NVMe disk. Accessible only to host root / cluster administrators. |

### 4.2 Key Recommendations

1. **Use the Memory-Only Checkpoint Approach:** Release the GPU when the workload is paused. During resume, restore the workload on any available physical GPU card on the same node that matches the user's configuration (GPU model and GPU count).

2. **Set an Appropriate Container Memory Limit:** The container memory limit must be higher than the GPU memory because checkpointing copies the GPU memory into container RAM during pause. It must also accommodate the workload's own memory usage and gVisor overhead. As a baseline, set the limit to at least **1.5× the GPU card's memory**.

3. **Validate Resources Before Checkpointing:** Before starting a checkpoint, check whether the node has enough free RAM and checkpoint disk space. If resources are insufficient, skip checkpointing and stop the workload normally, preserving PVC data. Notify the user instead of allowing the pod to be `OOMKilled`.

4. **Configure vLLM for Successful Checkpointing:** Start vLLM with `--enforce-eager` to avoid CUDA graph-related issues during restore. Before taking a checkpoint, lock and checkpoint all CUDA processes, including the API server and EngineCore, to capture the complete workload state and ensure successful restoration.

5. Always Pass `--exclude-committed-zero-pages` to `runsc checkpoint`

### 4.3 Open Risks

1. **Node and Container OOM During Snapshot:** Evacuating high-VRAM GPUs (such as RTX PRO 6000 96GB) into CPU RAM causes a large memory spike. If the container limit or node free RAM is insufficient, the operation fails with an out-of-memory error (`OOMKilled` or `EOF`).

2. **No Cross-Node Resume:** Checkpoint files are stored on node-local NVMe disks. Workloads can only resume on the same physical Kubernetes node. If the node fails, undergoes maintenance, or stays fully booked, the workload cannot resume on another node.

3. **Checkpoints Broken by Driver or gVisor Upgrades:** Any host-level upgrade to the NVIDIA driver or gVisor (`runsc`) invalidates existing checkpoints. Checkpoints created prior to the upgrade cannot be restored, requiring affected workloads to cold start.

4. **Local NVMe Storage Exhaustion:** Each checkpoint consumes 18 GB to 28 GB of local NVMe storage. Multiple paused workloads can quickly fill up node disk space, causing subsequent pause operations or container disk writes to fail.

5. **Loss of CUDA Graph Optimizations:** To prevent memory address invalidation and crashes on restore, workloads must run in eager mode (`--enforce-eager`), losing the latency and throughput improvements that CUDA graphs normally provide.


