# GPU Checkpoint and Restore under gVisor — Master Proof of Concept Report

**Date:** November 12, 2026  
**Environment:** Dev workload cluster  
**Worker Node:** `stg-nc-workload1-worker-3`  
**Runtime Class:** `runsc-root-self`

---

## 1. Overview

This Proof of Concept (PoC) evaluates whether GPU-based workloads can be checkpointed and restored using the existing platform workflow under gVisor.

The validation uses the existing `sandbox-sr-operator`, `checkpoint-csi-driver`, and gVisor runtime with the `runsc-root-self` RuntimeClass (`overlay2 = "root:self"`).

The tests cover GPU workloads including PyTorch training, vLLM inference, and ComfyUI. Additional tests evaluate GPU compatibility, multi-GPU workloads, resource changes, container templates, persistent storage, and filesystem-only snapshots.

This report documents the test configurations, execution results, observed runtime errors, limitations, and engineering actions identified during the PoC.

## 2. Background

The existing platform provides checkpoint and restore capabilities for containerized workloads using `sandbox-sr-operator`, gVisor, and `checkpoint-csi-driver`.

This PoC evaluates whether the same workflow can support GPU workloads running with the `runsc-root-self` runtime.

GPU workloads use CUDA to access GPU resources. For checkpoint and restore to work, the GPU-related state must be handled along with the container's process and memory state. The restore environment must also provide compatible GPU resources and the required CUDA checkpoint configuration.

The `cuda-checkpoint` utility is used as part of the GPU checkpoint workflow. The tests evaluate this integration across different workload types and configurations.

## 3. Checkpoint and Restore Workflow

The PoC uses the existing platform checkpoint and restore workflow, with GPU-specific handling in the checkpoint command.

### 3.1 Checkpoint Creation

The checkpoint process starts when a `SnapshotJob` is created with the required snapshot configuration, such as `mode: snapshot` and `snapshotType: full`.

The `sandbox-sr-operator` watches the SnapshotJob resource and handles the checkpoint request.

The operator follows these steps:

1. **Receive the checkpoint request:** The operator detects the new `SnapshotJob` and starts processing it.
2. **Identify the target Pod:** The operator identifies the Pod for which the checkpoint must be created.
3. **Check for GPU usage:** The operator examines the Pod specification for GPU-related resource information, including `spec.resourceClaims` (`resource.k8s.io/v1`) and `nvidia.com/gpu` resource requests.
4. **Set the GPU flag:** If GPU usage is detected, the operator sets its internal `hasGPU` flag to `true`. Otherwise, it follows the regular non-GPU checkpoint path.
5. **Prepare the checkpoint command:** When `hasGPU=true`, the operator adds the `--cuda-checkpoint-path` argument to the gVisor checkpoint command. This specifies the location of the CUDA checkpoint utility.
6. **Execute the checkpoint:** The operator invokes the gVisor checkpoint command for the running container through `nsenter` on the host node.
7. **Package and upload the checkpoint:** The generated checkpoint files, such as `checkpoint.img`, `pages.img`, and `spec.img`, are packaged into a `.tar.zst` archive, verified using SHA256, and uploaded to object storage (`s3://sandbox-snapshots/`).
8. **Update the SnapshotJob status:** The operator updates the SnapshotJob with its final status, storage URI, and SHA256 checksum.

The checkpoint command for a GPU workload is:

```bash
runsc --root /run/containerd/runsc/k8s.io checkpoint \
  --image-path=/var/lib/neev-checkpoint-scratch/<namespace>/<job>/checkpoint \
  --exclude-committed-zero-pages \
  --cuda-checkpoint-path=/opt/cuda-checkpoint/cuda-checkpoint \
  --leave-running <container_id>
```

**Important:** The `--cuda-checkpoint-path` argument is included when the operator identifies the workload as GPU-enabled (`hasGPU=true`). The `--leave-running` option allows the original container to continue running after checkpoint creation. It does not, by itself, guarantee consistency between the checkpointed memory state and data being written to persistent volumes.

### 3.2 Restore Process

The restore process uses the checkpoint created by the operator.

1. A new workload Pod is created with the restore configuration.
2. The restore configuration identifies the checkpoint through the checkpoint CSI volume and the gVisor restore annotation:
   - **Volume:** `checkpoint.csi.neev.io`, carrying `objectRef`, `sha256`, and `restoreId`.
   - **Annotation:** `dev.gvisor.internal.restore.host-image-path`, pointing to the checkpoint directory on the worker node.
3. The `checkpoint-csi-driver` handles `NodePublishVolume` before Kubelet completes container startup:
   - Downloads the `.tar.zst` archive from object storage.
   - Verifies the SHA256 checksum.
   - Extracts the checkpoint files into `/var/lib/neev-checkpoint-scratch/<namespace>/<restoreId>/checkpoint`.
4. The `containerd-shim-runsc-v1` reads the restore image path from the annotation and passes the checkpoint image path to `runsc restore`.
5. gVisor restores the container using the saved checkpoint state and the restore Pod configuration. The GPU-related restore path also depends on the required CUDA configuration and compatible GPU devices being available.
6. If restoration succeeds, the container transitions to `Running` with its saved process and memory state.

The restore tests show that the target container configuration, mounts, GPU device identity, and GPU count can affect whether restoration succeeds.

## 4. Overall Test Results

The following table summarizes the results of all 12 PoC tests.

| Test | Test Scenario | Result | Key Observation |
|---:|---|---|---|
| 1 | CPU and Memory Scaling During Restore | **PARTIAL** | Scale-up to 4 CPU / 8 Gi passed in 22 seconds. Scale-down to 1 CPU / 100 Mi failed with `cannot restore container "cuda" in state stopped`. |
| 2 | Cross-Template Restore | **FAIL** | Restore failed in `vfs.CompleteRestore()` because `/opt/venv` was missing from the target image. |
| 3 | Ceph PVC Consistency | **PARTIAL** | Live checkpointing produced a mismatch between the memory checkpoint and PVC contents. Coordinated pausing eliminated the observed drift. |
| 4 | Template Version Upgrade | **PASS** | The tested workload checkpointed under template `0.1.1` resumed under template `0.1.2` at tick 35. |
| 5 | Cross-Card GPU Restore | **FAIL** | Restore to a different physical GPU failed in `nvproxy` with `NV0000_CTRL_CMD_GPU_GET_ID_INFO status=0x1f`. |
| 6 | Restore Without CUDA Configuration | **FAIL** | Checkpointing without the CUDA checkpoint path caused an `nvproxy.beforeSave()` failure. A restore with changed mounts also failed spec validation. |
| 7 | Multi-GPU Workload Restore | **FAIL** | Workload execution on both GPUs and checkpoint creation succeeded, but restoration failed with `NV0000_CTRL_CMD_GPU_GET_ID_INFO status=0x1f`. |
| 8 | GPU Count Change During Restore | **FAIL** | A checkpoint created with one GPU failed to restore with a two-GPU configuration. |
| 9 | PyTorch Training and AdamW Optimizer | **PASS** | Checkpoint completed in 26.269 seconds and restore in 16.611 seconds. Training resumed from Step 21 through Step 40. |
| 10 | vLLM Inference Engine | **PASS** | Single-process vLLM with 32 GiB host memory restored successfully and served an inference request with HTTP 200. |
| 11 | ComfyUI GPU Workload | **PASS** | Checkpoint and restore completed successfully. HTTP availability and the `/dev/shm` proof file were verified after restore. |
| 12 | Filesystem-Only Snapshot | **PASS — SNAPSHOT ONLY** | Workspace snapshot completed in 4.002 seconds without pausing GPU execution. Native filesystem restore was not validated because the installed shim lacked the required support. |

### 4.1 Overall Result

The PoC demonstrated successful checkpoint and restore for the tested PyTorch training, vLLM inference, and ComfyUI workloads on NVIDIA GeForce RTX 5090 hardware.

The tests also identified limitations that prevent treating GPU checkpoint and restore as universally supported across all configurations:

- Restoring to a different physical GPU failed in the tested cross-card scenario.
- Restoring a multi-GPU workload was unsuccessful, even though the workload executed on both GPUs and checkpoint creation completed.
- Changing the GPU count between checkpoint and restore caused a restore failure.
- The target container filesystem and restore specification must be compatible with the saved checkpoint.
- The tested restore with a significantly reduced memory limit failed.
- Persistent-volume consistency requires coordination between the workload checkpoint and storage snapshot.
- Filesystem-only snapshot creation worked, but native filesystem-only restore was not validated.

The results therefore demonstrate working GPU checkpoint and restore for the tested single-GPU application configurations, while identifying hardware, configuration, and storage constraints that need further engineering work.

## 5. Detailed Test Results

### Test 1: CPU and Memory Scaling During Restore

**Objective**

To check whether a GPU workload can be restored when the CPU and memory resources assigned to the target Pod differ from those assigned to the original Pod.

**Test Configuration**

| Parameter | Baseline (Original) | Scale-Up Attempt | Scale-Down Attempt |
|---|---|---|---|
| Workload | PyTorch / GPT-2 | PyTorch / GPT-2 | PyTorch / GPT-2 |
| Active Memory Footprint | 644 MiB | 644 MiB | 644 MiB |
| Assigned CPU | 2 | 4 | 1 |
| Assigned Memory | 4 Gi | 8 Gi | 100 Mi |
| Snapshot Source | `sj-airuntime-gpu-01` | `sj-airuntime-gpu-01` | `sj-airuntime-gpu-01` |

**Test Execution and Results**

1. The baseline workload `airuntime-gpu-workload` was snapshotted through `sj-airuntime-gpu-01`. Its observed active memory footprint was approximately 644 MiB.
2. **Scale-up scenario (4 CPU / 8 Gi RAM):** The workload was restored using `restore-scale-up.yaml`. The Pod transitioned from `ContainerCreating` to `Running` in approximately 22 seconds. The `/health` and `/generate` endpoints were queried to verify that the workload was responding.
3. **Scale-down scenario (1 CPU / 100 Mi RAM):** The workload was restored using `restore-scale-down.yaml`. The Pod entered `StartError`.

**Exact Observed Kubelet Event Log:**

```text
Events:
  Type     Reason    Age   From     Message
  ----     ------    ----  ----     -------
  Warning  Failed    9s    kubelet  spec.containers{cuda}: Error: failed to start containerd task "cuda": OCI runtime restore failed: starting container: cannot restore container "cuda" in state stopped
```

**Technical Analysis**

The scale-up scenario completed successfully. The scale-down scenario failed during container restoration, and the OCI runtime reported that the container had stopped.

The original workload had an observed active memory footprint of approximately 644 MiB, while the target Pod was configured with a 100 MiB memory limit. This is a significant reduction from the original configuration and may have contributed to the failure.

However, the observed event alone does not establish the exact internal failure point. Additional runtime and cgroup diagnostics would be required to confirm the precise cause.

**Result: PARTIAL**

The tested scale-up configuration restored successfully. The tested scale-down configuration of 1 CPU and 100 MiB failed. Further testing is required to determine the minimum supported restore memory limit.

---

### Test 2: Cross-Template Restore

**Objective**

To check whether a checkpoint created using one container template can be restored using a different container template.

**Test Configuration**

| Parameter | Source Container | Target Container |
|---|---|---|
| Image Name | `harbor.neev.work/test-template/pytorch-v2.8.0-ubuntu-24.04:0.1.2` | `harbor.neev.work/test-template/ubuntu-24.04:0.1.3` |
| Image Contents | PyTorch 2.8.0, CUDA 13.2, Python virtual environment | Base Ubuntu 24.04 without PyTorch or CUDA |
| Required Directory | `/opt/venv` | Missing |

**Test Execution and Results**

1. A PyTorch GPU workload was snapshotted through `sj-airuntime-gpu-01`.
2. A restore Pod named `airuntime-gpu-cross-template` was deployed using the checkpoint SHA256 but specifying the plain Ubuntu image `ubuntu-24.04:0.1.3`.
3. The target image did not contain the `/opt/venv` directory present in the source container.
4. The restore Pod entered `StartError` during container creation.

**Exact Observed Kubelet Event Log:**

```text
Events:
  Type     Reason    Age   From     Message
  ----     ------    ----  ----     -------
  Warning  Failed    8s    kubelet  spec.containers{cuda}: Error: failed to start containerd task "cuda": OCI runtime restore failed: starting container: starting sub-container: failed to load kernel: vfs.CompleteRestore() failed: failed to complete restore for filesystem type "9p": failed to walk "opt/venv" of type 040000 in mount "cuda:/": no such file or directory
```

**Technical Analysis**

During restoration, gVisor validates and reconstructs the saved container filesystem state. The checkpoint referenced the `/opt/venv` directory, but the target container image did not contain that directory.

The filesystem restoration process therefore failed in `vfs.CompleteRestore()` with `no such file or directory`.

This test demonstrates that the tested source and target images were not filesystem-compatible for restore.

**Result: FAIL**

Restoration from the tested PyTorch image into the plain Ubuntu image failed. The target image must provide a filesystem layout compatible with the checkpoint. This test does not establish that every template version change will fail; the separate template-version test demonstrates a successful restore for a tested patch-version change.

---

### Test 3: Ceph PVC Consistency During Checkpoint

**Objective**

To check whether data written to a Ceph-backed persistent volume remains consistent when a checkpoint is created while the workload is actively writing.

**Test Execution and Results**

1. The workload `airuntime-gpu-pvc-workload` ran with a Ceph-backed PVC attached.
2. The workload continuously appended log entries to `/workspace/workload.log` every 500 milliseconds.
3. **Case 1 — Live checkpoint without pausing:** A SnapshotJob was triggered while the workload continued running. The Ceph snapshot and PVC cloning operation took approximately 14 seconds.
4. During this period, the live workload advanced from `tick=23` to `tick=46`.
5. After restoration, the workload's memory state resumed at `tick=23`, while the cloned PVC contained writes up to `tick=46`.

**Exact Observed Log Output on Restored Pod:**

```text
[2026-11-10T14:22:14Z] tick=45 val=45.0
[2026-11-10T14:22:15Z] tick=46 val=46.0   <-- Written by the live container before the clone finished
[2026-11-10T14:22:29Z] tick=23 val=23.0   <-- Workload memory resumed here
[2026-11-10T14:22:30Z] tick=24 val=24.0
```

6. **Case 2 — Coordinated checkpoint:** The container was paused before snapshotting. This eliminated the storage drift observed in the live-checkpoint scenario.

**Technical Analysis**

In the first scenario, the workload continued writing to the PVC while its memory checkpoint and the storage snapshot were being created. As a result, the restored memory state represented an earlier point in the workload's execution than the data present on the cloned PVC.

This produced an inconsistent application state: the restored process resumed from `tick=23`, while the PVC contained later writes up to `tick=46`.

In the second scenario, pausing the workload before snapshotting prevented the observed mismatch.

**Result: PARTIAL**

The tested live-checkpoint procedure did not provide a consistent application state across memory and the Ceph PVC. Coordinating workload suspension with the volume snapshot avoided the observed drift. A coordinated pause or application-level quiescing mechanism is required for stateful workloads.

---

### Test 4: Template Version Upgrade

**Objective**

To verify whether a workload checkpointed using an older version of a template can be restored using a newer version of the same template.

**Test Configuration**

| Parameter | Source Workload | Target Restore Workload |
|---|---|---|
| Container Image | `pytorch-v2.8.0-ubuntu-24.04:0.1.1` | `pytorch-v2.8.0-ubuntu-24.04:0.1.2` |
| Change Description | Baseline template | Patch upgrade |
| Checkpoint Tick | `tick=34` | Resumed at `tick=35` |

**Test Execution and Results**

1. The workload ran under template version `0.1.1` until `tick=34`.
2. A checkpoint was created through `sj-version-upgrade`.
3. The source Pod was deleted to release the GPU.
4. The restore Pod was configured to use the newer image `pytorch-v2.8.0-ubuntu-24.04:0.1.2`.
5. The workload restored successfully in approximately 18 seconds.

**Exact Observed Log Output on Restored Pod:**

```text
=== Restoring Workload from Snapshot on Template v0.1.2 ===
[PyTorch v0.1.2 Restore] Resuming from checkpoint...
[CARD-1] tick=35 sample=35.0 VRAM=256.0MB
[CARD-1] tick=36 sample=36.0 VRAM=256.0MB
```

**Technical Analysis**

The workload resumed from the expected next tick after being checkpointed at `tick=34`. This demonstrates successful restoration for the specific source and target template versions tested.

The result does not establish compatibility with arbitrary template changes. Changes to the filesystem, libraries, mounts, or container configuration may still affect restore compatibility.

**Result: PASS**

The tested template upgrade from version `0.1.1` to `0.1.2` restored successfully and resumed execution at `tick=35`.

---

### Test 5: Cross-Card GPU Restore

**Objective**

To check whether a GPU workload checkpointed on one physical GPU can be restored on another physical GPU of the same model.

**Test Configuration**

| Parameter | Source GPU (Card 1) | Target GPU (Card 0) |
|---|---|---|
| GPU Hardware | ASUS NVIDIA GeForce RTX 5090 | ASUS NVIDIA GeForce RTX 5090 |
| PCI Bus Address | `0000:c1:00.0` | `0000:81:00.0` |
| GPU UUID | `GPU-36df8b09-2435-5c42-7324-850c168ecc94` | `GPU-860e5920-e4bc-f9fd-18c3-365260cc3a01` |
| Workload | PyTorch Matrix Loop | Restore Target |

**Test Execution and Results**

1. The workload was deployed on Card 1 using a DRA ResourceClaim.
2. The workload was checkpointed successfully through `sj-cross-card-01` in approximately 14 seconds.
3. The source Pod and claim were deleted to release Card 1.
4. The restore Pod `neev-gpu-card0-restore` was deployed with a claim selecting Card 0 by UUID.
5. The restore Pod entered `StartError`.

**Exact Observed Kubelet Event Log:**

```text
Events:
  Type     Reason    Age   From     Message
  ----     ------    ----  ----     -------
  Warning  Failed    12s   kubelet  spec.containers{cuda}: Error: failed to start containerd task "cuda": OCI runtime restore failed: starting container: starting sub-container: failed to get nvproxy device IDs: NV0000_CTRL_CMD_GPU_GET_ID_INFO for minor=0 gpuId=0x100 failed: status=0x1f
```

**Technical Analysis**

The failure occurred during gVisor `nvproxy` initialization while it was retrieving GPU device information through the NVIDIA control command `NV0000_CTRL_CMD_GPU_GET_ID_INFO`.

The source and target devices were both RTX 5090 GPUs, but they had different PCI bus addresses and GPU UUIDs. The restore attempt failed when using the different physical device.

The observed result indicates that the tested restore workflow could not move this checkpoint from Card 1 to Card 0. The exact internal reason for the NVIDIA driver rejection would require further driver and `nvproxy` investigation.

**Result: FAIL**

Restoration to a different physical RTX 5090 failed. For the tested workflow, restore should be scheduled against the original physical GPU identity unless further testing establishes support for cross-card restoration.

---

### Test 6: Restore Without CUDA Configuration

**Objective**

To check whether GPU checkpoint and restore can work when the required CUDA checkpoint utility or GPU-related restore configuration is unavailable.

Two scenarios were evaluated.

#### Scenario 1: Checkpoint Without `--cuda-checkpoint-path`

An active GPU container allocating VRAM was checkpointed while omitting `--cuda-checkpoint-path`. The checkpoint operation failed inside gVisor's `nvproxy` save path.

**Exact Observed Runtime Log:**

```text
checkpoint failed: checkpointing container "8299eef405cc809eb93221e6740ed312d4aa2a9b845202bf8b934497367fbf97":
encoding error: nvproxy: object handle 0xc1d02d06:0x5c00007e (class 0x00000079, type *nvproxy.miscObject) is not restorable:

goroutine 587 [running]:
gvisor.dev/gvisor/pkg/state.safely.func1()
        pkg/state/state.go:309 +0x19e
panic({0x12fc9a0?, 0x2158717b1340?})
        GOROOT/src/runtime/panic.go:860 +0x13a
gvisor.dev/gvisor/pkg/sentry/devices/nvproxy.(*nvproxy).beforeSave(0x421b5f?)
        pkg/sentry/devices/nvproxy/save_restore.go:182 +0x1f9
```

#### Scenario 2: Restore Without the Original GPU Configuration

A valid GPU checkpoint was restored using a manifest that removed `resourceClaims` and omitted the `/opt/cuda-checkpoint` mount. The Pod entered `StartError`.

**Exact Observed Kubelet Event Log:**

```text
Events:
  Type     Reason    Age   From     Message
  ----     ------    ----  ----     -------
  Warning  Failed    4s    kubelet  spec.containers{cuda}: Error: failed to start containerd task "cuda": OCI runtime restore failed: starting container: starting sub-container [/bin/bash]: failed to handle restore spec validation: failed to validate restore spec: failed to validate spec for container "cuda": "Mounts" does not match across checkpoint restore for container: "agent"
```

**Technical Analysis**

The first scenario demonstrates that the tested GPU checkpoint operation failed when the CUDA checkpoint path was omitted. The `nvproxy.beforeSave()` stack trace identifies the failure during gVisor's save operation.

The second scenario failed during restore specification validation because the target Pod's mount configuration differed from the saved container specification. Since the test changed more than one aspect of the GPU restore configuration, the observed error specifically confirms the mount mismatch; it does not independently establish which of the removed GPU settings would have caused a later failure.

**Result: FAIL**

The tested checkpoint operation failed without `--cuda-checkpoint-path`. The tested restore also failed when the target mount configuration did not match the checkpoint. GPU restore configuration and container mounts must be preserved consistently.

---

### Test 7: Multi-GPU Workload Checkpoint and Restore

**Objective**

To check whether a workload using both RTX 5090 GPUs can be checkpointed and restored.

**Test Configuration**

| Parameter | Configuration |
|---|---|
| Workload | PyTorch Dual-GPU Matrix Multiplication |
| DRA ResourceClaim | `count: 2` |
| GPU Allocation | Device 0: Card 0; Device 1: Card 1 |
| Total Available VRAM | Approximately 63.6 GB |
| Checkpoint Job | `sj-multi-gpu-01` |

**Test Execution and Results**

1. The workload `airuntime-gpu-multi-source` was started with a DRA ResourceClaim requesting two GPUs.
2. PyTorch detected both devices as `cuda:0` and `cuda:1`.
3. The workload allocated approximately 256 MB on each GPU and successfully executed the dual-GPU workload.
4. SnapshotJob `sj-multi-gpu-01` completed checkpoint creation in approximately 10 seconds, and the archive was saved to S3.
5. The source Pod was deleted.
6. The restore Pod `airuntime-gpu-multi-restored` was deployed requesting two GPUs.
7. The restore Pod entered `StartError`.

**Exact Observed Kubelet Event Log:**

```text
Events:
  Type     Reason    Age   From     Message
  ----     ------    ----  ----     -------
  Warning  Failed    18s   kubelet  spec.containers{cuda}: Error: failed to start containerd task "cuda": OCI runtime restore failed: starting container: starting sub-container: failed to get nvproxy device IDs: NV0000_CTRL_CMD_GPU_GET_ID_INFO for minor=0 gpuId=0x100 failed: status=0x1f
```

**Technical Analysis**

The test verified three separate stages:

- **GPU allocation:** Kubernetes DRA allocated both GPUs to the workload.
- **Workload execution:** PyTorch detected both GPUs and executed the dual-GPU workload.
- **Checkpoint creation:** The SnapshotJob completed and produced a checkpoint archive.

However, the restore attempt did not succeed. The failure occurred during `nvproxy` GPU device initialization with `NV0000_CTRL_CMD_GPU_GET_ID_INFO status=0x1f`.

The observed error does not establish that DRA allocation or initial multi-GPU execution failed. Those stages succeeded. The failure occurred during restoration of the checkpointed workload.

**Result: FAIL**

Multi-GPU allocation, workload execution, and checkpoint creation were demonstrated successfully. Restoration of the multi-GPU workload was unsuccessful. Multi-GPU checkpoint and restore must therefore not be considered fully supported based on this test.

---

### Test 8: GPU Count Change During Restore

**Objective**

To check whether a workload checkpointed with one GPU can be restored using a target Pod requesting two GPUs.

**Test Execution and Results**

1. A baseline workload was started with one GPU.
2. The workload was checkpointed through `sj-gpu-scale-01`.
3. The target restore Pod was configured to request two GPUs using a DRA ResourceClaim with `count: 2`.
4. The restore Pod entered `StartError`.

**Exact Observed Kubelet Event Log:**

```text
Events:
  Type     Reason    Age   From     Message
  ----     ------    ----  ----     -------
  Warning  Failed    22s   kubelet  spec.containers{cuda}: Error: failed to start containerd task "cuda": OCI runtime restore failed: starting container: starting sub-container: failed to get nvproxy device IDs: NV0000_CTRL_CMD_GPU_GET_ID_INFO for minor=0 gpuId=0x100 failed: status=0x1f
```

**Technical Analysis**

The checkpoint was created with a one-GPU configuration, while the restore Pod requested two GPUs. The restore operation failed during `nvproxy` device initialization.

This test demonstrates that the tested workflow did not support changing the GPU count from one to two during restore. The observed error does not independently identify every internal device-topology mismatch; further testing would be required to determine the exact compatibility rules.

**Result: FAIL**

The tested one-GPU-to-two-GPU restore failed. The GPU configuration should be kept consistent between checkpoint and restore until additional testing establishes which GPU count changes, if any, are supported.

---

### Test 9: PyTorch Training Loop and AdamW Optimizer State

**Objective**

To verify whether an active PyTorch deep learning training workload can be checkpointed and restored with its model parameters, training progress, and AdamW optimizer state preserved.

**Test Configuration**

| Parameter | Configuration |
|---|---|
| Model Architecture | 5-Layer Linear Neural Network (2048 → 4096 → 2048 → 512) |
| Optimizer | AdamW (`lr=0.001`, `weight_decay=0.01`) |
| Loss Criterion | Mean Squared Error (`MSELoss`) |
| Batch Size / Feature Dim | Batch=256, Input=2048, Target=512 |
| Active VRAM Footprint | 292.9 MB |

**Test Execution and Results**

1. The workload `airuntime-gpu-training` executed training Steps 1–20 on the RTX 5090.
2. The loss decreased from `1.0046` to `0.0894`.
3. The optimizer state was checked. The observed values were:
   - Step count: `20`
   - First momentum norm: `4.1205`
   - Second momentum norm: `0.8931`
4. SnapshotJob `sj-training-01` was triggered.
5. Checkpoint creation completed in **26.269 seconds**.
6. The source Pod was deleted to release GPU memory.

**Restore Specification Issue and Fix**

The first restore attempt used an unquoted heredoc:

```bash
cat << EOF
```

Bash expanded `$PATH` in the manifest before it was applied. This changed the container arguments and caused gVisor restore specification validation to fail.

**Exact Observed Error:**

```text
spec.containers{cuda}: Error: failed to start containerd task "cuda": OCI runtime restore failed: starting container: starting sub-container [/bin/bash]: failed to handle restore spec validation: failed to validate restore spec: failed to validate spec for container "cuda": "Args" does not match across checkpoint restore for container: "cuda"
```

The issue was resolved by quoting the heredoc delimiter:

```bash
cat << 'EOF'
```

This prevented the shell from expanding variables in the manifest.

7. The restore Pod `airuntime-gpu-training-restored` was deployed.
8. The Pod reached `Running` in **16.611 seconds**.
9. Training resumed from Step 21 and continued through Step 40.

**Step-by-Step Training Continuity**

| Step | Phase | Loss | VRAM | Status | Observation |
|---:|---|---:|---:|---|---|
| 1 | Pre-Snapshot | 1.0046 | 292.9 MB | Training | Initial random weights |
| 10 | Pre-Snapshot | 0.3842 | 292.9 MB | Training | Loss decreased |
| 20 | Pre-Snapshot | 0.0894 | 292.9 MB | Checkpointed | AdamW state captured |
| — | Snapshot and Delete | — | 0.0 MB | Completed | Capture: 26.269 s |
| — | CSI Restore | — | 292.9 MB | Restored | Restore: 16.611 s |
| 21 | Post-Restore | 0.0679 | 292.9 MB | Resumed | Training continued |
| 30 | Post-Restore | 0.0211 | 292.9 MB | Training | Loss continued decreasing |
| 40 | Post-Restore | 0.0084 | 292.9 MB | Completed | Training continued successfully |

**Technical Analysis**

The workload resumed from Step 21 after being checkpointed at Step 20. The loss continued decreasing from `0.0679` at Step 21 to `0.0084` at Step 40.

The optimizer step count and momentum values were verified before checkpointing. The successful continuation of training provides evidence that the tested model and optimizer state were preserved across the checkpoint and restore operation.

**Result: PASS**

The tested PyTorch training workload restored successfully and continued from Step 21 through Step 40. No NaN values or gradient-explosion symptoms were observed in the recorded training results.

---

### Test 10: vLLM Inference Engine, Model Weights and KV Cache

**Objective**

To verify whether an active vLLM inference engine running GPT-2 can be checkpointed and restored with its model state and preallocated KV-cache memory preserved.

The test also evaluates how vLLM process configuration and host memory allocation affect checkpoint creation.

**Test Scenarios and Observed Results**

#### Scenario 1: Default vLLM Multiprocessing

The default vLLM configuration was deployed with the API server running as PID 1 and the GPU engine running as PID 30.

SnapshotJob `sj-vllm-01` failed.

**Exact Observed Error Log:**

```text
checkpoint failed: "cuda-checkpoint --toggle for PID 30" failed with exit status 256;
output: "Error toggling CUDA in process ID 30: OS call failed or operation not supported on this OS"
```

**Observation:** The CUDA toggle operation failed for the GPU engine process. This configuration did not complete checkpoint creation.

#### Scenario 2: Single-Process Mode with 16 GiB Host RAM

The workload was changed to single-process mode using:

```bash
VLLM_ENABLE_V1_MULTIPROCESSING=0
--distributed-executor-backend uni
```

Checkpointing progressed further but failed during the VRAM dump.

**Exact Observed Error Log:**

```text
urpc method "containerManager.Checkpoint" failed: EOF
```

**Observation:** The workload used `gpu_memory_utilization=0.90`, corresponding to approximately 28.8 GB of allocated GPU memory. The container had a 16 GiB host memory limit. The failure was consistent with insufficient host memory during checkpointing, although the `EOF` message alone does not prove the exact OOM sequence.

#### Scenario 3: Single-Process Mode with 32 GiB Host RAM

The container memory limit was increased to 32 GiB while retaining the single-process configuration.

Checkpoint creation completed successfully.

- Checkpoint time: **14 seconds**
- Archive: `sj-vllm-01.tar.zst`
- Archive uploaded to S3.

#### Scenario 4: Restore and Inference Verification

1. The source Pod was deleted to release GPU memory.
2. The workload was restored through the CSI driver.
3. Restore completed in approximately **26 seconds**.
4. An inference request was sent to:

```text
/generate?prompt=Antigravity
```

5. The endpoint returned HTTP 200 OK and generated 15 completion tokens.
6. The observed inference latency was less than 80 milliseconds.
7. The restored service responded without an observed model cold-start reload.

**Technical Analysis**

The default multiprocessing configuration did not complete checkpointing because the CUDA toggle operation failed for the GPU engine process.

The single-process configuration allowed the tested vLLM workload to complete checkpoint creation. The 16 GiB configuration failed during checkpointing, while the 32 GiB configuration completed successfully with approximately 28.8 GB of GPU memory allocated.

This demonstrates that host memory capacity is an important consideration when checkpointing a workload with a large GPU memory footprint. The test does not establish a universal host-RAM-to-VRAM sizing formula.

**Result: PASS**

The tested single-process vLLM configuration with 32 GiB host memory completed checkpoint and restore successfully. The restored inference endpoint returned HTTP 200 and generated a response with less than 80 ms observed latency.

---

### Test 11: ComfyUI GPU Workload Pause and Resume

**Objective**

To verify whether a running ComfyUI application can be checkpointed and restored with its application state and volatile shared-memory data preserved.

**Test Configuration**

| Parameter | Configuration |
|---|---|
| Container Image | `harbor.neev.work/gpu/comfyui:0.1.5` |
| Python | 3.12 |
| PyTorch | 2.14.1 |
| Application | ComfyUI with ComfyUI-Manager |
| Web UI Port | 3000 |
| Checkpoint Job | `sj-comfyui-02` |

**Test Execution and Results**

1. The workload `airuntime-gpu-comfyui` was deployed using `harbor.neev.work/gpu/comfyui:0.1.5`.
2. The ComfyUI server initialized on port 3000 using `cudaMallocAsync`.
3. HTTP availability on port 3000 was verified.
4. A proof file was written into shared memory:

```bash
kubectl exec -n gpu-gvisor-poc airuntime-gpu-comfyui -- \
  bash -c "echo 'CONFIRMED_RESTORE_1762908842' > /dev/shm/proof.txt"
```

5. SnapshotJob `sj-comfyui-02` completed checkpoint creation in **19.075 seconds**.
6. The source Pod was deleted to release the GPU.
7. The workload was restored as `airuntime-gpu-comfyui-restored`.
8. Restore completed in **20.081 seconds**.
9. The restored service returned HTTP 200 within the observed verification interval.
10. The file `/dev/shm/proof.txt` was read after restoration and its contents were preserved.

**Technical Analysis**

The test verified that the ComfyUI workload could be checkpointed and restored successfully. The HTTP endpoint was available after restoration, and the proof file stored in `/dev/shm` remained accessible.

This provides evidence that the tested workload's application and volatile shared-memory state were preserved through the checkpoint and restore workflow.

**Result: PASS**

The tested ComfyUI workload restored successfully. HTTP availability and preservation of the `/dev/shm` proof file were verified after restore.

---

### Test 12: Filesystem-Only Snapshot While GPU Workload Is Running

**Objective**

To check whether workspace files can be captured without creating a full GPU checkpoint or stopping the running GPU workload.

**Test Execution and Results**

1. A GPU workload continuously updated files under `/workspace/model_weights.bin` while executing GPU compute operations.
2. SnapshotJob `sj-gpu-fs-only-01` was triggered with `snapshotType: filesystem`.
3. Snapshot creation completed in **4.002 seconds**.
4. The generated `multitar.img` was approximately 4.6 KB.
5. The uploaded S3 archive was approximately 18 KB.
6. GPU compute ticks continued to be logged during the filesystem snapshot operation. No workload pause was observed in the recorded test.

**Restore Limitation**

A restore attempt was made using `airuntime-gpu-fs-restored`. The `/workspace` directory was empty.

**Observed Command Output:**

```text
kubectl exec -n gpu-gvisor-poc airuntime-gpu-fs-restored -- ls -la /workspace

total 0
drwxr-xr-x 2 root root 4096 Nov 12 04:30 .
drwxr-xr-x 1 root root 4096 Nov 12 04:30 ..
```

**Technical Analysis**

The installed `containerd-shim-runsc-v1` on `worker-3` was version `20260905`.

The installed shim supports the annotation:

```text
dev.gvisor.internal.restore.host-image-path
```

However, it does not support the filesystem restore annotation:

```text
dev.gvisor.internal.restore.fs-image-path
```

The filesystem restore support was added in commit `5decc67e` in October 2026. The installed shim therefore did not pass the filesystem image path to the appropriate gVisor filesystem restore operation.

As a result, filesystem snapshot creation succeeded, but the filesystem image was not applied during the tested restore attempt.

**Result: PASS — SNAPSHOT ONLY**

Filesystem-only snapshot creation completed in 4.002 seconds while GPU execution continued. Native filesystem-only restore was not validated because the installed containerd shim did not include the required filesystem restore support. The shim must be updated before this restore path can be validated.

## 6. Performance Measurements

The following table records the observed checkpoint and restore measurements from the successful workload tests.

| Workload | RAM and VRAM Footprint | Checkpoint Time | Restore Time | Checkpoint Size | Verification |
|---|---|---:|---:|---:|---|
| PyTorch Training | ~1.8 GB RAM + 292.9 MB VRAM | 26.269 s | 16.611 s | ~980 MB | Training resumed from Step 21 through Step 40 |
| ComfyUI | ~3.2 GB RAM + ~4.5 GB VRAM | 19.075 s | 20.081 s | ~2.1 GB | HTTP 200 and `/dev/shm` proof file verified |
| vLLM Inference | ~32 GB RAM + ~28.8 GB VRAM | 14.000 s | 26.000 s | ~1.8 GB | HTTP 200; 15 tokens generated; <80 ms observed latency |
| Filesystem-Only | Workspace files; no full GPU checkpoint | 4.002 s | Not validated | ~18 KB | Snapshot completed without observed GPU pause |

**Note:** These are measurements from the tested configurations, not guaranteed performance values. Checkpoint size, execution time, and restore time can vary with workload size, memory usage, filesystem contents, GPU state, and storage performance.

## 7. Key Findings and Restore Constraints

1. **Single-GPU workload restoration was demonstrated:** The tested PyTorch training, vLLM inference, and ComfyUI workloads completed checkpoint and restore successfully on the tested RTX 5090 configuration.

2. **GPU identity matters:** Restoring the tested checkpoint from Card 1 onto Card 0 failed during `nvproxy` initialization with `NV0000_CTRL_CMD_GPU_GET_ID_INFO status=0x1f`. The restore workflow should preserve the original GPU identity until cross-card compatibility is established.

3. **Multi-GPU restoration remains unsuccessful:** The multi-GPU workload detected both GPUs, executed its workload, and completed checkpoint creation. However, its restore attempt failed. Successful allocation and checkpoint creation must not be treated as proof of successful multi-GPU restoration.

4. **GPU configuration changes can cause restore failures:** The tested one-GPU-to-two-GPU restore failed during GPU device initialization. The tested workflow should preserve the GPU configuration between checkpoint and restore.

5. **Container restore specifications must remain compatible:** The cross-template test failed because the target image lacked `/opt/venv`. The PyTorch training test also demonstrated that changing shell-expanded container arguments can cause an `Args` mismatch during restore validation.

6. **Mount configuration must be preserved:** The restore without the original GPU configuration failed because the target mount specification did not match the saved checkpoint specification.

7. **Host memory affects checkpointing:** The tested vLLM configuration failed with 16 GiB host memory and succeeded with 32 GiB host memory while using approximately 28.8 GB of GPU memory. Additional testing is needed to establish suitable memory-sizing guidelines.

8. **Persistent storage requires coordination:** The live Ceph PVC test showed that storage contents can advance beyond the memory checkpoint when the workload continues writing. Pausing the workload before snapshotting eliminated the observed drift.

9. **Filesystem-only snapshot and restore are separate capabilities:** Filesystem snapshot creation succeeded without an observed GPU pause. Native filesystem restore remains unvalidated until the containerd shim includes the required support.

## 8. Conclusion

This PoC demonstrated that the existing `sandbox-sr-operator`, `checkpoint-csi-driver`, and gVisor workflow can checkpoint and restore the tested single-GPU PyTorch training, vLLM inference, and ComfyUI workloads using the `runsc-root-self` runtime on NVIDIA GeForce RTX 5090 hardware.

The PyTorch test demonstrated continued training after restoration, including preservation of the tested AdamW optimizer state. The vLLM test demonstrated successful inference after restoration using a single-process configuration with 32 GiB host memory. The ComfyUI test demonstrated successful application restoration and preservation of a proof file in `/dev/shm`.

The PoC also identified important limitations. Cross-card GPU restoration, multi-GPU restoration, and the tested GPU-count change did not succeed. Restore attempts can also fail when the target image, container arguments, or mount configuration differs from the checkpoint. The Ceph test showed the need to coordinate application state with persistent-volume snapshots. Filesystem-only snapshot creation succeeded, but native filesystem-only restore remains unvalidated until the containerd shim is updated.

The results provide a concrete basis for further engineering work. The next steps are to enforce GPU compatibility checks, investigate multi-GPU restoration, coordinate persistent-volume snapshots, preserve container restore specifications, and deploy the updated shim for filesystem-only restore support.

---

*End of Report*
