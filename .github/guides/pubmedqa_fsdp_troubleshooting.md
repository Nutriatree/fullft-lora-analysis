# PubMedQA 2-GPU FSDP smoke test and troubleshooting

## Purpose

Run this smoke test before the full study whenever the model, PyTorch/CUDA environment, GPU assignment, or FSDP implementation changes. It exercises:

1. rank-0 CPU float32 reference validation before FSDP wrapping
2. Qwen decoder-layer auto-wrap
3. one optimizer step across two GPUs
4. all-rank full-state collection with rank-0 CPU persistence
5. standalone CPU float32 checkpoint validation and test inference
6. Gloo control-group and NCCL process-group shutdown

## Run the smoke test

This script performs **real GPU training**. It is not the offline dry-run gate.
For a machine without GPUs, use `python scripts/test_pubmedqa_offline.py` instead.
The offline suite runs tiny CPU models and fake collectives; it does not verify
NCCL liveness, GPU OOM avoidance, convergence or performance.

Activate the same environment used for the full study, then run:

```bash
set -o pipefail
bash scripts/run_pubmedqa_fsdp_smoke.sh 2>&1 | tee fsdp_smoke.log
```

The script defaults to `Qwen/Qwen3-1.7B` and GPUs `0,1`. Override the GPU IDs when needed:

```bash
PUBMEDQA_GPU_IDS=2,3 \
bash scripts/run_pubmedqa_fsdp_smoke.sh
```

The smoke profile is intentionally small:

- 2 training examples
- 2 validation examples
- 2 test examples
- 1 epoch
- per-GPU batch size 1
- gradient accumulation 1
- checkpoint at 100%
- optimizer-state saving disabled
- no explicit input truncation by default, so answer labels are retained

Set `PUBMEDQA_MAX_INPUT_TOKENS` only when a specific sequence limit must be tested. Ensure the limit includes the answer tokens appended to the prompt.

## Expected artifacts

For the run ID printed by the script, verify the model-specific directory under `outputs/pubmedqa_train/<run-id>/` contains:

- `config.json`
- `summary.json`
- `distributed_runtime.json`
- `evaluations/validation_reference_summary.json`
- `evaluations/validation_pct_100_summary.json`
- `evaluations/test_summary.json`
- `checkpoints/scheduled_pct_100_epoch_001_step_000001/`

The checkpoint directory should contain model/tokenizer files and `training_state.json`. `optimizer.pt` should not exist in the default smoke run.

`config.json` and `distributed_runtime.json` should list the discovered FSDP layer class, such as `Qwen3DecoderLayer`, in `fsdp_auto_wrap_layer_classes`.

Also check `master_dtype=torch.float32`, `evaluation_dtype=torch.float32`,
`evaluation_device=cpu`, and that `per_rank` has one unique entry per worker.
Training peaks exclude checkpoint/evaluation work; CPU records explicitly have
`memory_measured=false`. Evaluation summaries use `loss_reduction=token_mean`.

## Runtime diagnostics

The smoke script enables these diagnostics by default:

```bash
NCCL_DEBUG=INFO
TORCH_DISTRIBUTED_DEBUG=DETAIL
TORCH_SHOW_CPP_STACKTRACES=1
PYTHONFAULTHANDLER=1
```

Record the environment alongside a failure:

```bash
nvidia-smi
python -c 'import torch, transformers; print(torch.__version__); print(torch.version.cuda); print(transformers.__version__); print(torch.cuda.nccl.version())'
```

Check kernel messages immediately after a crash:

```bash
dmesg -T | grep -Ei 'out of memory|killed process|NVRM|Xid|NCCL'
journalctl -k --since '-15 min' | grep -Ei 'oom|killed|NVRM|Xid|NCCL'
```

Some servers restrict `dmesg`; use `sudo` only if server policy permits it.

## Interpreting common failures

| Symptom | Likely cause | First checks |
|---|---|---|
| `exitcode: -6`, `SIGABRT` | NCCL watchdog, CUDA/NCCL native assertion, or driver Xid | Search earlier log lines for `Watchdog`, `NCCL`, `CUDA`, and inspect kernel logs |
| `exitcode: -9`, `SIGKILL` | Linux OOM killer or administrator/process manager kill | Check `dmesg`, RAM/swap, scheduler limits |
| `CUDA out of memory` Python traceback | GPU memory exhausted | Inspect `distributed_runtime.json`, sequence length, and other GPU processes |
| Hang before the first training step | rank mismatch, rendezvous failure, or one rank failed during loading | Confirm exactly two visible GPUs and compare rank-tagged logs |
| Failure while saving a checkpoint | rank-0 RAM/disk pressure or full-state collective failure | Check free RAM, disk space, and whether optimizer-state saving was enabled |

For a single-node PCIe/NCCL transport diagnosis only, retry once with:

```bash
NCCL_P2P_DISABLE=1 bash scripts/run_pubmedqa_fsdp_smoke.sh
```

If that succeeds, inspect PCIe peer-to-peer topology and driver configuration instead of keeping the workaround as the default.

## Memory-sensitive checkpoint behavior

- All ranks call full `state_dict`; `rank0_only=True` controls returned CPU state,
  not collective participation. Only rank 0 writes model/tokenizer/metadata files.
- Optimizer state is not saved by the smoke or full-study script unless explicitly enabled.
- If enabled, `optimizer.pt` remains rank-0 local state, **not a full resumable FSDP
  optimizer checkpoint**. Do not enable it expecting exact restart support.
- Layer-wise unsharding involves all ranks; only rank 0 retains full CPU values,
  computes analysis and writes files. Generation never runs inside that context.
- Validation generation runs from a standalone CPU float32 checkpoint on rank 0;
  non-main ranks wait on the long-timeout Gloo control group, not NCCL.

### Initialization, dtype and memory limits

Pretrained models and PEFT adapters are loaded as uniform float32 CPU master
weights. Transformer classes declared by `_no_split_modules` define nested wrap
units, and `device_id` moves/shards one unit at a time. There is no pre-wrap full
`model.to(cuda)`. Embeddings/tied heads remain in the root unit. Unsupported model
structures fail explicitly instead of falling back to a GPU replica.

This requires full CPU model RAM on every rank and GPU capacity for the largest
unit. Rank-zero CPU evaluation and analysis need additional RAM. FSDP training
uses the configured mixed-precision dtype; CPU evaluation uses float32/eager
attention. Full/LoRA/selective LoRA share this policy to avoid mixed-dtype flatten
failures. CPU metrics and latency are not directly comparable to older GPU runs.

FSDP uses its own global `clip_grad_norm_`. With accumulation, `no_sync` may retain
unsharded gradients; reducing accumulation can reduce this memory cost. Windowed
CUDA memory measurements also synchronize at training-window boundaries; actual
performance overhead has not been measured in the offline suite.

### DDP/FSDP control errors and shutdown

Both study and standalone training sessions use an explicit Gloo control group.
`train/pipeline.py::run_training` owns the run flow; `train/distributed.py::TrainingSession`
owns communication resources only, not the model, optimizer, datasets or artifacts. Set
`PUBMEDQA_CONTROL_TIMEOUT_SECONDS` to a positive number of seconds (default 86400)
for evaluation/storage control waits. NCCL tensor-operation timeouts are separate.
A study lends its group to each session; only the creator destroys it, Gloo before NCCL.
The pipeline closes run-owned resources in `finally` on success and failure, including
initialization/model-load failures. `EvaluationSettings` selects CPU float32 evaluation
without temporarily mutating the training session's device.

`continue_on_error` (CLI `--continue-on-error`) only applies to rank-acknowledged
local I/O or validation failures. Decisions are shared by all workers. A failed
forward/backward collective, rank crash or broken communicator aborts the study;
Python error broadcasting cannot safely recover that group. Inspect the original
failure and restart a fresh process group after correcting the cause.

Input rows without any shifted answer target fail before optimizer construction,
with pubid/length but no question/context in the error. Do not shorten sequences
until all answers disappear to make an OOM test pass. Evaluation restores the
original training mode and padding even on failure, and incomplete temporary
snapshots are cleaned up for recoverable errors.

After the smoke test passes, run the full study with:

```bash
bash scripts/run_pubmedqa_full_study.sh
```
