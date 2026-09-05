# PubMedQA 2-GPU FSDP smoke test and troubleshooting

## Purpose

Run this smoke test before the full study whenever the model, PyTorch/CUDA environment, GPU assignment, or FSDP implementation changes. It exercises:

1. rank-0 reference validation before FSDP wrapping
2. Qwen decoder-layer auto-wrap
3. one optimizer step across two GPUs
4. a rank-0 CPU-offloaded full checkpoint
5. standalone checkpoint validation and test inference
6. Gloo control-group and NCCL process-group shutdown

## Run the smoke test

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

- Model state is gathered only on rank 0 and offloaded to CPU before saving.
- Optimizer state is not saved by the smoke or full-study script unless explicitly enabled.
- Layer-wise full parameters are materialized only on rank 0 and offloaded to CPU.
- Validation generation runs from a standalone checkpoint on rank 0; non-main ranks wait on the long-timeout Gloo control group, not NCCL.

After the smoke test passes, run the full study with:

```bash
bash scripts/run_pubmedqa_full_study.sh
```
