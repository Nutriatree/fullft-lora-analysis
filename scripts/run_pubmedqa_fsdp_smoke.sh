#!/usr/bin/env bash
# Minimal two-GPU FSDP validation: initial eval, one update, checkpoint, test, shutdown.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

RUN_ID="${PUBMEDQA_RUN_ID:-fsdp_smoke_$(date +%Y%m%d_%H%M%S)}"
MODEL_NAME="${PUBMEDQA_MODEL_NAME:-Qwen/Qwen3-1.7B}"
GPU_IDS="${PUBMEDQA_GPU_IDS:-0,1}"
TRAIN_FILE="${PUBMEDQA_TRAIN_FILE:-data/processed/pqa_artificial/train.jsonl}"
VALIDATION_FILE="${PUBMEDQA_VALIDATION_FILE:-data/processed/pqa_artificial/validation.jsonl}"
TEST_FILE="${PUBMEDQA_TEST_FILE:-data/processed/pqa_labeled/test.jsonl}"
OUTPUT_DIR="${PUBMEDQA_TRAIN_OUTPUT_DIR:-outputs/pubmedqa_train}"
MAX_INPUT_TOKENS="${PUBMEDQA_MAX_INPUT_TOKENS:-}"

IFS=',' read -r -a GPU_LIST <<< "$GPU_IDS"
if [[ "${#GPU_LIST[@]}" -ne 2 ]]; then
  echo "PUBMEDQA_GPU_IDS must contain exactly two comma-separated GPU IDs." >&2
  exit 1
fi

for data_file in "$TRAIN_FILE" "$VALIDATION_FILE" "$TEST_FILE"; do
  if [[ ! -f "$data_file" ]]; then
    echo "Required dataset file does not exist: $data_file" >&2
    exit 1
  fi
done

echo "[fsdp-smoke] run_id=$RUN_ID model=$MODEL_NAME gpus=$GPU_IDS"

SMOKE_ARGS=(
  --run-id "$RUN_ID"
  --runs F1
  --model-name "$MODEL_NAME"
  --train-file "$TRAIN_FILE"
  --validation-file "$VALIDATION_FILE"
  --test-file "$TEST_FILE"
  --train-output-dir "$OUTPUT_DIR"
  --distributed-mode fsdp
  --device cuda
  --dtype bf16
  --gradient-checkpointing
  --num-epochs 1
  --train-batch-size 1
  --eval-batch-size 1
  --gradient-accumulation-steps 1
  --max-train-examples 2
  --max-validation-examples 2
  --max-test-examples 2
  --checkpoint-percents 100
  --no-save-optimizer-state
)

if [[ -n "$MAX_INPUT_TOKENS" ]]; then
  SMOKE_ARGS+=(--max-input-tokens "$MAX_INPUT_TOKENS")
fi

CUDA_VISIBLE_DEVICES="$GPU_IDS" \
NCCL_DEBUG="${NCCL_DEBUG:-INFO}" \
TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-DETAIL}" \
TORCH_SHOW_CPP_STACKTRACES="${TORCH_SHOW_CPP_STACKTRACES:-1}" \
PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}" \
PYTHONPATH=src \
torchrun --standalone --nproc_per_node=2 scripts/run_pubmedqa_experiments.py "${SMOKE_ARGS[@]}"

echo "[fsdp-smoke] completed: $OUTPUT_DIR/$RUN_ID"
