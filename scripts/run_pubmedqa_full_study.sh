#!/usr/bin/env bash
# Run the complete PubMedQA Full FT vs LoRA study defined in the experiment plan.
set -euo pipefail

STUDY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$STUDY_ROOT"

RUN_ID="${PUBMEDQA_RUN_ID:-study_$(date +%Y%m%d_%H%M%S)}"
MODEL_NAME="${PUBMEDQA_MODEL_NAME:-Qwen/Qwen3-1.7B}"
GPU_IDS="${PUBMEDQA_GPU_IDS:-0,1}"
TRAIN_FILE="${PUBMEDQA_TRAIN_FILE:-data/processed/posttrain_v1/pqa_artificial/train.jsonl}"
VALIDATION_FILE="${PUBMEDQA_VALIDATION_FILE:-data/processed/posttrain_v1/pqa_artificial/validation.jsonl}"
TEST_FILE="${PUBMEDQA_TEST_FILE:-data/processed/pqa_labeled/test.jsonl}"
TRAIN_OUTPUT_DIR="${PUBMEDQA_TRAIN_OUTPUT_DIR:-outputs/pubmedqa_train}"
BASELINE_OUTPUT_DIR="${PUBMEDQA_BASELINE_OUTPUT_DIR:-outputs/pubmedqa_eval}"
NUM_EPOCHS="${PUBMEDQA_NUM_EPOCHS:-3}"
TRAIN_BATCH_SIZE="${PUBMEDQA_TRAIN_BATCH_SIZE:-1}"
EVAL_BATCH_SIZE="${PUBMEDQA_EVAL_BATCH_SIZE:-2}"
GRAD_ACCUM_STEPS="${PUBMEDQA_GRAD_ACCUM_STEPS:-16}"
SELECTIVE_LAYER_COUNT="${PUBMEDQA_SELECTIVE_LAYER_COUNT:-4}"
INCLUDE_LL2="${PUBMEDQA_INCLUDE_LL2:-1}"
INCLUDE_LOW_DATA="${PUBMEDQA_INCLUDE_LOW_DATA:-0}"
SAVE_OPTIMIZER_STATE="${PUBMEDQA_SAVE_OPTIMIZER_STATE:-0}"
STUDY_DIR="$TRAIN_OUTPUT_DIR/$RUN_ID"
STUDY_MANIFEST="$STUDY_DIR/study_manifest.json"
RESULTS_FILE="$STUDY_DIR/study_results.json"

for data_file in "$TRAIN_FILE" "$VALIDATION_FILE" "$TEST_FILE"; do
  if [[ ! -f "$data_file" ]]; then
    echo "Required dataset file does not exist: $data_file" >&2
    exit 1
  fi
done

if ! command -v python >/dev/null 2>&1 || ! command -v torchrun >/dev/null 2>&1; then
  echo "Activate the jw conda environment before running this script." >&2
  exit 1
fi

mkdir -p "$STUDY_DIR"
python -c '
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
payload = {
    "run_id": sys.argv[2],
    "model_name": sys.argv[3],
    "gpu_ids": sys.argv[4],
    "train_file": sys.argv[5],
    "validation_file": sys.argv[6],
    "test_file": sys.argv[7],
    "full_study_runs": ["B0", "F1", "L1", "L2", "L3", "L4", "LL1", "LL2"],
    "include_ll2": sys.argv[8] == "1",
    "include_low_data": sys.argv[9] == "1",
    "selective_layer_count": int(sys.argv[10]),
}
path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
' "$STUDY_MANIFEST" "$RUN_ID" "$MODEL_NAME" "$GPU_IDS" "$TRAIN_FILE" "$VALIDATION_FILE" "$TEST_FILE" "$INCLUDE_LL2" "$INCLUDE_LOW_DATA" "$SELECTIVE_LAYER_COUNT"

TRAIN_ARGS=(
  --run-id "$RUN_ID"
  --distributed-mode fsdp
  --device cuda
  --dtype bf16
  --gradient-checkpointing
  --model-name "$MODEL_NAME"
  --train-file "$TRAIN_FILE"
  --validation-file "$VALIDATION_FILE"
  --test-file "$TEST_FILE"
  --baseline-eval-file "$TEST_FILE"
  --train-output-dir "$TRAIN_OUTPUT_DIR"
  --baseline-output-dir "$BASELINE_OUTPUT_DIR"
  --num-epochs "$NUM_EPOCHS"
  --train-batch-size "$TRAIN_BATCH_SIZE"
  --eval-batch-size "$EVAL_BATCH_SIZE"
  --gradient-accumulation-steps "$GRAD_ACCUM_STEPS"
  --checkpoint-percents 25,50,75,100
)

if [[ "$SAVE_OPTIMIZER_STATE" != "1" ]]; then
  TRAIN_ARGS+=(--no-save-optimizer-state)
fi

run_experiments() {
  CUDA_VISIBLE_DEVICES="$GPU_IDS" PYTHONPATH=src \
    torchrun --standalone --nproc_per_node=2 scripts/run_pubmedqa_experiments.py \
    "${TRAIN_ARGS[@]}" --runs "$1" "${@:2}"
}

echo "[study] run_id=$RUN_ID model=$MODEL_NAME gpus=$GPU_IDS"
echo "[phase 1] B0, F1, L1, L2, L3, L4"
run_experiments "B0,F1,L1,L2,L3,L4"

echo "[phase 2] select high/low layers from L1 final update"
CUDA_VISIBLE_DEVICES="$GPU_IDS" PYTHONPATH=src \
  python scripts/select_pubmedqa_lora_layers.py \
  --run-id "$RUN_ID" \
  --model-name "$MODEL_NAME" \
  --train-output-dir "$TRAIN_OUTPUT_DIR" \
  --layer-count "$SELECTIVE_LAYER_COUNT"

L1_ROOT="$TRAIN_OUTPUT_DIR/$RUN_ID/${MODEL_NAME//\//_}/lora"
HIGH_LAYER_FILE="$L1_ROOT/analysis/selective_layers_high.json"
LOW_LAYER_FILE="$L1_ROOT/analysis/selective_layers_low.json"
HIGH_LAYERS="$(python -c 'import json,sys; print(",".join(map(str, json.load(open(sys.argv[1]))["target_layers"])))' "$HIGH_LAYER_FILE")"
LOW_LAYERS="$(python -c 'import json,sys; print(",".join(map(str, json.load(open(sys.argv[1]))["target_layers"])))' "$LOW_LAYER_FILE")"

python -c '
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
payload["selective_layer_selection"] = {
    "high_update_file": sys.argv[2],
    "low_update_file": sys.argv[3],
    "high_update_layers": [int(value) for value in sys.argv[4].split(",") if value],
    "low_update_layers": [int(value) for value in sys.argv[5].split(",") if value],
}
path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
' "$STUDY_MANIFEST" "$HIGH_LAYER_FILE" "$LOW_LAYER_FILE" "$HIGH_LAYERS" "$LOW_LAYERS"

echo "[phase 3] LL1 high-update layers=$HIGH_LAYERS"
run_experiments "LL1" --target-layer-override "LL1=$HIGH_LAYERS"

COMPLETED_RUNS="B0,F1,L1,L2,L3,L4,LL1"
if [[ "$INCLUDE_LL2" == "1" ]]; then
  echo "[phase 4] LL2 low-update control layers=$LOW_LAYERS"
  run_experiments "LL2" --target-layer-override "LL2=$LOW_LAYERS"
  COMPLETED_RUNS="$COMPLETED_RUNS,LL2"
fi

if [[ "$INCLUDE_LOW_DATA" == "1" ]]; then
  echo "[phase 5] F2, L5 low-data extension"
  run_experiments "F2,L5"
  COMPLETED_RUNS="$COMPLETED_RUNS,F2,L5"
fi

echo "[validation] $COMPLETED_RUNS"
PYTHONPATH=src python scripts/validate_pubmedqa_runs.py \
  --run-id "$RUN_ID" \
  --runs "$COMPLETED_RUNS" \
  --model-name "$MODEL_NAME" \
  --baseline-output-dir "$BASELINE_OUTPUT_DIR" \
  --train-output-dir "$TRAIN_OUTPUT_DIR" \
  --output-file "$RESULTS_FILE"

echo "[complete] run_id=$RUN_ID study_manifest=$STUDY_MANIFEST results=$RESULTS_FILE"
