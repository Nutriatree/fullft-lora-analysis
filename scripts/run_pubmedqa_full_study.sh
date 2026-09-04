#!/usr/bin/env bash
# PubMedQA baseline, Full FT, LoRA 비교 실험 전체를 순서대로 실행한다.
#
# 기본 실행(single GPU 0):
#   scripts/run_pubmedqa_full_study.sh
#
# 실행 설정을 바꾸는 예시:
#   PUBMEDQA_RUN_ID=study_single_epoch1 \
#   PUBMEDQA_GPU_IDS=0 \
#   PUBMEDQA_DISTRIBUTED_MODE=single \
#   PUBMEDQA_NUM_EPOCHS=1 \
#   scripts/run_pubmedqa_full_study.sh
#
# 실행 순서:
#   1. B0: 학습하지 않은 baseline을 PQA-L test에서 평가
#   2. F1: 전체 parameter Full Fine-Tuning
#   3. L1~L4: target module과 rank가 서로 다른 LoRA 실험
#   4. L1의 layer별 update 크기로 high/low-update layer 선택
#   5. LL1 및 선택적으로 LL2: 선택된 layer에만 LoRA 적용
#   6. 선택적으로 F2/L5 low-data 실험
#   7. 모든 산출물 검사 및 결과표 생성
set -euo pipefail

# 어떤 위치에서 호출하더라도 repository root를 기준으로 상대 경로를 해석한다.
STUDY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$STUDY_ROOT"

# -----------------------------------------------------------------------------
# 공통 실험 설정
# 환경변수를 지정하지 않으면 아래의 :- 뒤에 있는 값이 사용된다.
# -----------------------------------------------------------------------------
RUN_ID="${PUBMEDQA_RUN_ID:-study_$(date +%Y%m%d_%H%M%S)}"
MODEL_NAME="${PUBMEDQA_MODEL_NAME:-Qwen/Qwen3-1.7B}"
# single은 첫 번째 GPU 한 장을 사용하고, ddp/fsdp는 쉼표로 구분한 GPU를 모두 사용한다.
GPU_IDS="${PUBMEDQA_GPU_IDS:-0}"
DISTRIBUTED_MODE="${PUBMEDQA_DISTRIBUTED_MODE:-single}"
# PQA-A balanced train/validation으로 학습·검증하고 PQA-L test는 최종 평가에만 사용한다.
TRAIN_FILE="${PUBMEDQA_TRAIN_FILE:-data/processed/posttrain_v1/pqa_artificial/train.jsonl}"
VALIDATION_FILE="${PUBMEDQA_VALIDATION_FILE:-data/processed/posttrain_v1/pqa_artificial/validation.jsonl}"
TEST_FILE="${PUBMEDQA_TEST_FILE:-data/processed/pqa_labeled/test.jsonl}"
TRAIN_OUTPUT_DIR="${PUBMEDQA_TRAIN_OUTPUT_DIR:-outputs/pubmedqa_train}"
BASELINE_OUTPUT_DIR="${PUBMEDQA_BASELINE_OUTPUT_DIR:-outputs/pubmedqa_eval}"
# Batch 1 × gradient accumulation 8 = effective batch 8 (single GPU 기준).
NUM_EPOCHS="${PUBMEDQA_NUM_EPOCHS:-3}"
TRAIN_BATCH_SIZE="${PUBMEDQA_TRAIN_BATCH_SIZE:-1}"
EVAL_BATCH_SIZE="${PUBMEDQA_EVAL_BATCH_SIZE:-4}"
GRAD_ACCUM_STEPS="${PUBMEDQA_GRAD_ACCUM_STEPS:-8}"
# LL1/LL2에서 사용할 high/low-update layer 개수다.
SELECTIVE_LAYER_COUNT="${PUBMEDQA_SELECTIVE_LAYER_COUNT:-4}"
# LL2와 low-data 확장 실험의 실행 여부를 1/0으로 제어한다.
INCLUDE_LL2="${PUBMEDQA_INCLUDE_LL2:-1}"
INCLUDE_LOW_DATA="${PUBMEDQA_INCLUDE_LOW_DATA:-0}"
# 재학습을 이어서 하지 않는 기본 실험에서는 optimizer state를 저장하지 않는다.
SAVE_OPTIMIZER_STATE="${PUBMEDQA_SAVE_OPTIMIZER_STATE:-0}"
STUDY_DIR="$TRAIN_OUTPUT_DIR/$RUN_ID"
STUDY_MANIFEST="$STUDY_DIR/study_manifest.json"
RESULTS_FILE="$STUDY_DIR/study_results.json"

# 긴 학습을 시작하기 전에 입력 데이터 누락을 먼저 검사한다.
for data_file in "$TRAIN_FILE" "$VALIDATION_FILE" "$TEST_FILE"; do
  if [[ ! -f "$data_file" ]]; then
    echo "Required dataset file does not exist: $data_file" >&2
    exit 1
  fi
done

if ! command -v python >/dev/null 2>&1; then
  echo "Activate the jw conda environment before running this script." >&2
  exit 1
fi

if [[ "$DISTRIBUTED_MODE" != "single" ]] && ! command -v torchrun >/dev/null 2>&1; then
  echo "torchrun is required for DDP/FSDP mode." >&2
  exit 1
fi

# GPU ID 개수로 torchrun process 수를 정한다. single은 항상 process 하나만 실행한다.
IFS=',' read -r -a GPU_ID_ARRAY <<< "$GPU_IDS"
NUM_VISIBLE_GPUS="${#GPU_ID_ARRAY[@]}"
if [[ "$DISTRIBUTED_MODE" == "single" ]]; then
  NUM_PROCESSES=1
else
  NUM_PROCESSES="$NUM_VISIBLE_GPUS"
  if [[ "$NUM_PROCESSES" -lt 2 ]]; then
    echo "$DISTRIBUTED_MODE mode requires at least 2 visible GPUs in PUBMEDQA_GPU_IDS." >&2
    exit 1
  fi
fi

# 실제 사용한 설정과 선택 layer 정보를 추후 재현할 수 있도록 manifest에 기록한다.
mkdir -p "$STUDY_DIR"
python -c '
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
payload = {
    "run_id": sys.argv[2],
    "model_name": sys.argv[3],
    "gpu_ids": sys.argv[4],
    "distributed_mode": sys.argv[5],
    "train_file": sys.argv[6],
    "validation_file": sys.argv[7],
    "test_file": sys.argv[8],
    "full_study_runs": ["B0", "F1", "L1", "L2", "L3", "L4", "LL1", "LL2"],
    "include_ll2": sys.argv[9] == "1",
    "include_low_data": sys.argv[10] == "1",
    "selective_layer_count": int(sys.argv[11]),
}
path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
' "$STUDY_MANIFEST" "$RUN_ID" "$MODEL_NAME" "$GPU_IDS" "$DISTRIBUTED_MODE" "$TRAIN_FILE" "$VALIDATION_FILE" "$TEST_FILE" "$INCLUDE_LL2" "$INCLUDE_LOW_DATA" "$SELECTIVE_LAYER_COUNT"

# 모든 run에 공통으로 전달할 학습 인자다.
# Full FT는 gradient checkpointing을 켜고 LoRA는 끈다.
# 학습 진행률 25/50/75/100%마다 checkpoint 저장 후 validation을 수행한다.
TRAIN_ARGS=(
  --run-id "$RUN_ID"
  --distributed-mode "$DISTRIBUTED_MODE"
  --device cuda
  --dtype bf16
  --full-ft-gradient-checkpointing
  --no-lora-gradient-checkpointing
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

# single이면 일반 Python process 하나를, ddp/fsdp이면 GPU당 torchrun process 하나를 실행한다.
# 첫 번째 인자는 쉼표로 연결한 run tag이며 나머지는 해당 호출에만 필요한 추가 인자다.
run_experiments() {
  if [[ "$DISTRIBUTED_MODE" == "single" ]]; then
    CUDA_VISIBLE_DEVICES="$GPU_IDS" PYTHONPATH=src \
      python scripts/run_pubmedqa_experiments.py \
      "${TRAIN_ARGS[@]}" --runs "$1" "${@:2}"
    return
  fi

  CUDA_VISIBLE_DEVICES="$GPU_IDS" PYTHONPATH=src \
    torchrun --standalone --nproc_per_node="$NUM_PROCESSES" \
    scripts/run_pubmedqa_experiments.py \
    "${TRAIN_ARGS[@]}" --runs "$1" "${@:2}"
}

echo "[study] run_id=$RUN_ID model=$MODEL_NAME gpus=$GPU_IDS mode=$DISTRIBUTED_MODE"
# Phase 1 run 조건:
#   B0: baseline
#   F1: Full FT, LR 2e-5
#   L1: q/v, rank 8, alpha 16
#   L2: q/k/v/o, rank 8, alpha 16
#   L3: q/v, rank 4, alpha 8
#   L4: q/v, rank 16, alpha 32
echo "[phase 1] B0, F1, L1, L2, L3, L4"
# Add B0, F1 When running the experiments to evaluate the baseline model without any fine-tuning.
run_experiments "L1,L2,L3,L4"

# L1의 100% checkpoint update 통계를 이용해 변화량이 큰/작은 layer를 각각 선택한다.
echo "[phase 2] select high/low layers from L1 final update"
CUDA_VISIBLE_DEVICES="$GPU_IDS" PYTHONPATH=src \
  python scripts/select_pubmedqa_lora_layers.py \
  --run-id "$RUN_ID" \
  --model-name "$MODEL_NAME" \
  --train-output-dir "$TRAIN_OUTPUT_DIR" \
  --layer-count "$SELECTIVE_LAYER_COUNT"

# 선택 결과 JSON에서 layer index를 읽어 후속 CLI에 전달할 쉼표 구분 문자열로 변환한다.
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

# LL1은 L1에서 update가 가장 컸던 layer에만 q/v LoRA(rank 8)를 적용한다.
echo "[phase 3] LL1 high-update layers=$HIGH_LAYERS"
run_experiments "LL1" --target-layer-override "LL1=$HIGH_LAYERS"

COMPLETED_RUNS="B0,F1,L1,L2,L3,L4,LL1"
# LL2는 update가 가장 작았던 layer를 사용하는 대조군이며 환경변수로 생략할 수 있다.
if [[ "$INCLUDE_LL2" == "1" ]]; then
  echo "[phase 4] LL2 low-update control layers=$LOW_LAYERS"
  run_experiments "LL2" --target-layer-override "LL2=$LOW_LAYERS"
  COMPLETED_RUNS="$COMPLETED_RUNS,LL2"
fi

# F2/L5는 train example을 제한하는 선택 실험이며 기본값에서는 실행하지 않는다.
if [[ "$INCLUDE_LOW_DATA" == "1" ]]; then
  echo "[phase 5] F2, L5 low-data extension"
  run_experiments "F2,L5"
  COMPLETED_RUNS="$COMPLETED_RUNS,F2,L5"
fi

# 필수 파일 존재 여부를 검사하고 JSON/CSV 비교 결과를 생성한다.
echo "[validation] $COMPLETED_RUNS"
PYTHONPATH=src python scripts/validate_pubmedqa_runs.py \
  --run-id "$RUN_ID" \
  --runs "$COMPLETED_RUNS" \
  --model-name "$MODEL_NAME" \
  --baseline-output-dir "$BASELINE_OUTPUT_DIR" \
  --train-output-dir "$TRAIN_OUTPUT_DIR" \
  --output-file "$RESULTS_FILE"

echo "[complete] run_id=$RUN_ID study_manifest=$STUDY_MANIFEST results=$RESULTS_FILE"
