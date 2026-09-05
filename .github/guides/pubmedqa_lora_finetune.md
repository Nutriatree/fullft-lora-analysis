# PubMedQA LoRA 실행 가이드

이 문서는 [LoRA strategy](../../src/pubmedqa/training/strategies/lora.py)와 [LoRA CLI](../../scripts/run_pubmedqa_lora_finetune.py)를 사용해 PubMedQA LoRA 학습을 실행하는 방법을 설명한다.

기준 환경은 서버의 `conda` 환경 `jw` 이다.

## 개요

LoRA 학습 코드는 Full FT와 같은 실험 계약을 유지한다.

- 같은 train / validation / test split 사용
- 같은 prompt / parser / 평가 방식 사용
- 같은 `0/25/50/75/100/Best` checkpoint 구조 사용
- 같은 `summary.json`, `run_metadata.json`, `analysis/*` 구조 사용

추가로 LoRA 전용 분석 artifact를 저장한다.

- `analysis/lora_configuration.json`
- `analysis/lora_dynamics.json`
- `analysis/lora_singular_values.json`
- `analysis/lora_effective_rank.json`
- `analysis/lora_adapter_norms.json`
- `analysis/lora_update_direction.json`
- `analysis/module_update_share.json`
- `analysis/val_update_alignment.json`

## 사전 조건

- `conda` 환경 `jw` 가 활성화되어 있어야 한다.
- `HF_TOKEN` 환경변수는 필요한 경우만 주입한다.
- 입력 데이터는 JSONL 형식이어야 한다.
- base model은 현재 `Qwen/Qwen3-1.7B` 기준으로 맞춰져 있다.

예시:

```bash
export HF_TOKEN=...
```

## 기본 실행 예시

대표 LoRA run `L1` 예시는 다음과 같다.

```bash
PYTHONPATH=src python scripts/run_pubmedqa_lora_finetune.py \
  --train-file data/processed/pqa_labeled/train.jsonl \
  --validation-file data/processed/pqa_labeled/validation.jsonl \
  --test-file data/processed/pqa_labeled/test.jsonl \
  --output-dir outputs/pubmedqa_train \
  --run-id lora_20260903 \
  --run-tag L1 \
  --method-name lora \
  --condition lora \
  --model-name Qwen/Qwen3-1.7B \
  --data-regime full-data \
  --data-fraction 1.0 \
  --hf-token "$HF_TOKEN" \
  --num-epochs 3 \
  --train-batch-size 2 \
  --eval-batch-size 4 \
  --gradient-accumulation-steps 8 \
  --learning-rate 2e-5 \
  --weight-decay 0.01 \
  --warmup-ratio 0.03 \
  --max-grad-norm 1.0 \
  --max-new-tokens 4 \
  --dtype bfloat16 \
  --target-modules q_proj,v_proj \
  --lora-rank 8 \
  --lora-alpha 16 \
  --lora-dropout 0.0 \
  --checkpoint-percents 25,50,75,100
```

## 2x3090 FSDP 실행

Full FT와 LoRA의 실행 조건을 맞추려면 LoRA도 같은 `torchrun` + FSDP 환경에서 실행할 수 있다. FSDP가 base model과 adapter parameter를 분산하고, PEFT adapter checkpoint는 rank 0이 Hugging Face adapter 형식으로 저장한다.

```bash
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=src \
  torchrun --standalone --nproc_per_node=2 scripts/run_pubmedqa_lora_finetune.py \
  --train-file data/processed/pqa_artificial/train.jsonl \
  --validation-file data/processed/pqa_artificial/validation.jsonl \
  --test-file data/processed/pqa_labeled/test.jsonl \
  --run-id l1_2x3090_20260903 \
  --run-tag L1 \
  --distributed-mode fsdp \
  --device cuda --dtype bf16 --gradient-checkpointing \
  --train-batch-size 1 --eval-batch-size 2 --gradient-accumulation-steps 16 \
  --target-modules q_proj,v_proj --lora-rank 8 --lora-alpha 16 \
  --no-save-optimizer-state
```

FSDP에서 validation과 layer-wise LoRA `A/B`, `Delta W` 분석은 모든 rank가 full-parameter gather에 참여하고 rank 0이 기록한다. 따라서 결과 artifact는 단일 GPU 실행과 같은 위치에 한 번만 생성된다.

## run별 권장 설정

### `L1`

- `--run-tag L1`
- `--target-modules q_proj,v_proj`
- `--lora-rank 8`

### `L2`

- `--run-tag L2`
- `--target-modules q_proj,k_proj,v_proj,o_proj`
- `--lora-rank 8`

### `L3`

- `--run-tag L3`
- `--target-modules q_proj,v_proj`
- `--lora-rank 4`

### `L4`

- `--run-tag L4`
- `--target-modules q_proj,v_proj`
- `--lora-rank 16`

### `LL1`

High-update layer가 외부 분석으로 결정된 뒤 실행한다.

```bash
PYTHONPATH=src python scripts/run_pubmedqa_lora_finetune.py \
  --train-file data/processed/pqa_labeled/train.jsonl \
  --validation-file data/processed/pqa_labeled/validation.jsonl \
  --test-file data/processed/pqa_labeled/test.jsonl \
  --output-dir outputs/pubmedqa_train \
  --run-id selective_lora_20260903 \
  --run-tag LL1 \
  --condition selective-lora \
  --model-name Qwen/Qwen3-1.7B \
  --target-modules q_proj,v_proj \
  --target-layers-file configs/selected_layers_high_update.json \
  --layer-scope high-update \
  --lora-rank 8
```

직접 layer를 주입할 수도 있다.

```bash
--target-layers 20,21,22,23
```

## 주요 인자

### 공통 학습 인자

| 인자 | 설명 |
| --- | --- |
| `--train-file` | train JSONL 경로 |
| `--validation-file` | validation JSONL 경로 |
| `--test-file` | test JSONL 경로 |
| `--run-id` | 실험 식별자 |
| `--run-tag` | `L1`, `L2`, `L3` 같은 run 이름 |
| `--condition` | 출력 경로용 condition |
| `--model-name` | base model 이름 |
| `--num-epochs` | 학습 epoch 수 |
| `--train-batch-size` | 학습 배치 크기 |
| `--eval-batch-size` | 평가 배치 크기 |
| `--gradient-accumulation-steps` | gradient accumulation 수 |
| `--learning-rate` | 학습률 |
| `--weight-decay` | weight decay |
| `--warmup-ratio` | scheduler warmup 비율 |
| `--max-input-tokens` | 입력 최대 길이 |
| `--max-new-tokens` | 생성 최대 길이 |
| `--dtype` | `float16`, `bfloat16`, `float32` |
| `--distributed-mode` | `single` 또는 `fsdp`; `fsdp`는 `torchrun --nproc_per_node=2` 필요 |
| `--fsdp-cpu-offload` | FSDP parameter CPU offload. 기본값은 `false` |
| `--strict-parser` | strict parser 사용 여부 |

### LoRA 전용 인자

| 인자 | 설명 |
| --- | --- |
| `--target-modules` | 쉼표 구분 target module 목록 |
| `--target-layers` | 쉼표 구분 target layer 목록 |
| `--target-layers-file` | target layer JSON 파일 |
| `--layer-scope` | `all`, `high-update`, `low-update`, `custom` 등 기록값 |
| `--lora-rank` | rank `r` |
| `--lora-alpha` | alpha |
| `--lora-dropout` | dropout |
| `--lora-bias` | `none`, `all`, `lora_only` |
| `--lora-task-type` | 기본값 `CAUSAL_LM` |
| `--modules-to-save` | adapter 외 추가 저장 module |
| `--merge-for-eval` | adapter merge 후 평가 여부 |

## 환경변수 방식

CLI 대신 환경변수로도 실행할 수 있다.

```bash
export HF_TOKEN=...
export PUBMEDQA_RUN_ID=lora_20260903
export PUBMEDQA_RUN_TAG=L1
export PUBMEDQA_METHOD_NAME=lora
export PUBMEDQA_CONDITION=lora
export PUBMEDQA_MODEL_NAME=Qwen/Qwen3-1.7B
export PUBMEDQA_TRAIN_PATH=data/processed/pqa_labeled/train.jsonl
export PUBMEDQA_VALIDATION_PATH=data/processed/pqa_labeled/validation.jsonl
export PUBMEDQA_TEST_PATH=data/processed/pqa_labeled/test.jsonl
export PUBMEDQA_OUTPUT_DIR=outputs/pubmedqa_train
export PUBMEDQA_NUM_EPOCHS=3
export PUBMEDQA_TRAIN_BATCH_SIZE=2
export PUBMEDQA_EVAL_BATCH_SIZE=4
export PUBMEDQA_GRAD_ACCUM_STEPS=8
export PUBMEDQA_LEARNING_RATE=2e-5
export PUBMEDQA_WEIGHT_DECAY=0.01
export PUBMEDQA_WARMUP_RATIO=0.03
export PUBMEDQA_MAX_NEW_TOKENS=4
export PUBMEDQA_DTYPE=bf16
export PUBMEDQA_LORA_TARGET_MODULES=q_proj,v_proj
export PUBMEDQA_LORA_RANK=8
export PUBMEDQA_LORA_ALPHA=16
export PUBMEDQA_LORA_DROPOUT=0.0
export PUBMEDQA_CHECKPOINT_PERCENTS=25,50,75,100

PYTHONPATH=src python -m pubmedqa.lora_finetune
```

## selective layer JSON 형식

`--target-layers-file` 또는 `PUBMEDQA_LORA_TARGET_LAYERS_FILE` 에 넣을 JSON 예시는 다음과 같다.

```json
{
  "layer_scope": "high-update",
  "target_layers": [20, 21, 22, 23]
}
```

현재 코드는 `target_layers` 배열만 실제 읽고, `layer_scope` 값은 metadata 기록용으로 사용한다.

## 출력 구조

출력 구조는 Full FT와 동일하다.

```text
outputs/pubmedqa_train/
  <run_id>/
    Qwen_Qwen3-1.7B/
      <condition>/
        summary.json
        run_metadata.json
        config.json
        artifacts.json
        checkpoints/
        logs/
        evaluations/
        layerwise_updates/
        prediction_transitions/
        analysis/
```

LoRA 전용으로 확인해야 하는 파일은 다음이다.

- `checkpoints/*/lora_adapter_metrics.jsonl`
- `checkpoints/*/lora_adapter_summary.json`
- `analysis/lora_dynamics.json`
- `analysis/lora_singular_values.json`
- `analysis/lora_effective_rank.json`
- `analysis/lora_adapter_norms.json`
- `analysis/lora_update_direction.json`
- `analysis/module_update_share.json`
- `analysis/val_update_alignment.json`

## 현재 구현 기준 주의점

- checkpoint 저장은 adapter-only 기준이다.
- `merge_for_eval` 은 로드 후 merge 평가를 지원하지만, merged checkpoint 파일을 별도로 저장하지는 않는다.
- `target_layers` 는 PEFT `layers_to_transform` 방식으로 전달된다.
- `layer_scope` 는 선택된 layer의 의미를 남기기 위한 metadata다. high-update / low-update 판정은 trainer가 하지 않는다.
- low-data preset 실행 스크립트는 아직 따로 없다. 현재는 `--max-train-examples` 또는 별도 low-data JSONL로 맞추는 방식이다.

## 권장 검증 순서

1. `L1` 조건으로 소규모 dry run 실행
2. `summary.json`과 `run_metadata.json` 확인
3. `checkpoints/*/lora_adapter_metrics.jsonl` 생성 여부 확인
4. `analysis/lora_effective_rank.json` 과 `analysis/module_update_share.json` 확인
5. 이후 `L2`, `L3`, `L4`를 설정만 바꿔 반복
