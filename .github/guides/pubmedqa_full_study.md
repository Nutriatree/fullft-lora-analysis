# PubMedQA 전체 실험 실행

[`scripts/run_pubmedqa_full_study.sh`](../../scripts/run_pubmedqa_full_study.sh)는 baseline, Full FT, LoRA configuration, selective LoRA를 순차 실행하고 최종 비교 파일을 저장한다. 현재 기본 실행 mode는 GPU 0 한 장을 사용하는 `single`이다.

## 실행 흐름

1. `B0`, `F1`, `L1`, `L2`, `L3`, `L4`를 실행한다.
2. `L1` 최종 checkpoint의 layer-wise relative update norm으로 high/low layer를 각각 선택한다.
3. 선택된 high-update layer로 `LL1`, low-update control layer로 `LL2`를 실행한다.
4. 모든 run의 필수 artifact를 검증하고 비교 지표를 JSON/CSV로 수집한다.

`LL2`는 기본적으로 포함되며, low-data 확장인 `F2/L5`는 기본적으로 제외된다. LL1/LL2 layer는 L1 artifact에서 동적으로 정하므로 사용자가 checkpoint나 layer를 미리 지정하지 않는다.

## 보고서 조건 재현

최종 보고서는 RTX 3090 한 장, 1 epoch 조건을 사용했다. Shell script 자체의 epoch 기본값은 3이므로 보고서 조건을 재현하려면 `PUBMEDQA_NUM_EPOCHS=1`을 명시한다.

```bash
PUBMEDQA_RUN_ID=study_single_epoch1 \
PUBMEDQA_GPU_IDS=0 \
PUBMEDQA_DISTRIBUTED_MODE=single \
PUBMEDQA_NUM_EPOCHS=1 \
scripts/run_pubmedqa_full_study.sh
```

필요하면 `HF_TOKEN`을 함께 전달한다.

```bash
HF_TOKEN=... \
PUBMEDQA_DISTRIBUTED_MODE=single \
PUBMEDQA_NUM_EPOCHS=1 \
scripts/run_pubmedqa_full_study.sh
```

## DDP / FSDP

`PUBMEDQA_DISTRIBUTED_MODE`는 `single`, `ddp`, `fsdp`를 지원한다. 두 GPU에서 single과 동일한 global effective batch 8을 유지하려면 per-device batch 1, gradient accumulation 4를 사용한다.

```bash
PUBMEDQA_GPU_IDS=0,1 \
PUBMEDQA_DISTRIBUTED_MODE=ddp \
PUBMEDQA_TRAIN_BATCH_SIZE=1 \
PUBMEDQA_GRAD_ACCUM_STEPS=4 \
PUBMEDQA_NUM_EPOCHS=1 \
scripts/run_pubmedqa_full_study.sh
```

FSDP는 환경별 PyTorch/CUDA 호환성의 영향을 받으므로 먼저 smoke test를 실행한다.

```bash
bash scripts/run_pubmedqa_fsdp_smoke.sh

PUBMEDQA_GPU_IDS=0,1 \
PUBMEDQA_DISTRIBUTED_MODE=fsdp \
PUBMEDQA_TRAIN_BATCH_SIZE=1 \
PUBMEDQA_GRAD_ACCUM_STEPS=4 \
PUBMEDQA_NUM_EPOCHS=1 \
scripts/run_pubmedqa_full_study.sh
```

Global effective batch는 `per-device batch × process 수 × gradient accumulation`이다.

## 주요 환경변수

| Variable | Default | Description |
|---|---|---|
| `PUBMEDQA_RUN_ID` | timestamp | Study identifier |
| `PUBMEDQA_MODEL_NAME` | `Qwen/Qwen3-1.7B` | Base model |
| `PUBMEDQA_GPU_IDS` | `0` | Visible GPU IDs |
| `PUBMEDQA_DISTRIBUTED_MODE` | `single` | `single`, `ddp`, or `fsdp` |
| `PUBMEDQA_NUM_EPOCHS` | `3` | Training epochs |
| `PUBMEDQA_TRAIN_BATCH_SIZE` | `1` | Per-device train batch |
| `PUBMEDQA_EVAL_BATCH_SIZE` | `4` | Per-device evaluation batch |
| `PUBMEDQA_GRAD_ACCUM_STEPS` | `8` | Gradient accumulation |
| `PUBMEDQA_SELECTIVE_LAYER_COUNT` | `4` | High/low layer count |
| `PUBMEDQA_INCLUDE_LL2` | `1` | Run low-update control |
| `PUBMEDQA_INCLUDE_LOW_DATA` | `0` | Run F2/L5 extension |
| `PUBMEDQA_SAVE_OPTIMIZER_STATE` | `0` | Save optimizer state |

## 저장 파일

`outputs/pubmedqa_train/<run_id>/`에 다음 파일을 남긴다.

- `study_manifest.json`: 데이터 경로, GPU, 학습 조건, 포함 run, high/low layer 선택 결과
- `study_results.json`: 모든 run의 수집 결과와 필수 artifact 검증 결과
- `study_results.csv`: validation 성능, parameter, memory, training time, checkpoint size, inference 지표 비교표

각 run의 최종 PQA-L 성능은 `<condition>/evaluations/test_summary.json`에 저장된다. 최상위 `study_results.csv`의 `accuracy`와 `macro_f1`은 checkpoint 선택에 사용한 validation 성능이므로 test 성능과 구분해야 한다.

L1 output의 `analysis/selective_layers_high.json`과 `analysis/selective_layers_low.json`에는 layer 선택 기준, source checkpoint, 전체 layer score가 저장된다.

## Low-data 확장

```bash
PUBMEDQA_INCLUDE_LOW_DATA=1 scripts/run_pubmedqa_full_study.sh
```

이 경우 `F2`, `L5`도 최종 `study_results.json`과 CSV에 포함된다.
