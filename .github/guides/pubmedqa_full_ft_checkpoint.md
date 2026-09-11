# PubMedQA Full Fine-Tuning Checkpoint 문서

이 문서는 PubMedQA의 Full Fine-Tuning 실험에서 checkpoint가 어떤 역할을 해야 하는지, 어떤 정보가 반드시 저장되어야 하는지, 그리고 현재 코드 기준으로 어느 수준까지 반영되어 있는지를 점검하기 위한 기준 문서다.

대상 실험은 다음 두 가지다.

- `B0`: 학습 전 baseline evaluation
- `F1`: Full Fine-Tuning 100% data

선택적으로 low-data 실험이 필요하면 다음도 포함한다.

- `F2`: Full Fine-Tuning low-data

이 문서는 특히 `F1`의 중간 checkpoint 저장과 layer-wise / temporal adaptation 분석 가능 여부를 확인하는 용도로 사용한다.  
실행 순서는 [train/pipeline.py](../../src/pubmedqa/train/pipeline.py),
checkpoint 저장은 [train/checkpoints.py](../../src/pubmedqa/train/checkpoints.py),
분석은 [train/full_ft.py](../../src/pubmedqa/train/full_ft.py)가 소유한다.
이전 `training/engine.py`는 제거했다. 기존 notebook은 공개 `pubmedqa.full_finetune` API를 사용한다.

## 1. 연구 목적과 checkpoint의 역할

Full Fine-Tuning에서 checkpoint는 단순 복구 지점이 아니라, 학습 과정 전체를 시간축으로 분석하기 위한 관측 지점이다.

checkpoint를 통해 확인하려는 내용은 다음과 같다.

- 성능이 언제 얼마나 향상되는가
- validation 성능이 언제 정점에 도달하는가
- parameter 변화가 어떤 layer / component에 집중되는가
- 변화가 초기에 몰리는가, 후반까지 계속 누적되는가
- 성능 포화 이후에도 weight drift가 계속 증가하는가

즉 checkpoint는 아래 질문에 답할 수 있어야 한다.

> Full Fine-Tuning은 PubMedQA에 적응하면서 Transformer의 어떤 Layer와 Component를, 학습의 어느 시점에, 얼마나 변화시키는가?

## 2. 분석 대상 component

Full FT의 layer-wise 분석 대상은 다음 projection / MLP weight다.

- `q_proj`
- `k_proj`
- `v_proj`
- `o_proj`
- `gate_proj`
- `up_proj`
- `down_proj`

현재 구현은 Llama 계열 파라미터 이름 기준으로 다음 suffix를 추적 대상으로 사용한다.

- `self_attn.q_proj.weight`
- `self_attn.k_proj.weight`
- `self_attn.v_proj.weight`
- `self_attn.o_proj.weight`
- `mlp.gate_proj.weight`
- `mlp.up_proj.weight`
- `mlp.down_proj.weight`

## 3. 권장 checkpoint 시점

한 run을 여러 번 반복하지 않고, 한 번의 F1 학습 과정에서 다음 시점을 저장하는 것을 기본 원칙으로 한다.

| 시점 | 의미 | 용도 |
| --- | --- | --- |
| `0%` | Pretrained 상태 | 기준점 |
| `25%` | 학습 초기 | 초기 adaptation 관찰 |
| `50%` | 학습 중기 | 변화 누적 확인 |
| `75%` | 학습 후기 | 후반 drift 확인 |
| `100%` | 최종 상태 | 최종 모델 |
| `Best` | Validation 기준 최고 checkpoint | 최적 성능 지점 |

주의:

- `Best`는 별도 checkpoint 파일을 추가로 복제하는 방식이 아니라, 저장된 checkpoint들 중 validation 기준 최적 지점을 식별하는 개념이다.
- 구현상 `epoch_end` checkpoint가 추가로 생길 수 있다.
- 구현상 `100%`는 `scheduled`, `epoch_end`, `final` 중 하나 이상으로 중복 기록될 수 있다.

## 4. 필수 수식

### 4.1 누적 변화량

checkpoint `t`에서 layer `l`, module `m`에 대해:

$$
C_{l,m}^{(t)} = \frac{ \left\|W_{l,m}^{(t)}-W_{l,m}^{(0)}\right\|_F }{ \left\|W_{l,m}^{(0)}\right\|_F }
$$

이 값은 pretrained 상태에서 현재 checkpoint까지 얼마나 누적 변화가 일어났는지 나타낸다.

### 4.2 구간별 변화량

checkpoint `t`에서 직전 checkpoint `t-1` 대비:

$$
I_{l,m}^{(t)} = \frac{ \left\|W_{l,m}^{(t)}-W_{l,m}^{(t-1)}\right\|_F }{ \left\|W_{l,m}^{(0)}\right\|_F }
$$

이 값은 특정 학습 구간에서 어느 layer / module이 많이 갱신되었는지 보여준다.

### 4.3 Update Share

각 checkpoint에서 전체 update 중 특정 layer / module이 차지하는 비율도 함께 남겨야 한다.

- cumulative update share
- incremental update share

이 값은 adaptation이 특정 위치에 집중되는지를 확인하는 데 필요하다.

## 5. 필수 기록 요소

### 5.1 Performance

- Accuracy
- Macro-F1
- Class-wise F1
- Invalid Rate
- Confusion Matrix

목적:

- task 성능 비교
- class별 불균형 영향 확인
- sample 오류 패턴 분석

### 5.2 Parameter

- Total Params
- Trainable Params
- Trainable Ratio

목적:

- Full FT의 학습 규모 확인
- LoRA와의 후속 비교 기준 제공

### 5.3 Memory

- Loaded VRAM
- Peak Allocated
- Peak Reserved
- Idle memory

목적:

- 학습 메모리 비용 측정

### 5.4 Training Efficiency

- Total Time
- sec/step
- samples/sec
- tokens/sec

목적:

- compute cost 비교

### 5.5 Optimization

- Train Loss
- Validation Loss
- Learning Rate
- Gradient Norm

목적:

- convergence 해석
- overfitting 시점 해석

### 5.6 Learning Timeline

- checkpoint kind
- checkpoint percent
- step
- epoch
- elapsed time

목적:

- 변화량과 성능을 시간축에 정렬

### 5.7 Layer-wise Update

- parameter name
- layer index
- module / component name
- base weight norm
- current weight norm
- cumulative update norm
- cumulative relative update norm
- incremental update norm
- incremental relative update norm

목적:

- 어느 위치가 얼마나 변했는지 분석

### 5.8 Update Concentration

- cumulative update share
- incremental update share
- by-layer aggregate
- by-component aggregate

목적:

- adaptation 집중 위치 추적

### 5.9 Update Direction

- checkpoint 간 update cosine similarity

목적:

- 학습 방향이 초중후반에 바뀌는지 확인

### 5.10 Prediction Transition

- Wrong → Correct
- Correct → Wrong
- sample-level prediction timeline

목적:

- 성능 향상이 어떤 샘플 변화에서 오는지 확인

### 5.11 Storage / Inference

- Checkpoint Size
- Inference Latency
- Inference Throughput
- Inference Peak VRAM

목적:

- 배포 비용 측정

## 6. 핵심 시각화와 필요한 입력

이 문서의 목적은 시각화를 코드로 직접 그리는 것이 아니라, 시각화가 가능하도록 필요한 raw artifact가 저장되는지 확인하는 것이다.

| 시각화 | 필요한 입력 |
| --- | --- |
| Layer × Component Heatmap | checkpoint별 layerwise cumulative update |
| Checkpoint별 Heatmap | `25/50/75/100%` 시점별 layerwise update |
| Layer Update Share Curve | checkpoint별 update share |
| Cumulative Update vs Progress | checkpoint percent + cumulative update |
| Incremental Update vs Progress | checkpoint percent + incremental update |
| Val Macro-F1 vs Update Magnitude | validation metric + layerwise update norm |
| Prediction Transition Curve | sample별 checkpoint 예측 결과 |

## 7. 현재 구현 기준 점검 항목

아래 표는 `train/pipeline.py`, `train/checkpoints.py`, `train/full_ft.py`의
산출물을 기준으로 요구사항 충족 여부를 확인하기 위한 체크리스트다.

상태 정의:

- `완료`: 현재 코드에서 바로 저장됨
- `부분`: 일부 raw 데이터는 저장되지만 연구 요구사항을 그대로 만족하지는 않음
- `미흡`: 아직 직접 저장되지 않음

| 항목 | 상태 | 점검 기준 |
| --- | --- | --- |
| `0% reference checkpoint` | 부분 | `reference` checkpoint record와 layerwise artifact는 저장되지만, model weight 파일 자체는 저장하지 않음 |
| `25/50/75/100% checkpoint` | 완료 | `checkpoint_percent` 기준 scheduled checkpoint 존재 여부 |
| `Best checkpoint 식별` | 완료 | summary에 best checkpoint kind/percent/dir 존재 여부 |
| checkpoint별 validation accuracy / macro-F1 | 완료 | checkpoint timeline에 validation metric 존재 여부 |
| checkpoint별 class-wise F1 | 완료 | `validation_class_f1` 저장 여부 |
| checkpoint별 train loss | 부분 | checkpoint record에는 저장되지만, epoch 평균과 누적 평균이 혼재될 수 있음 |
| step별 LR / gradient norm | 완료 | `logs/train_steps.jsonl` 존재 여부 |
| cumulative update `C_(l,m)^(t)` | 완료 | layerwise JSONL에 `relative_update_norm` 존재 여부 |
| incremental update `I_(l,m)^(t)` | 완료 | layerwise JSONL에 `relative_incremental_update_norm` 존재 여부 |
| update share | 완료 | `cumulative_update_share`, `incremental_update_share` 존재 여부 |
| by-layer aggregate | 완료 | layerwise summary의 `by_layer` 존재 여부 |
| by-component aggregate | 완료 | layerwise summary의 `by_component` 존재 여부 |
| checkpoint percent / elapsed time | 완료 | checkpoint record에 `checkpoint_percent`, `elapsed_seconds` 존재 여부 |
| layerwise base reference summary | 완료 | `layerwise_updates/base_reference_summary.json` 존재 여부 |
| confusion matrix | 완료 | 각 evaluation summary에 `confusion_matrix`, `label_order` 저장 여부 |
| update cosine similarity | 부분 | layerwise summary에는 평균 cosine이 저장되지만, 별도 독립 분석 파일은 없음 |
| prediction transition | 완료 | `prediction_transitions/` 아래 checkpoint 간 전이 요약/상세 파일 존재 여부 |
| train accuracy / train macro-F1 timeline | 미흡 | 현재 train loss 중심이며 train prediction metric은 없음 |
| `train_val_gap_accuracy` / `train_val_gap_macro_f1` | 미흡 | summary 필드는 있으나 현재 `None`으로 저장됨 |
| explicit overfitting start marker | 미흡 | 현재 해석용 raw metric만 저장하고 시점 판정은 하지 않음 |
| checkpoint별 best model 파일 별도 보존 | 미흡 | best는 식별만 하고 별도 복사/symlink는 만들지 않음 |

## 8. 현재 구현에서 반드시 확인해야 할 파일

Full FT 실행 후 아래 파일들이 생성되는지 확인한다.

### 8.1 run 메타데이터

- `summary.json`
- `run_metadata.json`
- `artifacts.json`

### 8.2 checkpoint / timeline

- `logs/train_steps.jsonl`
- `logs/checkpoints.jsonl`
- `analysis/checkpoint_timeline.jsonl`
- `analysis/learning_dynamics.json`
- `analysis/performance_dynamics.json`

추가 확인:

- `logs/checkpoints.jsonl` 각 row에 `checkpoint_kind`, `checkpoint_percent`, `epoch`, `step_in_epoch`, `global_step`, `elapsed_seconds`가 모두 있는지 확인한다.
- `reference` row가 있어도 해당 checkpoint 디렉터리에는 model file이 없을 수 있다. 이건 현재 구현 의도다.

### 8.3 layer-wise update

- `layerwise_updates/base_reference_summary.json`
- `layerwise_updates/<checkpoint>.jsonl`
- `layerwise_updates/<checkpoint>_summary.json`
- `analysis/update_distribution.json`
- `analysis/prediction_transitions.json`
- `prediction_transitions/<from>__to__<to>_summary.json`
- `prediction_transitions/<from>__to__<to>.jsonl`

추가 확인:

- 각 `layerwise_updates/<checkpoint>.jsonl` row에 아래 필드가 있어야 한다.
  - `update_norm`
  - `relative_update_norm`
  - `incremental_update_norm`
  - `relative_incremental_update_norm`
  - `cumulative_update_share`
  - `incremental_update_share`
- 각 `<checkpoint>_summary.json`에 `by_component`, `by_layer`, `total_cumulative_update_norm`, `total_incremental_update_norm`가 있어야 한다.

### 8.4 최종 분석 그룹

- `analysis/parameter_model_size.json`
- `analysis/memory.json`
- `analysis/training_efficiency.json`
- `analysis/optimization.json`
- `analysis/performance_generalization.json`
- `analysis/inference_deployment.json`
- `analysis/lora_configuration.json`

## 9. 미흡 구현을 판별하는 방법

다음 질문에 `아니오`가 나오면 해당 영역은 미흡하다고 판단한다.

### 9.1 Checkpoint 시간축

- `0/25/50/75/100%` 시점이 모두 존재하는가
- 각 checkpoint에 `global_step`, `epoch`, `elapsed_seconds`가 있는가
- `100%` 시점이 하나만 있는 것이 아니라 `scheduled`, `epoch_end`, `final`로 중복될 수 있음을 알고 있는가

### 9.2 성능 추적

- 각 checkpoint에 validation accuracy, macro-F1, class-wise F1이 있는가
- checkpoint 성능과 layerwise update 파일을 step 기준으로 연결할 수 있는가
- confusion matrix가 각 evaluation summary에 저장되는가

### 9.3 Layer-wise 분석

- 각 tracked component에 대해 base norm과 update norm이 저장되는가
- cumulative / incremental 상대 변화량이 모두 있는가
- by-layer / by-component share를 구할 수 있는가
- base 기준은 저장되지만 reference checkpoint에 model file까지 필요한지 별도로 판단했는가

### 9.4 시각화 가능성

- `25/50/75/100%` 시점별 heatmap 입력이 모두 존재하는가
- validation 성능과 같은 checkpoint percent 축으로 연결 가능한가

### 9.5 부족한 부분

아래 중 하나라도 필요하면 추가 구현이 요구된다.

- update cosine similarity를 checkpoint 비교 전용 독립 artifact로 더 세분화할 필요가 있는가
- train metric의 generation-based timeline 저장
- `train_val_gap_accuracy`, `train_val_gap_macro_f1` 실제 계산
- best checkpoint 전용 alias 또는 복사본 저장

## 10. 현재 문서의 사용 방식

이 문서는 다음 순서로 사용한다.

1. Full FT run 실행
2. output 디렉터리에서 checkpoint / analysis / layerwise 파일 생성 여부 확인
3. `7. 현재 구현 기준 점검 항목` 표로 충족 여부 판정
4. `9. 미흡 구현을 판별하는 방법`으로 부족한 영역 식별
5. 이후 시각화와 해석은 별도 분석 코드에서 수행

## 11. 현재 기준 결론

현재 구현은 다음까지는 지원한다.

- `0/25/50/75/100%` checkpoint 기반 timeline
- checkpoint별 validation 성능 기록
- layer / component별 cumulative update
- checkpoint 간 incremental update
- update share 기반 concentration 분석 입력

반면 아래는 아직 미흡하다.

- update direction cosine similarity의 독립 분석 artifact
- train accuracy / train macro-F1 timeline
- `train_val_gap_accuracy`, `train_val_gap_macro_f1` 계산
- best checkpoint 별도 파일 보존

즉 이 문서는 현재 구현이 `Full FT의 layer-wise / temporal adaptation 분석을 위한 핵심 raw artifact를 저장하는지`를 검증하는 기준 문서다.
