# PubMedQA 결과 검증 및 수집 가이드

이 문서는 [scripts/validate_pubmedqa_runs.py](../../scripts/validate_pubmedqa_runs.py)를 사용해 실행이 끝난 PubMedQA run들의 결과를 검증하고, 비교용 표를 수집하는 방법을 설명한다.

기준 환경은 서버의 `conda` 환경 `jw` 이다.

## 코드 검증과 결과 수집의 차이

이 문서의 결과 수집 CLI는 이미 생성된 실험 파일을 읽는다. 코드 검증은
저장소 루트에서 다음 명령으로 별도로 수행한다. Python 3.10 이상과 PyTorch,
`requirements.txt`, 개발 의존성(`pip install -e '.[dev]'`)이 필요하다.

```bash
.venv/bin/python -m compileall -q src scripts tests
.venv/bin/python -m ruff check src scripts tests
bash -n scripts/run_pubmedqa_full_study.sh
bash -n scripts/run_pubmedqa_fsdp_smoke.sh
git diff --check
.venv/bin/python -m coverage run --source=src/pubmedqa scripts/test_pubmedqa_offline.py
.venv/bin/python -m coverage report -m
```

동적 테스트는 실제 학습 자료 없이 합성 데이터·소형 CPU 모델·PEFT adapter를 사용하며,
네트워크 및 GPU 실행을 차단한다. 학습·저장·재로딩과 report reader 호환성을 확인한다.
필수 테스트의 skip도 실패로 처리한다. 실제 GPU 성능과 다중 rank collective는 검증 범위 밖이다.
`run_pubmedqa_experiments.py --dry-run`은 manifest만 생성하므로 이 테스트를 대체하지 않는다.

추가 offline 검증은 Full/LoRA/selective × single/DDP/FSDP 9개 조합의 CPU step·저장·reload,
두 fake rank의 제어 흐름, 실패 전파와 계속/중단 결정, launcher 명령 생성까지 포함한다.
launcher 검사는 Python/torchrun을 인자 기록용 stub으로 대체하며 실제 학습을 실행하지 않는다.
소형 모델은 실제 CPU 수치 계산, 분산 wrapper/collective는 fake라는 경계를 유지한다.

offline suite는 다음 실행 계약도 확인한다.

- `train/pipeline.py::run_training` 및 standalone/환경변수/study 진입점이 같은 학습 흐름을 사용한다.
- `TrainingSession`·`RunFiles`·`EvaluationSettings`·`AdapterHistory`는 각자 필요한 상태만 보관한다.
- 초기화·모델 적재·평가·writer 오류가 나도 정리/실패 전파가 유지되고 borrowed group은 파괴하지 않는다.
- 산출물 계약과 고정 CPU 수치 기준을 유지하며, 학습 모델이 해제된 뒤 best checkpoint를 적재한다.

새 구조는 실제 GPU 검증 완료를 의미하지 않는다. GPU 수렴·성능·메모리와 NCCL 동작은 별도 검증이 필요하다.

## 새 결과의 provenance 확인

- `evaluations/*_summary.json`: `loss_reduction=token_mean`. nonignored `labels[:,1:]`
  수로 가중한 validation NLL이다. 기존 batch 평균 loss나 학습 로그의 microbatch 평균과 다르다.
  loss 차이로 best checkpoint 선택이 달라질 수 있다.
- `run_metadata.json`: `validation_loss_reduction`, `training_loss_reduction`,
  `analysis_schema_version=2`를 확인한다. 이전 파일에 필드가 없으면 새 정의로 추정하지 않는다.
- Layerwise summary: reference dtype, float32 accumulation, reference/snapshot/increment 바이트 수.
  이전 float16 reference 분석과 미세한 update 수치가 달라질 수 있다.
- `distributed_runtime.json`: `per_rank` 길이와 `world_size` 일치, rank 중복/누락 없음,
  training peak만 집계했는지 확인한다. CPU는 `memory_measured=false`; numeric max 0은
  실제 GPU 메모리 사용 0이라는 뜻이 아니다.
- FSDP distributed metadata: CPU float32 evaluation, float32 master, auto-wrap layer classes,
  Gloo control timeout. CPU 평가의 latency·memory는 과거 GPU 평가와 직접 비교하지 않는다.

truncation으로 shifted target가 없는 row는 pubid·length 오류로 거절하며 sample을 조용히
버리지 않는다. 0/음수 epoch/batch/accumulation, 빈 train/validation도 오류다.
`save_every_epoch=true`는 fresh validation metric을 위해 `eval_every_epoch=true`를 요구한다.
기존 outputs/reports는 재작성하지 않았고 기존 reader의 경로·필드는 유지된다.

## 개요

이 스크립트는 두 가지 역할을 한다.

- 각 run의 필수 결과 파일이 모두 생성됐는지 검증
- 핵심 metric을 JSON/CSV로 모아서 비교표 생성

추가로 train 계열 run에 대해서는 `F1`의 `config.json` 을 기준으로 공통 환경 조건이 달라진 항목이 있는지도 확인한다.

## 검증 대상

### baseline run

다음을 확인한다.

- `summary.json`
- `outputs.jsonl`
- `run.json`
- `metrics/ACC.json`
- `metrics/Macro_F1.json`
- `metrics/Invalid_rate.json`

### Full FT / LoRA run

공통으로 다음을 확인한다.

- `summary.json`
- `config.json`
- `run_metadata.json`
- `artifacts.json`
- `logs/train_steps.jsonl`
- `logs/checkpoints.jsonl`
- `analysis/parameter_model_size.json`
- `analysis/memory.json`
- `analysis/training_efficiency.json`
- `analysis/optimization.json`
- `analysis/learning_dynamics.json`
- `analysis/performance_dynamics.json`
- `analysis/performance_generalization.json`
- `analysis/inference_deployment.json`
- `analysis/update_distribution.json`
- `analysis/prediction_transitions.json`
- `analysis/checkpoint_timeline.jsonl`
- `analysis/optimization_timeline.jsonl`

LoRA run은 추가로 다음도 확인한다.

- `analysis/lora_configuration.json`
- `analysis/lora_dynamics.json`
- `analysis/lora_singular_values.json`
- `analysis/lora_effective_rank.json`
- `analysis/lora_adapter_norms.json`
- `analysis/lora_update_direction.json`
- `analysis/module_update_share.json`
- `analysis/val_update_alignment.json`
- `analysis/lora_adapter_metrics.jsonl`

## 기본 사용 예시

```bash
PYTHONPATH=src python scripts/validate_pubmedqa_runs.py \
  --run-id exp_20260903 \
  --runs B0,F1,L1,L2,L3,L4,LL1
```

이 명령은 지정한 run들의 출력 경로를 찾아서 검증하고, 리포트를 생성한다.

## 전체 run 검증 예시

```bash
PYTHONPATH=src python scripts/validate_pubmedqa_runs.py \
  --run-id exp_20260903 \
  --runs all
```

`all` 을 주면 현재 등록된 모든 run tag를 대상으로 확인한다.

## 주요 인자

| 인자 | 설명 |
| --- | --- |
| `--run-id` | 검증할 실험 묶음 ID |
| `--runs` | 쉼표 구분 run 목록 또는 `all` |
| `--baseline-output-dir` | baseline 결과 루트 |
| `--train-output-dir` | Full FT / LoRA 결과 루트 |
| `--model-name` | 결과 경로 계산에 사용할 모델 이름 |
| `--output-file` | 검증 리포트 JSON 경로 |

## 출력 파일

기본 출력은 다음 두 파일이다.

- `outputs/pubmedqa_train/<run_id>/run_validation_report.json`
- `outputs/pubmedqa_train/<run_id>/run_validation_report.csv`

### JSON 리포트

JSON에는 다음이 포함된다.

- 전체 `run_id`
- `model_name`
- run별 경로
- run별 파일 누락 여부
- run별 `summary.json`
- run별 `config.json`
- `F1` 대비 config 차이
- 비교용 metric row

### CSV 비교표

CSV에는 다음 핵심 비교 값이 들어간다.

- `run_tag`
- `method`
- `condition`
- `accuracy`
- `macro_f1`
- `trainable_params`
- `trainable_ratio`
- `peak_train_allocated_gb`
- `total_training_time_seconds`
- `best_checkpoint_size_bytes`
- `inference_examples_per_second`
- `inference_avg_latency_seconds`
- `invalid_rate`

즉, `F1 ↔ L1`, `L1 ↔ L2`, `L3 ↔ L1 ↔ L4` 비교용 1차 테이블로 바로 쓸 수 있다.

## F1 대비 공통 설정 비교

validator는 train 계열 run에서 `F1`의 `config.json` 을 기준으로 다음 항목이 달라졌는지 확인한다.

- `model_name`
- `num_epochs`
- `train_batch_size`
- `eval_batch_size`
- `gradient_accumulation_steps`
- `learning_rate`
- `weight_decay`
- `warmup_ratio`
- `max_grad_norm`
- `max_input_tokens`
- `max_new_tokens`
- `device`
- `cpu_threads`
- `strict_parser`
- `seed`

즉, LoRA 비교에서 바뀌면 안 되는 공통 조건이 달라졌는지 빠르게 확인할 수 있다.

## 콘솔 출력 해석

정상인 run은 다음처럼 출력된다.

```text
[OK] L1 -> outputs/pubmedqa_train/exp_20260903/Qwen_Qwen3-1.7B/lora
```

누락이 있으면 다음처럼 출력된다.

```text
[MISSING] L1 -> ...
  - missing: analysis/lora_effective_rank.json
```

또한 `F1` 대비 공통 조건 차이가 있으면 다음처럼 출력된다.

```text
  - config_diff_vs_F1: ['learning_rate', 'seed']
```

이 경우는 의도적 변경인지 확인해야 한다.

## 종료 코드

- 모든 run이 정상이면 `0`
- 하나라도 필수 결과 파일이 누락되면 `1`

즉, 배치 실험 후 자동 검증 단계에 바로 연결할 수 있다.

## 권장 사용 순서

1. `run_pubmedqa_experiments.py` 로 실행
2. `validate_pubmedqa_runs.py` 로 파일 누락 여부 확인
3. `run_validation_report.csv` 로 1차 비교표 확인
4. 세부 분석은 각 run의 `analysis/*` 와 checkpoint artifact 사용

## 주의점

- validator는 artifact의 존재와 기본 비교표 수집을 담당한다.
- heatmap 생성, singular value 시각화, prediction transition 분석 자체는 하지 않는다.
- selective LoRA의 layer 선택이 적절했는지 판단하지도 않는다.
- low-data subset이 실질적으로 동일했는지는 run manifest와 central spec를 함께 봐야 한다.
