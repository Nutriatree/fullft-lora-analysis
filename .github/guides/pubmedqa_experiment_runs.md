# PubMedQA 전체 Run 실행 가이드

이 문서는 [scripts/run_pubmedqa_experiments.py](../../scripts/run_pubmedqa_experiments.py)를 사용해 PubMedQA의 baseline, Full Fine-Tuning, LoRA run들을 중앙 정의 기반으로 실행하는 방법을 설명한다.

실험 정의의 단일 기준은 [src/pubmedqa/config/experiments.py](../../src/pubmedqa/config/experiments.py)이다.
`train/study.py::build_training_config`는 설정만 반환하고,
`train/study.py`는 `train/pipeline.py::run_training`을 직접 호출한다.
`build_runner`와 기존 Trainer는 공개 import 호환용이며 핵심 학습 경로에서 사용하지 않는다.
study는 통신 그룹을 소유하고 각 `TrainingSession`에 빌려준다. 학습 성공·실패 모두
세션을 정리하되 빌린 그룹은 study 종료 시에만 정리한다.
`--dry-run`은 manifest/표만 만들고 모델 적재나 분산 초기화를 실행하지 않는다.

기준 환경은 서버의 `conda` 환경 `jw` 이다. 이 문서의 실행 예시는 모두 이 환경 안에서 바로 실행한다고 가정한다.

## 개요

전체 run 실행 스크립트는 다음 목적을 가진다.

- `B0`, `F1`, `L1`, `L2`, `L3`, `L4`, `LL1`, `LL2`, `F2`, `L5` 같은 run 정의를 한 곳에서 관리
- 공통 학습 조건을 한 번만 지정하고 각 run이 참조하도록 유지
- 실행 전 `manifest`를 생성해서 실제 사용된 조건을 기록
- `--dry-run` 으로 실행 계획만 확인
- `run_id` 는 기본적으로 자동 생성된다. 필요할 때만 `--run-id` 로 덮어쓴다.

즉, low-data subset을 별도 코드 분기로 고정하기보다, 중앙 run 정의와 공통 인자를 하나의 기준으로 참조하게 만드는 구조다.

## 현재 기본 run 정의

현재 등록된 run은 다음과 같다.

| Run | Method | Condition | 목적 |
| --- | --- | --- | --- |
| `B0` | Baseline Eval | `baseline` | 학습 전 기준점 |
| `F1` | Full FT | `full-ft` | Full FT 대표 |
| `L1` | LoRA | `lora` | LoRA 대표 |
| `L2` | LoRA | `lora-qkvo` | target module 비교 |
| `L3` | LoRA | `lora-r4` | rank 4 비교 |
| `L4` | LoRA | `lora-r16` | rank 16 비교 |
| `LL1` | LoRA | `selective-lora-high-update` | high-update selective LoRA |
| `LL2` | LoRA | `selective-lora-low-update` | low-update control |
| `F2` | Full FT | `full-ft-low-data` | low-data Full FT |
| `L5` | LoRA | `lora-low-data` | low-data LoRA |

실제 정의는 [src/pubmedqa/config/experiments.py](../../src/pubmedqa/config/experiments.py)의 `RUN_SPECS`를 기준으로 한다.

## 기본 데이터 경로

중앙 정의의 기본 경로는 다음과 같다.

- train: `data/processed/pqa_artificial/train.jsonl`
- validation: `data/processed/pqa_artificial/validation.jsonl`
- test: `data/processed/pqa_labeled/test.jsonl`
- baseline eval: `data/processed/pqa_labeled/test.jsonl`

즉 현재 기본 구성은

- 학습: `PQA-A`
- 평가: `PQA-L test`

형태다.

필요하면 실행 시 CLI 인자로 덮어쓸 수 있다.

## 실행 전 run 목록 확인

```bash
PYTHONPATH=src python scripts/run_pubmedqa_experiments.py --list-runs
```

이 명령은 등록된 run tag와 method, condition, target module, rank, layer scope를 출력한다.

## dry-run

실제 학습 전에 manifest와 run 계획만 확인하려면 `--dry-run` 을 사용한다.

이 옵션은 optimizer나 모델 평가를 실행하지 않는다. 코드 변경 후에는 별도로
`python scripts/test_pubmedqa_offline.py`를 실행해 소형 CPU 모델의 실제 학습·저장·재로딩을 검증한다.
로컬 dry-run의 `--train-output-dir`과 `--baseline-output-dir`은 임시 디렉터리로 지정해
기존 실험 결과와 분리한다. 실행 환경 준비와 정적 검사 명령은 README의 Validation 절을 참고한다.

```bash
PYTHONPATH=src python scripts/run_pubmedqa_experiments.py \
  --runs B0,F1,L1,L2,L3,L4,LL1,F2,L5 \
  --dry-run
```

이때 생성되는 파일:

- `outputs/pubmedqa_train/<run_id>/run_manifest.json`

manifest에는 다음이 저장된다.

- 요청한 run 목록
- 공통 데이터 경로
- 공통 학습 기본값
- 각 run의 개별 spec

## 전체 실행 예시

```bash
PYTHONPATH=src python scripts/run_pubmedqa_experiments.py \
  --runs B0,F1,L1,L2,L3,L4,LL1 \
  --train-file data/processed/pqa_artificial/train.jsonl \
  --validation-file data/processed/pqa_artificial/validation.jsonl \
  --test-file data/processed/pqa_labeled/test.jsonl \
  --baseline-eval-file data/processed/pqa_labeled/test.jsonl \
  --model-name Qwen/Qwen3-1.7B \
  --num-epochs 3 \
  --train-batch-size 2 \
  --eval-batch-size 4 \
  --gradient-accumulation-steps 8 \
  --learning-rate 2e-5 \
  --weight-decay 0.01 \
  --warmup-ratio 0.03 \
  --max-grad-norm 1.0 \
  --dtype bf16 \
  --checkpoint-percents 25,50,75,100
```

이 스크립트는 run을 순차 실행한다.

- `B0`: baseline evaluation
- `F1`: Full FT
- `L1/L2/L3/L4/LL1`: LoRA

## 2x3090 FSDP 실행

전체 실행 전에 [FSDP smoke test 및 문제 해결 가이드](pubmedqa_fsdp_troubleshooting.md)의 소규모 검증을 먼저 수행하는 것을 권장한다.

`--distributed-mode fsdp` 는 DDP처럼 모델을 각 GPU에 복제하지 않는다. PyTorch FSDP `FULL_SHARD` 를 사용해 parameter, gradient, Adam optimizer state를 두 GPU에 분산한다. 현재 기본 모델인 `Qwen/Qwen3-1.7B`를 24GB RTX 3090 두 장에서 Full FT하기 위한 모드다.

실행은 반드시 `torchrun` 으로 한다.

```bash
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=src \
  torchrun --standalone --nproc_per_node=2 scripts/run_pubmedqa_experiments.py \
  --runs B0,F1,L1,L2,L3,L4 \
  --distributed-mode fsdp \
  --device cuda \
  --dtype bf16 \
  --gradient-checkpointing \
  --train-batch-size 1 \
  --eval-batch-size 2 \
  --gradient-accumulation-steps 16 \
  --no-save-optimizer-state
```

각 `train-batch-size` 는 GPU당 micro-batch다. 위 예시의 유효 global batch는 `1 x 2 GPU x 16 accumulation = 32` 이다. 이전 단일 GPU 기준과 유효 batch를 맞추려면 `world_size` 변화까지 포함해 accumulation을 조정해야 하며, 실제 값은 `run_manifest.json` 에 남는다.

FSDP 모드에서 checkpoint full-state 수집에는 모든 rank가 참여하고 rank 0만 파일을 쓴다.
초기 모델은 float32 CPU에 로드한 뒤 Transformer block별로 GPU에 옮겨 sharding한다.
validation/test는 rank 0 CPU의 독립 float32/eager 모델로 실행하며 다른 rank는 전용 Gloo
control group에서 결과를 기다린다. 학습 계산 dtype은 `--dtype`의 mixed precision이다.
CPU full-model RAM은 각 rank에 필요하다. Layer-wise 분석도 모든 rank가 unshard에
참여하고 rank 0만 CPU full values를 분석한다. 기존 artifact 디렉터리 구조는 유지한다.

`distributed_runtime.json` 에는 각 GPU의 idle, model-loaded, peak allocated/reserved VRAM과 두 GPU 중 최대 peak 값이 저장된다. `summary.json` 의 memory 값은 rank 0 값이므로, 2-GPU memory 비교에는 이 artifact를 사용한다.

`--no-save-optimizer-state`를 권장한다. 현재 `optimizer.pt`는 rank 0 로컬 optimizer 상태이며
FSDP full optimizer state 수집·정확한 resume 기능은 제공하지 않는다.

DDP도 같은 저장/오류 제어 흐름을 사용하지만 모델은 GPU별로 복제한다. study와 standalone
CLI 모두 `PUBMEDQA_CONTROL_TIMEOUT_SECONDS`(기본 86400초)를 평가·저장 제어 통신에 사용한다.
이는 NCCL 학습 timeout을 늘리는 옵션이 아니다. `--continue-on-error`는 모든 rank가
확인한 로컬 I/O/검증 오류에만 적용하며 collective 실패나 rank crash에서는 study를 중단한다.

중간에 실패가 나면 기본적으로 거기서 중단한다.

## 실패 무시하고 계속 진행

```bash
PYTHONPATH=src python scripts/run_pubmedqa_experiments.py \
  --runs B0,F1,L1,L2,L3,L4,LL1 \
  --continue-on-error
```

실패한 run은 콘솔에 남기고, 다음 run으로 넘어간다.

## low-data run

`F2`, `L5`는 별도 하드코딩된 subset 파일을 쓰지 않는다.  
중앙 run 정의의 `max_train_examples` 값을 참조한다.

현재 기본값은 [src/pubmedqa/config/experiments.py](../../src/pubmedqa/config/experiments.py)의

- `DEFAULT_LOW_DATA_MAX_TRAIN_EXAMPLES = 2048`

이다.

즉 현재 구조에서는

- `F2`
- `L5`

가 같은 기준 값을 공유하므로, 별도 subset 고정 파일이 없어도 run 정의 기준으로는 일관성이 유지된다.

## 주요 인자

| 인자 | 설명 |
| --- | --- |
| `--run-id` | 전체 실험 묶음 ID |
| `--runs` | 쉼표 구분 run tag 목록 |
| `--list-runs` | 등록된 run 목록만 출력 |
| `--dry-run` | manifest만 생성하고 종료 |
| `--continue-on-error` | 실패해도 다음 run 계속 |
| `--train-file` | train JSONL 경로 |
| `--validation-file` | validation JSONL 경로 |
| `--test-file` | test JSONL 경로 |
| `--baseline-eval-file` | baseline용 평가 파일 |
| `--model-name` | 공통 base model |
| `--num-epochs` | 공통 epoch |
| `--train-batch-size` | 공통 학습 배치 |
| `--eval-batch-size` | 공통 평가 배치 |
| `--gradient-accumulation-steps` | 공통 accumulation |
| `--learning-rate` | 공통 learning rate |
| `--weight-decay` | 공통 weight decay |
| `--warmup-ratio` | 공통 warmup 비율 |
| `--max-grad-norm` | 공통 grad clip |
| `--max-input-tokens` | 공통 입력 길이 제한 |
| `--max-new-tokens` | 공통 생성 길이 제한 |
| `--device` | 공통 device |
| `--distributed-mode` | `single`, `ddp`, `fsdp` |
| `--fsdp-cpu-offload` | FSDP parameter CPU offload. PCIe 환경에서는 기본값 `false` 권장 |
| `--dtype` | 공통 dtype |
| `--checkpoint-percents` | checkpoint 저장 비율 |
| `--hf-token` | Hugging Face 토큰 |

## 출력 구조

baseline과 train 계열 출력 루트가 다르다.

### baseline

```text
outputs/pubmedqa_eval/
  <run_id>/
    <model_name>/
      baseline/
```

### Full FT / LoRA

```text
outputs/pubmedqa_train/
  <run_id>/
    <model_name>/
      <condition>/
```

추가로 manifest는 아래에 저장된다.

```text
outputs/pubmedqa_train/
  <run_id>/
    run_manifest.json
```

## selective layer run 주의점

`LL1`, `LL2`는 layer-wise 분석 결과가 나온 뒤에만 실행한다. layer 목록 없이 selective run을 요청하면 코드가 즉시 실패하므로, 실수로 all-layer LoRA가 실행되지 않는다.

분석 결과를 실행 시점에 주입하려면 다음 형식을 사용한다.

```bash
PYTHONPATH=src python scripts/run_pubmedqa_experiments.py \
  --run-id selective_20260903 \
  --runs LL1,LL2 \
  --target-layer-override LL1=20,21,22,23 \
  --target-layer-override LL2=0,1,2,3
```

각 override는 `run_manifest.json` 의 `target_layer_overrides` 에 저장된다. 따라서 high-update layer와 low-update control layer가 어떤 기준으로 선택됐는지는 analysis 결과와 함께 별도로 보관해야 한다.

## 권장 사용 순서

1. `--list-runs` 로 등록 상태 확인
2. `--dry-run` 으로 manifest 생성
3. `B0,F1,L1` 소규모 우선 실행
4. 이후 `L2,L3,L4`
5. selective layer가 정해진 뒤 `LL1`, 필요 시 `LL2`
6. low-data가 필요하면 `F2,L5`
