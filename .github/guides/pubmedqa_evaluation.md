# PubMedQA 평가 실행 가이드

이 문서는 `src/pubmedqa/evaluation.py`와 `scripts/run_pubmedqa_eval.py`를 사용해 PubMedQA baseline QA 평가를 실행하는 방법을 설명합니다.  
현재 기준으로 평가 대상은 base model이며, 이후 full fine-tune과 LoRA 결과도 같은 출력 형식으로 맞춰 비교할 수 있게 구성되어 있습니다.

## 개요

평가 코드는 다음 역할을 수행합니다.

- PubMedQA test JSONL 로드
- 모델별 prompt 생성
- torch 기반 batched generation 실행
- 응답 파싱
- `ACC`, `Macro F1`, `Invalid rate` 계산
- 입력 prompt와 모델 응답 전체 저장

출력 디렉터리 구조는 다음과 같습니다.

```text
outputs/
  <run_id>/
    <model_name>/
      <condition>/
        summary.json
        outputs.jsonl
        run.json
        metrics/
          ACC.json
          Macro_F1.json
          Invalid_rate.json
```

## 사전 조건

- conda 환경이 준비되어 있어야 합니다.
- `requirements.txt` 기준 패키지 설치가 끝나 있어야 합니다.
- Hugging Face 모델 접근이 필요하면 `HF_TOKEN` 환경변수를 설정합니다.
- 입력 데이터는 JSONL 형식이어야 하며, 각 row는 `question`, `context.contexts`, `final_decision` 등을 포함해야 합니다.

예시:

```bash
export HF_TOKEN=...
```

## 실행 방식 1: 단일 모델 평가

단일 모델만 평가할 때는 `scripts/run_pubmedqa_eval.py`를 사용하는 편이 간단합니다.

```bash
PYTHONPATH=src python scripts/run_pubmedqa_eval.py \
  --data-file data/processed/pqa_labeled/test.jsonl \
  --output-dir outputs/pubmedqa_eval \
  --run-id baseline_20260903 \
  --model-name Qwen/Qwen3-0.6B \
  --condition baseline \
  --hf-token "$HF_TOKEN" \
  --batch-size 8 \
  --max-new-tokens 4 \
  --torch-dtype bfloat16 \
  --expected-size 500
```

### 주요 인자

| 인자 | 설명 |
| --- | --- |
| `--data-file` | 평가할 JSONL 파일 경로 |
| `--output-dir` | 결과 저장 루트 경로 |
| `--run-id` | 실험 식별자 |
| `--model-name` | 평가할 모델 이름 |
| `--condition` | `baseline`, `full-ft`, `lora` 등 조건 이름 |
| `--hf-token` | Hugging Face 토큰 |
| `--batch-size` | generation 배치 크기 |
| `--max-new-tokens` | 최대 생성 토큰 수 |
| `--torch-dtype` | `float16`, `bfloat16`, `float32` |
| `--max-input-tokens` | 입력 길이 제한 |
| `--expected-size` | 기대 샘플 수 검증 |
| `--strict-parser` | 엄격한 파서 사용 여부 |

## 실행 방식 2: 다중 모델 일괄 평가

`src/pubmedqa/evaluation.py`는 환경변수를 통해 여러 모델을 순차 평가하는 entrypoint도 제공합니다.

```bash
export HF_TOKEN=...
export PUBMEDQA_TEST_PATH=data/processed/pqa_labeled/test.jsonl
export PUBMEDQA_OUTPUT_DIR=outputs/pubmedqa_eval
export PUBMEDQA_RUN_ID=baseline_20260903
export PUBMEDQA_CONDITION=baseline
export PUBMEDQA_EXPECTED_TEST_SIZE=500
export PUBMEDQA_MODELS="Qwen/Qwen3-0.6B,meta-llama/Llama-3.2-1B-Instruct,Qwen/Qwen3-1.7B"
export PUBMEDQA_BATCH_SIZE=8
export PUBMEDQA_MAX_NEW_TOKENS=4
export PUBMEDQA_DTYPE=bf16

PYTHONPATH=src python -m pubmedqa.evaluation
```

이 방식은 지정한 모델들을 하나씩 순차 실행합니다.  
단일 GPU 환경에서는 동시에 여러 모델을 올리지 않기 때문에, timing과 memory 비교가 더 일관됩니다.

## 사용 가능한 환경변수

| 환경변수 | 설명 | 기본값 |
| --- | --- | --- |
| `HF_TOKEN` | Hugging Face 토큰 | 없음 |
| `PUBMEDQA_TEST_PATH` | 테스트 JSONL 경로 | 필수 |
| `PUBMEDQA_OUTPUT_DIR` | 출력 경로 | `outputs/pubmedqa_eval` |
| `PUBMEDQA_RUN_ID` | 실험 식별자 | 현재 시각 기반 자동 생성 |
| `PUBMEDQA_CONDITION` | 실험 조건 | `baseline` |
| `PUBMEDQA_EXPECTED_TEST_SIZE` | 샘플 수 검증 | `500` |
| `PUBMEDQA_MODELS` | 쉼표 구분 모델 목록 | 기본 모델 목록 |
| `PUBMEDQA_BATCH_SIZE` | 배치 크기 | `8` |
| `PUBMEDQA_MAX_NEW_TOKENS` | 최대 생성 토큰 수 | `4` |
| `PUBMEDQA_MAX_INPUT_TOKENS` | 입력 길이 제한 | 없음 |
| `PUBMEDQA_DTYPE` | `bf16`, `fp16`, `fp32` | `bf16` |
| `PUBMEDQA_USE_CUDA` | CUDA 사용 여부 | `true` |
| `PUBMEDQA_DEVICE_MAP` | Hugging Face device map | 없음 |
| `PUBMEDQA_ATTN_IMPLEMENTATION` | attention 구현 설정 | `sdpa` |
| `PUBMEDQA_TRUST_REMOTE_CODE` | remote code 허용 | `false` |
| `PUBMEDQA_CPU_THREADS` | CPU thread 수 | `os.cpu_count()-2` |
| `PUBMEDQA_STRICT_PARSER` | strict parser 사용 | `false` |

## 출력 파일 설명

### `summary.json`

실험 전체 요약입니다.

- `run_id`
- `model_name`
- `condition`
- `title`
- `start_time`
- `end_time`
- `elapsed_seconds`
- `num_examples`
- `num_parsed`
- `accuracy`
- `macro_f1`
- `invalid_rate`

### `outputs.jsonl`

샘플 단위 결과입니다. 각 row에는 다음이 포함됩니다.

- 입력 prompt
- 원본 모델 응답
- 파싱된 라벨
- 파서 사용 경로
- `run_id`, `model_name`, `condition`, `title`
- `start_time`, `end_time`

즉, baseline 분석이나 오류 사례 점검 시 이 파일을 그대로 사용하면 됩니다.

### `metrics/*.json`

지표별 파일이 분리되어 저장됩니다.

- `metrics/ACC.json`
- `metrics/Macro_F1.json`
- `metrics/Invalid_rate.json`

후속 자동 집계 스크립트에서 metric 단위로 직접 읽기 쉽게 하기 위한 구조입니다.

## 현재 프롬프트 동작과의 관계

평가 코드는 `src/pubmedqa/prompt_builder.py`를 사용합니다. 현재 프롬프트는 `question`과
`context`만 모델에 제공하고, `yes`, `no`, `maybe` 중 하나를 출력하도록 지시합니다. 이는
decoding vocabulary를 제한하는 hard constraint가 아닙니다.

생성된 응답은 parser를 통해 최종 prediction label로 변환하며, 유효한 label을 추출하지 못한
응답은 invalid prediction으로 처리합니다. Qwen3 계열에서는 chat template 렌더링 시
`enable_thinking=False`를 우선 적용해 direct-answer-only 형식을 유도합니다.

## 권장 실행 순서

처음에는 전체 500개 test를 바로 돌리기보다, 소규모 샘플로 dry run을 먼저 보는 편이 낫습니다.

예시:

1. `pqa_labeled/test.jsonl`에서 일부 샘플만 분리
2. `Qwen/Qwen3-0.6B` 단일 모델로 실행
3. `outputs.jsonl`에서 parser invalid 여부 확인
4. 출력 구조가 기대와 맞는지 확인
5. 전체 500개 test 실행

## 주의 사항

- `google/gemma-3-4b-it`는 `transformers` 버전이 충분히 최신이어야 합니다.
- Hugging Face 토큰 없이도 공개 모델 접근은 가능할 수 있지만, rate limit과 다운로드 속도 문제가 생길 수 있습니다.
- `strict parser`를 켜면 verbose 응답 허용 폭이 줄어들기 때문에 baseline invalid rate가 올라갈 수 있습니다.
- `device_map`을 사용할 경우 GPU 메모리 사용량 측정 방식이 단일 device 배치와 다르게 보일 수 있습니다.
