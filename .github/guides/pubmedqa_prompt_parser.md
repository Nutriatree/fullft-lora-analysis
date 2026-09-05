# PubMedQA 프롬프트 및 파서 가이드

이 저장소는 baseline, full fine-tune, LoRA를 같은 QA 정의 위에서 비교할 수 있도록 프롬프트 계층과 파서 계층을 분리해 두었습니다.  
핵심 목표는 모델별 tokenizer 템플릿은 존중하되, 태스크 정의 자체는 바꾸지 않는 것입니다.

## 관련 모듈

- `src/pubmedqa/prompt_builder.py`: PubMedQA 입력 프롬프트 생성
- `src/pubmedqa/instruction.py`: system prompt, user prompt, assistant answer 템플릿 정의
- `src/pubmedqa/labels.py`: `yes`, `no`, `maybe` 라벨 정규화 및 검증
- `src/pubmedqa/answer_parser.py`: 모델 출력에서 최종 라벨 파싱
- `scripts/pubmedqa_prompt.py`: 프롬프트 및 파서 동작을 빠르게 점검하는 CLI

환경에서는 다음 둘 중 하나를 사용하면 됩니다.

```bash
pip install -e .
```

또는

```bash
PYTHONPATH=src
```

## 프롬프트 설계 원칙

현재 QA 프롬프트는 매우 단순하게 유지되어 있습니다.

- 모델에게 제공되는 입력은 `question`과 `context.contexts`뿐입니다.
- `pubid`, `labels`, `meshes`, `long_answer`, `gold label`은 추론 프롬프트에 넣지 않습니다.
- 출력은 반드시 `yes`, `no`, `maybe` 중 하나여야 합니다.
- 영어 이외의 응답이나 추론 체인을 유도하지 않도록 직접 답변만 요구합니다.

즉, 모델은 아래 형태의 입력을 받습니다.

- system message: PubMedQA QA 역할 정의
- user message: 질문, abstract context, 출력 제약

그리고 최종 출력은 아래처럼 한 단어만 기대합니다.

```text
yes
```

## 추론 프롬프트 사용

추론 시에는 `include_answer=False` 상태로 prompt를 만듭니다.

```python
from pubmedqa.prompt_builder import build_tokenizer_prompt, example_from_record
from pubmedqa.answer_parser import parse_pubmedqa_answer

example = example_from_record(record)
prompt = build_tokenizer_prompt(tokenizer, example, include_answer=False)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=16, do_sample=False)
decoded = tokenizer.decode(
    outputs[0][inputs["input_ids"].shape[-1]:],
    skip_special_tokens=True,
)
prediction = parse_pubmedqa_answer(decoded)
```

`build_tokenizer_prompt()`는 tokenizer에 chat template이 있으면 그것을 사용하고, 없으면 plain prompt fallback을 사용합니다.

또한 Qwen3 계열에서는 `enable_thinking=False`를 우선 적용해서 direct-answer-only 형식을 유지합니다.

## 학습용 타깃 구성

full fine-tune과 LoRA에서는 gold answer를 assistant message로 붙여서 사용합니다.

```python
from pubmedqa.prompt_builder import build_messages, example_from_record

example = example_from_record(record)
messages = build_messages(example, include_answer=True)
```

이때 assistant target은 아래처럼 한 단어입니다.

```text
yes
```

SFT에서는 일반적으로 prompt 부분은 loss에서 제외하고, assistant answer 구간만 loss 대상으로 두는 방식이 적절합니다.

## 파서 동작

`parse_pubmedqa_answer(text)`는 `ParseResult`를 반환합니다.

- `label`: 파싱된 최종 라벨 또는 `None`
- `raw_text`: 원본 출력
- `normalized_text`: 정규화된 출력
- `matched_text`: 라벨 추출에 사용된 부분 문자열
- `method`: 어떤 경로로 파싱했는지 표시
- `error`: 실패 사유

기본적으로는 label-only 출력을 가장 우선합니다.

```text
yes
```

다만 baseline dry run 단계에서 모델 출력이 완전히 정돈되지 않을 수 있으므로, 아래 같은 변형도 받아들입니다.

```text
Answer: no
{"answer": "maybe"}
yes, based on the abstract.
```

## strict 모드

더 엄격한 평가가 필요하면 strict 파서를 사용할 수 있습니다.

```python
parse_pubmedqa_answer(text, strict=True)
```

strict 모드에서는 JSON, `Answer: ...` 형식은 허용하지만, 일반 문장 fallback은 첫 토큰이 `yes|no|maybe`일 때만 인정합니다.

## 빠른 점검 CLI

### chat 형식 프롬프트 출력

```bash
PYTHONPATH=src python scripts/pubmedqa_prompt.py \
  --input-file data/processed/pqa_labeled/test.jsonl \
  --index 0 \
  --format chat
```

### plain fallback 프롬프트 출력

```bash
PYTHONPATH=src python scripts/pubmedqa_prompt.py --format plain
```

### 파서 동작 확인

```bash
PYTHONPATH=src python scripts/pubmedqa_prompt.py --parse "maybe"
```

## 현재 실험 대상 모델

현재 baseline 기준으로 고려 중인 모델은 다음과 같습니다.

- `Qwen/Qwen3-0.6B`
- `meta-llama/Llama-3.2-1B-Instruct`
- `Qwen/Qwen3-1.7B`
- `Qwen/Qwen3-4B`
- `google/gemma-3-4b-it`

이 문서의 프롬프트와 파서는 위 모델들에서 공통으로 재사용하는 기준 계층입니다.
