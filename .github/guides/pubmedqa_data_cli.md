# PubMedQA 데이터 CLI 가이드

이 저장소에서는 `scripts/pubmedqa_data.py` 하나로 PubMedQA 데이터 준비, 검증, 소스 비교를 처리합니다.  
생성된 데이터 파일은 Git에서 추적하지 않기 때문에, 동일한 구조를 다시 만들 수 있는 재현 가능한 진입점이 필요합니다.

## 목적

이 스크립트는 다음 작업을 담당합니다.

- Hugging Face `qiaojin/PubMedQA`의 `pqa_artificial` subset을 기준으로 PQA-A 생성
- 공식 GitHub `pubmedqa/pubmedqa` 데이터를 기준으로 PQA-L 생성
- 생성된 split 구조와 메타데이터 검증
- Hugging Face와 GitHub의 PQA-L 구성이 실제로 일치하는지 비교

특히 PQA-A와 PQA-L이 혼합되지 않도록, 출력 경로와 각 row 메타데이터에 `dataset_id`와 `dataset_collection`을 명시적으로 기록합니다.

## 데이터 구성

현재 기준으로 생성되는 구조는 다음과 같습니다.

```text
data/processed/
  metadata.json
  pqa_artificial/
    train.json
    train.jsonl
    validation.json
    validation.jsonl
  pqa_labeled/
    cv.json
    cv.jsonl
    test.json
    test.jsonl
    folds/
      fold_0/
        train.json
        train.jsonl
        validation.json
        validation.jsonl
      ...
      fold_9/
        train.json
        train.jsonl
        validation.json
        validation.jsonl
```

## 기본 split 정책

- `pqa_artificial/train`: 200,000개
- `pqa_artificial/validation`: 11,269개
- `pqa_labeled/cv`: 500개
- `pqa_labeled/test`: 500개
- `pqa_labeled/folds/fold_*/train`: 450개
- `pqa_labeled/folds/fold_*/validation`: 50개

즉, PQA-A는 대규모 학습용, PQA-L은 CV와 최종 테스트용으로 분리됩니다.

## 주요 명령어

### 1. 데이터 생성

```bash
python scripts/pubmedqa_data.py prepare --clean-output-dir
```

이 명령은 `data/processed/` 아래에 PQA-A, PQA-L, fold 분할, 메타데이터를 모두 생성합니다.

### 2. 생성된 split 설명 및 검증

```bash
python scripts/pubmedqa_data.py describe --strict
```

이 명령은 생성된 JSONL 파일을 스캔해서 다음 정보를 출력합니다.

- 샘플 수
- `dataset_id`
- `split`
- label 분포

`--strict`를 주면 경로와 메타데이터가 불일치하거나, 하나의 파일에 다른 데이터셋이 섞여 있을 때 비정상 종료합니다.

### 3. Hugging Face와 GitHub 소스 비교

```bash
python scripts/pubmedqa_data.py verify-sources
```

이 명령은 Hugging Face `pqa_labeled`와 GitHub의 공식 `ori_pqal.json`, `test_ground_truth.json`을 비교합니다.

비교 대상은 다음과 같습니다.

- PMID 집합
- 질문 텍스트
- context 문장
- MeSH terms
- reasoning prediction 관련 필드
- long answer
- 최종 정답 라벨

## 자주 쓰는 옵션

| 옵션 | 기본값 | 설명 |
| --- | --- | --- |
| `--output-dir` | `data/processed` | 생성 결과 저장 경로 |
| `--seed` | `0` | 공식 split 재현용 seed |
| `--clean-output-dir` | 비활성 | 생성 전 output 디렉터리 삭제 |
| `--pqal-source` | `github` | PQA-L 소스를 `github` 또는 `hf` 중 선택 |
| `--pqal-test-ground-truth` | 없음 | 로컬 `test_ground_truth.json` 경로 지정 |
| `--skip-github-test-download` | 비활성 | 공식 test label이 없을 때 seed 기반 재구성 허용 |

GitHub raw 접근이 불안정하면 아래처럼 로컬 파일을 직접 줄 수 있습니다.

```bash
python scripts/pubmedqa_data.py prepare \
  --pqal-test-ground-truth path/to/test_ground_truth.json
```

## 생성 row 스키마

생성된 각 row는 아래와 같은 정규화된 구조를 따릅니다.

```json
{
  "pubid": "21645374",
  "dataset_id": "pqa_labeled",
  "dataset_collection": "PQA-L",
  "source": "https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/ori_pqal.json",
  "split": "test",
  "split_role": "test",
  "question": "...",
  "context": {
    "contexts": ["..."],
    "labels": ["BACKGROUND", "RESULTS"],
    "meshes": ["..."]
  },
  "long_answer": "...",
  "final_decision": "yes"
}
```

PQA-L row에는 공식 데이터 기준의 추가 필드가 포함될 수 있습니다. 예를 들어 `year`, `context.reasoning_required_pred`, `context.reasoning_free_pred`, `fold` 등이 있습니다.

## 재현성 메모

- PQA-A는 공식 방식대로 seed `0`으로 shuffle 후 앞 200,000개를 train, 나머지 11,269개를 validation으로 사용합니다.
- PQA-L은 공식 방식대로 500개 CV, 500개 Test로 나눈 뒤, CV 500개에 대해 10-fold를 구성합니다.
- `metadata.json`에는 소스 URL, split 경로, 샘플 수, label 분포, fold 정보가 기록됩니다.
- 공식 GitHub test split을 정확히 재현할 수 있으면 `metadata.json`에 그 상태가 함께 기록됩니다.
