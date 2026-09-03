# PubMedQA-Fine-Tune

Compare baseline prompting, full fine-tuning, and PEFT/LoRA on PubMedQA.

See [docs/pubmedqa_data_cli.md](docs/pubmedqa_data_cli.md) for the data CLI
details.
See [docs/pubmedqa_prompt_parser.md](docs/pubmedqa_prompt_parser.md) for the
shared QA prompt and parser contract.

```bash
pip install -e .
```

## Dataset Splits

The project uses `qiaojin/PubMedQA` from Hugging Face and follows the official
PubMedQA split policy from https://github.com/pubmedqa/pubmedqa:

- PQA-A (`pqa_artificial`): 200,000 train examples + 11,269 validation examples.
- PQA-L (`pqa_labeled`): 500 CV examples + 500 held-out test examples.
- PQA-L CV: 10 folds, each with 450 train examples + 50 validation examples.

The PQA-L test set is fixed with the official PubMedQA
`data/test_ground_truth.json` PMIDs. PQA-A and PQA-L CV folds use the official
seed `0` split behavior.

## Prepare Data

```bash
python scripts/pubmedqa_data.py prepare
```

Outputs are written to `data/processed/` as both JSON and JSONL:

- `pqa_artificial/train.jsonl`
- `pqa_artificial/validation.jsonl`
- `pqa_labeled/cv.jsonl`
- `pqa_labeled/test.jsonl`
- `pqa_labeled/folds/fold_*/train.jsonl`
- `pqa_labeled/folds/fold_*/validation.jsonl`
- `metadata.json`

Each row includes explicit dataset metadata so experiment code can distinguish
PQA-A from PQA-L even after loading or concatenating files:

- `dataset_id`: `pqa_artificial` or `pqa_labeled`
- `dataset_collection`: `PQA-A` or `PQA-L`
- `split`: canonical split name
- `split_role`: `train`, `validation`, `cv`, or `test`
- `source`: source URL used to create the row

If GitHub access is unavailable, download
`https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/test_ground_truth.json`
manually and pass:

```bash
python scripts/pubmedqa_data.py prepare --pqal-test-ground-truth path/to/test_ground_truth.json
```

Generated split files are intentionally ignored by Git. To recreate the same
dataset structure after cloning:

```bash
python scripts/pubmedqa_data.py prepare --clean-output-dir
python scripts/pubmedqa_data.py describe --strict
```

## Verify Sources

To confirm that Hugging Face `pqa_labeled` matches the official GitHub
`ori_pqal.json` and `test_ground_truth.json`:

```bash
python scripts/pubmedqa_data.py verify-sources
```

To inspect generated split files and catch accidental PQA-A/PQA-L mixing:

```bash
python scripts/pubmedqa_data.py describe --strict
```
