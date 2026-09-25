# Business Entity Resolution pipeline

Status: Phase 1 (data preparation, EDA, scorer, all-empty baseline). Matching model not built yet.

## Setup
```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
```
Run all commands below from this folder (`code/business_entity_resolution/`), with the organiser data in
`../../dataset/` (the TSVs).

## Steps
1. Convert the TSVs to Parquet (all columns as strings):
   `python src/convert_to_parquet.py --tsv-dir ../../dataset --out-dir ../../data_parquet`
2. Optional, for local smoke tests: a 1% structure-preserving sample:
   `python src/make_sample.py --in-dir ../../data_parquet --out-dir ../../data_sample`
3. EDA and the S1-grouped train/validation split (`g25_split.parquet`). The full data needs ~20 GB RAM (run on Kaggle):
   `python src/eda.py --data-dir ../../data_parquet --out-dir ../../outputs/eda`
4. All-empty baseline: scores the validation split and writes an all-empty test submission:
   `python src/baseline_empty.py --data-dir ../../data_parquet --split ../../outputs/eda/g25_split.parquet --out-dir ../../output`

## Modules
- `src/metric.py`: macro F0.5 exactly as the challenge defines it (tests in `tests/test_metric.py`, run with `python -m pytest tests`).
- `src/submission.py`: writes `matching_results.tsv` and `candidate_pairs.tsv` in the required format.
