# Business Entity Resolution pipeline

Status: Phase 1 done (data preparation, EDA, scorer, all-empty baseline); shared normalisation built (norm-v3). Blocking and matching model not built yet.

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
4. Normalisation of every source file (state aliases learned from train pairs excluding validation S1, city
   vocabulary from train + test records, quality report). The full data runs on Kaggle (~10 GB RAM):
   `python src/run_normalise.py --data-dir ../../data_parquet --split ../../outputs/eda/g25_split.parquet --out-dir ../../artifacts/normalised`
   Outputs `normalised/{split}_source{k}.parquet` (raw + normalised columns), `state_aliases_norm-v3.json`,
   `city_vocab_norm-v3.parquet` and `normalisation_report.md`. Rules and lists: `docs/normalisation.md`.
5. All-empty baseline: scores the validation split and writes an all-empty test submission:
   `python src/baseline_empty.py --data-dir ../../data_parquet --split ../../outputs/eda/g25_split.parquet --out-dir ../../output`

## Modules
- `src/metric.py`: macro F0.5 exactly as the challenge defines it (tests in `tests/test_metric.py`, run with `python -m pytest tests`).
- `src/submission.py`: writes `matching_results.tsv` and `candidate_pairs.tsv` in the required format.
- `src/normalise.py`: shared name/address normalisation (`normalise_records()`, `Normaliser.names()`,
  `Normaliser.addresses()`), `learn_state_aliases()` (train pairs only) and `Normaliser.load()` for the saved
  aliases. Version string `NORM_VERSION`. Tests in `tests/test_normalise.py`.
- `src/eval_normalise.py`: learns state aliases and reports same-city/same-state agreement of true pairs:
  `python src/eval_normalise.py --data-dir ../../data_sample --aliases ../../artifacts/state_aliases.json`
