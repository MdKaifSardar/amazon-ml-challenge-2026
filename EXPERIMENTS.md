# Experiments

One row per experiment. Val F0.5 is macro F0.5 on held-out S1 entities (validation split: `role == "val"` in
`outputs/eda/g25_split.parquet`, 99,994 S1), full pipeline end to end.

| Date/time | Change | Blocking recall | Val F0.5 overall | Val F0.5 US / India | Singleton / non-singleton | Kept? |
|---|---|---|---|---|---|---|
| 2026-09-25 13:05 | All-empty baseline (`src/baseline_empty.py`) | n/a (no candidates) | 0.0559 | 0.0562 / 0.0554 | 1.0 / 0.0 | Reference |

## Run log

- **2026-09-25 12:41–12:50, Kaggle CPU job `phase0-eda` v1** (`src/eda.py` via `jobs/build_notebook.py`).
  Full train and test data, 24.2M records. Runtime 403 s, peak RAM 17.9 GB. Outputs in `outputs/eda/`, summary in
  `docs/eda_summary.md`. Created the S1-grouped split `g25_split.parquet`: 5 folds stratified by country ×
  match-count bucket; val = 99,994 S1 (fold 0), train = 299,728 S1 (folds 1–4), rest unused.
- **2026-09-25, scorer unit tests** (`tests/test_metric.py`): 11/11 passed. Covers the PS example (0.714), singleton
  empty = 1.0, singleton with prediction = 0.0, empty prediction with true matches = 0.0, and macro averaging.
- **2026-09-25, all-empty test submission** written to `output/` (1,732,544 rows per file).
  `utils/validate_submission.py`: PASS, both default and `--check-ids`.
