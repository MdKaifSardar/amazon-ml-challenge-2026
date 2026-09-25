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
- **2026-09-25, normalisation module** (`src/normalise.py`, tests in `tests/test_normalise.py`, eval in
  `src/eval_normalise.py`). Local 1% sample only (`data_sample/`, 76,786 true pairs). Not yet a pipeline change, so
  there's no val F0.5.
  - 44/44 unit tests pass (11 metric + 33 normalisation, using real strings from `outputs/eda/`).
  - Learned state aliases from the sample's train pairs: 14 India (e.g. mharastr, krnatk, telmgan, dilli, pscimbng),
    1 US (atl -> ga, a false alias for Atlanta; recheck on full data). Saved to `artifacts/state_aliases.json`.
  - Same-city agreement of true pairs, before (EDA heuristic) -> after:

    | country | source | before | after | same state after |
    |---|---|---|---|---|
    | India | S2 | 61.5% | 73.1% | 98.6% |
    | India | S3 | 46.1% | 90.8% | 98.6% |
    | US | S2 | 75.1% | 80.7% | 99.9% |
    | US | S3 | 7.5% | 82.6% | 99.9% |
    | all | all | 45.7% | 81.9% | 99.3% |

  - Remaining city mismatches are mostly typos (phenix/phoenix, austni/austin), county vs city, and merged words
    (saintmichael). Leave these to fuzzy city features.
  - Speed on 1M test S2 rows: names 3.3 s, addresses 8.1 s, peak 2.1 GB. France: city found for 97%, legal form 55%,
    département 32%.
