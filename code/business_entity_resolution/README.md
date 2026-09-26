# Business Entity Resolution pipeline

Status:
- Done: data preparation, EDA, scorer and all-empty baseline; shared normalisation (norm-v3).
- Done: blocking and candidate selection (v3). Default operating point: validation recall 96.8% at 6.6
  candidates per S1.
- Done: Job A candidate generation (300k training S1 sample -> 1.97M train pairs, 96.86% recall).
- Done: Job B full test candidate generation (1.73M test S1 -> 12.67M candidate pairs, 99.98% coverage,
  output/candidate_pairs.tsv verified, test_candidates.parquet published to Kaggle v3 dataset).
- Next: Feature extraction on test candidates (Person 2), model scoring (Person 3), and pipeline integration.

## Setup
```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
```
Optional imports, not pinned: `torch` (GPU top-k; the CPU path is used without it and was faster here) and
`matplotlib` (the recall-vs-candidates plot). Both are preinstalled on Kaggle.
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

6. Blocking and candidate selection. The full data runs on Kaggle as pieces (CPU is faster than GPU here):
   - search, one piece per country and side:
     `python src/run_blocking_eval.py --stage search --side train --countries India --scopes state ...`
     (`--side test` queries test S1; `--shard i/n` splits further);
   - evaluate: merges the pieces, trains the pruner, reports recall vs candidates per S1 and the operating points:
     `python src/run_blocking_eval.py --stage evaluate --pieces <dirs> --scopes state ...`;
   - France / test check, no labels:
     `python src/run_blocking_test_check.py --pieces <dirs> --model-dir <evaluate output>`.
   Kaggle jobs: `blocking-train-india`, `blocking-train-us`, `blocking-test-france`, `blocking-test-usin`,
   `blocking-eval-v3`, `blocking-test-check`. Results are in `EXPERIMENTS.md`.
7. Pair features (table 6) for train, val and test, one country at a time, with checks (schema, duplicates, inf,
   NaN share and median per split x country, test-vs-train shift flags) and per-feature AUC on the labelled splits:
   `python src/run_features.py --input <dir with *_candidates.parquet and normalised/> --out-dir <out> --splits train val test --prune val`
   (`--prune val`: the blocking-eval-v3 val table is the unpruned starting list; train and test are already pruned).

7. Model + selection + test outputs (Kaggle CPU job `model-v1`, ~50 min): LightGBM variants, F0.5-tuned selection,
   France checks, both output files and the validator:
   `python src/run_model.py --input <folder with features-v2, blocking-v3 candidates, split, raw data> --out-dir DIR --validator ../../utils/validate_submission.py`
   Current submission: variant C, tau 0.65, margin 0.7, validation macro F0.5 0.9717.

## Shared table formats (interfaces between the streams)
All tables are Parquet with string ids. `s1` = Source 1 entity_id and `cand` = S2/S3 entity_id. The pair key
everywhere is `(s1, cand)`.

| # | table | produced by | columns |
|---|---|---|---|
| 1 | normalised records `normalised/{train,test}_source{1,2,3}.parquet` | Kaggle `normalise-v3` | entity_id, business_name, business_address, country, full_name, core_name, legal, addr_norm, city, state, dept, numbers (list of str), norm_version |
| 2 | split `g25_split.parquet` | Kaggle `phase0-eda` | source1_entity_id, country, n_matches, fold, role (`val` 99,994 / `train` 299,728 / `unused`) |
| 3 | blocking results, long `pieces/piece_{train,test}_{country}_{i}of{n}.parquet` | search jobs | scope, method (name, name_city, name_addr, reverse, rare), score, rank, best (reverse only: the pool record's top-1 score), s1, cand |
| 4 | candidate table, wide `val_candidates.parquet` (validation S1) | `blocking-eval-v3`; any split via `to_wide()` in `run_blocking_eval.py` | scope, s1, cand, rank_{name,name_city,name_addr,reverse,rare} (null = not found by that method), score_* (same), best_reverse, best_{name,name_city,name_addr} (the S1's top score per method), order (best cosine), is_match (validation only), role, p_u20, p_u50 (pruner probability) |
| 5 | submitted candidate set (default operating point) | `candidates.prune()` | rows of table 4 in the starting list `u50` with p_u50 >= 0.01, at most 10 per S1 by p_u50 (`config.json` -> operating_points, target 7) |
| 6 | pair features (stream A -> B) | `features.py` + `run_features.py` (feat-v2, 86 features) | s1, cand, f_* (float32; one column per feature, NaN = missing). One file per split: train, val, test (rows grouped by S1 country). No label, country or language column. One source feature: `f_cand_is_s3` (1 = S3 candidate, 0 = S2). All counts (name / token frequency, rare words) and the state maps come from ALL train + test records of the country, the same for every split; no feature counts how many S1 lists a candidate is in. No département or postcode feature (they almost never fire). Missing states for `f_state_agree` are filled from the city (blocking's maps; no département → state map is learned on the real data) |
| 7 | pair scores (B -> selection -> C) | `model.py` + `run_model.py` | s1, cand, score (match probability, 0–1) |
| 8 | selection config (B -> C) | `selection.py` + `run_model.py` (`selection_config.json`) | JSON: model file, threshold, margin, one_to_one (bool), plus the validation F0.5 it was tuned for |
| 9 | output files | `submission.py` | `matching_results.tsv`, `candidate_pairs.tsv` (spec in the problem statement; candidates = table 5 for test S1) |

Rules for every stream:
- Use `metric.py` and the `g25_split.parquet` validation S1 for every number.
- Tune on validation, never on test.
- Keep features and rules country-agnostic (France exists only in test).

Known issue for model training: the saved pruners (`pruner_state_u*.pkl`) were trained on 100k train-split S1,
so their scores on those S1 are in-sample (too confident). Train the model on other train-split S1, or use
cross-fitted pruner scores.

## Modules
- `src/metric.py`: macro F0.5 exactly as the challenge defines it (tests in `tests/test_metric.py`, run with `python -m pytest tests`).
- `src/submission.py`: writes `matching_results.tsv` and `candidate_pairs.tsv` in the required format.
- `src/normalise.py`: shared name/address normalisation (`normalise_records()`, `Normaliser.names()`,
  `Normaliser.addresses()`), `learn_state_aliases()` (train pairs only) and `Normaliser.load()` for the saved
  aliases. Version string `NORM_VERSION`. Tests in `tests/test_normalise.py`.
- `src/blocking.py`: TF-IDF searches (forward name / name + city / name + address, reverse S2/S3 -> S1, rare
  tokens), state buckets with missing states inferred from the city (a département map is also tried, but none passes the
  >= 20 records / >= 90% rule on this data), GPU/CPU top-k.
- `src/candidates.py`: selection rules, the cheap pruner (features, training, chunked scoring), list statistics.
  Tests in `tests/test_candidates.py`.
- `src/model.py`: LightGBM training / prediction and the feature sets A / B / C.
- `src/selection.py`: threshold, one owner per S2/S3, margin; vectorised macro F0.5 (tests in `tests/test_selection.py`).
- `src/run_model.py`: model + selection end to end (training, validation, stress test, test outputs, validator).
- `src/write_candidates.py`: `candidate_pairs.tsv` from the blocking output, with checks and the validator.
- `src/run_blocking_eval.py`, `src/run_blocking_test_check.py`: blocking evaluation and the test-data France check.
- `src/features.py`: pair features (table 6): names, rare words, legal form, address (city / state
  agreement, numbers), blocking scores, per-list context, source flag. Tests in
  `tests/test_features.py`. `src/run_features.py` runs it per split and country and writes the report.
- `src/eval_normalise.py`: learns state aliases and reports same-city/same-state agreement of true pairs:
  `python src/eval_normalise.py --data-dir ../../data_sample --aliases ../../artifacts/state_aliases.json`
