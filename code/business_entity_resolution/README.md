# Business Entity Resolution pipeline

For every test Source 1 (S1) business, the pipeline finds the matching Source 2 / Source 3 (S2/S3) records.
It produces `output/matching_results.tsv` and `output/candidate_pairs.tsv`.

**Final submission (leaderboard 0.972):**
- validation macro F0.5 **0.9786** (US 0.987, India 0.967; precision 0.997, recall 0.947);
- model: two-stage LightGBM (v4), with a fine-tuned multilingual cross-encoder re-scoring the borderline pairs;
- candidate set: 7.3 candidates per S1 on test (validation recall 96.85% at 6.6 per S1);
- organiser validator: PASS.

Methodology: `../../SOLUTION_REPORT.md`. Experiment log: `../../EXPERIMENTS.md`.

## Pipeline at a glance
```
TSV -> Parquet -> EDA + S1-grouped split -> normalisation (norm-v3)
    -> blocking v3: TF-IDF searches + pruner  -> candidates (train / val / test)  = candidate_pairs.tsv
    -> pair features feat-v2 (86; 82 used)
    -> model v4: stage-1 LightGBM (5-fold OOF) -> set + decoy + cross-S1 features -> stage-2 LightGBM (3 seeds)
    -> cross-encoder (distilbert-base-multilingual-cased, fine-tuned) re-scores pairs with 0.02 < p < 0.98
    -> blend p = 0.35 p_model + 0.65 p_ce -> selection (tau 0.625, margin 0.7, one owner per S2/S3)
    -> matching_results.tsv (+ candidate_pairs.tsv) -> organiser validator
```

## Setup
```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
```
- Python 3.12+; every package the pipeline imports is pinned in `requirements.txt`.
- `torch` and `transformers` are needed only for the two cross-encoder steps (10, 11), which need a CUDA GPU (a T4
  is enough). Blocking can also use `torch` for GPU top-k, but the CPU path was faster here.
- `matplotlib` (one plot in the blocking report) is optional and not pinned.
- Tests: `python -m pytest tests` (pytest is a dev tool, not a pipeline dependency).
- No external data, APIs or lookups are used. Models:
  - trained here from the provided training data: LightGBM (MIT licence) and a scikit-learn
    HistGradientBoosting pruner (BSD-3-Clause);
  - one pretrained model, `distilbert-base-multilingual-cased` (Apache-2.0, ~135M parameters). It is downloaded
    from Hugging Face by step 10 and fine-tuned on the provided training pairs.

Run every command below **from this folder** (`code/business_entity_resolution/`), with the organiser data in
`../../dataset/{train,test}/*.tsv`.

**Input search:** several scripts search their `--input` / `--norm-dir` folder recursively and take the first
file with the expected name. Keep only one copy of each file under those folders (e.g. no old smoke-test
outputs).

## Reproduce end to end
The seeds are fixed (42). Runtimes and peak RAM are from our runs:
- Kaggle CPU sessions: 4 cores, ~30 GB;
- AWS EC2 for Jobs A and B: c6i.2xlarge (8 vCPU, 15 GB + swap) and r6i.2xlarge (8 vCPU, 64 GB).

`jobs/` in the repository holds the Kaggle notebooks that ran each step; each embeds the script verbatim.

| # | step | command | runtime / peak RAM | outputs |
|---|---|---|---|---|
| 1 | TSV -> Parquet (all columns as strings) | `python src/convert_to_parquet.py --tsv-dir ../../dataset --out-dir ../../data_parquet` | minutes / low | `data_parquet/{train,test}/*.parquet` |
| 2 | EDA + S1-grouped split (5 folds; val = fold 0) | `python src/eda.py --data-dir ../../data_parquet --out-dir ../../outputs/eda` | 7 min / 18 GB | `outputs/eda/g25_split.parquet` |
| 3 | normalisation norm-v3 (state aliases from non-validation train pairs; city vocabulary from train + test records) | `python src/run_normalise.py --data-dir ../../data_parquet --split ../../outputs/eda/g25_split.parquet --out-dir ../../artifacts/normalised` | 18 min / 12 GB | `artifacts/normalised/normalised/{train,test}_source{1,2,3}.parquet`, `state_aliases_norm-v3.json` |
| 4a | blocking search, validation + 100k train S1 (pruner data), one country per run | `python src/run_blocking_eval.py --stage search --side train --countries India --scopes state --norm-dir ../../artifacts/normalised --data-dir ../../data_parquet --split ../../outputs/eda/g25_split.parquet --out-dir ../../artifacts/blocking_eval` (then the same with `--countries US`) | 39 + 46 min / 18 GB | `artifacts/blocking_eval/pieces/piece_train_*.parquet` |
| 4b | blocking evaluate: pruner training, operating points, validation candidates | `python src/run_blocking_eval.py --stage evaluate --pieces ../../artifacts/blocking_eval/pieces --scopes state --norm-dir ../../artifacts/normalised --data-dir ../../data_parquet --split ../../outputs/eda/g25_split.parquet --out-dir ../../artifacts/blocking_eval` | 25 min / 23 GB | `pruner_state_u50.pkl`, `config.json`, `val_candidates.parquet` (unpruned starting list), report |
| 5 | Job A: train candidates for all 299,728 train-split S1 (frozen v3 rule: u50, pruner tau 0.01, cap 10) | `python src/run_blocking_job_a.py --norm-dir ../../artifacts/normalised --split ../../outputs/eda/g25_split.parquet --data-dir ../../data_parquet --model-dir ../../artifacts/blocking_eval --out-dir ../../artifacts/blocking_eval` | 56 min / ~15 GB (8 vCPU) | `artifacts/blocking_eval/train_candidates.parquet` |
| 6 | Job B: test candidates for all 1,732,544 test S1 (same rule) | `python src/run_blocking_job_b.py --norm-dir ../../artifacts/normalised --model-dir ../../artifacts/blocking_eval --out-dir ../../artifacts/blocking_eval --tsv-out ../../artifacts/blocking_eval/job_b_pairs.tsv` | 99 min / **34 GB** | `artifacts/blocking_eval/test_candidates.parquet` (the `--tsv-out` file is a pair list for inspection, **not** the official format) |
| 7 | pair features feat-v2 for train, val (pruned to the submitted set) and test | `python src/run_features.py --input ../../artifacts --out-dir ../../artifacts/features --splits train val test --prune val` | 28 min / 14 GB | `artifacts/features/features_{train,val,test}.parquet`, report |
| 8 | final model v4: stage 1 (5-fold OOF), set / decoy / cross-S1 features, stage 2 (3 seeds), unseen-country simulation, validation report, test pair scores | `python src/run_model_v4.py --input ../../artifacts/model_input --config configs/best_config.json --out-dir ../../artifacts/model_v4 --validator ../../utils/validate_submission.py` | 2.5 h / 21 GB (CPU) | `artifacts/model_v4/{val_scores,test_scores}.parquet`, `model/model_config.json`, `report.md`, `output/*.tsv` (v4 alone) |
| 9 | (reference only) model v1 | `python src/run_model.py --input ../../artifacts/model_input --out-dir ../../artifacts/model --variants C` | 7 min | superseded by steps 8–11 |
| 10 | cross-encoder fine-tune (GPU) | `python src/cross_encoder_train.py --input ../../artifacts/model_input --out-dir ../../artifacts/ce --pairs 3000000 --batch 64 --max-train-min 90` | 90 min on a T4 | `artifacts/ce/ce_model/`, `ce_config.json`, `ce_report.md` |
| 11 | blend re-tuned on v4 validation scores, borderline test pairs re-scored, **both output files** + validator (GPU) | `python src/cross_encoder_score.py --input ../../artifacts/model_input --ce-dir ../../artifacts/ce --v4-dir ../../artifacts/model_v4 --out-dir ../../artifacts/ce_blend --validator ../../utils/validate_submission.py` | ~25 min on a T4 | `artifacts/ce_blend/{matching_results,candidate_pairs}.tsv`, `ce_test_report.md` |
| 12 | copy the outputs and validate | `cp ../../artifacts/ce_blend/*.tsv ../../output/ && python3 ../../utils/validate_submission.py --matching ../../output/matching_results.tsv --candidate ../../output/candidate_pairs.tsv --test-dir ../../dataset/test` | seconds | `output/` (PASS) |

Steps 8–11 read one folder. Link the inputs into it, so each file exists once:
```bash
M=../../artifacts/model_input && mkdir -p $M
ln -sf "$(realpath ../../outputs/eda/g25_split.parquet)" $M/
for f in train_source1 train_source2 train_source3 train_ground_truth; do ln -sf "$(realpath ../../data_parquet/train/$f.parquet)" $M/; done
ln -sfn "$(realpath ../../artifacts/normalised/normalised)" $M/normalised
for k in 1 2 3; do ln -sf "$(realpath ../../data_parquet/test/test_source$k.parquet)" $M/; done
for s in train val test; do ln -sf "$(realpath ../../artifacts/blocking_eval/${s}_candidates.parquet)" $M/;
                           ln -sf "$(realpath ../../artifacts/features/features_$s.parquet)" $M/; done
# step 10 also reads validation scores to report its own blend check (step 11 re-tunes on v4's):
ln -sf "$(realpath ../../artifacts/model_v4/val_scores.parquet)" $M/   # after step 8
```

Notes:
- Steps 4–6 are the frozen blocking v3.
  - Step 4 alone (`--stage all`) also runs on a laptop-sized sample from `src/make_sample.py`.
  - `src/run_blocking_test_check.py` (France check on test data) and `src/write_candidates.py` (candidate file
    and validator from the blocking output alone) are diagnostics, not needed for the outputs.
- Step 6 needs ~34 GB RAM as written. On a ~30 GB machine, run it per country.
- Step 8 trains four LightGBM variants (A, B, C and A_all) and picks one with a fixed rule. The rule: the most
  France-robust variant within 0.5 F0.5 points of the best on the validation report half.
  - The submitted choice is C: 82 features, without f_n_methods and the pruner-derived scores.
  - `--variants C` trains only the chosen variant (~7 min instead of 30).
- `candidate_pairs.tsv` = the test candidate set of step 6: exactly the pairs the model scores. Matches are
  always a subset of it (checked when writing).
- Every test S1 appears in both files; 280 S1 without candidates have empty lists.

## Shared table formats (interfaces between the stages)
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

Pruner overlap: the saved pruners (`pruner_state_u*.pkl`) were trained on 100k train-split S1, so their scores
on those S1 are in-sample. `run_model.py` therefore trains on the other ~200k train S1 (variant A_all, with all
S1, scores only +0.0005 higher on validation, so the effect is small).

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
- `src/two_stage.py`: stage-1 out-of-fold training and the 8 set features; `src/pair_extra.py`: the 18 targeted
  decoy features; `src/pipeline.py`: the shared stage-2 frame (C + set + decoy features).
- `src/run_model_v4.py`: the final model v4 (cross-S1 features, unseen-country simulation, validation report,
  test pair scores and v4 output files). `configs/best_config.json`: the tuned stage-2 settings (from
  `src/run_model_v3.py`).
- `src/cross_encoder_train.py`: fine-tunes `distilbert-base-multilingual-cased` on training candidate pairs and
  checks the blend on validation. `src/cross_encoder_score.py`: re-tunes the blend on v4's validation scores,
  re-scores the borderline test pairs and writes the final output files.
- `src/train_model.py`, `src/predict.py`, `src/run_final.py`, `src/run_two_stage.py`, `src/run_model_v3.py`:
  the v2 / v3 experiment runners (kept for reference; the final outputs come from steps 8–11).
- `src/selection.py`: threshold, one owner per S2/S3, margin; vectorised macro F0.5 (tests in `tests/test_selection.py`).
- `src/run_model.py`: model + selection end to end (training, validation, stress test, test outputs, validator).
- `src/write_candidates.py`: `candidate_pairs.tsv` from the blocking output, with checks and the validator.
- `src/run_blocking_eval.py`, `src/run_blocking_test_check.py`: blocking evaluation and the test-data France check.
- `src/features.py`: pair features (table 6): names, rare words, legal form, address (city / state
  agreement, numbers), blocking scores, per-list context, source flag. Tests in
  `tests/test_features.py`. `src/run_features.py` runs it per split and country and writes the report.
- `src/eval_normalise.py`: learns state aliases and reports same-city/same-state agreement of true pairs:
  `python src/eval_normalise.py --data-dir ../../data_sample --aliases ../../artifacts/state_aliases.json`
