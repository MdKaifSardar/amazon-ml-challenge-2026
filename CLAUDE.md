# Amazon ML Challenge: Business Entity Resolution

Full problem statement: `docs/ps_ML.pdf` (read it when details are needed).
Team goal: maximise macro F0.5 on the private leaderboard, with a clean, reproducible pipeline.

## Environment
- My laptop has NO GPU. Heavy work (embeddings, transformer training) runs on Kaggle.
  Classical work (normalisation, TF-IDF blocking, rapidfuzz features, LightGBM) runs locally on CPU.
- Kaggle CLI: `.venv/bin/kaggle`, authenticated via `kaggle auth login` (no token file).
- Kaggle username: sayanchatterjee264. Challenge data dataset id: sayanchatterjee264/amazon-ml-2026.
- Never write any credential or token into any file.

## Kaggle job workflow
- Jobs live in `jobs/<job-name>/` as .ipynb plus `kernel-metadata.json` with: id "sayanchatterjee264/<job-name>",
  title "<job-name>", code_file "<notebook>.ipynb", language "python", kernel_type "notebook",
  is_private true, enable_gpu true, enable_internet true,
  dataset_sources ["sayanchatterjee264/amazon-ml-2026"], kernel_sources [] (add earlier jobs to reuse outputs).
- In notebooks, read from /kaggle/input/ (print os.walk first to confirm paths); save all outputs to /kaggle/working/.
- Push: `.venv/bin/kaggle kernels push -p jobs/<job-name>`
- Status: `.venv/bin/kaggle kernels status sayanchatterjee264/<job-name>`
- Output: `.venv/bin/kaggle kernels output sayanchatterjee264/<job-name> -p outputs/<job-name>`
- On failure: download the log, fix, push again. Test on a small sample before full runs.

## Project layout
- `dataset/train/` and `dataset/test/`: organiser TSVs, the source of truth (never commit; never modify).
  The validator reads `dataset/test/`.
- `data_parquet/train/` and `data_parquet/test/`: Parquet copies of every TSV (all columns str), built by
  `python code/business_entity_resolution/src/convert_to_parquet.py`. Use these for EDA and modelling.
- `utils/validate_submission.py`: organiser validator.
- `code/business_entity_resolution/src/`: ALL pipeline code (this is what gets submitted).
- `code/business_entity_resolution/README.md` and `requirements.txt` (pinned versions): keep updated as we go.
  Every package the pipeline imports goes in that requirements.txt with its exact installed version.
- `requirements.txt` (root): full dev environment. It includes the submission file via `-r` and adds dev-only tools
  (kaggle). Whenever you pip install something, add it to the right file with its pinned version.
- `.gitignore`: keep it covering data, data_parquet/, artifacts/, outputs/, generated TSVs, .venv and credentials.
- `output/`: `matching_results.tsv` and `candidate_pairs.tsv`.
- `artifacts/`: cached intermediate files (embeddings, candidate sets, features, models).
- `EXPERIMENTS.md`: log of every experiment (see Working rules).
- `Documentation_template.md`: methodology write-up, fill in progressively.

## The task
- 3 sources, each TSV with columns: entity_id, business_name, business_address, country.
  Source is given by the ID prefix S1-/S2-/S3-. IDs carry no matching information.
- Source 1 is deduplicated: every S1 record is a distinct business.
- For every S1 entity, find ALL matching S2 and/or S3 records: zero, one, or many, possibly several from the same source.
  Never output S1 IDs, never match S2 to S3 directly.
- Train has ground truth (`train_ground_truth.tsv`: source1_entity_id, matched_entity_ids comma-separated, empty for none).
- Test: new, unseen businesses, no labels. Test S1 is matched only against test S2/S3.
- The pipeline must do everything automatically; no manual labelling of test records.

## Reading files (always)
- EDA and modelling: polars on Parquet, e.g. `pl.read_parquet("data_parquet/train/train_source1.parquet")`,
  or `pl.scan_parquet(...)` to stay lazy. Don't use pandas on the full data.
- TSVs only for the validator and final outputs. If a TSV must be read directly:
  `pl.scan_csv(path, separator="\t", quote_char='"', infer_schema=False, empty_string_is_null=False)`.
  Fields use standard CSV quoting (`""` = literal quote); never disable quoting.
- Kaggle jobs read the private Kaggle dataset sayanchatterjee264/amazon-ml-2026 (uploaded from
  data_parquet/ or dataset/), at the path shown by os.walk("/kaggle/input").

## Data facts (verified 2026-09-25)
| file | rows | countries |
|---|---|---|
| train_source1 | 2,206,821 | US 1.32M, India 0.88M |
| train_source2 | 5,034,616 | US 3.02M, India 2.02M |
| train_source3 | 5,285,603 | US 3.17M, India 2.12M |
| train_ground_truth | 2,206,821 (one row per train S1) | |
| test_source1 | 1,732,544 | India 0.81M, US 0.66M, France 0.26M |
| test_source2 | 4,887,273 | India 2.31M, US 1.87M, France 0.70M |
| test_source3 | 5,082,316 | India 2.41M, US 1.95M, France 0.73M |
- IDs unique per file, no nulls, names never empty. About 3% of S2/S3 addresses are empty; S1 addresses never are.
- S2/S3 names and addresses include non-Latin scripts (Devanagari, Tamil, ...), so normalisation must transliterate.

## Compute budget
Local: 4 cores, 7.4 GB RAM (~4 GB free), no swap. Each full S2/S3 table is ~450 MB in polars.
- Local: loading, normalisation, EDA, and developing on samples (e.g. 50k S1 plus their true matches plus
  random distractors). Samples have fewer distractors than the full pool, so their scores look better.
- Kaggle (~30 GB RAM): full-data TF-IDF blocking, feature building for all candidate pairs, LightGBM on
  full training pairs, and test inference.

## Noise to expect
- Names: abbreviations (Pvt/Private, Ltd/Limited, Corp/Corporation), legal suffix differences, DBA/trade names,
  & vs "and", punctuation, word-order swaps, typos, transliterations.
- Addresses: Rd/Road, St/Street, transliteration (Bengaluru/Bangalore), missing PIN/state, landmarks ("Near SBI ATM"),
  house-number formats, reordered components.
- Traps: chains (same name, different address), generic names, empty or partial addresses.

## FRANCE (critical)
Train has only US and India; test also has France. Treat country as an open set of strings.
- Never hard-code, filter or one-hot only {US, India}. Every test S1 entity (France included) must appear in the output.
- Prefer country-agnostic features (char n-grams, digit/postcode overlap); strip accents (unidecode).
- Legal-suffix lists must be general and include French forms (sarl, sas, sa, eurl, sasu, sci).
- No country-specific hard rules; use country as a feature. Sanity-check French predictions by eye after test inference.

## Metric: macro F0.5 per S1 entity, averaged over all S1 entities
F0.5 = 1.25*P*R / (0.25*P + R). Precision weighted 2x over recall.
- Singleton (no true matches): empty prediction = 1.0, any prediction = 0.0.
- Adding one wrong match usually costs more than missing one true match. When unsure, leave it out.
- "All empty" baseline scores exactly the singleton fraction; every model must beat it.
- Implement the scorer exactly like this, and use it for every decision.

## Rules (disqualification risk)
- NO external data, APIs or lookups: no geocoding, business registries, web search, scraping, external datasets.
- Pretrained models allowed only if MIT or Apache 2.0 licence and <= 8B parameters. Verify the licence on the
  model card before use and record model name + licence in the documentation.
- Code must reproduce both output files end to end from the provided data.

## Output format (both files tab-separated)
- `matching_results.tsv`: columns source1_entity_id, matched_entity_ids. Exactly one row per test S1 entity,
  comma-separated IDs, no quoting, no duplicates in a list, only S2/S3 IDs that exist in test, empty if no match.
- `candidate_pairs.tsv`: columns source1_entity_id, candidate_entity_ids. Same rules. Must be the EXACT final
  candidate set the model scores (last filtering stage before the model). Matches must be a subset of candidates.
- Always run before any upload:
  `python3 utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test`

## Pipeline plan
1. Normalisation (one shared function): lowercase, unidecode, & -> and, strip punctuation, collapse spaces,
   canonical abbreviations (pvt->private, ltd->limited, rd->road, st->street, ...), core name without legal
   suffixes, extracted digit tokens and postcodes from address.
2. Blocking (union of methods, report recall and candidates per S1):
   - TF-IDF char_wb 3-4-grams on core name, top-k cosine per S1 (sparse matmul).
   - Same on name + address.
   - Inverted index on rare tokens and postcodes.
   - Later: embedding nearest neighbours (FAISS).
   Choose smallest k reaching ~95-98% blocking recall on train. Country is not a hard filter.
3. Pair features (rapidfuzz): name ratio, Jaro-Winkler, token_set/token_sort, Jaccard, TF-IDF cosine on full and
   core names; first-token match; address similarities; digit/postcode overlap with states match/conflict/missing;
   blocking rank and score; gap to best other candidate; candidate popularity (how many S1 it is close to);
   name frequency (rare vs common); same country; candidate source (S2/S3).
4. Model: LightGBM binary classifier on candidate pairs.
5. Selection (tune on validation for macro F0.5): global threshold sweep; one-to-one assignment if EDA confirms each
   S2/S3 ID matches at most one S1 (greedy by descending score); relative margin to best candidate;
   prefer a threshold slightly on the conservative side of the peak.
6. Later improvements, driven by error analysis: multilingual embeddings (Kaggle GPU), cross-encoder re-ranker,
   optional small LLM judge for borderline pairs only.

## Validation protocol
- Split by S1 entity (GroupKFold / 80-20 on S1 IDs), never by random pairs.
- Validate the FULL pipeline end to end (blocking -> features -> model -> selection) on held-out S1 entities.
- Report: blocking recall, macro F0.5 overall, per country, on singletons vs non-singletons, precision and recall.

## Working rules
- Every change gets a validation F0.5 before it is kept. Log it in EXPERIMENTS.md:
  date/time, change, blocking recall, val F0.5 (overall and per country), kept or reverted.
- Keep the best pipeline runnable at all times; never break the last working version.
- Cache expensive steps in artifacts/ keyed by config so reruns are fast.
- Set random seeds. Keep one entry point: `python src/run_pipeline.py` (train + inference + write outputs).
- Update README.md, requirements.txt and Documentation_template.md as the approach changes, not at the end.
- Show me a short summary after each step: what was done, key numbers, suggested next step.
- Ask before long-running jobs or anything that uses Kaggle GPU quota.
