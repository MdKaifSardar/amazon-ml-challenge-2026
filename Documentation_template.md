# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Your Team Name]  
**Team Members:** Sayan Chatterjee, Md Kaif Sardar, Ananya Ghosh [complete if needed]  
**Submission Date:** 2026-09-27

---

## 1. Executive Summary
We resolve entities in three steps, with one design goal: keep candidate sets small without losing true
matches.
- **Blocking** runs country-partitioned TF-IDF searches in both directions (S1 → S2/S3 and S2/S3 → S1) plus a
  rare-token index. A cheap learned pruner then cuts each S1's list to **~7 candidates** while keeping
  **96.85%** of true pairs.
- **Matching:** a LightGBM classifier on 82 country-agnostic similarity features scores each candidate.
- **Selection:** a threshold and margin tuned for F0.5 decide the final matches.

On held-out training entities this gives **macro F0.5 = 0.9717** (precision 0.992, recall 0.938), against 0.056
for the all-empty baseline.

Key choices:
- Legal forms, addresses and house numbers are compared **separately** from the core name. The data is full of
  decoys that share a name but differ in legal form or house number.
- Every step treats country as an open set, so the test-only country France is handled by the same generic rules.
  The final model variant was chosen for robustness on French data.

---

## 2. Methodology

### 2.1 Problem Analysis
Full-data EDA (`src/eda.py`; train 12.5M and test 11.7M records):
- **Match structure.**
  - 5.6% of S1 entities have no match (singletons), so the all-empty baseline scores 0.0559 on our validation
    split.
  - Each S1 has 3.46 matches on average, and 76.8% have 2 or more from the same source.
  - No S2/S3 record is linked to more than one S1 (0 of 7.6M), and no match crosses countries.
- **Names.** Legal-suffix variants (Pvt / Private, Ltd / Limited, Inc / Incorporated, L.L.C.), typos (12% of
  true pairs), word reordering (6%), added honorifics (M/s, Smt, Shri) and generic words. 18% of Indian S2/S3
  names are in Devanagari, Tamil or other Indic scripts. 9.6% of true pairs have name token_set_ratio < 70.
- **Addresses.**
  - Postcodes are almost absent (< 0.5% in India and France).
  - S2/S3 reorder components, write states differently from S1 (TX vs Texas, Maharashtra vs MH vs
    transliterated forms), and add landmarks ("Near ...", 10% of Indian pairs).
  - 4% of matched S2/S3 addresses are empty.
- **France (test only, 15% of test S1).** It shares no top vocabulary with train. Legal forms SARL / SAS / EURL
  / SA end 55–67% of names, "St" means Saint, and S2/S3 add départements that S1 lacks.
- **Decoys.** Name similarity alone separates true matches from the hardest same-country non-match poorly
  (AUC ≤ 0.54). A manual check shows two kinds of hard non-match:
  - same-name businesses in other cities (chains, generic names);
  - near-identical decoys: same name, different legal form, house number off by a little ("Unit 2" vs "9",
    "24" vs "25 Greenwood Ln").

  So address, house-number and legal-form agreement must decide, not the name.

### 2.2 Normalisation (`src/normalise.py`)
One shared, deterministic, vectorised (polars) module is used by blocking, features and inference. Rules are
looked up by country label; countries without their own table get only the general rules.
- **Names.**
  - Transliteration with anyascii (ISC licence), lowercase, & → and, punctuation removed.
  - Indian honorific prefixes are stripped (India only), and abbreviations are expanded (intl, mfg, svcs, ...).
  - Legal suffixes (the trailing run, or a leading run after word reordering) are mapped to one canonical token
    per form. This includes French forms and transliterated Indian forms (praivet, limitet, pra li, elelpi).
  - Outputs: `full_name`, `core_name` (suffix removed) and `legal`.
- **Addresses.**
  - Split into comma components, then cleaned.
  - States are detected first: a hand-written US, India and France (région) list, plus 13 aliases learned from
    train pairs, e.g. Devanagari "mharastr" → maharashtra. An alias is accepted only if it never appears in S1
    and does not concentrate on one S1 city, which keeps out old city names like "bombay".
  - Country-aware abbreviations follow: st → street (US) vs saint (France).
  - Outputs: `city` (the most frequent city-like component in the corpus, so reordered addresses agree),
    `state`, `dept` (French départements) and `numbers`.
- **Leakage and open set.**
  - State aliases use labels, so they are learned only from train pairs whose S1 is not in the validation
    split, then saved and reused at inference.
  - The city frequency vocabulary uses no labels (train + test records).
  - Full rule list: `docs/normalisation.md` (in the repository).
- **Effect.** On validation pairs, normalisation raises name-similarity AUC against the hardest non-match by
  +0.09–0.12 for India and +0.05 for the US. True pairs agree on the city in 72–90% of cases and on the state in
  98.7–99.9%.

### 2.3 Solution Strategy
**Approach Type:** Blocking + learned pruning + gradient-boosted pair classifier + F0.5-tuned selection.

**Core Innovation:**
- **Two-sided, candidate-size-aware blocking.** Because the candidate set size is part of the evaluation, each
  S2/S3 record searches for its best S1 (reverse search), as well as each S1 searching the pool. A cheap pruner
  then keeps each list close to the true match count (6.6 candidates per S1 for 3.46 true matches).
- **Decoy-aware, country-agnostic features and a France-robust model choice.** The final variant does not use
  search-method counts, which behave differently on French test data.

Validation protocol:
- Split by S1 entity: 5 folds stratified by country × match count. Validation = fold 0 (99,994 S1); training =
  folds 1–4 (299,728 S1).
- Nothing is tuned on test.
- Blocking and selection settings are chosen on one half of the validation S1 (even ids) and reported on the
  other half (odd ids), so the reported numbers are not tuned on themselves.
- Every validation S1 counts, including singletons and the few without candidates.

---

## 3. Candidate Generation (Blocking)
Code: `src/blocking.py`, `src/candidates.py`, `src/run_blocking_eval.py`, `src/run_blocking_job_a.py`,
`src/run_blocking_job_b.py`.

- **Blocking keys used:**
  1. **TF-IDF char_wb 3–4-grams**, fitted per country on train + test texts (no labels; min_df 2, max_df 2%,
     sublinear tf). Three text views:
     - `core_name`;
     - `core_name + city`;
     - `core_name + normalised address`.
  2. **Forward top-k:** each S1 finds its top-50 S2/S3 records per view (sparse top-k cosine, `sparse_dot_topn`).
  3. **Reverse top-m:** each S2/S3 record finds its top-5 S1 on name + address, with all S1 of the country
     competing. This makes the reverse direction central: a record rarely has more than one plausible owner.
  4. **Rare-token index:** S2/S3 records sharing a core-name token with document frequency ≤ 20 (3+ characters)
     in the country; top 20 per S1.
  5. **Partitioning** (a compute split, not a filter):
     - by country (no true pair crosses countries);
     - then by state: a query searches its own state plus pool records without a state.
     - A missing state is inferred from the city, via city → state maps learned from records that have both
       (≥ 20 records, ≥ 90% agreement, no labels).
     - Pool records still without a state search the whole country.
     - Test France: all S1 fall in 3 régions, and 64% of French S2/S3 get their région from the city.
- **Learned pruner (last stage before the model):**
  - A HistGradientBoosting classifier (scikit-learn) on 22 cheap features: per-method scores, ranks and gaps,
    the number of methods that found the pair, list position and size, name / address token-set similarity,
    same city / state.
  - Trained on 100k train-split S1 only.
  - The union of the searches (≈117 candidates per S1) is cut to candidates with pruner probability ≥ 0.01, at
    most 10 per S1.
  - The operating point was chosen on a recall vs candidates-per-S1 curve over ~500 configurations (fixed k,
    reverse-first with margins, adaptive margins, pruned lists). Alternatives: ~5 per S1 at 95.8% recall, ~10 at
    97.2%.
  - The pruned list is exactly what the model scores, and is what `candidate_pairs.tsv` contains.
- **Candidate pairs generated:**
  - test: **12,666,305** pairs for 1,732,544 S1, i.e. 7.31 per S1 (France 7.53, India 7.66, US 6.80; median 7,
    p95 10);
  - 280 S1 (0.02%) have no candidate.
  - Validation: 6.56 per S1 (median 6, p95 10).
  - Reduction ratio vs all S1 × S2/S3 pairs of the country: ≈ 99.9999%.
- **How we ensured true matches were not lost:**
  - Recall is measured on validation against the full train pool (4.1M India / 6.2M US records):
    - **96.85%** of true pairs are kept (India 94.0%, US 98.8%);
    - maximum reachable with the unpruned union: 97.5%.
  - Group recall at the operating point:
    - common names 96.1%;
    - low name similarity (token-set < 70) 90.0%;
    - Indian-script records 83.7%;
    - match records without a state 82.4% (79.9% before the no-state fallback).
  - The reverse search and the rare-token index recover pairs the name searches miss, and the country-wide
    fallback covers records without a state.
  - The **French test data** was checked without labels. Lists were 7.5 per S1, 0.01% empty, and the pruner was
    equally confident about the best candidate (median 0.997). There was no sign of worse blocking.
- **Scalability** (the organisers mention billions of records):
  - Normalisation, feature building, pruning and model scoring are **linear**: O(records) or O(candidate pairs),
    and candidate pairs grow linearly with S1 (~7 per S1). They are embarrassingly parallel by S1 shard (the
    code has `--shard i/n` and per-country processing).
  - The only super-linear step is the similarity search. Its cost is Σ over partitions of
    |queries| × nnz(pool vectors), kept down in three ways:
    - partitioning by country and state (7–12% of whole-country comparisons for US / India, 36% for France);
    - dropping n-grams that occur in > 2% of texts (max_df) from the sparse vectors, which removes about two
      thirds of the product cost;
    - sparse top-k with a fixed k, so memory is O(queries × k).
  - Measured throughput: ~1.5–2 ns per query-record comparison on 4 CPU cores. A test run takes ~2.8 CPU hours
    in total, of which fitting and transforming the TF-IDF matrices is ~15 min per country.
  - At billions of records, the exact search would be replaced by:
    - finer partitions (city / postcode prefix);
    - approximate nearest neighbours: MinHash-LSH on character shingles, or HNSW / IVF indices (e.g. FAISS) on
      hashed n-gram or learned embeddings, giving ~O(N log N);
    - the rare-token inverted index, which is already linear because tokens are capped at df ≤ 20.

    The reverse search and the pruner carry over unchanged, since both work on fixed-size top-k lists.

---

## 4. Matching Model
Code: `src/features.py`, `src/run_features.py`, `src/model.py`, `src/selection.py`, `src/run_model.py`.

**Features used** (feat-v2: 86 in total, 82 in the submitted variant C). The pipeline is country-agnostic: no
country, source-language or label column. Counts come from all train + test records of the country, identical
for every split.
- **Name features:**
  - Levenshtein ratio, Jaro-Winkler, token-set / token-sort ratio, token Jaccard and character 3-gram Jaccard,
    each on `full_name`, `core_name` and the name with **all legal words removed at any position** (fixes
    names like "Willow LLC Center");
  - partial ratio, first / last token equal, tokens only on one side, length difference, containment;
  - IDF-weighted token overlap and the rarest differing token;
  - rare-word overlap (tokens with df ≤ 20: shared, one-sided, how rare the rarest shared one is).
- **Legal-form features:** the edge legal form and legal words found anywhere in the name, each coded as same /
  different / missing; legal-word Jaccard.
- **Address features:**
  - token-set / token-sort / ratio / partial ratio, token Jaccard and 3-gram Jaccard on the normalised address;
  - address missing;
  - city equal / fuzzy / found in the other address;
  - state agreement (with city-inferred states);
  - **house-number features**: shared numbers, numbers only on one side, number Jaccard, first / last number
    equal. These are the strongest single signals (AUC 0.85–0.92), because decoys differ in house number.
- **Blocking features:** per-method cosine scores, ranks, gaps to the S1's best, and the reverse-search gap.
- **Context features:**
  - name frequency (log count of the name among S1 / S2+S3 records of the country);
  - the pair's name / address / number similarity minus the best **other** candidate of the same S1;
  - duplicates of the candidate's name in the list;
  - the number of exact-name candidates.
- **Other:** S3 vs S2 source flag; list size.

**Model type:** LightGBM binary classifier (MIT licence; library version 4.7.0). Settings:
- 127 leaves, ≥ 100 rows per leaf, 80% feature and row subsampling, L2 1.0, learning rate 0.05;
- early stopping on a held-out 10% of the training S1 (split by S1, never by pair); best iteration 1,453;
- trained on the 199,699 train-split S1 the blocking pruner never saw (1.18M fit pairs, 51% positives), so that
  in-sample pruner scores cannot leak in.
- Four variants were compared on validation:

| variant | features | validation F0.5 (report half) |
|---|---|---|
| A: all features | 86 | 0.9719 |
| B: without the number of search methods | 85 | 0.9718 |
| **C: B without the pruner-derived features (pruner score, gap, list position)** | 82 | **0.9717** (submitted) |
| A_all: A trained with the pruner's 100k S1 too | 86 | 0.9724 |

- Why C:
  - French test candidates are found by fewer search methods than training pairs (median 2 vs 4), and the
    pruner itself uses the method count. So features that depend on method counts could under-score French
    true pairs.
  - The rule fixed before looking at test: take the most France-robust variant within 0.5 F0.5 points of the
    best. C costs 0.07 points on validation.
- A stress test on validation (true pairs found by only 1–2 methods) showed no variant under-scores them much;
  C selects 86% of 1-method and 96% of 2-method true pairs (A: 85% / 95%).

**Threshold selection method:** Macro F0.5 is optimised on the validation tune half with the exact challenge
metric (vectorised; checked equal to the reference implementation to 1e-9). The grid:
- probability threshold τ ∈ {0.10, …, 0.95};
- relative margin m ∈ {0, 0.3, 0.5, 0.7}: keep a match only if p ≥ m × the S1's best p;
- one owner per S2/S3 (on / off), motivated by the EDA finding that no S2/S3 record matches two S1.

Chosen for C: **τ = 0.65, margin 0.7**; one owner off (it made no difference at this threshold). Tune-half and
report-half F0.5 differ by < 0.001.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** **0.9717** on 50,032 held-out validation S1 (report half). Breakdown:

| group | macro F0.5 | precision | recall | predicted per S1 |
|---|---|---|---|---|
| overall | **0.9717** | 0.992 | 0.938 | 3.27 |
| US | 0.9804 | 0.993 | 0.956 | 3.33 |
| India | 0.9586 | 0.992 | 0.911 | 3.19 |
| non-singletons | 0.9720 | 0.994 | 0.938 | 3.46 |
| singletons | 0.9677 | – | – | 0.03 |

- **Test (no labels):**
  - predicted matches per S1: France 3.36, India 3.18, US 3.34;
  - S1 with no match: 5.0%, 6.6% and 5.7% (training singleton share: 5.6%);
  - French matches sit at the top of capped candidate lists (mean position 3.1 of 10; 1.7% at positions 9–10);
  - a by-eye check of random French entities found correct matches across abbreviations (R. / Rue, Sàrl),
    reordered addresses and département-vs-région differences.
- **Common false positives (wrong merges):** precision is 0.992, so few. The known risks, from the data
  analysis, are:
  - near-identical decoys: same name, a different or missing legal form, a close house number;
  - chains or generic names in the same city.

  The legal-form and house-number features target both, and the high threshold (τ = 0.65) keeps precision
  high. Singletons are the main place where a wrong match costs a full point (singleton F0.5 0.968).
- **Common false negatives (missed matches):** recall 0.938 is the main loss, and most of it is set by blocking:
  - 3.15% of true pairs never become candidates (India 6.0%, US 1.2%);
  - Indian-script records (83.7% blocking recall) and records without a state (82.4%) are the weakest groups;
  - pairs found by only one search method are the ones the pruner drops most (it keeps 88.7% of them vs 99.7%
    for 3+ methods);
  - inside the candidate set, the model misses mostly low-similarity true pairs: heavy name changes or
    transliterations, and empty or partial addresses.

---

## 6. Conclusion
The approach has three parts:
- Recall-oriented, size-aware blocking (two-way TF-IDF search, a rare-token index and a learned pruner) keeps
  ~97% of true pairs in ~7 candidates per S1.
- Decoy-aware features on name, legal form, address and house numbers let a LightGBM model reach macro F0.5
  0.972 with 99% precision.
- Keeping every rule country-agnostic, and choosing the model variant for robustness to the French shift,
  matters as much as the validation score, because the test set contains a country never seen in training.

The next gains are in blocking recall for India / Indic scripts, e.g. an embedding-based search.

---

## Appendix

### A. Code Artefacts
All code is in `code/business_entity_resolution/src/`. `README.md` there has the exact end-to-end commands, with
runtimes and memory; `requirements.txt` pins every dependency.

| stage | script / module |
|---|---|
| data preparation, EDA, split | `convert_to_parquet.py`, `eda.py`, `make_sample.py` |
| normalisation | `normalise.py` (rules), `run_normalise.py` (all files) |
| blocking + pruner | `blocking.py`, `candidates.py`, `run_blocking_eval.py` (search / evaluate / pruner), `run_blocking_job_a.py` (train candidates), `run_blocking_job_b.py` (test candidates) |
| features | `features.py`, `run_features.py` |
| model + selection + outputs | `model.py`, `selection.py`, `run_model.py`, `submission.py` |
| metric, checks | `metric.py` (challenge F0.5), `write_candidates.py`, `run_blocking_test_check.py` (France check) |

Reproduction order:
1. `convert_to_parquet.py`
2. `eda.py`
3. `run_normalise.py`
4. `run_blocking_eval.py` (search + evaluate)
5. `run_blocking_job_a.py`
6. `run_blocking_job_b.py`
7. `run_features.py`
8. `run_model.py`

This writes `output/matching_results.tsv` and `output/candidate_pairs.tsv`, which pass
`utils/validate_submission.py`.

Unit tests (`tests/`, 99): metric, normalisation, candidate selection, features, match selection.

**Compute:** Kaggle CPU sessions (4 cores, ~30 GB) and two AWS EC2 CPU instances for the full-data blocking runs.
No GPU is required; on this data, CPU sparse top-k was faster than a T4 GPU.

**Rules compliance:**
- No external data, APIs, lookups or pretrained models; every model is trained from the provided training data.
- Libraries: LightGBM (MIT), scikit-learn (BSD-3-Clause), rapidfuzz (MIT), anyascii (ISC), polars (MIT),
  sparse_dot_topn (Apache-2.0).

### B. Additional Results
- Blocking operating points, validation (recall / candidates per S1):
  - ~5 per S1: 95.8% / 4.8;
  - **~7 per S1 (chosen): 96.8% / 6.6**;
  - ~10 per S1: 97.2% / 8.7;
  - ~12 per S1: 97.3% / 11.7;
  - without the pruner: 97.0% at 49, 97.5% at 118.
- Most important model features (variant C, gain share):
  - house-number Jaccard 0.37;
  - numbers only on the S1 side 0.10;
  - address token-set similarity 0.08;
  - numbers only on the candidate side 0.04;
  - reverse-search gap / rank 0.03 each;
  - name token-sort / token Jaccard 0.02 each.
- The full experiment log with every run, setting and number: `EXPERIMENTS.md` (in the repository).
