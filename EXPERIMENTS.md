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
- **2026-09-25, normalisation finished (norm-v1 → norm-v2)**
  - Code changes:
    - atl/Georgia fixed by a general rule: an alias must *replace* the state.
    - core_name falls back to full_name; raw columns kept.
    - Edge cases handled: empty, null, punctuation only, very long, mixed script, repeated suffixes.
    - addr_norm keeps ", " between components, which makes it idempotent.
    - Postcode + city components ("75001 paris", "10115 berlin"); unit and box words excluded.
    - Aliases learned only from train pairs whose S1 is not in validation; saved as a versioned JSON.
  - Local 1% sample, same-city agreement 82.0% (US S3 83.3%); unit tests 65/65 pass (11 metric + 54 normalisation).
  - **Kaggle CPU `normalise-v1`** (16:24–16:41, 1033 s, peak 10.5 GB): all 24.2M records.
    - The report showed international legal forms misfiring in US and India ("Ramey Spa Inc" → core "ramey";
      "Sas Nagar" place name), about 0.05% of records. Fixed: those forms now apply to France and unknown
      countries only.
    - Bumped to norm-v2. The v1 outputs are superseded.
  - **Kaggle CPU `normalise-v2`** (949 s, peak 10.5 GB). This is the current version; later jobs use it via
    `kernel_sources: sayanchatterjee264/normalise-v2`. Reports are in `outputs/normalise-v2/` (Parquet files stay
    on Kaggle).
    - State aliases: 13 India, 0 US, all purity ≥ 0.999 and ≥ 3,586 pairs. Learned from a seeded 3M subsample of
      7,292,528 non-validation train pairs. No flags; atl gone. dilli (Delhi) is no longer learned, because Delhi
      addresses also carry "Delhi" as a state.
    - Same-city / same-state on true pairs (all train):

      | country | source | same city | same state |
      |---|---|---|---|
      | India | S2 | 72.5% | 98.7% |
      | India | S3 | 89.7% | 98.7% |
      | US | S2 | 81.9% | 99.9% |
      | US | S3 | 83.3% | 99.9% |

      On validation S1 only (not used for aliases) the numbers are within 0.3 pt, so there is no leakage effect.
    - Quality: core_name empty 0% everywhere. City found 96–100%, state found 95–100% (US and India).
      Legal form found in France S1/S2/S3 69/56/55% (EDA 55–67%). Test vs train differ by < 2.5 pt for every
      metric and country, and no flags were raised. France S2/S3: département 31–32%, region 32–35%.
    - Open issues for review:
      - 5.7% of names keep a legal word inside core_name (noise word after the suffix: "Willow LLC Center").
      - "& Cie" leaves a dangling "and".
      - Gurmukhi "limtid" (Limited) is not in the list.
- **2026-09-25, norm-v3 + effectiveness check.** Kaggle CPU `normalise-v3` (2783 s: normalisation 1075 s,
  effectiveness 1708 s; peak 12.1 GB). This is the current version; later jobs use `kernel_sources:
  sayanchatterjee264/normalise-v3`. Reports are in `outputs/normalise-v3/`.
  - Changes:
    - `cie` → co as an international legal form (France and unknown countries).
    - Trailing `and` removed from core_name when a legal form is stripped ("Elsa & Cie SARL" → elsa,
      "Smith & Co" → smith).
    - Malayalam praivrr/limirrd and Gurmukhi limtid added. limirrd was the most common missed Indic spelling
      (3.6k per 110k Indian-script names).
    - Unit tests 68/68 pass. No report flags. Aliases and agreement are unchanged from v2.
  - Effectiveness on the validation split: 2,000 val S1 per country. raw = lowercased name, full = full_name,
    norm = core_name. The hard non-match is the most name-similar non-matching record in the same country and
    source, searched over the full pool (2–3M) separately per representation.

    | country | src | AUC tsr raw / full / norm | AUC jac3 raw / full / norm | true pairs tsr<70: raw → norm |
    |---|---|---|---|---|
    | India | S2 | 0.313 / 0.386 / 0.407 | 0.268 / 0.353 / 0.389 | 29.7% → 18.8% |
    | India | S3 | 0.373 / 0.423 / 0.452 | 0.280 / 0.354 / 0.400 | 20.7% → 15.1% |
    | US | S2 | 0.476 / 0.532 / 0.492 | 0.468 / 0.510 / 0.501 | 9.0% → 7.8% |
    | US | S3 | 0.482 / 0.539 / 0.499 | 0.446 / 0.488 / 0.468 | 9.7% → 7.9% |

  - Reading:
    - Normalisation helps: +0.09 to +0.12 AUC for India (transliteration, legal forms), +0.05 for US with
      full_name. core_name alone is no better than raw for US, because it drops the legal form.
    - All AUCs are ≤ 0.54: name alone cannot beat the hardest impostor. Its token_set_ratio median is 100 in every
      group.
    - A manual check of 16 hard negatives showed two kinds: same-name businesses in other cities (chains, generic
      names), and deliberate decoys with the same name, a different legal form and a near-identical address (Unit
      2 vs 9, house 9788 vs 9795, 24 vs 25 Greenwood Ln, extra token "West").
  - Implications:
    - Blocking must not rely on name alone.
    - Features: address, city and house-number agreement; legal-form agreement (same / different / missing); and
      extra or missing name tokens.
    - Keep both full_name and core_name similarities (plus the all-legal-words-removed variant agreed for
      mid-name legal words).

- **2026-09-25, blocking + candidate selection (no model yet, so no val F0.5).** The organiser update makes the
  candidate set size part of the ranking, so every design is judged on recall AND candidates per S1.
  - Setup:
    - Validation split = 99,994 S1 against the full train S2/S3 pool (India 4.1M, US 6.2M).
    - Validation is halved by id parity: configs are chosen on the tune half and reported on the report half
      (50k S1, 173k true pairs). The floor is 3.46 true matches per S1.
    - The pruner is trained on 100k train-split S1 only (asserted disjoint from validation).
    - Code: `src/blocking.py`, `src/candidates.py`, `src/run_blocking_eval.py`.
  - **Kaggle `blocking-eval-v2`** (65 min, peak 18.4 GB; GPU check passed but CPU was 11x faster, so searches ran
    on CPU). Within-state buckets; the union for the pruner is fwd top-20 + reverse top-3 + rare top-10 (u20).
    - ~5 candidates per S1: recall 95.4% (India 91.8, US 97.8), 4.8 per S1 (median 5, p95 8).
    - ~7 candidates per S1: recall 96.5% (India 93.2, US 98.7), 7.1 per S1 (median 6, p95 14).
    - Without the pruner, u20 = 96.9% at 49 per S1; the maximum is 97.4% at 118 per S1.
    - Groups at ~7: common names 95.8%, low name similarity 89.0%, Indian script 81.3%, match without a state
      79.9%, state differs 2.6% (0.5% of pairs).
  - **Kaggle `blocking-eval-v3`** (search jobs `blocking-train-india` 39 min and `blocking-train-us` 46 min, then
    evaluate 25 min, peak 22.8 GB, all CPU). Changes:
    - A missing state is inferred from the city, then the département. The maps are learned from records that have
      both (>= 20 records, >= 90% agreement; train + test, no labels) and are used for bucket placement and pruner
      features.
    - Pool records still without a state run the reverse search against all S1 of the country.
    - Pruner also on a larger starting list u50 = fwd top-50 + reverse top-5 + rare top-20. Training rows are capped
      at 6M (all positives kept).

    | operating point | config | recall ALL | India | US | avg / median / p95 per S1 |
    |---|---|---|---|---|---|
    | ~5 | pruner u50, tau 0.1, cap 20 | 95.8% | 92.6% | 98.0% | 4.8 / 5 / 8 |
    | **~7 (default)** | pruner u50, tau 0.01, cap 10 | **96.8%** | 94.0% | 98.8% | **6.6 / 6 / 10** |
    | ~10 | pruner u50, tau 0.005 | 97.2% | 94.4% | 99.0% | 8.7 / 7 / 18 |
    | ~12 | pruner u50, tau 0.002 | 97.3% | 94.6% | 99.2% | 11.7 / 9 / 26 |

    - Effects at ~7, v2 → v3: recall 96.5% at 7.1 → 96.8% at 6.6, better on both axes.
      - The no-state fallback alone (same u20 pruner): 96.4% at 6.6 → 96.7% at 6.9. The blocking union u20 goes
        96.9 → 97.0% at the same size, and reverse top-1 90.4 → 92.7%.
      - The larger starting list (u50 vs u20 in v3): 96.7% at 6.9 → 96.9% at 6.6 on the tune half, i.e. +0.2 pt
        recall with fewer candidates. That is small, as expected: u50 holds about 1% more true pairs than u20.
    - Group "match has no state": 79.9% (v2) → **82.4%** at ~7 (88.1% at ~12). Indian script 81.3 → 83.7%,
      low name similarity 89.0 → 90.0%. State differs is unchanged at 2.5% (882 pairs, 0.5%).
    - Cost: the reverse search does more comparisons (India 3.5 → 4.7e11, US 3.6 → 6.6e11 pairs; +34% / +81%).
      Estimated test blocking is 2.8 h CPU in total (TF-IDF build 15 min, searches 1.9 h, pruner 37 min).
    - Pruner audit:
      - 22 generic features: blocking scores / ranks / gaps, method agreement, list rank and size, rapidfuzz name
        and address similarity, same city / state. No country, source or language feature.
      - Trained only on train-split S1.
      - 22–34% of candidates have no city/state even after inference (empty or unparseable addresses, which appear
        as candidates far more often than their 3–5% share of records).
    - Decision: keep the v3 design with the ~7 operating point (pruner u50, tau 0.01, cap 10).
  - **France check on TEST data** (no labels; Kaggle `blocking-test-france` 14 min over all 259k French test S1,
    `blocking-test-usin` 71 min over 10% of US/India test S1 with the full pool, then `blocking-test-check`
    ~15 min). It applies the v3 pruner and the ~7 operating point chosen on validation.

    | | France | India | US |
    |---|---|---|---|
    | S2/S3 state detected / from city / none | 33% / 64% / 3% | 95% / 2% / 2% | 97% / 0% / 3% |
    | search groups (S1 buckets) | 3 | 26 | 46 |
    | comparisons vs whole-country search | 36% | 12% | 7% |
    | candidates per S1 after pruner (avg / median / p95) | 7.5 / 8 / 10 | 7.7 / 8 / 10 | 6.8 / 7 / 10 |
    | S1 with no candidate | 0.01% | 0.03% | 0.01% |
    | S1 lists at the cap (8–10) | 53% | 55% | 38% |
    | median best pruner score per S1 | 0.997 | 0.994 | 0.996 |
    | same-state / same-city feature missing | 20% / 20% | 22% / 23% | 31% / 31% |

    - France gets a state from its région. Normalisation finds it for only 33% of S2/S3 records; city inference
      fills 64% more. All French test S1 fall in just 3 régions (Hauts-de-France, Nouvelle-Aquitaine, Pays de la
      Loire), so the within-state search compares 36% of all pairs instead of 7–12%. That is the only flag, and it
      is a cost issue (France took 14 min), not quality.
    - No sign of worse blocking for France: similar list sizes, almost no empty lists, the pruner confident on
      the best candidate, and location features missing less often than for US/India.
    - Watch item: slightly more French candidates pass tau (8.6% vs 6.4–6.7%). With cap 10, lists hit the cap
      as often as India's. If French S1 have more true matches than the cap allows, the cap could cut some. It
      cannot be measured without labels; check by eye in the prediction sanity check.
