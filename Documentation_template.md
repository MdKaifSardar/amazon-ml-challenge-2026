# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Your Team Name]  
**Team Members:** [List all team members]  
**Submission Date:** [Date]

---

## 1. Executive Summary
*Provide a brief 2-3 sentence overview of your approach and key innovations.*

---

## 2. Methodology

### 2.1 Problem Analysis
Full-data EDA (`src/eda.py`; train 12.5M and test 11.7M records):
- **Match structure.** 5.6% of S1 entities have no match (singletons), so the all-empty baseline scores 0.0559 on
  our validation split. Each S1 has 3.46 matches on average, and 76.8% have 2 or more from the same source. No S2/S3
  record is linked to more than one S1 (0 of 7.6M), and no match crosses countries.
- **Names.** Legal-suffix variants (Pvt/Private, Ltd/Limited, Inc/Incorporated, L.L.C.), typos (12% of true pairs),
  word reordering (6%), added honorifics (M/s, Smt, Shri) and generic words. 18% of Indian S2/S3 names are in
  Devanagari, Tamil or other Indic scripts. 9.6% of true pairs have name token_set_ratio < 70.
- **Addresses.** Postcodes are almost absent (< 0.5% in India and France). S2/S3 reorder components, write states
  differently from S1 (TX vs Texas, Maharashtra vs MH vs transliterated forms), and add landmarks ("Near ...", 10%
  of Indian pairs). 4% of matched S2/S3 addresses are empty.
- **France (test only, 15% of test S1).** It shares no top vocabulary with train. Legal forms SARL/SAS/EURL/SA end
  55-67% of names, "St" means Saint, and S2/S3 add départements that S1 lacks.

### 2.2 Normalisation (`src/normalise.py`)
One shared, deterministic, vectorised (polars) module for blocking, features and inference. Rules are looked up by
country label. Countries without their own table get only the general rules.
- **Names.** Transliteration with anyascii (ISC licence), lowercase, & -> and, punctuation removed. Indian
  honorific prefixes are stripped (India only), and abbreviations are expanded (intl, mfg, svcs, ...). Legal suffixes
  (trailing run, or leading after word reordering) are mapped to one canonical token per form, including French
  forms and transliterated Indian forms (praivet, limitet, pra li, elelpi). Outputs: `full_name`, `core_name`
  (suffix removed) and `legal`.
- **Addresses.** Split into comma components, then cleaned. States are detected first: a hand-written US, India and
  France region list plus aliases learned from train pairs, e.g. Devanagari "mharastr" -> maharashtra. An alias is
  accepted only if it never appears in S1 and does not concentrate on one S1 city, which keeps out old city names
  like "bombay". Country-aware abbreviations follow: st -> street (US) vs saint (France). Outputs: `city` (the most
  frequent city-like component in the corpus, so reordered addresses agree), `state`, `dept` (French
  départements) and `numbers`.
- **Effect.** On true pairs (1% sample), same-city agreement rose from 45.7% to 81.9%. For US S3 it rose from
  7.5% to 82.6%. Same state is 99.3%.

### 2.3 Solution Strategy
*Outline your high-level approach.*

**Approach Type:** [Blocking + Classifier / End-to-End / Graph-Based / Hybrid, etc]  
**Core Innovation:** [Brief description of your main technical contribution]

---

## 3. Candidate Generation (Blocking)
*Describe how you reduced the comparison space to a manageable candidate set.*

- **Blocking keys used:** [e.g., PIN code, phonetic name encoding, TF-IDF, etc.]
- **Candidate pairs generated:** [total]
- **How you ensured true matches were not lost:**

---

## 4. Matching Model

**Features used:**
- Name features: [e.g., Jaccard, Levenshtein, phonetic encoding]
- Address features: [e.g., token overlap, edit distance, PIN code matching]
- Other: []

**Model type:** [e.g., XGBoost, Siamese Network, Transformer, etc.]  
**Threshold selection method:** [e.g., F_0.5 optimization on validation set]

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** [your best validation score]
- **Common false positives (wrong merges):** [brief description]
- **Common false negatives (missed matches):** [brief description]

---

## 6. Conclusion
*Summarize your approach, key achievements, and lessons learned in 2-3 sentences.*

---

## Appendix

### A. Code Artefacts
*Your complete, runnable code ships in the submission zip under
`code/business_entity_resolution/` (all source in `src/`, with a `README.md` and
`requirements.txt`). Summarise its structure and the entry point(s) to reproduce
`output/matching_results.tsv` and `output/candidate_pairs.tsv` here.*

### B. Additional Results
*Include any additional charts, graphs, or detailed results.*

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and technical depth.
