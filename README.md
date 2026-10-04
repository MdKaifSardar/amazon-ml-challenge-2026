# Amazon ML Challenge 2026: Business Entity Resolution

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0 / MIT](https://img.shields.io/badge/License-MIT%2FApache%202.0-green.svg)](LICENSE)
[![Validation F0.5](https://img.shields.io/badge/Val%20Macro%20F0.5-0.9786-brightgreen.svg)](#results)
[![Leaderboard](https://img.shields.io/badge/Live%20LB%20Rank-~1756-orange.svg)](#results)

An end-to-end, high-performance, country-agnostic entity resolution pipeline designed for the **Amazon ML Challenge 2026**. The system resolves millions of business entities across heterogeneous data sources (`Source 1`, `Source 2`, and `Source 3`) under strict runtime, candidate set size, and precision constraints.

---

## 👥 Team: Heisenberg

- **Md Kaif Sardar** (Team Lead) – [GitHub](https://github.com/MdKaifSardar/) | [LinkedIn](https://www.linkedin.com/in/mdkaifsardar/)
- **Sayan Chatterjee** – [GitHub](https://github.com/sayanChaterjee) | [LinkedIn](https://www.linkedin.com/in/sayan-chatterjee-devch/)
- **Ananya Ghosh** – [GitHub](https://github.com/Ananya9304) | [LinkedIn](https://www.linkedin.com/in/ananya-ghosh-014b00290/)

---

## 📊 Results Summary

- **Live Leaderboard Rank:** ~1756 (Public LB Score: **0.9720**)
- **Validation Macro $F_{0.5}$:** **0.9786** (Precision: **0.9965**, Recall: **0.9475**)
  - United States: **0.9866**
  - India: **0.9666**
- **Candidate Size Efficiency:** Average **~7.3 candidates per entity** while retaining **96.85%** ground-truth recall.
- **Unseen Domain Robustness:** Zero-shot handling of **France** (~15% of test data, 0 training labels) using country-agnostic feature engineering and multilingual transformer scoring.

---

## 🏗️ System Architecture

```text
Raw TSV Data (24M+ records)
  │
  ├──> Parquet Conversion & Stratified 5-Fold Split (S1-grouped by country × match count)
  │
  ├──> Step 1: Normalisation (norm-v3)
  │      ├── anyascii script transliteration (Indic / Devanagari / Latin)
  │      ├── Legal form canonicalization & honorific stripping
  │      └── State alias resolution & corpus-level city frequency extraction
  │
  ├──> Step 2: Blocking & Learned Pruning (blocking-v3)
  │      ├── Country & state partitioned TF-IDF char n-gram search (forward + reverse)
  │      ├── Rare-token inverted index (document frequency <= 20)
  │      └── Fast HistGradientBoosting Pruner (caps candidate pool at ~7/query)
  │
  ├──> Step 3: Pair Feature Extraction (feat-v2: 82 features)
  │      ├── House number matching & Jaccard similarity (strongest decoy signal)
  │      ├── Legal-form mismatch detection
  │      └── Address token overlaps, city/state agreement, context & gap features
  │
  ├──> Step 4: Two-Stage Gradient-Boosted Classification
  │      ├── Stage 1: 5-Fold Out-of-Fold LightGBM
  │      └── Stage 2: LightGBM trained with list-level set, decoy, and cross-S1 competition features
  │
  ├──> Step 5: Multilingual Cross-Encoder Re-Scoring
  │      └── Fine-tuned `distilbert-base-multilingual-cased` re-evaluates borderline pairs (0.02 < p < 0.98)
  │
  └──> Step 6: Optimal Selection & Bipartite Enforcement
         ├── Calibrated threshold tau & margin filtering
         └── Strict `one_owner = True` bipartite constraint (eliminates duplicate merges)
```

---

## 🚀 Key Innovations & Engineering Highlights

1. **Two-Sided, Size-Aware Candidate Generation:**
   - Both forward queries ($S1 \to S2/S3$) and reverse queries ($S2/S3 \to S1$) compete under a learned pruner.
   - Shrinks the search space by **99.9999%**, outputting only ~7 high-quality candidate pairs per query.
2. **Decoy Discrimination via Number & Legal Form Features:**
   - Near-identical decoys (e.g., same business name but differing unit/house number or differing suffix such as LLC vs PLLC) are disentangled using explicit number tokenization and legal entity parsing.
3. **One-Owner Bipartite Assignment:**
   - Guarantees that each $S2/S3$ record links to at most one $S1$ query entity, removing >10,000 duplicate merge errors.
4. **Generalization to Unseen Regions (France):**
   - No country labels or dataset-specific leakage features were used.
   - Features sensitive to distribution shifts (like raw search method counts) were explicitly dropped in favor of invariant representations and multilingual transformer embeddings.

---

## 📁 Repository Structure

```text
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       │   ├── normalise.py            # Vectorized name and address normalisation
│       │   ├── blocking.py             # Two-way TF-IDF cosine blocking & indexing
│       │   ├── candidates.py           # HistGradientBoosting candidate pruner
│       │   ├── features.py             # 82 country-agnostic pair features
│       │   ├── two_stage.py            # Two-stage LightGBM pipeline
│       │   ├── cross_encoder_train.py  # GPU fine-tuning for multilingual transformer
│       │   ├── cross_encoder_score.py  # Borderline pair re-scoring & blending
│       │   ├── selection.py            # Bipartite matching & F0.5 optimization
│       │   └── metric.py               # Macro F0.5 exact challenge metric
│       ├── tests/                      # 99+ unit and integration tests
│       └── requirements.txt            # Pinned dependencies
├── docs/                               # EDA findings, normalisation rules, and notes
├── submissions/                        # Submission archives and templates
├── EXPERIMENTS.md                      # Comprehensive log of all runs, metrics, and ablations
└── Documentation_template.md           # Formal competition technical report
```

---

## 🛠️ Quickstart & Reproduction

### Prerequisites
- Python 3.12+
- CUDA-compatible GPU (recommended for cross-encoder inference; all other steps run fast on CPU)

### Installation
```bash
# Clone repository
git clone https://github.com/MdKaifSardar/amazon-ml-challenge-2026.git
cd amazon-ml-challenge-2026/code/business_entity_resolution

# Create virtual environment & install dependencies
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Running the Pipeline
Place competition TSV datasets in `dataset/train/` and `dataset/test/`:
1. **Convert to Parquet:** `python src/convert_to_parquet.py --tsv-dir ../../dataset --out-dir ../../data_parquet`
2. **Run Normalisation:** `python src/run_normalise.py --data-dir ../../data_parquet --split ../../outputs/eda/g25_split.parquet --out-dir ../../artifacts/normalised`
3. **Generate Candidates:** `python src/run_blocking_job_a.py` and `python src/run_blocking_job_b.py`
4. **Extract Features:** `python src/run_features.py --input ../../artifacts --out-dir ../../artifacts/features --splits train val test --prune val`
5. **Train & Score Model v4:** `python src/run_model_v4.py --input ../../artifacts/model_input --config configs/best_config.json --out-dir ../../artifacts/model_v4`
6. **Cross-Encoder Re-scoring:** `python src/cross_encoder_score.py`
7. **Validate Outputs:** `python ../../utils/validate_submission.py --matching ../../output/matching_results.tsv --candidate ../../output/candidate_pairs.tsv --test-dir ../../dataset/test`

*(Detailed step-by-step commands and hardware configurations are documented in [code/business_entity_resolution/README.md](code/business_entity_resolution/README.md).)*

---

## 📜 Compliance & Open Source Licenses
- Built strictly using open-source libraries: `LightGBM` (MIT), `scikit-learn` (BSD-3), `polars` (MIT), `rapidfuzz` (MIT), `anyascii` (ISC), `PyTorch` (BSD-3), and `transformers` (Apache-2.0).
- Pretrained weights: `distilbert-base-multilingual-cased` (Apache-2.0, ~135M params, strictly within competition parameter caps).
- No external lookups, private APIs, or test-label leakage.
