# Exploratory Data Analysis (EDA) Findings & Benchmarks

**Dataset Analyzed**: Full Dataset (24.2 Million records across Train & Test) + Stratified Sample (44k S1 records with 152k true match pairs).

---

## 1. Global Dataset Health & Completeness

Across the entire dataset (100% full scan using Polars):

| Table | Total Records | Empty / Null Address | Empty Address % | Countries Present |
| :--- | :--- | :--- | :--- | :--- |
| **`train_source1`** | **2,206,821** | 0 | **0.00%** | US (1.32M), India (883k) |
| **`train_source2`** | **5,034,616** | 168,967 | **3.36%** | US (3.02M), India (2.02M) |
| **`train_source3`** | **5,285,603** | 175,916 | **3.33%** | US (3.17M), India (2.12M) |
| **`test_source1`**  | **1,732,544** | 0 | **0.00%** | India (810k), US (663k), France (259k) |
| **`test_source2`**  | **4,887,273** | 129,408 | **2.65%** | India (2.31M), US (1.87M), France (703k) |
| **`test_source3`**  | **5,082,316** | 136,098 | **2.68%** | India (2.41M), US (1.95M), France (732k) |

### Key Takeaways:
1. **Source 1 is Pristine**: `business_name` and `business_address` have **zero** nulls or empty strings in both train and test.
2. **Missing Addresses in S2 & S3**: Between **2.65% and 3.36%** of records in Source 2 and Source 3 have completely blank addresses.
   * *Actionable*: We must include a binary feature `is_target_address_empty = 1` in our model so it falls back to high-confidence name matching when addresses are missing.
3. **Country Separation**:
   * Cross-country matches = **0.00%**.
   * Test adds **France** (~15% of test records). Country is a hard partition boundary.

---

## 2. Legal Suffixes & Domain Patterns by Country

From frequency extraction across business names:

### United States (US):
* **Top Single Words**: `llc`, `inc`, `corp`, `pc`, `group`, `pllc`, `associates`, `center`, `lp`, `partners`, `co`, `corporation`, `company`.
* **Top Two-Word Phrases**: `associates llc`, `associates inc`, `partners llc`, `center llc`, `care llc`, `dds pc`.

### India:
* **Top Single Words**: `limited`, `ltd`, `llp`, `co`, `corp`, `corporation`, `company`, `trust`, `group`, `society`, `enterprises`.
* **Top Two-Word Phrases**: `private limited`, `pvt ltd`, `public limited`, `india limited`, `seva samiti`, `charitable trust`.

### France (Test set):
* **Top Single Words**: `sarl`, `sas`, `eurl`, `sa`, `sasu`, `sci`, `ei`, `club`, `ecole`.
* **Top Two-Word Phrases**: `sarl unipersonnelle`, `societe par`, `actions simplifiee`.

### Domain Name Aliasing:
* Many S2/S3 names are website URLs (e.g. `domain.com`, `brand.co.in`, `company.fr`).
* *Actionable*: Build a normalizer that strips suffixes (`.com`, `.org`, `.net`, `.co.in`, `.fr`, `www.`) and standardizes legal suffixes (`pvt ltd` $\rightarrow$ `private limited`, `ltd` $\rightarrow$ `limited`, `inc` $\rightarrow$ `incorporated`).

---

## 3. Address Anatomy & Number Extraction

| Metric | US | India | France |
| :--- | :--- | :--- | :--- |
| **Contains Digits (%)** | **100.0%** | **91.4%** | **99.4%** |
| **5/6-digit Number Present (%)** | **11.1%** | **0.0%** (omitted in text) | **0.3%** |

### Key Takeaways:
* Full postal codes (ZIP / PIN codes) are often omitted from raw address strings in India and France.
* **However, house numbers / street digits exist in >91% to 100% of all non-empty addresses!**
* *Actionable*: Exact numerical digit extraction (e.g., matching `"85"` in `"85 Wayne Ave"` vs `"85 Wanye Ave"`) is an extremely high-precision discriminator.

---

## 4. Pairwise True Match Behavior (152,904 True Pairs Evaluated)

Comparing reference `S1` against actual ground truth matches from `S2` and `S3`:

| Metric | Measured Value | Practical Meaning |
| :--- | :--- | :--- |
| **Exact Name Match** | **10.70%** | Only ~1 in 10 true matches have exact character-for-character identical names. 89.3% have noise, typos, or abbreviation swaps. |
| **Exact Address Match** | **7.27%** | Over 92% of addresses have component reordering, landmark additions, or abbreviations. |
| **Target Address Empty** | **4.42%** | Matches where address is completely blank. |
| **Average Token Jaccard** | **0.6156** | True matches share an average of 61.6% of word tokens. |

---

## 5. Candidate Blocking Recall Ceiling (Simulated Benchmark)

We tested candidate generation using character 3-gram TF-IDF cosine similarity on the sample:

| Top $K$ Candidates Kept | Recall on True Matches |
| :--- | :--- |
| **Top 10 ($k=10$)** | **84.82%** |
| **Top 15 ($k=15$)** | **87.25%** |
| **Top 20 ($k=20$)** | **88.72%** |
| **Top 30 ($k=30$)** | **90.77%** |

### Key Takeaway:
* Keeping just **top 15–20 candidates** per business captures **~88% of all true matches** while reducing the search space by **99.999%**!
* Adding a secondary fallback key (first word of name + house number) will push the recall ceiling well above **93–95%**.

---

## 6. Concrete Dictionaries Generated for Code Pipeline

### Legal Suffixes Set (`LEGAL_SUFFIXES`):
```python
LEGAL_SUFFIXES = {
    # English / US / International
    "inc", "incorporated", "llc", "corp", "corporation", "ltd", "limited", 
    "co", "company", "pc", "pllc", "lp", "llp", "group", "holdings",
    # India specific
    "pvt ltd", "private limited", "public limited", "trust", "samiti",
    # France specific
    "sarl", "sas", "eurl", "sa", "sasu", "sci", "ei"
}
```

### Address Normalization Map (`ADDR_MAP`):
```python
ADDR_MAP = {
    "rd": "road", "st": "street", "ave": "avenue", "dr": "drive",
    "blvd": "boulevard", "ln": "lane", "ct": "court", "ste": "suite",
    "apt": "apartment", "pkwy": "parkway", "hwy": "highway", "fl": "floor"
}
```

---

## 7. Official Validation Benchmark & Baseline Score

To ensure all teammates evaluate models consistently:

* **Official 5-Fold Stratified Split (`g25_split.parquet`)**:
  * **Validation Set (Fold 0)**: Exactly **99,994 S1 entities** (stratified by country $\times$ match-count bucket).
  * **Training Pool (Folds 1–4)**: **299,728 S1 entities**.
  * Stored in `outputs/eda/g25_split.parquet`.
* **The "All-Empty" Baseline Score**:
  * Predicts 0 matches for all entities.
  * **Macro $F_{0.5}$ Baseline Score**: **`0.0559`** (derived from the ~5.58% singleton rate).
  * Any trained model must significantly beat `0.0559` on the 99,994 validation entities.
