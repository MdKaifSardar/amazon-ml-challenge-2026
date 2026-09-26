"""Pair features for the main model (stream A -> B, table 6 in the README).

One row per candidate pair (s1, cand), in the input order, with columns s1, cand, f_* (float32, NaN = missing).
No label, country, source or language column: France exists only in test, so every feature is country-agnostic.
The record's country is used only to pick the legal-word list, exactly as normalisation does.

Inputs:
- pairs: the candidate table (table 4/5): s1, cand, score_*/rank_* per blocking method, best_* (optional; else
  computed over the given list), p_u50 (optional). Missing blocking columns give all-NaN features, so the schema
  is the same for every split.
- s1_rec / pool_rec: normalised records (table 1) of the S1 and S2/S3 side, passed through prepare_records().
- stats: PoolStats from the WHOLE side (all S1 and all S2+S3 records of train, or of test), no labels.

Groups:
1. name: similarities on full_name, core_name and `nolegal` (legal words removed at ANY position, the agreed fix
   for "Willow LLC Center"); first/last token, extra tokens, IDF-weighted overlap, rarest differing token.
2. legal form: agreement of the edge legal form (`legal`) and of legal words found anywhere in the name.
3. address: fuzzy similarities, city agreement, city found in the other address, number tokens (shared,
   conflicting, first/last, long 5+ digit tokens such as postcodes). No state / département agreement: blocking
   searches within the state (always equal in train), and train has no département (France is test-only).
4. blocking: scores, ranks, gaps to the S1's best, number of methods, pruner score and list position.
5. context: name frequency in the side's files; the pair's value minus the best OTHER candidate of the same S1;
   duplicates of the candidate's name in the S1's list.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

from normalise import _legal_for

FEATURE_VERSION = "feat-v1"
METHODS = ("name", "name_city", "name_addr", "reverse", "rare")
FORWARD = ("name", "name_city", "name_addr")
REC_COLS = ("entity_id", "country", "full_name", "core_name", "legal", "addr_norm", "city", "numbers")
TRAILING_ONLY = {"pra", "li"}  # Indian legal spellings that are legal only as a trailing token (normalisation rule)
NAME_VARIANTS = ("full", "core", "nolegal")
LONG_NUM = 5  # digit tokens this long are postcodes (US ZIP, Indian PIN, French CP) or long house numbers
CONTEXT = ("f_name_core_tsr", "f_name_core_ratio", "f_name_idf_jacc", "f_addr_tsr", "f_addr_c3_jacc", "f_num_jacc")


# ---------------------------------------------------------------------------- records and side statistics

def legal_words(country: str | None) -> dict[str, str]:
    """Legal word -> canonical form for this country, for removal at any position (trailing-only forms excluded)."""
    table = _legal_for((country or "").strip().lower())
    return {w: c for w, c in table.items() if w not in TRAILING_ONLY}


def prepare_records(rec: pl.DataFrame) -> pl.DataFrame:
    """Normalised records -> the columns used here, nulls filled, plus `nolegal` (full_name without legal words at
    any position; falls back to core_name) and `legal_any` (sorted canonical legal words found anywhere)."""
    rec = rec.select(REC_COLS).with_columns(
        *[pl.col(c).fill_null("") for c in ("country", "full_name", "core_name", "legal", "addr_norm")],
        pl.col("numbers").fill_null([]),
    )
    key = pl.col("country").str.to_lowercase().str.strip_chars()
    parts = []
    for k, g in rec.with_columns(key.alias("_k")).group_by("_k", maintain_order=True):
        table = legal_words(k[0])
        words = list(table)
        toks = pl.col("full_name").str.split(" ")
        parts.append(g.with_columns(
            toks.list.eval(pl.element().filter(~pl.element().is_in(words))).list.join(" ").alias("nolegal"),
            toks.list.eval(pl.element().filter(pl.element().is_in(words)).replace_strict(table, default=None))
            .list.unique().list.sort().list.join(" ").alias("legal_any"),
        ))
    out = pl.concat(parts) if parts else rec.with_columns(pl.lit("").alias("nolegal"), pl.lit("").alias("legal_any"), pl.lit("").alias("_k"))
    return out.with_columns(pl.when(pl.col("nolegal") == "").then(pl.col("core_name")).otherwise(pl.col("nolegal")).alias("nolegal")).drop("_k")


@dataclass
class PoolStats:
    """Side statistics (no labels): token document frequency of pool core names, and name counts."""
    n_pool: int
    token_df: dict[str, int]
    pool_name_n: pl.DataFrame  # core_name, n: S2+S3 records with this core_name
    s1_name_n: pl.DataFrame  # core_name, n: S1 records with this core_name


def pool_stats(s1_names: pl.Series, pool_names: pl.Series) -> PoolStats:
    """s1_names / pool_names: core_name of ALL S1 / ALL S2+S3 records of one side (train or test)."""
    pool_names = pool_names.fill_null("")
    df = (pool_names.str.split(" ").list.unique().explode(empty_as_null=False).drop_nulls().to_frame("t")
          .filter(pl.col("t") != "").group_by("t").len())
    count = lambda s: s.fill_null("").to_frame("core_name").group_by("core_name").len("n")
    return PoolStats(len(pool_names), dict(zip(df["t"].to_list(), df["len"].to_list())), count(pool_names), count(s1_names))


# ---------------------------------------------------------------------------- scalar helpers

def _f32(name: str, a) -> pl.Series:
    return pl.Series(name, np.asarray(a, dtype=np.float32)).fill_nan(None)


def _grams(s: str, n: int) -> set[str]:
    if n == 0:
        return set(s.split())
    s = f" {s} "
    return {s[i:i + n] for i in range(len(s) - n + 1)} if len(s) >= n else {s}


def _jaccard(a: list[str], b: list[str], n: int = 0) -> np.ndarray:
    """Token (n=0) or character n-gram Jaccard; NaN when either side is empty."""
    out = np.full(len(a), np.nan, np.float32)
    for i, (s, t) in enumerate(zip(a, b)):
        if s and t:
            A, B = _grams(s, n), _grams(t, n)
            out[i] = len(A & B) / len(A | B)
    return out


def _fuzzy(scorer, a: list[str], b: list[str]) -> np.ndarray:
    """rapidfuzz scorer in 0..1 over aligned pairs; NaN when either side is empty."""
    s = process.cpdist(a, b, scorer=scorer, workers=-1).astype(np.float32)
    if scorer is not JaroWinkler.normalized_similarity:
        s /= 100.0
    empty = (np.array([len(x) for x in a]) == 0) | (np.array([len(x) for x in b]) == 0)
    s[empty] = np.nan
    return s


# ---------------------------------------------------------------------------- feature groups (row level)

def _name_features(q: dict, p: dict, stats: PoolStats) -> list[pl.Series]:
    out = []
    for v, col in zip(NAME_VARIANTS, ("full_name", "core_name", "nolegal")):
        a, b = q[col], p[col]
        out += [_f32(f"f_name_{v}_ratio", _fuzzy(fuzz.ratio, a, b)),
                _f32(f"f_name_{v}_tsr", _fuzzy(fuzz.token_set_ratio, a, b)),
                _f32(f"f_name_{v}_tsort", _fuzzy(fuzz.token_sort_ratio, a, b)),
                _f32(f"f_name_{v}_jw", _fuzzy(JaroWinkler.normalized_similarity, a, b)),
                _f32(f"f_name_{v}_tok_jacc", _jaccard(a, b, 0)),
                _f32(f"f_name_{v}_c3_jacc", _jaccard(a, b, 3))]
    a, b = q["core_name"], p["core_name"]
    out.append(_f32("f_name_core_partial", _fuzzy(fuzz.partial_ratio, a, b)))
    n = len(a)
    first, last, only_q, only_p, len_diff, contains, idf_j, idf_max = (np.full(n, np.nan, np.float32) for _ in range(8))
    w_unknown = math.log(stats.n_pool + 1)
    idf = lambda t: math.log((stats.n_pool + 1) / (stats.token_df.get(t, 0) + 1))
    for i, (s, t) in enumerate(zip(a, b)):
        if not (s and t):
            continue
        ts, tt = s.split(), t.split()
        A, B = set(ts), set(tt)
        first[i], last[i] = ts[0] == tt[0], ts[-1] == tt[-1]
        only_q[i], only_p[i] = len(A - B), len(B - A)
        len_diff[i] = abs(len(s) - len(t))
        contains[i] = f" {s} " in f" {t} " or f" {t} " in f" {s} "
        w = {x: idf(x) for x in A | B}
        union = sum(w.values())
        idf_j[i] = sum(w[x] for x in A & B) / union if union > 0 else 1.0
        diff = [w[x] for x in A ^ B]
        idf_max[i] = max(diff) / w_unknown if diff else 0.0
    out += [_f32("f_name_first_eq", first), _f32("f_name_last_eq", last), _f32("f_name_only_s1", only_q),
            _f32("f_name_only_cand", only_p), _f32("f_name_len_diff", len_diff), _f32("f_name_contains", contains),
            _f32("f_name_idf_jacc", idf_j), _f32("f_name_idf_max_diff", idf_max)]
    return out


def _legal_features(q: dict, p: dict) -> list[pl.Series]:
    out = []
    for tag, col in (("", "legal"), ("_any", "legal_any")):
        a, b = q[col], p[col]
        ea, eb = np.array([not x for x in a]), np.array([not x for x in b])
        same = np.array([x == y for x, y in zip(a, b)], np.float32)
        same[ea | eb] = np.nan
        out += [_f32(f"f_legal{tag}_both_missing", ea & eb), _f32(f"f_legal{tag}_one_missing", ea ^ eb),
                _f32(f"f_legal{tag}_same", same)]
    out.append(_f32("f_legal_tok_jacc", _jaccard(q["legal"], p["legal"], 0)))
    return out


def _eq(a: list, b: list) -> np.ndarray:
    """1/0 equality; NaN when either value is missing (null or empty)."""
    return np.array([np.nan if not x or not y else float(x == y) for x, y in zip(a, b)], np.float32)


def _address_features(q: dict, p: dict) -> list[pl.Series]:
    a = [s.replace(",", " ").replace("  ", " ") for s in q["addr_norm"]]
    b = [s.replace(",", " ").replace("  ", " ") for s in p["addr_norm"]]
    qc, pc = q["city"], p["city"]
    city_fz = _fuzzy(fuzz.ratio, [c or "" for c in qc], [c or "" for c in pc])
    city_in = np.full(len(a), np.nan, np.float32)
    for i, (c1, c2, s, t) in enumerate(zip(qc, pc, a, b)):
        hits = [f" {c} " in f" {addr} " for c, addr in ((c1, t), (c2, s)) if c and addr]
        if hits:
            city_in[i] = any(hits)
    out = [_f32("f_addr_missing", [not s or not t for s, t in zip(a, b)]),
           _f32("f_addr_ratio", _fuzzy(fuzz.ratio, a, b)),
           _f32("f_addr_tsr", _fuzzy(fuzz.token_set_ratio, a, b)),
           _f32("f_addr_tsort", _fuzzy(fuzz.token_sort_ratio, a, b)),
           _f32("f_addr_partial", _fuzzy(fuzz.partial_ratio, a, b)),
           _f32("f_addr_tok_jacc", _jaccard(a, b, 0)),
           _f32("f_addr_c3_jacc", _jaccard(a, b, 3)),
           _f32("f_city_same", _eq(qc, pc)), _f32("f_city_ratio", city_fz), _f32("f_city_in_addr", city_in)]
    n = len(a)
    cols = {k: np.full(n, np.nan, np.float32) for k in
            ("n_s1", "n_cand", "shared", "only_s1", "only_cand", "jacc", "first_eq", "last_eq", "long_shared", "long_conflict")}
    for i, (x, y) in enumerate(zip(q["numbers"], p["numbers"])):
        x, y = list(x or []), list(y or [])
        A, B = set(x), set(y)
        cols["n_s1"][i], cols["n_cand"][i] = len(x), len(y)
        if not (x and y):
            continue
        cols["shared"][i], cols["only_s1"][i], cols["only_cand"][i] = len(A & B), len(A - B), len(B - A)
        cols["jacc"][i] = len(A & B) / len(A | B)
        cols["first_eq"][i], cols["last_eq"][i] = x[0] == y[0], x[-1] == y[-1]
        LA, LB = {t for t in A if len(t) >= LONG_NUM}, {t for t in B if len(t) >= LONG_NUM}
        if LA and LB:
            cols["long_shared"][i] = len(LA & LB)
            cols["long_conflict"][i] = not (LA & LB)
    return out + [_f32(f"f_num_{k}", v) for k, v in cols.items()]


def _blocking_features(x: pl.DataFrame) -> list[pl.Expr]:
    """From the candidate table; absent columns give all-NaN features so every split has the same schema."""
    has = set(x.columns)
    col = lambda c: pl.col(c).cast(pl.Float32) if c in has else pl.lit(None, dtype=pl.Float32)
    best = lambda m: col(f"best_{m}") if f"best_{m}" in has else col(f"score_{m}").max().over("s1")
    out = [*[col(f"score_{m}").alias(f"f_score_{m}") for m in METHODS],
           *[col(f"rank_{m}").alias(f"f_rank_{m}") for m in METHODS],
           *[(best(m) - col(f"score_{m}")).alias(f"f_gap_{m}") for m in FORWARD],
           (col("best_reverse") - col("score_reverse")).alias("f_gap_reverse"),
           pl.sum_horizontal(*[col(f"rank_{m}").is_not_null() for m in METHODS]).cast(pl.Float32).alias("f_n_methods"),
           col("p_u50").alias("f_p_u50"),
           (col("p_u50").max().over("s1") - col("p_u50")).alias("f_p_gap"),
           col("p_u50").rank("ordinal", descending=True).over("s1").cast(pl.Float32).alias("f_list_rank"),
           pl.len().over("s1").cast(pl.Float32).alias("f_list_size")]
    return out


# ---------------------------------------------------------------------------- assembly

def row_features(pairs: pl.DataFrame, s1_rec: pl.DataFrame, pool_rec: pl.DataFrame, stats: PoolStats) -> pl.DataFrame:
    """Per-pair features that need only the pair itself (safe to compute in row chunks). Keeps pairs' order and
    carries q_core / p_core for the context step."""
    cols = ("full_name", "core_name", "nolegal", "legal", "legal_any", "addr_norm", "city", "numbers")
    x = (pairs.select("s1", "cand").with_row_index("_i")
         .join(s1_rec.select(pl.col("entity_id").alias("s1"), *[pl.col(c).alias(f"q_{c}") for c in cols]), on="s1", how="left")
         .join(pool_rec.select(pl.col("entity_id").alias("cand"), *[pl.col(c).alias(f"p_{c}") for c in cols]), on="cand", how="left")
         .sort("_i"))
    missing = x.filter(pl.col("q_core_name").is_null() | pl.col("p_core_name").is_null()).height
    if missing:
        raise ValueError(f"{missing} pairs have no normalised record for s1 or cand")
    q = {c: x[f"q_{c}"].to_list() for c in cols}
    p = {c: x[f"p_{c}"].to_list() for c in cols}
    feats = [*_name_features(q, p, stats), *_legal_features(q, p), *_address_features(q, p)]
    return (x.select("s1", "cand", pl.col("q_core_name").alias("q_core"), pl.col("p_core_name").alias("p_core"))
            .with_columns(*feats)
            .join(stats.s1_name_n.rename({"core_name": "q_core", "n": "_s1n"}), on="q_core", how="left", maintain_order="left")
            .join(stats.pool_name_n.rename({"core_name": "q_core", "n": "_pq"}), on="q_core", how="left", maintain_order="left")
            .join(stats.pool_name_n.rename({"core_name": "p_core", "n": "_pp"}), on="p_core", how="left", maintain_order="left")
            .with_columns(*[pl.col(c).fill_null(0).cast(pl.Float32).log1p().alias(f)
                            for c, f in (("_s1n", "f_freq_s1_name_in_s1"), ("_pq", "f_freq_s1_name_in_pool"),
                                         ("_pp", "f_freq_cand_name_in_pool"))])
            .drop("_s1n", "_pq", "_pp"))


def context_features(x: pl.DataFrame) -> pl.DataFrame:
    """Needs every candidate of an S1 in x: value minus the best OTHER candidate of the same S1 (null when the list
    has one candidate), and how many other candidates share this candidate's core name."""
    x = x.with_row_index("_i")
    top = x.group_by("s1").agg(*[pl.col(c).drop_nulls().top_k(2).alias(f"_top_{c}") for c in CONTEXT])
    x = x.join(top, on="s1", how="left").sort("_i")
    other = lambda c: (pl.when(pl.col(c) == pl.col(f"_top_{c}").list.get(0, null_on_oob=True))
                       .then(pl.col(f"_top_{c}").list.get(1, null_on_oob=True))
                       .otherwise(pl.col(f"_top_{c}").list.get(0, null_on_oob=True)))
    return x.with_columns(
        *[(pl.col(c) - other(c)).alias(f"f_ctx_{c[2:]}") for c in CONTEXT],
        (pl.len().over("s1", "p_core") - 1).cast(pl.Float32).alias("f_ctx_same_name_in_list"),
        (pl.col("f_name_core_tsr") >= 1.0).sum().over("s1").cast(pl.Float32).alias("f_ctx_n_name_exact"),
    ).drop("_i", *[f"_top_{c}" for c in CONTEXT])


def build(pairs: pl.DataFrame, s1_rec: pl.DataFrame, pool_rec: pl.DataFrame, stats: PoolStats,
          chunk: int = 1_000_000) -> pl.DataFrame:
    """Table 6 for one split: s1, cand, f_* (float32, NaN = missing), rows in the order of `pairs`.
    `pairs` must hold every candidate of each S1 it contains (context and list features are per S1)."""
    dup = pairs.select("s1", "cand").is_duplicated().sum()
    if dup:
        raise ValueError(f"{dup} duplicated (s1, cand) rows")
    blocking = pairs.select("s1", "cand", *_blocking_features(pairs))
    rows = pl.concat([row_features(pairs[lo:lo + chunk], s1_rec, pool_rec, stats) for lo in range(0, pairs.height, chunk)]
                     or [row_features(pairs.clear(), s1_rec, pool_rec, stats)])
    x = context_features(rows).drop("q_core", "p_core")
    assert x.select("s1", "cand").equals(blocking.select("s1", "cand")), "row order changed"
    x = x.hstack(blocking.drop("s1", "cand"))
    f = [c for c in x.columns if c.startswith("f_")]
    return x.select("s1", "cand", *[pl.col(c).cast(pl.Float32).fill_null(float("nan")) for c in f])


def feature_names(x: pl.DataFrame) -> list[str]:
    return [c for c in x.columns if c.startswith("f_")]
