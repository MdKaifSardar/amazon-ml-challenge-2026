"""Pair features for the main model (stream A -> B, table 6 in the README).

One row per candidate pair (s1, cand), in the input order, with columns s1, cand, f_* (float32, NaN = missing).
No label, country or language column: France exists only in test, so every feature is country-agnostic. The
record's country is used only to pick the legal-word list (as normalisation does) and as the key of the counts
and state maps (as blocking does). The one source feature is f_cand_is_s3 (S2 vs S3 candidate).

Inputs:
- pairs: the candidate table (table 4/5): s1, cand, score_*/rank_* per blocking method, best_* (optional; else
  computed over the given list), p_u50 (optional). Missing blocking columns give all-NaN features, so the schema
  is the same for every split.
- s1_rec / pool_rec: normalised records (table 1) of the S1 and S2/S3 side, with missing states filled in
  (fill_record_states), passed through prepare_records().
- stats: SideStats, counts over ALL train + test records (S1, S2 and S3) of each country, no labels. The same
  object serves every split, so a count means the same in train, val and test.

Groups:
1. name: similarities on full_name, core_name and `nolegal` (legal words removed at ANY position, the agreed fix
   for "Willow LLC Center"); first/last token, extra tokens, IDF-weighted overlap, rarest differing token.
2. rare words: core-name tokens that are rare in the country (document frequency <= RARE_DF and 3+ characters,
   the definition of blocking's rare-token search): shared, only on one side, and how rare the rarest shared one is.
3. legal form: agreement of the edge legal form (`legal`) and of legal words found anywhere in the name.
4. address: fuzzy similarities; city and state agreement (missing states filled in from the city, then the
   département, as in blocking); city found in the other address; number tokens (shared, conflicting,
   first/last). No département or postcode feature: on real data they almost never fire (features-v2-sample:
   French S1 never has a département; 5-6 digit postcodes are in ~0.1% of addresses, most such numbers being
   US house numbers). The département still counts through the state filling.
5. blocking: scores, ranks, gaps to the S1's best, number of methods, pruner score and list position. The rare
   method's score/rank are left out (98% NaN, no signal); the rare-word group replaces them.
6. context: name frequency in the country; the pair's value minus the best OTHER candidate of the same S1;
   duplicates of the candidate's name in the S1's list. Every context feature looks only inside one S1's list:
   nothing counts how many S1 lists a candidate appears in (that number depends on which S1 were queried).
7. source: f_cand_is_s3.
"""
from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

from blocking import RARE_DF, fill_states, state_maps
from normalise import _legal_for

FEATURE_VERSION = "feat-v2"
METHODS = ("name", "name_city", "name_addr", "reverse", "rare")
SCORED = ("name", "name_city", "name_addr", "reverse")  # rare score/rank dropped in v2 (see group 5)
FORWARD = ("name", "name_city", "name_addr")
REC_COLS = ("entity_id", "country", "full_name", "core_name", "legal", "addr_norm", "city", "state", "numbers")
TRAILING_ONLY = {"pra", "li"}  # Indian legal spellings that are legal only as a trailing token (normalisation rule)
NAME_VARIANTS = ("full", "core", "nolegal")
RARE_MIN_LEN = 3  # blocking.rare_tokens: rare tokens have df <= RARE_DF and at least 3 characters
CONTEXT = ("f_name_core_tsr", "f_name_core_ratio", "f_name_idf_jacc", "f_addr_tsr", "f_addr_c3_jacc", "f_num_jacc")


# ---------------------------------------------------------------------------- records, states and counts

def legal_words(country: str | None) -> dict[str, str]:
    """Legal word -> canonical form for this country, for removal at any position (trailing-only forms excluded)."""
    table = _legal_for((country or "").strip().lower())
    return {w: c for w, c in table.items() if w not in TRAILING_ONLY}


def country_state_maps(loc: pl.DataFrame) -> dict[str, dict[str, pl.DataFrame]]:
    """Per country, blocking's city -> state and dept -> state maps. loc: country, city, dept, state of ALL train +
    test records (S1, S2, S3), no labels: the same maps blocking used for its search buckets."""
    return {c: state_maps(g) for (c,), g in loc.group_by("country")}


def fill_record_states(rec: pl.DataFrame, maps: dict[str, dict[str, pl.DataFrame]]) -> pl.DataFrame:
    """Missing states filled from city, then département, per country (blocking.fill_states). Row order is not kept."""
    parts = [fill_states(g, maps[c]) if c in maps else g for (c,), g in rec.group_by("country")]
    return pl.concat(parts, how="diagonal_relaxed").drop("state_src", strict=False) if parts else rec


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
class SideStats:
    """Counts over ALL train + test records of each country (S1, S2 and S3), no labels. Keyed by the country string
    exactly as blocking keys its per-country token counts."""
    n: dict[str, int]  # country -> number of records
    token_df: pl.DataFrame  # country, t, df: records whose core_name contains token t
    name_n: pl.DataFrame  # country, core_name, n_s1, n_pool: S1 / S2+S3 records with this core_name


def count_stats(s1: Iterable[pl.DataFrame | pl.LazyFrame], pool: Iterable[pl.DataFrame | pl.LazyFrame]) -> SideStats:
    """s1 / pool: frames (or lazy scans, one per file) with country and core_name of ALL S1 / ALL S2+S3 records of
    train AND test. Pass the same frames for every split."""
    names, toks = [], []
    for col, frames in (("n_s1", s1), ("n_pool", pool)):
        for f in frames:
            f = f.lazy().select(pl.col("country").fill_null(""), pl.col("core_name").fill_null(""))
            names.append(f.group_by("country", "core_name").agg(pl.len().alias(col)).collect())
            toks.append(f.select("country", pl.col("core_name").str.split(" ").list.unique().alias("t"))
                        .explode("t", empty_as_null=True).filter(pl.col("t").is_not_null() & (pl.col("t") != ""))
                        .group_by("country", "t").agg(pl.len().alias("df")).collect())
    empty = {"country": pl.String, "core_name": pl.String, "n_s1": pl.UInt32, "n_pool": pl.UInt32}
    name_n = (pl.concat(names, how="diagonal_relaxed") if names else pl.DataFrame(schema=empty))
    for c in ("n_s1", "n_pool"):
        if c not in name_n.columns:
            name_n = name_n.with_columns(pl.lit(0, pl.UInt32).alias(c))
    name_n = name_n.group_by("country", "core_name").agg(pl.col("n_s1").sum(), pl.col("n_pool").sum())
    token_df = (pl.concat(toks).group_by("country", "t").agg(pl.col("df").sum()) if toks
                else pl.DataFrame(schema={"country": pl.String, "t": pl.String, "df": pl.UInt32}))
    n = name_n.group_by("country").agg((pl.col("n_s1") + pl.col("n_pool")).sum().alias("n"))
    return SideStats(dict(zip(n["country"].to_list(), n["n"].to_list())), token_df, name_n)


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


def _eq(a: list, b: list) -> np.ndarray:
    """1 = same, 0 = different, NaN = missing on either side (null or empty)."""
    return np.array([np.nan if not x or not y else float(x == y) for x, y in zip(a, b)], np.float32)


def _df_lookup(stats: SideStats, country: list[str], names: list[str]) -> dict[tuple[str, str], int]:
    """(country, token) -> document frequency, for the tokens of the given names only."""
    t = (pl.DataFrame({"country": country, "t": pl.Series(names, dtype=pl.String).str.split(" ")})
         .explode("t", empty_as_null=True).filter(pl.col("t").is_not_null() & (pl.col("t") != "")).unique()
         .join(stats.token_df, on=["country", "t"], how="left"))
    return dict(zip(zip(t["country"].to_list(), t["t"].to_list()), t["df"].fill_null(0).to_list()))


# ---------------------------------------------------------------------------- feature groups (row level)

def _name_features(q: dict, p: dict, country: list[str], stats: SideStats) -> list[pl.Series]:
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
    dfl = _df_lookup(stats, country + country, a + b)
    n = len(a)
    first, last, only_q, only_p, len_diff, contains, idf_j, idf_max = (np.full(n, np.nan, np.float32) for _ in range(8))
    r_shared, r_only_q, r_only_p, r_idf = (np.full(n, np.nan, np.float32) for _ in range(4))
    for i, (s, t, c) in enumerate(zip(a, b, country)):
        if not (s and t):
            continue
        ts, tt = s.split(), t.split()
        A, B = set(ts), set(tt)
        first[i], last[i] = ts[0] == tt[0], ts[-1] == tt[-1]
        only_q[i], only_p[i] = len(A - B), len(B - A)
        len_diff[i] = abs(len(s) - len(t))
        contains[i] = f" {s} " in f" {t} " or f" {t} " in f" {s} "
        df = {x: dfl.get((c, x), 0) for x in A | B}
        n_c = stats.n.get(c, 0)
        w_unknown = math.log(n_c + 1) or 1.0
        w = {x: math.log((n_c + 1) / (d + 1)) for x, d in df.items()}
        union = sum(w.values())
        idf_j[i] = sum(w[x] for x in A & B) / union if union > 0 else 1.0
        diff = [w[x] for x in A ^ B]
        idf_max[i] = max(diff) / w_unknown if diff else 0.0
        rare = {x for x, d in df.items() if d <= RARE_DF and len(x) >= RARE_MIN_LEN}
        shared = rare & A & B
        r_shared[i], r_only_q[i], r_only_p[i] = len(shared), len((rare & A) - B), len((rare & B) - A)
        r_idf[i] = max(w[x] for x in shared) / w_unknown if shared else 0.0
    out += [_f32("f_name_first_eq", first), _f32("f_name_last_eq", last), _f32("f_name_only_s1", only_q),
            _f32("f_name_only_cand", only_p), _f32("f_name_len_diff", len_diff), _f32("f_name_contains", contains),
            _f32("f_name_idf_jacc", idf_j), _f32("f_name_idf_max_diff", idf_max),
            _f32("f_rare_shared", r_shared), _f32("f_rare_only_s1", r_only_q), _f32("f_rare_only_cand", r_only_p),
            _f32("f_rare_shared_idf", r_idf)]
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
           _f32("f_city_same", _eq(qc, pc)), _f32("f_city_ratio", city_fz), _f32("f_city_in_addr", city_in),
           _f32("f_state_agree", _eq(q["state"], p["state"]))]
    n = len(a)
    cols = {k: np.full(n, np.nan, np.float32) for k in
            ("n_s1", "n_cand", "shared", "only_s1", "only_cand", "jacc", "first_eq", "last_eq")}
    for i, (x, y) in enumerate(zip(q["numbers"], p["numbers"])):
        x, y = list(x or []), list(y or [])
        A, B = set(x), set(y)
        cols["n_s1"][i], cols["n_cand"][i] = len(x), len(y)
        if not (x and y):
            continue
        cols["shared"][i], cols["only_s1"][i], cols["only_cand"][i] = len(A & B), len(A - B), len(B - A)
        cols["jacc"][i] = len(A & B) / len(A | B)
        cols["first_eq"][i], cols["last_eq"][i] = x[0] == y[0], x[-1] == y[-1]
    return out + [_f32(f"f_num_{k}", v) for k, v in cols.items()]


def _blocking_features(x: pl.DataFrame) -> list[pl.Expr]:
    """From the candidate table; absent columns give all-NaN features so every split has the same schema."""
    has = set(x.columns)
    col = lambda c: pl.col(c).cast(pl.Float32) if c in has else pl.lit(None, dtype=pl.Float32)
    best = lambda m: col(f"best_{m}") if f"best_{m}" in has else col(f"score_{m}").max().over("s1")
    out = [*[col(f"score_{m}").alias(f"f_score_{m}") for m in SCORED],
           *[col(f"rank_{m}").alias(f"f_rank_{m}") for m in SCORED],
           *[(best(m) - col(f"score_{m}")).alias(f"f_gap_{m}") for m in FORWARD],
           (col("best_reverse") - col("score_reverse")).alias("f_gap_reverse"),
           pl.sum_horizontal(*[col(f"rank_{m}").is_not_null() for m in METHODS]).cast(pl.Float32).alias("f_n_methods"),
           col("p_u50").alias("f_p_u50"),
           (col("p_u50").max().over("s1") - col("p_u50")).alias("f_p_gap"),
           col("p_u50").rank("ordinal", descending=True).over("s1").cast(pl.Float32).alias("f_list_rank"),
           pl.len().over("s1").cast(pl.Float32).alias("f_list_size"),
           pl.col("cand").str.starts_with("S3-").cast(pl.Float32).alias("f_cand_is_s3")]
    return out


# ---------------------------------------------------------------------------- assembly

def row_features(pairs: pl.DataFrame, s1_rec: pl.DataFrame, pool_rec: pl.DataFrame, stats: SideStats) -> pl.DataFrame:
    """Per-pair features that need only the pair itself (safe to compute in row chunks). Keeps pairs' order and
    carries q_core / p_core for the context step."""
    cols = ("country", "full_name", "core_name", "nolegal", "legal", "legal_any", "addr_norm", "city", "state", "numbers")
    x = (pairs.select("s1", "cand").with_row_index("_i")
         .join(s1_rec.select(pl.col("entity_id").alias("s1"), *[pl.col(c).alias(f"q_{c}") for c in cols]), on="s1", how="left")
         .join(pool_rec.select(pl.col("entity_id").alias("cand"), *[pl.col(c).alias(f"p_{c}") for c in cols]), on="cand", how="left")
         .sort("_i"))
    missing = x.filter(pl.col("q_core_name").is_null() | pl.col("p_core_name").is_null()).height
    if missing:
        raise ValueError(f"{missing} pairs have no normalised record for s1 or cand")
    q = {c: x[f"q_{c}"].to_list() for c in cols}
    p = {c: x[f"p_{c}"].to_list() for c in cols}
    feats = [*_name_features(q, p, q["country"], stats), *_legal_features(q, p), *_address_features(q, p)]
    names = stats.name_n.select("country", "core_name", "n_s1", "n_pool")
    return (x.select("s1", "cand", pl.col("q_core_name").alias("q_core"), pl.col("p_core_name").alias("p_core"),
                     "q_country", "p_country")
            .with_columns(*feats)
            .join(names.rename({"country": "q_country", "core_name": "q_core", "n_s1": "_s1n", "n_pool": "_pq"}),
                  on=["q_country", "q_core"], how="left", maintain_order="left")
            .join(names.select(pl.col("country").alias("p_country"), pl.col("core_name").alias("p_core"), pl.col("n_pool").alias("_pp")),
                  on=["p_country", "p_core"], how="left", maintain_order="left")
            .with_columns(*[pl.col(c).fill_null(0).cast(pl.Float32).log1p().alias(f)
                            for c, f in (("_s1n", "f_freq_s1_name_in_s1"), ("_pq", "f_freq_s1_name_in_pool"),
                                         ("_pp", "f_freq_cand_name_in_pool"))])
            .drop("_s1n", "_pq", "_pp", "q_country", "p_country"))


def context_features(x: pl.DataFrame) -> pl.DataFrame:
    """Needs every candidate of an S1 in x: value minus the best OTHER candidate of the same S1 (null when the list
    has one candidate), and how many other candidates share this candidate's core name. Only within one S1's list."""
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


def build(pairs: pl.DataFrame, s1_rec: pl.DataFrame, pool_rec: pl.DataFrame, stats: SideStats,
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
