"""Candidate selection on top of blocking: turn the union of search results into a SMALL list per S1.

The candidate set size is part of the ranking (organiser update), so a list should be close to the S1's true
match count (~3.4 on average). Three stages, all tuned on validation:
1. Selection rules on the blocking table (one row per S1 x candidate, rank_*/score_* per method):
   - reverse: each S2/S3 record keeps its top-1 S1, and its top-2..m S1 only if within `rev_margin` of its best;
   - forward: rank <= fwd_k and score within `fwd_margin` of the S1's best for that method (or >= `fwd_min`);
   - rare-token hits (rank <= rare_k);
   - a cap on the list length per S1, ordered by the best cosine over methods.
2. Optional pruner: a small gradient-boosted model on cheap features (blocking scores, ranks, margins,
   agreement of methods, rapidfuzz on core name and address, same city/state). It is trained on candidates of
   TRAIN S1 only and keeps a candidate if p >= tau, up to a cap. Its output is the submitted candidate set.
3. List statistics: recall, candidates per S1 (average, median, p95, S1 with no candidates included) and
   reduction ratio against all S1 x pool pairs of the country.
"""
from __future__ import annotations

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from sklearn.ensemble import HistGradientBoostingClassifier

SEED = 42
FORWARD = ("name", "name_city", "name_addr")
METHODS = (*FORWARD, "reverse", "rare")
COSINE = (*FORWARD, "reverse")
RANK_NA = 10**6


def add_best(wide: pl.DataFrame) -> pl.DataFrame:
    """best_<method>: the S1's top forward score per method (margin reference). Needs rank/score columns."""
    return wide.with_columns(*[pl.col(f"score_{m}").max().over("scope", "s1").alias(f"best_{m}") for m in FORWARD],
                             pl.max_horizontal(*[pl.col(f"score_{m}") for m in COSINE]).fill_null(0.0).alias("order"))


def rule(rev_m: int = 0, rev_margin: float | None = None, fwd_k: int = 0, fwd_margin: float | None = None,
         fwd_min: float | None = None, rare_k: int = 0) -> pl.Expr:
    """Boolean expression over the blocking table (after add_best) for one selection config."""
    e = pl.lit(False)
    if rev_m:
        near = pl.lit(True) if rev_margin is None else pl.col("score_reverse") >= pl.col("best_reverse") - rev_margin
        e = e | ((pl.col("rank_reverse") == 1) | ((pl.col("rank_reverse") <= rev_m) & near)).fill_null(False)
    if fwd_k:
        for m in FORWARD:
            ok = pl.col(f"rank_{m}") <= fwd_k
            if fwd_margin is not None:
                near = pl.col(f"score_{m}") >= pl.col(f"best_{m}") - fwd_margin
                if fwd_min is not None:
                    near = near | (pl.col(f"score_{m}") >= fwd_min)
                ok = ok & near
            e = e | ok.fill_null(False)
    if rare_k:
        e = e | (pl.col("rank_rare") <= rare_k).fill_null(False)
    return e


def select(wide: pl.DataFrame, expr: pl.Expr, cap: int | None = None, order: str = "order") -> pl.DataFrame:
    """Rows passing expr, then at most `cap` per (scope, S1) by descending `order`."""
    out = wide.filter(expr)
    if cap:
        out = out.filter(pl.col(order).rank("ordinal", descending=True).over("scope", "s1") <= cap)
    return out


# ---------------------------------------------------------------- pruner

def features(cand: pl.DataFrame, s1_attr: pl.DataFrame, pool_attr: pl.DataFrame) -> pl.DataFrame:
    """Cheap per-pair features. cand: blocking rows (after add_best); *_attr: id + core_name, addr_norm, city, state."""
    a = s1_attr.select(pl.col("entity_id").alias("s1"), *[pl.col(c).alias(f"q_{c}") for c in ("core_name", "addr_norm", "city", "state")])
    b = pool_attr.select(pl.col("entity_id").alias("cand"), *[pl.col(c).alias(f"p_{c}") for c in ("core_name", "addr_norm", "city", "state")])
    x = cand.join(a, on="s1", how="left").join(b, on="cand", how="left")
    fz = lambda scorer, c: process.cpdist(x[f"q_{c}"].fill_null("").to_list(), x[f"p_{c}"].fill_null("").to_list(),
                                          scorer=scorer, workers=-1).astype(np.float32)
    x = x.with_columns(
        pl.Series("f_name_tsr", fz(fuzz.token_set_ratio, "core_name")),
        pl.Series("f_name_ratio", fz(fuzz.ratio, "core_name")),
        pl.Series("f_addr_tsr", fz(fuzz.token_set_ratio, "addr_norm")),
        (pl.col("q_city") == pl.col("p_city")).cast(pl.Int8).fill_null(-1).alias("f_same_city"),
        (pl.col("q_state") == pl.col("p_state")).cast(pl.Int8).fill_null(-1).alias("f_same_state"),
    )
    return x


def feature_cols() -> list[str]:
    return [*[f"score_{m}" for m in METHODS], *[f"rank_{m}" for m in METHODS], *[f"gap_{m}" for m in FORWARD],
            "gap_reverse", "n_methods", "list_rank", "list_size",
            "f_name_tsr", "f_name_ratio", "f_addr_tsr", "f_same_city", "f_same_state"]


def matrix(x: pl.DataFrame) -> np.ndarray:
    x = x.with_columns(
        *[(pl.col(f"best_{m}") - pl.col(f"score_{m}")).alias(f"gap_{m}") for m in FORWARD],
        (pl.col("best_reverse") - pl.col("score_reverse")).alias("gap_reverse"),
        pl.sum_horizontal(*[pl.col(f"rank_{m}").is_not_null() for m in METHODS]).alias("n_methods"),
        pl.col("order").rank("ordinal", descending=True).over("scope", "s1").alias("list_rank"),
        pl.len().over("scope", "s1").alias("list_size"),
    ).with_columns(*[pl.col(f"rank_{m}").fill_null(RANK_NA) for m in METHODS])
    return x.select(feature_cols()).to_numpy().astype(np.float32)  # NaN (missing score) is handled by the model


def train_pruner(x: pl.DataFrame, y: np.ndarray) -> HistGradientBoostingClassifier:
    m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=31, min_samples_leaf=50,
                                       l2_regularization=1.0, random_state=SEED)
    return m.fit(matrix(x), y)


def prune(x: pl.DataFrame, p: np.ndarray, tau: float, cap: int | None = None) -> pl.DataFrame:
    """Keep candidates with pruner probability >= tau, at most `cap` per (scope, S1) by descending probability."""
    out = x.with_columns(pl.Series("p_keep", p)).filter(pl.col("p_keep") >= tau)
    if cap:
        out = out.filter(pl.col("p_keep").rank("ordinal", descending=True).over("scope", "s1") <= cap)
    return out


# ---------------------------------------------------------------- list statistics

def list_stats(found: pl.DataFrame, truth: pl.DataFrame, s1s: pl.DataFrame, pool_size: dict[str, int]) -> pl.DataFrame:
    """found: (s1, cand) selected; truth: (s1, cand) true pairs; s1s: (s1, country) of all evaluated S1;
    pool_size: S2+S3 records per country. Per country + ALL: recall, avg/median/p95 candidates per S1 (S1 with
    no candidates count as 0), share of S1 with an empty list, reduction ratio vs all S1 x pool pairs."""
    per_s1 = lambda df, name: df.join(s1s.select("s1"), on="s1", how="semi").group_by("s1").len(name)
    d = (s1s.join(per_s1(found, "n"), on="s1", how="left")
         .join(per_s1(truth, "true"), on="s1", how="left")
         .join(per_s1(truth.join(found.select("s1", "cand"), on=["s1", "cand"], how="semi"), "hit"), on="s1", how="left")
         .with_columns(pl.col("n", "true", "hit").fill_null(0),
                       pl.col("country").replace_strict(pool_size, default=0, return_dtype=pl.Float64).alias("pool")))
    d = pl.concat([d, d.with_columns(pl.lit("ALL").alias("country"))])
    return d.group_by("country").agg(
        (pl.col("hit").sum() / pl.col("true").sum()).alias("recall"),
        pl.col("n").mean().alias("avg"), pl.col("n").median().alias("median"),
        pl.col("n").quantile(0.95, "nearest").alias("p95"), (pl.col("n") == 0).mean().alias("empty_share"),
        (1 - pl.col("n").sum() / pl.col("pool").sum()).alias("reduction_ratio"),
        pl.len().alias("n_s1"), pl.col("true").sum(), pl.col("n").sum().alias("cands"),
    ).sort(pl.col("country") == "ALL", "country")


def pareto(points: pl.DataFrame, x: str = "avg", y: str = "recall") -> pl.DataFrame:
    """Configs not dominated on (fewer candidates, higher recall)."""
    p = points.sort(x, -pl.col(y))
    best, keep = -1.0, []
    for v in p[y].to_list():
        keep.append(v > best)
        best = max(best, v)
    return p.filter(pl.Series(keep))
