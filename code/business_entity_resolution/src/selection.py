"""Match selection: from scored candidate pairs to the predicted matches per S1, tuned for macro F0.5.

Rules (all tuned on validation, never on test):
- threshold: keep a pair only if its match probability p >= tau;
- one owner: every S2/S3 record matches at most one S1 (EDA: 0 of 7.6M train records match two), so each
  candidate is kept only for the S1 that gives it the highest p;
- margin: keep a pair only if p >= margin * (best p of that S1), which drops weak extra matches next to a strong one.
F0.5 weighs precision twice, so a wrong match costs more than a missed one: the tuned threshold is usually high.

f05_table() is a vectorised version of metric.py (same definition, checked against it in the tests and in
run_model.py): macro F0.5 over ALL evaluated S1, an S1 without predictions scores 1.0 if it has no true match.
"""
from __future__ import annotations

import itertools

import polars as pl

TAUS = [round(0.05 * i, 2) for i in range(2, 20)]  # 0.10 .. 0.95
MARGINS = [0.0, 0.3, 0.5, 0.7]
BETA2 = 0.25


def select(scored: pl.DataFrame, tau: float, margin: float = 0.0, one_owner: bool = True, tau2: float | None = None,
           sim: float = 0.9, owner_delta: float = 0.0) -> pl.DataFrame:
    """scored: s1, cand, p (+ s_sim_name, s_num_eq for the secondary rule). Returns the kept (s1, cand, p) rows.
    tau2: also keep a pair with tau2 <= p < tau if it strongly resembles a confident candidate of the same S1
    (s_sim_name >= sim and the same first house number). owner_delta: with one_owner, an S2/S3 claimed by two S1
    goes to the higher p only if it leads by more than owner_delta; otherwise it is dropped from both."""
    keep = pl.col("p") >= tau
    if tau2 is not None and tau2 < tau:
        keep = keep | ((pl.col("p") >= tau2) & (pl.col("s_sim_name") >= sim) & (pl.col("s_num_eq") >= 1)).fill_null(False)
    x = scored.filter(keep)
    if one_owner:  # ties: the smaller s1 id wins (deterministic)
        x = x.sort(["cand", "p", "s1"], descending=[False, True, False])
        if owner_delta > 0:
            lead = pl.col("p") - pl.col("p").shift(-1).over("cand")
            x = x.with_columns(pl.int_range(pl.len()).over("cand").alias("_r"), lead.fill_null(1.0).alias("_lead"))
            x = x.filter((pl.col("_r") == 0) & (pl.col("_lead") > owner_delta)).drop("_r", "_lead")
        else:
            x = x.unique("cand", keep="first", maintain_order=True)
    if margin > 0:
        x = x.filter(pl.col("p") >= margin * pl.col("p").max().over("s1"))
    return x.select("s1", "cand", "p")


def f05_table(pred: pl.DataFrame, truth: pl.DataFrame, s1s: pl.DataFrame) -> pl.DataFrame:
    """Per evaluated S1: n_pred, n_true, tp, precision, recall, f05. pred / truth: s1, cand; s1s: s1 (+ any
    attribute columns, e.g. country), the S1 to evaluate. S1 missing from pred count as empty predictions."""
    keys = s1s.select("s1")
    pred = pred.join(keys, on="s1", how="semi").select("s1", "cand").unique()
    truth = truth.join(keys, on="s1", how="semi").select("s1", "cand").unique()
    tp = pred.join(truth, on=["s1", "cand"], how="semi").group_by("s1").len("tp")
    x = (s1s.join(pred.group_by("s1").len("n_pred"), on="s1", how="left")
         .join(truth.group_by("s1").len("n_true"), on="s1", how="left")
         .join(tp, on="s1", how="left")
         .with_columns(pl.col("n_pred", "n_true", "tp").fill_null(0)))
    prec = pl.when(pl.col("n_pred") > 0).then(pl.col("tp") / pl.col("n_pred"))
    rec = pl.when(pl.col("n_true") > 0).then(pl.col("tp") / pl.col("n_true"))
    f = (pl.when(pl.col("n_true") == 0).then((pl.col("n_pred") == 0).cast(pl.Float64))
         .when(pl.col("tp") == 0).then(0.0)
         .otherwise((1 + BETA2) * prec * rec / (BETA2 * prec + rec)))
    return x.with_columns(prec.alias("precision"), rec.alias("recall"), f.alias("f05"))


def summary(tab: pl.DataFrame, by: str | None = None) -> pl.DataFrame:
    """Macro F0.5, mean precision / recall (over S1 where defined), per group and overall (+ singletons)."""
    aggs = [pl.len().alias("s1"), pl.col("f05").mean().alias("macro_f05"), pl.col("precision").mean().alias("precision"),
            pl.col("recall").mean().alias("recall"), pl.col("n_pred").mean().alias("pred_per_s1")]
    kind = pl.when(pl.col("n_true") == 0).then(pl.lit("singleton")).otherwise(pl.lit("non_singleton"))
    parts = [tab.select(pl.lit("overall").alias("group"), *aggs),
             tab.with_columns(kind.alias("_k")).group_by("_k").agg(aggs).sort("_k")
             .select(pl.format("kind={}", "_k").alias("group"), *[a.meta.output_name() for a in aggs])]
    if by:
        parts.insert(1, tab.group_by(by).agg(aggs).sort(by).select(pl.format(f"{by}={{}}", by).alias("group"),
                                                                     *[a.meta.output_name() for a in aggs]))
    return pl.concat(parts, how="vertical_relaxed")


def tune(scored: pl.DataFrame, truth: pl.DataFrame, s1s: pl.DataFrame, taus=TAUS, margins=MARGINS,
         owners=(True, False)) -> pl.DataFrame:
    """Macro F0.5 on s1s for every (tau, margin, one_owner). Sorted best first."""
    rows = []
    for tau, margin, owner in itertools.product(taus, margins, owners):
        tab = f05_table(select(scored, tau, margin, owner), truth, s1s)
        rows.append({"tau": tau, "margin": margin, "one_owner": owner, "macro_f05": tab["f05"].mean(),
                     "pred_per_s1": tab["n_pred"].mean()})
    return pl.DataFrame(rows).sort("macro_f05", descending=True)
