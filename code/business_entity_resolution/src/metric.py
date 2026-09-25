"""Macro F0.5 exactly as the challenge defines it.

Per S1 entity: F0.5 = 1.25*P*R / (0.25*P + R).
- No true matches (singleton): 1.0 if the prediction is empty, else 0.0.
- True matches but empty prediction, or no correct IDs: 0.0.
Macro average over ALL S1 entities in the evaluation set; an S1 missing from the
predictions counts as an empty prediction.
"""
from collections.abc import Iterable, Mapping

import polars as pl

BETA2 = 0.25  # beta = 0.5


def parse_ids(s: str | None) -> set[str]:
    """'S2-1,S3-2' -> {'S2-1', 'S3-2'}; '' or None -> set()."""
    return {x.strip() for x in s.split(",") if x.strip()} if s else set()


def f05(pred: Iterable[str], truth: Iterable[str]) -> float:
    pred, truth = set(pred), set(truth)
    if not truth:
        return 1.0 if not pred else 0.0
    tp = len(pred & truth)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(truth)
    return (1 + BETA2) * p * r / (BETA2 * p + r)


def per_entity(pred: Mapping[str, Iterable[str]], truth: Mapping[str, Iterable[str]]) -> pl.DataFrame:
    """One row per S1 in `truth`: f05, precision, recall (P/R null when undefined)."""
    rows = []
    for s1, t in truth.items():
        p, t = set(pred.get(s1, ())), set(t)
        tp = len(p & t)
        rows.append({
            "s1": s1,
            "f05": f05(p, t),
            "precision": tp / len(p) if p else None,
            "recall": tp / len(t) if t else None,
            "n_true": len(t),
            "n_pred": len(p),
        })
    return pl.DataFrame(rows, schema={"s1": pl.String, "f05": pl.Float64, "precision": pl.Float64,
                                      "recall": pl.Float64, "n_true": pl.Int64, "n_pred": pl.Int64})


def report(pred: Mapping[str, Iterable[str]], truth: Mapping[str, Iterable[str]],
           country: Mapping[str, str] | None = None) -> pl.DataFrame:
    """Macro F0.5 overall, per country, and for singletons vs non-singletons."""
    df = per_entity(pred, truth).with_columns(
        pl.when(pl.col("n_true") == 0).then(pl.lit("singleton")).otherwise(pl.lit("non_singleton")).alias("kind"),
        pl.col("s1").replace_strict(country or {}, default="ALL").alias("country"),
    )
    aggs = [
        pl.len().alias("s1"),
        pl.col("f05").mean().alias("macro_f05"),
        pl.col("precision").mean().alias("mean_precision"),
        pl.col("recall").mean().alias("mean_recall"),
    ]
    parts = [df.select(pl.lit("overall").alias("group"), *aggs)]
    if country:
        parts.append(df.group_by("country").agg(aggs).sort("country").select(pl.format("country={}", "country").alias("group"), *[a.meta.output_name() for a in aggs]))
    parts.append(df.group_by("kind").agg(aggs).sort("kind").select(pl.format("kind={}", "kind").alias("group"), *[a.meta.output_name() for a in aggs]))
    return pl.concat(parts, how="vertical_relaxed")
