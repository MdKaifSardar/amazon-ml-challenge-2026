"""Does normalisation separate true matches from hard non-matches better than raw names?

On a sample of VALIDATION S1 entities (per country), for each match source (S2, S3) and each name
representation (raw = lowercased business_name; full = full_name, canonical legal form kept;
norm = core_name, legal form removed):
- true pairs: name token_set_ratio and char-3gram Jaccard of S1 vs each true match;
- hard non-match: the most name-similar record in the same country and source that is NOT a true match
  (top candidates by fuzz.ratio over the full pool, then the highest token_set_ratio among them). It is
  searched separately in each representation, so each one faces its own hardest impostor;
- AUC of each similarity for true pairs (positives) vs hard non-matches (negatives).
"""
from __future__ import annotations

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process

SEED = 42
TOP = 10     # candidates by fuzz.ratio before picking the hardest by token_set_ratio
CHUNK = 100  # queries per cdist call (100 x 3M uint8 = 300 MB)


def raw_name(e: pl.Expr) -> pl.Expr:
    return e.str.to_lowercase().str.replace_all(r"\s+", " ").str.strip_chars()


def jaccard3(a: str, b: str) -> float:
    ga = {a[i:i + 3] for i in range(max(len(a) - 2, 1))} if a else set()
    gb = {b[i:i + 3] for i in range(max(len(b) - 2, 1))} if b else set()
    return 100.0 * len(ga & gb) / len(ga | gb) if ga | gb else 0.0


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Mann-Whitney AUC with average ranks for ties: P(score_pos > score_neg) + 0.5 P(tie)."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    s = pl.Series(np.concatenate([pos, neg])).rank("average").to_numpy()
    return float((s[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def hardest_negatives(queries: list[str], true_sets: list[set[str]], choices: list[str], ids: list[str]) -> list[int | None]:
    """Index into choices of the hardest non-match per query."""
    out: list[int | None] = []
    for i in range(0, len(queries), CHUNK):
        q = queries[i:i + CHUNK]
        scores = process.cdist(q, choices, scorer=fuzz.ratio, dtype=np.uint8, workers=-1)
        for j, row in enumerate(scores):
            true = true_sets[i + j]
            k = min(TOP + len(true), len(row) - 1)
            cand = np.argpartition(-row.astype(np.int16), k)[: k + 1]
            cand = [c for c in cand if ids[c] not in true]
            if not cand:
                out.append(None)
                continue
            best = max(cand, key=lambda c: (fuzz.token_set_ratio(q[j], choices[c]), row[c], -c))
            out.append(int(best))
    return out


def effectiveness(s1: pl.DataFrame, pools: dict[str, pl.DataFrame], gt: pl.DataFrame, val_ids: pl.Series,
                  per_country: int = 2000) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """s1: train S1 (entity_id, country, business_name, full_name, core_name); pools: {"S2": df, "S3": df}, same columns.
    Returns (summary, auc, examples)."""
    s1 = s1.filter(pl.col("entity_id").is_in(val_ids.implode()))
    s1 = s1.filter(pl.int_range(pl.len()).shuffle(seed=SEED).over("country") < per_country).sort("entity_id")
    truth = (
        gt.filter(pl.col("source1_entity_id").is_in(s1["entity_id"].implode()))
        .select(pl.col("source1_entity_id").alias("s1"), pl.col("matched_entity_ids").str.split(",").alias("m"))
        .explode("m", empty_as_null=True).filter(pl.col("m").is_not_null() & (pl.col("m") != ""))
    )
    truth_map: dict[str, set[str]] = {}
    for a, b in truth.iter_rows():
        truth_map.setdefault(a, set()).add(b)
    reps = {"raw": raw_name(pl.col("business_name")), "full": pl.col("full_name"), "norm": pl.col("core_name")}
    rows, examples = [], []
    for src, pool_all in pools.items():
        for country in s1["country"].unique().sort().to_list():
            q = s1.filter(pl.col("country") == country)
            pool = pool_all.filter(pl.col("country") == country)
            ids = pool["entity_id"].to_list()
            by_id = {e: i for i, e in enumerate(ids)}
            for rep, expr in reps.items():
                qs = q.select(expr.alias("x"))["x"].to_list()
                ch = pool.select(expr.alias("x"))["x"].to_list()
                sets = [truth_map.get(e, set()) for e in q["entity_id"]]
                # true pairs (only those whose match is in this source)
                for e, qx, st in zip(q["entity_id"], qs, sets):
                    for m in st:
                        if m in by_id:
                            y = ch[by_id[m]]
                            rows.append((country, src, rep, "true", e, m, fuzz.token_set_ratio(qx, y), jaccard3(qx, y)))
                negs = hardest_negatives(qs, sets, ch, ids)
                for e, qx, n in zip(q["entity_id"], qs, negs):
                    if n is not None:
                        y = ch[n]
                        rows.append((country, src, rep, "hard_neg", e, ids[n], fuzz.token_set_ratio(qx, y), jaccard3(qx, y)))
    res = pl.DataFrame(rows, schema=["country", "src", "rep", "kind", "s1", "other", "tsr", "jac3"], orient="row")

    summary = (
        res.group_by("country", "src", "rep", "kind")
        .agg(
            pl.len().alias("pairs"),
            pl.col("tsr").median().alias("tsr_median"),
            (pl.col("tsr") < 50).mean().alias("tsr_below_50"),
            (pl.col("tsr") < 70).mean().alias("tsr_below_70"),
            pl.col("jac3").median().alias("jac3_median"),
            (pl.col("jac3") < 50).mean().alias("jac3_below_50"),
            (pl.col("jac3") < 70).mean().alias("jac3_below_70"),
        )
        .sort("country", "src", "kind", "rep")
    )
    auc_rows = []
    for (country, src, rep), g in res.group_by("country", "src", "rep"):
        pos, neg = g.filter(pl.col("kind") == "true"), g.filter(pl.col("kind") == "hard_neg")
        auc_rows.append({"country": country, "src": src, "rep": rep, "n_true": pos.height, "n_hard_neg": neg.height,
                         "auc_tsr": auc(pos["tsr"].to_numpy(), neg["tsr"].to_numpy()),
                         "auc_jac3": auc(pos["jac3"].to_numpy(), neg["jac3"].to_numpy())})
    aucs = pl.DataFrame(auc_rows).sort("country", "src", "rep")

    # a few hard non-matches side by side (normalised representation) for eyeballing
    name_of = {**dict(zip(s1["entity_id"], s1["business_name"]))}
    for df in pools.values():
        name_of.update(zip(df["entity_id"], df["business_name"]))
    ex = res.filter((pl.col("rep") == "norm") & (pl.col("kind") == "hard_neg")).sort("tsr", descending=True)
    ex = ex.group_by("country", "src", maintain_order=True).head(8).with_columns(
        pl.col("s1").replace_strict(name_of, default=None).alias("s1_name"),
        pl.col("other").replace_strict(name_of, default=None).alias("hard_neg_name"),
    )
    examples.append(ex)
    return summary, aucs, pl.concat(examples)
