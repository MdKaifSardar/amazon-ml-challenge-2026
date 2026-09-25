"""Blocking check on TEST data (no labels), focused on France (test-only country).

Reads test search pieces (run_blocking_eval.py --stage search --side test), the pruners and the chosen operating
point (default ~7 candidates per S1) from a finished evaluation, applies them to test candidates and compares
France with US / India:
  - states: share of S1 and S2/S3 with a detected / inferred (city, dept) / no state; how France gets a state;
  - within-state search groups: number of buckets, the largest groups and their share of all comparisons;
  - candidates per S1 before (starting list) and after the pruner: average, median, p95, empty share, histogram;
  - pruner scores: all candidates and the best candidate per S1; missing-value share of the location features.
Flags anything that suggests French records get worse blocking.

Usage: python src/run_blocking_test_check.py --norm-dir DIR --pieces DIR... --model-dir DIR --out-dir DIR
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import polars as pl

try:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
except NameError:  # Kaggle notebook: modules are written to /tmp/src
    sys.path.insert(0, "/tmp/src")
from candidates import add_list_features, features, prune, rule, score  # noqa: E402
from run_blocking_eval import COLS, DEFAULT_TARGET, UNIONS, filled_states, find, load_pieces, log, md, to_wide  # noqa: E402

HIST = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 7), (8, 10), (11, 20), (21, 10**9)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--norm-dir", default="artifacts/normalised")
    ap.add_argument("--pieces", nargs="+", required=True, help="folders holding piece_test_* files")
    ap.add_argument("--model-dir", required=True, help="folder with config.json and pruner_*.pkl of an evaluation")
    ap.add_argument("--out-dir", default="artifacts/blocking_test_check")
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    norm = lambda sp, k: pl.read_parquet(find(Path(a.norm_dir), f"{sp}_source{k}.parquet", "normalised"), columns=COLS)
    tr = {k: norm("train", k) for k in (1, 2, 3)}
    te = {k: norm("test", k) for k in (1, 2, 3)}
    countries = te[1]["country"].unique().sort().to_list()

    # operating point and pruner from the evaluation (chosen on validation, never on test)
    cfg = json.loads(find(Path(a.model_dir), "config.json").read_text())
    op = next(o for o in cfg["operating_points"] if o["target"] == DEFAULT_TARGET and o["scope"] == "state")
    opc = json.loads(op["config"])
    assert op["family"] == "pruned", f"default operating point is not a pruned list: {op}"
    union = opc["union"]
    with open(find(Path(a.model_dir), f"pruner_state_{union}.pkl"), "rb") as f:
        model = pickle.load(f)["model"]
    log(f"operating point <= {DEFAULT_TARGET}: {op['config']} (validation recall {op['recall']:.4f} at {op['avg']:.2f})")

    long, timing = load_pieces([Path(d) for d in a.pieces], countries, side="test")
    wide = to_wide(long.filter(pl.col("scope") == "state"))
    del long
    tef = filled_states(te, tr, countries)  # same maps as the search (train + test records of the country)
    s1_attr = tef[1].select("entity_id", "core_name", "addr_norm", "city", "state")
    pool_attr = pl.concat([tef[2], tef[3]]).select("entity_id", "core_name", "addr_norm", "city", "state")
    country_of = te[1].select(pl.col("entity_id").alias("s1"), "country")

    base = add_list_features(wide.filter(rule(**UNIONS[union])))
    p = score(model, base, s1_attr, pool_attr)
    base = base.with_columns(pl.Series("p_keep", p)).join(country_of, on="s1")
    kept = prune(base, p, opc["tau"], opc.get("cap"))
    log(f"test candidates: {base.height:,} in the starting list, {kept.height:,} kept")

    # ---- candidates per S1 (S1 with nothing found count as 0)
    tim = pl.DataFrame([t for t in timing if t["method"] == "queried"])
    queried = {r["country"]: r["s1"] for r in tim.iter_rows(named=True)}
    rows, hist = [], []
    for c in countries:
        for stage, df in (("starting list", base), ("after pruner", kept)):
            n = df.filter(pl.col("country") == c).group_by("s1").len()["len"].to_numpy()
            n = np.concatenate([n, np.zeros(max(queried.get(c, 0) - len(n), 0), dtype=np.int64)])
            rows.append({"country": c, "stage": stage, "s1_queried": queried.get(c, 0), "avg": n.mean(),
                         "median": float(np.median(n)), "p95": float(np.percentile(n, 95)), "empty_share": (n == 0).mean()})
            if stage == "after pruner":
                hist.append({"country": c, **{(f"{lo}" if lo == hi else f"{lo}-{hi}" if hi < 10**9 else f"{lo}+"):
                                              float(((n >= lo) & (n <= hi)).mean()) for lo, hi in HIST}})
    sizes, hist = pl.DataFrame(rows), pl.DataFrame(hist)

    # ---- pruner scores and location-feature coverage
    top = base.group_by("country", "s1").agg(pl.col("p_keep").max().alias("p_top"))
    q = lambda col, qq: pl.col(col).quantile(qq, "nearest")
    scores = base.group_by("country").agg(*[q("p_keep", x).alias(f"p_all_q{int(x * 100)}") for x in (0.5, 0.9, 0.99)],
                                          (pl.col("p_keep") >= opc["tau"]).mean().alias("share_ge_tau")).join(
        top.group_by("country").agg(*[q("p_top", x).alias(f"p_top1_q{int(x * 100)}") for x in (0.1, 0.25, 0.5)],
                                    (pl.col("p_top") < opc["tau"]).mean().alias("s1_no_cand_ge_tau")), on="country").sort("country")
    feat = []
    for c in countries:
        smp = base.filter(pl.col("country") == c)
        smp = smp.sample(min(smp.height, 500_000), seed=42)
        x = features(smp, s1_attr, pool_attr)
        feat.append({"country": c, "rows": x.height, "same_state_missing": (x["f_same_state"] == -1).mean(),
                     "same_city_missing": (x["f_same_city"] == -1).mean(), "same_state_true": (x["f_same_state"] == 1).mean(),
                     "same_city_true": (x["f_same_city"] == 1).mean(), "name_tsr_median": x["f_name_tsr"].median(),
                     "addr_tsr_median": x["f_addr_tsr"].median(), "reverse_found": x["rank_reverse"].is_not_null().mean()})
    feat = pl.DataFrame(feat)

    # ---- states and search groups (from the search pieces)
    src = pl.DataFrame([t for t in timing if t["method"] == "state_src"]).group_by("country", "side_table", "src").agg(
        pl.col("n").first()).with_columns((pl.col("n") / pl.col("n").sum().over("country", "side_table")).alias("share"))
    src = src.pivot(on="src", index=["country", "side_table"], values="share").fill_null(0.0).sort("country", "side_table")
    bk = pl.DataFrame([t for t in timing if t["method"] == "bucket"]).unique(["country", "bucket"]).sort(
        "country", "pairs_share", descending=[False, True]).select("country", "bucket", "s1", "pool", "pairs_share")
    bsum = pl.DataFrame([t for t in timing if t["method"] == "bucket_summary"]).unique("country").select(
        "country", "n_buckets", "pairs", "s1_total", "pool_total").with_columns(
        (pl.col("pairs") / (pl.col("s1_total") * pl.col("pool_total"))).alias("pairs_vs_whole_country")).sort("country")

    # ---- flags: France vs the mean of US and India
    flags = []
    ref = lambda df, col: df.filter(pl.col("country") != "France")[col].mean()
    fr = lambda df, col: df.filter(pl.col("country") == "France")[col].item() if "France" in df["country"].to_list() else None
    ks = sizes.filter(pl.col("stage") == "after pruner")
    checks = [("candidates per S1 (after pruner)", fr(ks, "avg"), ref(ks, "avg"), lambda f, r: f < 0.7 * r or f > 1.5 * r),
              ("share of S1 with no candidate", fr(ks, "empty_share"), ref(ks, "empty_share"), lambda f, r: f > max(2 * r, r + 0.02)),
              ("median best pruner score per S1", fr(scores, "p_top1_q50"), ref(scores, "p_top1_q50"), lambda f, r: f < r - 0.15),
              ("same-state feature missing", fr(feat, "same_state_missing"), ref(feat, "same_state_missing"), lambda f, r: f > 2 * r + 0.05),
              ("same-city feature missing", fr(feat, "same_city_missing"), ref(feat, "same_city_missing"), lambda f, r: f > 2 * r + 0.05),
              ("comparisons vs whole-country search", fr(bsum, "pairs_vs_whole_country"), ref(bsum, "pairs_vs_whole_country"),
               lambda f, r: f > 3 * r)]
    for name, f, r, bad in checks:
        if f is not None and r is not None:
            flags.append({"check": name, "France": f, "US/India mean": r, "flag": "FLAG" if bad(f, r) else "ok"})
    flags = pl.DataFrame(flags)

    for name, df in (("test_list_sizes", sizes), ("test_list_hist", hist), ("test_pruner_scores", scores),
                     ("test_feature_coverage", feat), ("test_state_sources", src), ("test_buckets", bk),
                     ("test_bucket_summary", bsum), ("test_flags", flags)):
        df.write_csv(out / f"{name}.csv")
    report = [
        "# Blocking on TEST data: France vs US / India (no labels)\n",
        f"Operating point (chosen on validation): <= {DEFAULT_TARGET} candidates per S1, `{op['config']}`; validation recall "
        f"{op['recall']:.4f} at {op['avg']:.2f} candidates per S1. Test S1 queried per country: {queried}.\n",
        "France gets a state in normalisation from its région (18 names, current and pre-2016 régions, hand-written); "
        "départements are a separate field (95 hand-written names) with no département -> région table. Blocking then "
        "infers a missing state from the city, then the département, using maps learned from records that have both "
        "(>= 20 records, >= 90% agreement; train + test records, no labels).\n",
        "## Flags\n", md(flags) + "\n",
        "## Where states come from (share of records)\n", md(src) + "\n",
        "## Within-state search groups\n", md(bsum) + "\n", md(bk) + "\n",
        "## Candidates per S1\n", md(sizes) + "\n", md(hist) + "\n",
        "## Pruner scores\n", md(scores) + "\n",
        "## Pruner feature coverage (sample of starting-list candidates)\n", md(feat) + "\n",
    ]
    (out / "test_check_report.md").write_text("\n".join(report))
    log("flags: " + "; ".join(f"{r['check']}: {r['flag']}" for r in flags.iter_rows(named=True)))


if __name__ == "__main__":
    main()
