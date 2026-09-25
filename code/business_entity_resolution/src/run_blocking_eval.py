"""Blocking evaluation on the validation split against the FULL train S2/S3 pool of each country.

Usage: python src/run_blocking_eval.py --norm-dir artifacts/normalised --data-dir data_parquet \
           --split outputs/eda/g25_split.parquet --out-dir artifacts/blocking_eval
--norm-dir / --split may be folders to search (e.g. /kaggle/input). Outputs in --out-dir:
  val_candidates.parquet  one row per (S1, candidate): rank and score per method (null = not found)
  recall_by_method.csv, recall_union_grid.csv, recall_groups.csv, timing.csv, test_estimate.csv, blocking_report.md
Every table has a `scope` column: "state" (searches within state buckets) or "country" (whole country).
Country-scope searches whose projected time would pass --budget-min are skipped and reported.
"""
import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process

try:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
except NameError:  # Kaggle notebook: modules are written to /tmp/src
    sys.path.insert(0, "/tmp/src")
from blocking import FORWARD, NONE, SCOPES, _pairs_evaluated, block_country, buckets, gpu_info, texts, token_df  # noqa: E402

K_GRID = [5, 10, 20, 30, 50, 75, 100]
M_GRID = [0, 1, 2, 3, 5, 10]
METHODS = [*FORWARD, "reverse", "rare"]
COLS = ["entity_id", "country", "business_name", "core_name", "city", "addr_norm", "state"]
NON_LATIN = r"[\p{L}&&[^\p{Latin}]]"
T0 = time.time()


def log(msg: str) -> None:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    print(f"[{time.time() - T0:7.1f}s peak {peak:5.1f} GB] {msg}", flush=True)


def find(root: Path, name: str, parent: str | None = None) -> Path:
    if root.is_file():
        return root
    hits = sorted(p for p in root.rglob(name) if parent is None or p.parent.name == parent)
    if not hits:
        raise FileNotFoundError(f"{name} (parent {parent}) not found under {root}")
    return hits[0]


def md(df: pl.DataFrame, max_rows: int = 60) -> str:
    df = df.head(max_rows)
    fmt = lambda v: f"{v:.4f}" if isinstance(v, float) else str(v)
    rows = [" | ".join(df.columns), " | ".join("---" for _ in df.columns)] + [" | ".join(fmt(v) for v in r) for r in df.iter_rows()]
    return "\n".join(f"| {r} |" for r in rows)


def test_pairs(te: dict, country: str, scopes: list[str]) -> list[dict]:
    """Similarity evaluations a test run would need for this country, per scope (same cost model as timing)."""
    t1 = te[1].filter(pl.col("country") == country)
    tp = pl.concat([te[2].filter(pl.col("country") == country), te[3].filter(pl.col("country") == country)])
    out = []
    for sc in scopes:
        tsb, tpb = buckets(t1, tp, sc)
        out.append({"scope": sc, "country": country, "test_s1": t1.height, "test_pool": tp.height,
                    "forward_pairs": _pairs_evaluated(tsb, tpb),
                    "reverse_pairs": _pairs_evaluated(tpb[tpb != NONE], tsb, with_none=False)})
    return out


rates: dict[str, float] = {}  # seconds per similarity evaluation, "forward" / "reverse", measured so far


def measured_rates(timing: list[dict]) -> dict[str, float]:
    out = {}
    for kind in ("forward", "reverse"):
        rows = [t for t in timing if t.get("pairs_evaluated") and not t.get("skipped")
                and (t["method"] == "reverse") == (kind == "reverse")]
        if rows:
            out[kind] = sum(t["search_s"] for t in rows) / sum(t["pairs_evaluated"] for t in rows)
    return out


def over_budget(tim: list[dict]):
    """skip() for block_country: skip a search if its projected time (pairs x measured rate) would pass the
    budget, keeping 20% of the budget for evaluation. State-scope searches are never skipped."""
    def skip(scope: str, method: str, pairs: float) -> bool:
        if scope == "state":
            return False
        r = {**rates, **measured_rates(tim)}.get("reverse" if method == "reverse" else "forward")
        return r is not None and (time.time() - T0) + pairs * r > 0.8 * BUDGET_S
    return skip


BUDGET_S = 600 * 60.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--norm-dir", default="artifacts/normalised")
    ap.add_argument("--data-dir", default="data_parquet")
    ap.add_argument("--split", default="outputs/eda/g25_split.parquet")
    ap.add_argument("--out-dir", default="artifacts/blocking_eval")
    ap.add_argument("--k-max", type=int, default=max(K_GRID))
    ap.add_argument("--m-max", type=int, default=max(M_GRID))
    ap.add_argument("--scopes", nargs="+", default=list(SCOPES), choices=SCOPES)
    ap.add_argument("--budget-min", type=float, default=600, help="wall-clock budget for the searches (minutes)")
    a = ap.parse_args()
    global BUDGET_S
    BUDGET_S = a.budget_min * 60
    log(f"search device: {gpu_info()}; scopes {a.scopes}; budget {a.budget_min:.0f} min")
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    norm = lambda sp, k, cols=COLS: pl.read_parquet(find(Path(a.norm_dir), f"{sp}_source{k}.parquet", "normalised"), columns=cols)
    split = pl.read_parquet(find(Path(a.split), "g25_split.parquet"))
    val_ids = split.filter(pl.col("role") == "val")["source1_entity_id"]
    gt = pl.read_parquet(find(Path(a.data_dir), "train_ground_truth.parquet"))
    truth = (
        gt.filter(pl.col("source1_entity_id").is_in(val_ids.implode()))
        .select(pl.col("source1_entity_id").alias("s1"), pl.col("matched_entity_ids").str.split(",").alias("cand"))
        .explode("cand", empty_as_null=True).filter(pl.col("cand").is_not_null() & (pl.col("cand") != ""))
        .with_columns(pl.col("cand").str.slice(0, 2).alias("msrc"))
    )
    tr = {k: norm("train", k) for k in (1, 2, 3)}
    te = {k: norm("test", k, ["country", "core_name", "city", "addr_norm", "state"]) for k in (1, 2, 3)}
    log(f"loaded; {val_ids.len():,} validation S1, {truth.height:,} validation true pairs")

    cands, timing, test_est, attrs = [], [], [], []
    for country in tr[1]["country"].unique().sort().to_list():
        log(f"country {country}")
        s1 = tr[1].filter(pl.col("country") == country)
        pool = pl.concat([tr[2].filter(pl.col("country") == country), tr[3].filter(pl.col("country") == country)])
        test_all = pl.concat([t.filter(pl.col("country") == country) for t in te.values()], how="diagonal_relaxed")
        train_all = pl.concat([s1, pool])
        corpus = {rep: pl.concat([texts(train_all)[rep], texts(test_all)[rep]]) for rep in FORWARD}
        df_counts = token_df(pl.concat([train_all["core_name"], test_all["core_name"]]))
        qmask = s1["entity_id"].is_in(val_ids.implode()).to_numpy()
        tim: list[dict] = []
        long, _ = block_country(s1, pool, qmask, corpus, df_counts, a.k_max, a.m_max, scopes=tuple(a.scopes),
                                skip=over_budget(tim), log=log, gpu_check=not timing, timing=tim)
        cands.append(long.with_columns(pl.Series("s1", s1["entity_id"].to_numpy()[long["qi"].to_numpy()]),
                                       pl.Series("cand", pool["entity_id"].to_numpy()[long["pi"].to_numpy()])).drop("qi", "pi"))
        timing += [{"country": country, **t} for t in tim]
        rates.update(measured_rates(timing))
        # test-time cost from the test bucket sizes of this country (same cost model as timing)
        test_est += test_pairs(te, country, a.scopes)
        # attributes of validation S1 for group recall
        cnt = pl.concat([train_all["core_name"], test_all["core_name"]]).alias("core_name").value_counts(name="n_core")
        attrs.append(s1.filter(pl.Series(qmask)).select(pl.col("entity_id").alias("s1"), "country", "core_name")
                     .join(cnt, on="core_name", how="left"))
        del s1, pool, test_all, train_all, corpus
        log(f"country {country} done")
    # France etc. (test-only countries) still need a test-time estimate
    for country in sorted(set(te[1]["country"].unique().to_list()) - set(tr[1]["country"].unique().to_list())):
        test_est += test_pairs(te, country, a.scopes)

    long = pl.concat(cands)
    del cands
    wide = long.pivot(on="method", index=["scope", "s1", "cand"], values=["rank", "score"], aggregate_function="min")
    del long
    for mth in METHODS:
        for c in ("rank", "score"):
            if f"{c}_{mth}" not in wide.columns:
                wide = wide.with_columns(pl.lit(None, dtype=pl.Int64 if c == "rank" else pl.Float32).alias(f"{c}_{mth}"))
    wide = wide.join(truth.select("s1", "cand", pl.lit(True).alias("is_match")), on=["s1", "cand"], how="left").with_columns(
        pl.col("is_match").fill_null(False))
    wide.write_parquet(out / "val_candidates.parquet")
    log(f"candidates: {wide.height:,} (scope, S1, cand) rows at k={a.k_max}, m={a.m_max}")

    val_s1 = pl.concat(attrs)
    n_s1 = val_s1.group_by("country").len("n_s1")
    t = truth.join(val_s1.select("s1", "country"), on="s1")
    tim = pl.DataFrame(timing, infer_schema_length=None)
    tim.write_csv(out / "timing.csv")
    skipped = {(r["scope"], r["method"]) for r in tim.filter(pl.col("skipped").fill_null(False)).iter_rows(named=True)} \
        if "skipped" in tim.columns else set()

    def union(k: int, m: int, rare: bool = True) -> pl.Expr:
        e = pl.any_horizontal([pl.col(f"rank_{f}") <= k for f in FORWARD])
        if m:
            e = e | (pl.col("rank_reverse") <= m)
        if rare:
            e = e | pl.col("rank_rare").is_not_null()
        return e.fill_null(False)

    # attributes of validation true pairs for group recall
    matches = pl.concat([tr[2].select("entity_id", "core_name", "business_name", "state"),
                         tr[3].select("entity_id", "core_name", "business_name", "state")]).rename(
        {"entity_id": "cand", "core_name": "m_core", "business_name": "m_name", "state": "m_state"})
    s1_state = tr[1].select(pl.col("entity_id").alias("s1"), pl.col("state").alias("s1_state"))
    g = t.join(val_s1.select("s1", "core_name", "n_core"), on="s1").join(matches, on="cand", how="left").join(s1_state, on="s1")
    g = g.with_columns(pl.Series("tsr", process.cpdist(g["core_name"].to_list(), g["m_core"].fill_null("").to_list(),
                                                       scorer=fuzz.token_set_ratio, workers=-1)))
    groups = {
        "all": pl.lit(True),
        "common name (core_name 5+ in country)": pl.col("n_core") >= 5,
        "low name similarity (tsr < 70)": pl.col("tsr") < 70,
        "Indian-script match record": pl.col("m_name").str.contains(NON_LATIN),
        "state differs (both known)": pl.col("m_state").is_not_null() & (pl.col("m_state") != pl.col("s1_state")),
        "match has no state": pl.col("m_state").is_null(),
    }

    by_method_all, grid_all, grp_all, recs, report = [], [], [], {}, []
    for sc in a.scopes:
        w = wide.filter(pl.col("scope") == sc)

        def evaluate(sel: pl.Expr, label: dict) -> list[dict]:
            found = w.filter(sel).select("s1", "cand")
            hit = t.join(found, on=["s1", "cand"], how="semi").group_by("country").len("hit")
            size = found.join(val_s1.select("s1", "country"), on="s1").group_by("country").len("cands")
            r = (t.group_by("country").len("true").join(hit, on="country", how="left").join(size, on="country", how="left")
                 .join(n_s1, on="country").fill_null(0))
            tot = r.select(pl.lit("ALL").alias("country"), *[pl.col(c).sum() for c in ("true", "hit", "cands", "n_s1")])
            r = pl.concat([r.select(tot.columns), tot])
            return [{"scope": sc, **label, "country": x["country"], "recall": x["hit"] / max(x["true"], 1),
                     "cands_per_s1": x["cands"] / max(x["n_s1"], 1)} for x in r.iter_rows(named=True)]

        rows = []
        for mth in METHODS:
            ks = M_GRID[1:] if mth == "reverse" else ([10**9] if mth == "rare" else K_GRID)
            for k in ks:
                rows += evaluate(pl.col(f"rank_{mth}") <= k, {"method": mth, "k": k if mth != "rare" else None})
        by_method = pl.DataFrame(rows)
        rows = []
        for k in K_GRID:
            for m in M_GRID:
                rows += evaluate(union(k, m), {"k": k, "m": m})
        grid = pl.DataFrame(rows)

        # recommendation: cheapest (k, m) with overall union recall >= 97% (else the best recall)
        overall = grid.filter(pl.col("country") == "ALL").sort("cands_per_s1")
        ok = overall.filter(pl.col("recall") >= 0.97)
        rec = (ok if ok.height else overall.sort("recall", descending=True)).row(0, named=True)
        k_rec, m_rec = rec["k"], rec["m"]
        recs[sc] = {"k": k_rec, "m": m_rec, "recall": rec["recall"], "cands_per_s1": rec["cands_per_s1"]}

        configs = {f"recommended k={k_rec}, m={m_rec}": union(k_rec, m_rec), f"max k={a.k_max}, m={a.m_max}": union(a.k_max, a.m_max),
                   **{f"{mth} only (k={a.k_max if mth != 'reverse' else a.m_max})": pl.col(f"rank_{mth}").is_not_null() for mth in METHODS}}
        rows = []
        for cname, sel in configs.items():
            found = w.filter(sel.fill_null(False)).select("s1", "cand", pl.lit(True).alias("found"))
            gg = g.join(found, on=["s1", "cand"], how="left").with_columns(pl.col("found").fill_null(False))
            for gname, cond in groups.items():
                for country, sub in [("ALL", gg.filter(cond))] + [(c, d) for (c,), d in gg.filter(cond).group_by("country")]:
                    rows.append({"scope": sc, "config": cname, "group": gname, "country": country, "true_pairs": sub.height,
                                 "recall": sub["found"].mean() if sub.height else None})
        grp = pl.DataFrame(rows).sort("config", "group", "country")
        by_method_all.append(by_method); grid_all.append(grid); grp_all.append(grp)
        sk = sorted(m for s_, m in skipped if s_ == sc)
        report += [
            f"# Scope: {sc}\n",
            f"Recommended: **k = {k_rec}, m = {m_rec}**, union recall {rec['recall']:.4f}, {rec['cands_per_s1']:.1f} candidates per S1."
            + (f" Skipped over budget (not in the union): {', '.join(sk)}." if sk else "") + "\n",
            "## Union recall grid (ALL countries)\n", md(overall.sort("k", "m").drop("country", "scope"), 60) + "\n",
            "## Union recall per country at the recommended config\n",
            md(grid.filter((pl.col("k") == k_rec) & (pl.col("m") == m_rec)).drop("scope")) + "\n",
            "## Recall per method (ALL countries)\n", md(by_method.filter(pl.col("country") == "ALL").drop("country", "scope"), 60) + "\n",
            "## Recall by group (ALL countries)\n", md(grp.filter(pl.col("country") == "ALL").drop("country", "scope"), 80) + "\n",
        ]
        log(f"[{sc}] recommended k={k_rec}, m={m_rec}: recall {rec['recall']:.4f}, {rec['cands_per_s1']:.1f} cands/S1")
    pl.concat(by_method_all).write_csv(out / "recall_by_method.csv")
    pl.concat(grid_all).write_csv(out / "recall_union_grid.csv")
    pl.concat(grp_all).write_csv(out / "recall_groups.csv")

    # test-time estimate: seconds per similarity evaluation measured here x test pair counts
    rate = measured_rates(timing)
    fit = tim.filter(pl.col("method").str.ends_with("_fit_transform"))
    fit_rate = fit["search_s"].sum() / max(fit["pool"].sum() + tr[1].height, 1)  # seconds per record per representation
    est = pl.DataFrame(test_est).with_columns(
        (pl.col("forward_pairs") * rate.get("forward", 0.0) * len(FORWARD) / 60).alias("forward_min"),
        (pl.col("reverse_pairs") * rate.get("reverse", 0.0) / 60).alias("reverse_min"),
        ((pl.col("test_s1") + pl.col("test_pool")) * fit_rate / 60).alias("fit_transform_min"),
    ).with_columns(pl.sum_horizontal("forward_min", "reverse_min", "fit_transform_min").alias("total_min"))
    est.write_csv(out / "test_estimate.csv")
    est_tot = est.group_by("scope").agg(pl.col("^.*_min$").sum()).sort("scope")

    report += [
        "# Timing on validation (seconds)\n", md(tim, 80) + "\n",
        f"Measured rate: forward {rate.get('forward', 0) * 1e9:.2f} ns, reverse {rate.get('reverse', 0) * 1e9:.2f} ns per "
        f"similarity evaluation ({gpu_info()}).\n",
        "# Test-time estimate (minutes)\n", md(est) + "\n", md(est_tot) + "\n",
        f"Peak memory of this run (host): {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20:.1f} GB.\n",
    ]
    (out / "blocking_report.md").write_text("\n".join(report))
    (out / "config.json").write_text(json.dumps({"recommended": recs, "k_grid": K_GRID, "m_grid": M_GRID,
                                                 "scopes": a.scopes, "device": gpu_info()}, indent=1))
    log(f"done: {recs}")


if __name__ == "__main__":
    main()
