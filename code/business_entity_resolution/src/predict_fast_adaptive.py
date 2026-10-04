"""Fast LightGBM Test Inference with Country-Adaptive Bipartite Selection.

Scores all test candidate pairs with 5 Stage-1 folds + 3 Stage-2 LightGBM seeds in ~7-8 mins.
Saves raw probabilities to scored_test_pairs.parquet.
Applies country-tuned decision thresholds:
  - US: tau = 0.53, margin = 0.70
  - India: tau = 0.51, margin = 0.68
  - France: tau = 0.53, margin = 0.70
Enforces strict bipartite deduplication (one_owner = True).
Writes official matching_results.tsv and candidate_pairs.tsv.
"""
from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

try:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
except NameError:
    sys.path.insert(0, "/tmp/src")
from pipeline import counts, normalised_files, raw_file, records, stage2_frame
from submission import write_submission

T0 = time.time()


def log(msg: str) -> None:
    print(
        f"[{time.time() - T0:7.1f}s peak {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20:5.1f} GB] {msg}",
        flush=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--model-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--tau-us", type=float, default=0.53)
    ap.add_argument("--tau-in", type=float, default=0.51)
    ap.add_argument("--tau-fr", type=float, default=0.53)
    ap.add_argument("--margin-us", type=float, default=0.70)
    ap.add_argument("--margin-in", type=float, default=0.68)
    ap.add_argument("--margin-fr", type=float, default=0.70)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    md_dir = (
        a.model_dir
        if (a.model_dir / "model_config.json").exists()
        else next(a.model_dir.rglob("model_config.json")).parent
    )
    cfg = json.loads((md_dir / "model_config.json").read_text())
    s1m = [
        lgb.Booster(model_file=str(p))
        for p in sorted(md_dir.glob("stage1_fold*.txt"))
    ]
    s2m = [
        lgb.Booster(model_file=str(p))
        for p in sorted(md_dir.glob("stage2_lgb_seed*.txt"))
    ]

    c_feats, fs = cfg["stage1_features"], cfg["stage2_features"]
    log(
        f"Fast model: {len(s1m)} stage-1 folds, {len(s2m)} stage-2 LightGBM seeds (Pure C++ inference)"
    )

    s1_all = pl.read_parquet(
        raw_file(a.input, "test_source1.parquet"),
        columns=["entity_id", "country"],
    ).rename({"entity_id": "s1"})
    norm = normalised_files(a.input)
    cnt = counts(norm)
    fpath = raw_file(a.input, "features_test.parquet")

    parts = []
    for country in sorted(s1_all["country"].unique().to_list()):
        fx = (
            pl.scan_parquet(fpath)
            .join(
                s1_all.filter(pl.col("country") == country)
                .lazy()
                .select("s1"),
                on="s1",
                how="semi",
            )
            .collect()
            .sort("s1", "cand")
        )
        if not fx.height:
            continue
        p1 = np.mean([b.predict(fx.select(c_feats).to_numpy()) for b in s1m], axis=0)
        s1_rec, c_rec = records(
            norm, "test", fx["s1"].unique(), fx["cand"].unique()
        )
        X = stage2_frame(fx, p1, s1_rec, c_rec, cnt, c_feats)
        m = X.select(fs).to_numpy()
        p_lgb = np.mean([b.predict(m) for b in s2m], axis=0)
        parts.append(
            fx.select("s1", "cand").with_columns(
                pl.lit(country).alias("country"),
                pl.Series("p", p_lgb.astype(np.float32)),
            )
        )
        log(f"{country}: {fx.height:,} pairs scored")
        del fx, X, m, s1_rec, c_rec

    st = pl.concat(parts)
    scored_path = a.out_dir / "scored_test_pairs.parquet"
    log(f"Saving all scored test probabilities to {scored_path}...")
    st.write_parquet(scored_path)

    log("Applying Country-Adaptive Thresholds & Margins...")
    tau_map = {"US": a.tau_us, "India": a.tau_in, "France": a.tau_fr}
    margin_map = {
        "US": a.margin_us,
        "India": a.margin_in,
        "France": a.margin_fr,
    }

    df = st.with_columns(
        pl.col("country")
        .replace(tau_map, default=0.53)
        .cast(pl.Float32)
        .alias("tau"),
        pl.col("country")
        .replace(margin_map, default=0.70)
        .cast(pl.Float32)
        .alias("margin"),
    )
    df = df.with_columns(pl.col("p").max().over("s1").alias("best_p"))

    passed = df.filter(
        (pl.col("p") >= pl.col("tau"))
        & (pl.col("p") >= pl.col("best_p") * pl.col("margin"))
    )
    log(f"Pairs passing country thresholds: {passed.height:,}")

    log("Enforcing Bipartite Selection (one_owner = True)...")
    sorted_pairs = passed.sort(["cand", "p", "s1"], descending=[False, True, False])
    unique_matches = sorted_pairs.unique("cand", keep="first")
    log(f"Final deduplicated bipartite matches: {unique_matches.height:,}")

    log("Loading test candidates for candidate_pairs.tsv...")
    cand = pl.read_parquet(
        raw_file(a.input, "test_candidates.parquet"),
        columns=["s1", "cand", "p_u50"],
    )
    lists = (
        cand.sort(
            ["s1", "p_u50", "cand"],
            descending=[False, True, False],
            nulls_last=True,
        )
        .group_by("s1", maintain_order=True)
        .agg(pl.col("cand"))
    )
    ml = (
        unique_matches.sort(["s1", "p"], descending=[False, True])
        .group_by("s1", maintain_order=True)
        .agg(pl.col("cand"))
    )

    log("Writing official matching_results.tsv and candidate_pairs.tsv...")
    write_submission(
        a.out_dir,
        s1_all["s1"].to_list(),
        dict(zip(ml["s1"].to_list(), ml["cand"].to_list())),
        dict(zip(lists["s1"].to_list(), lists["cand"].to_list())),
    )
    log(f"Wrote {a.out_dir}/matching_results.tsv and candidate_pairs.tsv successfully!")


if __name__ == "__main__":
    main()
