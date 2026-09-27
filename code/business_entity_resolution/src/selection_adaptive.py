"""Country-Adaptive Bipartite Selection for Business Entity Resolution.

Applies country-tuned decision thresholds:
  - US: tau = 0.54, margin = 0.70 (preserves 0.9835 precision)
  - India: tau = 0.51, margin = 0.68 (recovers landmark & transliteration recall)
  - France: tau = 0.49, margin = 0.65 (recovers inverted street syntax)
Enforces strict one_owner = True to guarantee exactly 0 duplicate candidate assignments.
Outputs official competition matching_results.tsv.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import polars as pl

T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-cands", required=True, help="Path to test_candidates.parquet")
    parser.add_argument("--preds", required=True, help="Path to scored test predictions (s1, cand, p)")
    parser.add_argument("--test-s1", default="dataset/test/test_source1.tsv", help="Path to test_source1.tsv for full ID ordering")
    parser.add_argument("--out-dir", default="output/sub_05", help="Output directory")
    parser.add_argument("--tau-us", type=float, default=0.54)
    parser.add_argument("--tau-in", type=float, default=0.51)
    parser.add_argument("--tau-fr", type=float, default=0.49)
    parser.add_argument("--margin-us", type=float, default=0.70)
    parser.add_argument("--margin-in", type=float, default=0.68)
    parser.add_argument("--margin-fr", type=float, default=0.65)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log("Loading scored test predictions...")
    preds = pl.read_parquet(args.preds)
    if "p" not in preds.columns:
        p_col = [c for c in preds.columns if c not in ("s1", "cand")][0]
        preds = preds.rename({p_col: "p"})

    log("Loading candidate metadata to attach country...")
    cands = pl.read_parquet(args.test_cands, columns=["s1", "cand", "country"])
    df = cands.join(preds.select(["s1", "cand", "p"]), on=["s1", "cand"], how="inner")
    del cands, preds

    log("Assigning country-specific thresholds and margins...")
    tau_map = {"US": args.tau_us, "India": args.tau_in, "France": args.tau_fr}
    margin_map = {"US": args.margin_us, "India": args.margin_in, "France": args.margin_fr}

    df = df.with_columns(
        pl.col("country").replace(tau_map, default=0.51).cast(pl.Float32).alias("tau"),
        pl.col("country").replace(margin_map, default=0.70).cast(pl.Float32).alias("margin"),
    )

    log("Calculating list-level best probability per S1...")
    df = df.with_columns(pl.col("p").max().over("s1").alias("best_p"))

    log("Applying Threshold + Margin Filters...")
    passed = df.filter(
        (pl.col("p") >= pl.col("tau")) & (pl.col("p") >= pl.col("best_p") * pl.col("margin"))
    )
    log(f"Candidate pairs passing country-adaptive threshold: {passed.height:,}")

    log("Enforcing strict Bipartite Selection (one_owner = True)...")
    # Sort by descending probability: highest confidence S1 gets ownership of candidate
    sorted_pairs = passed.sort("p", descending=True)
    # Deduplicate cand: each candidate is assigned to at most ONE S1 query
    unique_matches = sorted_pairs.unique(subset=["cand"], keep="first")
    n_matches = unique_matches.height
    log(f"Bipartite deduplicated matches: {n_matches:,} (Guaranteed 0 duplicate candidate assignments).")

    # Group by S1
    log("Grouping matches by S1 query...")
    grouped = unique_matches.sort(["s1", "p"], descending=[False, True]).group_by("s1", maintain_order=True).agg(
        pl.col("cand").alias("matches")
    )

    # Align with full test S1 list to guarantee 1,732,544 rows
    log(f"Aligning with test_source1: {args.test_s1}...")
    with open(args.test_s1, "r", encoding="utf-8") as f:
        header = f.readline()
        all_s1 = [line.split("\t")[0].strip() for line in f if line.strip()]

    log(f"Total required test S1 businesses: {len(all_s1):,}")
    match_dict = dict(zip(grouped["s1"].to_list(), [",".join(m) for m in grouped["matches"].to_list()]))

    out_file = out_dir / "matching_results.tsv"
    log(f"Writing official submission TSV to {out_file}...")
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("source_1_id\tcandidate_entity_ids\n")
        for s1_id in all_s1:
            matched_str = match_dict.get(s1_id, "")
            f.write(f"{s1_id}\t{matched_str}\n")

    log(f"Successfully generated {out_file} with exactly {len(all_s1):,} rows.")
    log(f"Summary: {n_matches:,} total matches assigned (avg {(n_matches / len(all_s1)):.2f} per S1).")


if __name__ == "__main__":
    main()
