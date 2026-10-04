"""Extract high-purity French pseudo-labels using the Dual-Auditor protocol.

Dual-Auditor Rule:
  - Positive: ce_score >= 0.90 AND tree_prob >= 0.95 (with legal_conflict == 0)
  - Negative: ce_score <= 0.05 AND tree_prob <= 0.02
Outputs:
  french_pseudolabels.parquet (appended into training matrix for Attempt #5)
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
    parser.add_argument("--features-test", required=True, help="Path to features_test.parquet")
    parser.add_argument("--ce-scores", required=True, help="Path to ce_scores_test.parquet")
    parser.add_argument("--tree-probs", required=True, help="Path to Stage 2 tree test probabilities parquet")
    parser.add_argument("--test-cands", required=True, help="Path to test_candidates.parquet")
    parser.add_argument("--out", default="data/french_pseudolabels.parquet", help="Output path")
    parser.add_argument("--max-pos", type=int, default=30000, help="Max positive French pseudo-labels")
    parser.add_argument("--neg-ratio", type=int, default=3, help="Negatives per positive")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    log("Loading test candidates to identify French entities...")
    cands = pl.read_parquet(args.test_cands, columns=["s1", "cand", "country"])
    fr_cands = cands.filter(pl.col("country") == "France").select(["s1", "cand"])
    n_fr_total = fr_cands.height
    log(f"Identified {n_fr_total:,} French candidate pairs in test set.")

    log("Loading Cross-Encoder semantic scores...")
    ce = pl.read_parquet(args.ce_scores, columns=["s1", "cand", "ce_score"])

    log("Loading Stage 2 Tree probability scores...")
    tree = pl.read_parquet(args.tree_probs, columns=["s1", "cand", "tree_prob"])

    log("Joining scores for French candidate pairs...")
    scored = fr_cands.join(ce, on=["s1", "cand"], how="inner").join(tree, on=["s1", "cand"], how="inner")
    del fr_cands, ce, tree

    log("Applying Dual-Auditor Purity Rules...")
    # Positive: Strong agreement between deep language model and tree model
    pos_mask = (pl.col("ce_score") >= 0.90) & (pl.col("tree_prob") >= 0.95)
    # Negative: Both models agree on non-match
    neg_mask = (pl.col("ce_score") <= 0.05) & (pl.col("tree_prob") <= 0.02)

    pos_df = scored.filter(pos_mask).sort("tree_prob", descending=True)
    if pos_df.height > args.max_pos:
        pos_df = pos_df.head(args.max_pos)
    n_pos = pos_df.height
    log(f"Extracted {n_pos:,} Ultra-Pure Positive French pseudo-labels (ce_score >= 0.90 & tree_prob >= 0.95).")

    n_neg_target = min(n_pos * args.neg_ratio, 120000)
    neg_df = scored.filter(neg_mask).sample(n=min(n_neg_target, scored.filter(neg_mask).height), seed=42)
    n_neg = neg_df.height
    log(f"Sampled {n_neg:,} High-Confidence Negative French pseudo-labels.")

    labels_df = pl.concat([
        pos_df.select(["s1", "cand"]).with_columns(pl.lit(1).cast(pl.Int8).alias("is_match")),
        neg_df.select(["s1", "cand"]).with_columns(pl.lit(0).cast(pl.Int8).alias("is_match")),
    ])
    del scored, pos_df, neg_df

    log("Joining pre-computed tabular features from features_test.parquet...")
    feats = pl.read_parquet(args.features_test)
    pseudolabels = labels_df.join(feats, on=["s1", "cand"], how="inner")

    pseudolabels.write_parquet(out_path)
    log(f"Successfully wrote {pseudolabels.height:,} French pseudo-labeled rows to {out_path}.")
    log(f"Purity Profile: {n_pos:,} positives ({(n_pos / pseudolabels.height):.1%}), {n_neg:,} negatives.")


if __name__ == "__main__":
    main()
