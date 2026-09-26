"""Job A: Generate candidate pairs for the ~300k training S1 sample.

Uses the frozen blocking v3 setup (state scope, starting union u50: fwd 50, rev 5, rare 20)
and the pre-trained pruner (tau = 0.01, cap = 10) to generate train_candidates.parquet.

Runs India and US sequentially to keep peak physical RAM well within 12 GB.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import resource
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

try:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
except NameError:
    sys.path.insert(0, "/tmp/src")

import blocking
from blocking import FORWARD, NONE, block_country, fill_states, gpu_info, state_maps, texts, token_df
from candidates import add_best, add_list_features, prune, rule, score
from run_blocking_eval import COLS, DEFAULT_TARGET, UNIONS, find, log, to_wide

SEED = 42
T0 = time.time()


def log_mem(msg: str) -> None:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    print(f"[{time.time() - T0:7.1f}s peak {peak:5.1f} GB] {msg}", flush=True)


def process_country(country: str, tr: dict, te: dict, train_s1_ids: set[str],
                    model, truth: pl.DataFrame, out_dir: Path) -> Path:
    log_mem(f"=== Starting {country} ===")
    cc = lambda d: d.filter(pl.col("country") == country)
    train_all = pl.concat([cc(t) for t in tr.values()], how="diagonal_relaxed")
    test_all = pl.concat([cc(t) for t in te.values()], how="diagonal_relaxed")
    maps = state_maps(pl.concat([train_all, test_all], how="diagonal_relaxed"))

    s1 = fill_states(cc(tr[1]), maps)
    pool = fill_states(pl.concat([cc(tr[2]), cc(tr[3])]), maps)
    corpus = {rep: pl.concat([texts(train_all)[rep], texts(test_all)[rep]]) for rep in FORWARD}
    df_counts = token_df(pl.concat([train_all["core_name"], test_all["core_name"]]))

    wanted = s1["entity_id"].is_in(list(train_s1_ids)).to_numpy()
    log_mem(f"{country}: {wanted.sum():,} train queries of {s1.height:,} S1; pool {pool.height:,} records")

    qmask = wanted
    tim: list[dict] = []
    long, _ = block_country(s1, pool, qmask, corpus, df_counts, k=50, m=5,
                            scopes=("state",), log=log_mem, gpu_check=False, timing=tim,
                            reverse_keep=wanted)

    long = long.with_columns(pl.Series("s1", s1["entity_id"].to_numpy()[long["qi"].to_numpy()]),
                             pl.Series("cand", pool["entity_id"].to_numpy()[long["pi"].to_numpy()])).drop("qi", "pi")
    log_mem(f"{country}: raw search returned {long.height:,} rows")

    # Pivot to wide format with rank/score per method
    wide = to_wide(long)
    del long

    # Filter to starting union u50 (fwd 50, rev 5, rare 20)
    ukw = UNIONS["u50"]
    u50_cand = add_list_features(wide.filter((pl.col("scope") == "state") & rule(**ukw)))
    del wide
    log_mem(f"{country}: u50 starting list has {u50_cand.height:,} candidates")

    # Attributes for pruner scoring
    s1_attr = s1.select("entity_id", "core_name", "addr_norm", "city", "state")
    pool_attr = pool.select("entity_id", "core_name", "addr_norm", "city", "state")
    del s1, pool, corpus, df_counts, train_all, test_all, maps

    # Score candidates using pruner model
    p = score(model, u50_cand, s1_attr, pool_attr)
    u50_cand = u50_cand.with_columns(pl.Series("p_u50", p))

    # Apply frozen operating point: tau = 0.01, cap = 10 (ordered by p_u50)
    pruned = prune(u50_cand, p, tau=0.01, cap=10)
    del u50_cand, p, s1_attr, pool_attr

    # Attach is_match ground truth label
    pruned = pruned.join(truth.select("s1", "cand", pl.lit(True).alias("is_match")),
                         on=["s1", "cand"], how="left").with_columns(
        pl.col("is_match").fill_null(False),
        pl.lit(country).alias("country")
    )

    out_file = out_dir / f"train_candidates_{country}.parquet"
    pruned.write_parquet(out_file)
    n_queries = int(wanted.sum())
    recall = pruned["is_match"].sum() / max(truth.filter(pl.col("s1").is_in(list(train_s1_ids))).height, 1)
    log_mem(f"{country} done: {pruned.height:,} candidates for {n_queries:,} S1 ({pruned.height / max(n_queries, 1):.2f} avg); recall {recall:.4f}")

    return out_file


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--norm-dir", default="artifacts/normalised")
    ap.add_argument("--split", default="outputs/eda/g25_split.parquet")
    ap.add_argument("--data-dir", default="data_parquet")
    ap.add_argument("--model-dir", default="artifacts/blocking_eval")
    ap.add_argument("--out-dir", default="artifacts/blocking_eval")
    a = ap.parse_args()

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    log_mem(f"Job A starting. CPU cores: {os.cpu_count()}; search device: {gpu_info()}")

    # Load pruner model
    pruner_path = find(Path(a.model_dir), "pruner_state_u50.pkl")
    with open(pruner_path, "rb") as f:
        pruner_obj = pickle.load(f)
        model = pruner_obj["model"]
    log_mem(f"Loaded pruner model from {pruner_path}")

    # Load split and identify the ~300k training S1 queries
    split = pl.read_parquet(find(Path(a.split), "g25_split.parquet"))
    trn_ids = set(split.filter(pl.col("role") == "train")["source1_entity_id"].to_list())
    log_mem(f"Identified {len(trn_ids):,} train S1 query IDs from split")

    # Load normalized tables
    norm = lambda sp, k: pl.read_parquet(find(Path(a.norm_dir), f"{sp}_source{k}.parquet", "normalised"), columns=COLS)
    tr = {k: norm("train", k) for k in (1, 2, 3)}
    te = {k: norm("test", k) for k in (1, 2, 3)}

    # Ground truth
    gt = pl.read_parquet(find(Path(a.data_dir), "train_ground_truth.parquet"))
    truth = (
        gt.filter(pl.col("source1_entity_id").is_in(list(trn_ids)))
        .select(pl.col("source1_entity_id").alias("s1"), pl.col("matched_entity_ids").str.split(",").alias("cand"))
        .explode("cand", empty_as_null=True).filter(pl.col("cand").is_not_null() & (pl.col("cand") != ""))
    )
    log_mem(f"Loaded ground truth: {truth.height:,} true pairs for train queries")

    # Run countries sequentially
    piece_files = []
    for country in ("India", "US"):
        pf = process_country(country, tr, te, trn_ids, model, truth, out)
        piece_files.append(pf)

    # Combine into single train_candidates.parquet
    log_mem("Merging country candidate pieces into train_candidates.parquet...")
    dfs = [pl.read_parquet(f) for f in piece_files]
    combined = pl.concat(dfs).sort("s1", "cand")
    
    final_out = out / "train_candidates.parquet"
    combined.write_parquet(final_out)
    log_mem(f"SUCCESS: wrote {final_out} with {combined.height:,} rows across {combined['s1'].n_unique():,} unique S1")

    # Final summary statistics
    total_true = truth.height
    recalled_true = combined["is_match"].sum()
    overall_recall = recalled_true / max(total_true, 1)
    avg_cands = combined.height / max(len(trn_ids), 1)
    print(f"\n=======================================================", flush=True)
    print(f"JOB A COMPLETE SUMMARY:", flush=True)
    print(f"Total Train S1 Queries: {len(trn_ids):,}", flush=True)
    print(f"Total Candidate Pairs:  {combined.height:,}", flush=True)
    print(f"Average Candidates/S1:  {avg_cands:.2f}", flush=True)
    print(f"True Pairs Recalled:    {recalled_true:,} / {total_true:,}", flush=True)
    print(f"Overall Train Recall:   {overall_recall:.4%}", flush=True)
    print(f"Output File:            {final_out}", flush=True)
    print(f"=======================================================\n", flush=True)

    # Cleanup temporary country pieces
    for f in piece_files:
        if f.exists():
            f.unlink()
    log_mem("Temporary country files cleaned up.")


if __name__ == "__main__":
    main()
