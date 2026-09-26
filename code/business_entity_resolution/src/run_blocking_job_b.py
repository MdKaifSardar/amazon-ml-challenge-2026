"""Job B: Generate candidate pairs for the FULL test set (France, India, US).

Outputs two files:
1. artifacts/blocking_eval/test_candidates.parquet (wide table with scores, ranks, p_u50 for Person 2's features)
2. output/candidate_pairs.tsv (official competition TSV with source_1_id and candidate_id)

Uses the frozen blocking v3 setup (state scope, starting union u50: fwd 50, rev 5, rare 20)
and the pre-trained pruner (tau = 0.01, cap = 10).
Processes France, India, and US sequentially with direct parquet streaming to bound memory.
"""
from __future__ import annotations

import argparse
import gc
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


def process_test_country(country: str, tr: dict, te: dict, model, out_dir: Path) -> Path:
    log_mem(f"=== Starting Test Country: {country} ===")
    cc = lambda d: d.filter(pl.col("country") == country)
    train_all = pl.concat([cc(t) for t in tr.values()], how="diagonal_relaxed")
    test_all = pl.concat([cc(t) for t in te.values()], how="diagonal_relaxed")
    maps = state_maps(pl.concat([train_all, test_all], how="diagonal_relaxed"))

    s1 = fill_states(cc(te[1]), maps)
    pool = fill_states(pl.concat([cc(te[2]), cc(te[3])]), maps)
    corpus = {rep: pl.concat([texts(train_all)[rep], texts(test_all)[rep]]) for rep in FORWARD}
    df_counts = token_df(pl.concat([train_all["core_name"], test_all["core_name"]]))

    n_s1 = s1.height
    n_pool = pool.height
    log_mem(f"{country}: {n_s1:,} test S1 queries; pool {n_pool:,} records")

    qmask = np.ones(n_s1, dtype=bool)
    tim: list[dict] = []
    long, _ = block_country(s1, pool, qmask, corpus, df_counts, k=50, m=5,
                            scopes=("state",), log=log_mem, gpu_check=False, timing=tim,
                            reverse_keep=qmask)

    long = long.with_columns(pl.Series("s1", s1["entity_id"].to_numpy()[long["qi"].to_numpy()]),
                             pl.Series("cand", pool["entity_id"].to_numpy()[long["pi"].to_numpy()])).drop("qi", "pi")
    log_mem(f"{country}: raw search returned {long.height:,} rows")

    # Pivot to wide format with rank/score per method
    wide = to_wide(long)
    del long
    gc.collect()

    # Filter to starting union u50 (fwd 50, rev 5, rare 20)
    ukw = UNIONS["u50"]
    u50_cand = add_list_features(wide.filter((pl.col("scope") == "state") & rule(**ukw)))
    del wide
    gc.collect()
    log_mem(f"{country}: u50 starting list has {u50_cand.height:,} candidates")

    # Attributes for pruner scoring
    s1_attr = s1.select("entity_id", "core_name", "addr_norm", "city", "state")
    pool_attr = pool.select("entity_id", "core_name", "addr_norm", "city", "state")
    del s1, pool, corpus, df_counts, train_all, test_all, maps
    gc.collect()

    # Score candidates using pruner model
    p = score(model, u50_cand, s1_attr, pool_attr)
    u50_cand = u50_cand.with_columns(pl.Series("p_u50", p))

    # Apply frozen operating point: tau = 0.01, cap = 10 (ordered by p_u50)
    pruned = prune(u50_cand, p, tau=0.01, cap=10).with_columns(pl.lit(country).alias("country"))
    del u50_cand, p, s1_attr, pool_attr
    gc.collect()

    out_file = out_dir / f"test_candidates_{country}.parquet"
    pruned.write_parquet(out_file)

    avg_cands = pruned.height / max(n_s1, 1)
    n_at_cap = pruned.group_by("s1").len().filter(pl.col("len") >= 10).height
    log_mem(f"{country} done: {pruned.height:,} candidates for {n_s1:,} test S1 ({avg_cands:.2f} avg); {n_at_cap / max(n_s1, 1):.2%} at cap 10")

    return out_file


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--norm-dir", default="artifacts/normalised")
    ap.add_argument("--model-dir", default="artifacts/blocking_eval")
    ap.add_argument("--out-dir", default="artifacts/blocking_eval")
    ap.add_argument("--tsv-out", default="output/candidate_pairs.tsv")
    a = ap.parse_args()

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tsv_path = Path(a.tsv_out)
    tsv_path.parent.mkdir(parents=True, exist_ok=True)

    log_mem(f"Job B starting. CPU cores: {os.cpu_count()}; search device: {gpu_info()}")

    # Load pruner model
    pruner_path = find(Path(a.model_dir), "pruner_state_u50.pkl")
    with open(pruner_path, "rb") as f:
        pruner_obj = pickle.load(f)
        model = pruner_obj["model"]
    log_mem(f"Loaded pruner model from {pruner_path}")

    # Load normalized tables
    norm = lambda sp, k: pl.read_parquet(find(Path(a.norm_dir), f"{sp}_source{k}.parquet", "normalised"), columns=COLS)
    tr = {k: norm("train", k) for k in (1, 2, 3)}
    te = {k: norm("test", k) for k in (1, 2, 3)}

    countries = te[1]["country"].unique().sort().to_list()
    # Prioritize France first (test-only country) for early QA check
    if "France" in countries:
        countries = ["France"] + [c for c in countries if c != "France"]
    log_mem(f"Test countries to process sequentially: {countries}")

    piece_files = []
    for country in countries:
        pf = process_test_country(country, tr, te, model, out)
        piece_files.append(pf)

    # Combine into single test_candidates.parquet (Wide Table for Features)
    log_mem("Merging country candidate pieces into test_candidates.parquet...")
    dfs = [pl.read_parquet(f) for f in piece_files]
    combined = pl.concat(dfs).sort("s1", "cand")

    wide_out = out / "test_candidates.parquet"
    combined.write_parquet(wide_out)
    log_mem(f"SUCCESS: wrote wide candidate table to {wide_out} with {combined.height:,} rows across {combined['s1'].n_unique():,} unique S1")

    # Write official submission TSV (2 columns: source_1_id \t candidate_id)
    log_mem(f"Writing official submission TSV to {tsv_path}...")
    tsv_df = combined.select(pl.col("s1").alias("source_1_id"), pl.col("cand").alias("candidate_id"))
    tsv_df.write_csv(tsv_path, separator="\t")
    log_mem(f"SUCCESS: wrote official TSV to {tsv_path} with {tsv_df.height:,} pairs")

    # Final summary statistics
    total_test_s1 = te[1].height
    covered_s1 = combined["s1"].n_unique()
    avg_cands = combined.height / max(total_test_s1, 1)

    print(f"\n=======================================================", flush=True)
    print(f"JOB B COMPLETE SUMMARY (TEST SET):", flush=True)
    print(f"Total Test S1 Queries:    {total_test_s1:,}", flush=True)
    print(f"Covered Test S1 Queries:  {covered_s1:,} ({covered_s1 / max(total_test_s1, 1):.2%})", flush=True)
    print(f"Total Candidate Pairs:    {combined.height:,}", flush=True)
    print(f"Average Candidates/S1:    {avg_cands:.2f}", flush=True)
    print(f"Wide Parquet Output:      {wide_out} (for Person 2 features)", flush=True)
    print(f"Official Submission TSV:  {tsv_path} (for competition validator)", flush=True)
    print(f"=======================================================\n", flush=True)

    # Cleanup temporary country pieces
    for f in piece_files:
        if f.exists():
            f.unlink()
    log_mem("Temporary country files cleaned up.")


if __name__ == "__main__":
    main()
