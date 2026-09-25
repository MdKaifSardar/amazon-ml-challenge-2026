"""Build a small, structure-preserving sample of the Parquet data for local smoke tests.

Usage: python src/make_sample.py --in-dir data_parquet --out-dir data_sample --frac 0.01
Train: a random fraction of S1, all of their true matches, and the same fraction of the
S2/S3 records that match nothing (so the distractor ratio is roughly preserved).
Test: the same random fraction of each file. File names match data_parquet/.
"""
import argparse
from pathlib import Path

import polars as pl

SEED = 42


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="data_parquet")
    ap.add_argument("--out-dir", default="data_sample")
    ap.add_argument("--frac", type=float, default=0.01)
    args = ap.parse_args()
    src, dst = Path(args.in_dir), Path(args.out_dir)
    (dst / "train").mkdir(parents=True, exist_ok=True)
    (dst / "test").mkdir(parents=True, exist_ok=True)

    gt = pl.read_parquet(src / "train/train_ground_truth.parquet")
    gt_s = gt.sample(fraction=args.frac, seed=SEED)
    gt_s.write_parquet(dst / "train/train_ground_truth.parquet")
    s1 = pl.read_parquet(src / "train/train_source1.parquet")
    s1.join(gt_s.select(pl.col("source1_entity_id").alias("entity_id")), on="entity_id").write_parquet(
        dst / "train/train_source1.parquet"
    )
    del s1

    ids = pl.col("matched_entity_ids").str.split(",").explode()
    all_matched = gt.select(ids.alias("entity_id")).filter(pl.col("entity_id") != "").unique()
    kept_matched = gt_s.select(ids.alias("entity_id")).filter(pl.col("entity_id") != "").unique()
    for name in ("train_source2", "train_source3"):
        df = pl.read_parquet(src / f"train/{name}.parquet")
        distractors = df.join(all_matched, on="entity_id", how="anti").sample(fraction=args.frac, seed=SEED)
        pl.concat([df.join(kept_matched, on="entity_id"), distractors]).write_parquet(dst / f"train/{name}.parquet")
        del df

    for p in sorted((src / "test").glob("*.parquet")):
        pl.read_parquet(p).sample(fraction=args.frac, seed=SEED).write_parquet(dst / "test" / p.name)

    for p in sorted(dst.glob("*/*.parquet")):
        print(f"{p}: {pl.scan_parquet(p).select(pl.len()).collect().item():,} rows")


if __name__ == "__main__":
    main()
