"""All-empty baseline: score it on the validation split and write the all-empty test submission.

Usage: python src/baseline_empty.py --data-dir data_parquet --split outputs/eda/g25_split.parquet --out-dir output
"""
import argparse
from pathlib import Path

import polars as pl

from metric import parse_ids, report
from submission import write_submission


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data_parquet")
    ap.add_argument("--split", default="outputs/eda/g25_split.parquet")
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--report", default="artifacts/baseline_empty_val.csv")
    a = ap.parse_args()
    data = Path(a.data_dir)

    val = pl.read_parquet(a.split).filter(pl.col("role") == "val")
    gt = pl.read_parquet(data / "train/train_ground_truth.parquet").join(
        val.select("source1_entity_id", "country"), on="source1_entity_id"
    )
    truth = {s1: parse_ids(m) for s1, m in zip(gt["source1_entity_id"], gt["matched_entity_ids"])}
    rep = report({}, truth, dict(zip(gt["source1_entity_id"], gt["country"])))
    Path(a.report).parent.mkdir(parents=True, exist_ok=True)
    rep.write_csv(a.report)
    with pl.Config(tbl_rows=20, tbl_cols=10):
        print(f"All-empty baseline on {len(truth):,} validation S1:\n{rep}")

    s1_ids = pl.read_parquet(data / "test/test_source1.parquet", columns=["entity_id"])["entity_id"].to_list()
    write_submission(Path(a.out_dir), s1_ids, matches={}, candidates={})
    print(f"wrote all-empty submission for {len(s1_ids):,} test S1 to {a.out_dir}/")


if __name__ == "__main__":
    main()
