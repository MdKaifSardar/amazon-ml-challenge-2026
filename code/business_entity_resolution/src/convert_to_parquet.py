"""Convert the organiser TSVs to Parquet (all columns as strings) for fast loading.

Usage: python src/convert_to_parquet.py --tsv-dir dataset --out-dir data_parquet
The TSVs stay the source of truth; this only caches them. Rows are streamed, so
memory stays well below the size of the largest file.
"""
import argparse
from pathlib import Path

import polars as pl


def read_tsv_lazy(path: Path) -> pl.LazyFrame:
    # Standard CSV quoting ("" escapes a quote inside a quoted field), all columns as
    # strings so IDs and postcodes stay intact, and empty cells stay "" rather than null.
    return pl.scan_csv(
        path,
        separator="\t",
        quote_char='"',
        infer_schema=False,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv-dir", default="dataset")
    ap.add_argument("--out-dir", default="data_parquet")
    args = ap.parse_args()

    for split in ("train", "test"):
        out = Path(args.out_dir) / split
        out.mkdir(parents=True, exist_ok=True)
        for tsv in sorted((Path(args.tsv_dir) / split).glob("*.tsv")):
            dst = out / (tsv.stem + ".parquet")
            read_tsv_lazy(tsv).sink_parquet(dst)
            print(f"{tsv} -> {dst}")


if __name__ == "__main__":
    main()
