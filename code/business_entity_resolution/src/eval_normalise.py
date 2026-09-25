"""Learn state aliases from train pairs and measure same-city / same-state agreement of true pairs,
before (EDA heuristic keys) vs after (normalise.py), per country and match source.

Usage: python src/eval_normalise.py --data-dir data_sample --split outputs/eda/g25_split.parquet \
           --aliases artifacts/state_aliases_norm-v1.json --out artifacts/normalise_eval.csv
Aliases are learned only from pairs of S1 entities that are NOT in the validation split (role != "val").
"""
import argparse
import sys
import time
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
import eda  # noqa: E402  (reuses the EDA heuristic keys as the "before" baseline)
from normalise import NORM_VERSION, Normaliser, learn_state_aliases, save_state_aliases  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data_sample")
    ap.add_argument("--split", default="outputs/eda/g25_split.parquet")
    ap.add_argument("--aliases", default=f"artifacts/state_aliases_{NORM_VERSION}.json")
    ap.add_argument("--out", default="artifacts/normalise_eval.csv")
    a = ap.parse_args()
    t0 = time.time()

    recs, gt = eda.load(Path(a.data_dir))
    recs = eda.add_keys(recs.filter(pl.col("split") == "train"))
    pairs = (
        gt.select(pl.col("source1_entity_id").alias("s1"), pl.col("matched_entity_ids").str.split(",").alias("m"))
        .explode("m")
        .filter(pl.col("m").is_not_null() & (pl.col("m") != ""))
        .with_columns(pl.col("m").str.slice(0, 2).alias("msrc"))
    )
    cols = ["entity_id", "country", "addr", "city_key"]
    s1 = recs.filter(pl.col("src") == "S1").select(cols).rename(lambda c: f"{c}_1")
    mt = recs.filter(pl.col("src") != "S1").select(cols).rename(lambda c: f"{c}_2")
    pairs = pairs.join(s1, left_on="s1", right_on="entity_id_1").join(mt, left_on="m", right_on="entity_id_2")
    print(f"{pairs.height:,} true pairs")

    val = pl.read_parquet(a.split).filter(pl.col("role") == "val").select(pl.col("source1_entity_id").alias("s1"))
    learn = pairs.join(val, on="s1", how="anti")
    print(f"learning aliases from {learn.height:,} pairs (validation S1 excluded: {pairs.height - learn.height:,} pairs)")
    aliases = learn_state_aliases(learn.select(pl.col("country_1").alias("country"), "addr_1", "addr_2"))
    save_state_aliases(aliases, a.aliases, {"learned_from": f"{a.data_dir} train pairs, role != val",
                                            "pairs": learn.height})
    print("learned aliases:", {c: len(m) for c, m in aliases.items()})
    for c, m in aliases.items():
        print(f"  {c}: {dict(list(m.items())[:25])}")

    norm = Normaliser(aliases)
    t1 = time.time()
    na = norm.addresses(recs["addr"], recs["country"])  # city vocab fitted on all train records
    print(f"normalised {recs.height:,} addresses in {time.time() - t1:.1f}s")
    keyed = recs.select("entity_id").with_columns(na.select("city", "state", "dept"))
    after = pairs.join(keyed.rename(lambda c: f"{c}_1"), left_on="s1", right_on="entity_id_1").join(
        keyed.rename(lambda c: f"{c}_2"), left_on="m", right_on="entity_id_2"
    )

    def same(a: str, b: str) -> pl.Expr:
        return pl.col(a).is_not_null() & (pl.col(a) == pl.col(b))

    aggs = [
        pl.len().alias("pairs"),
        same("city_key_1", "city_key_2").mean().alias("same_city_before"),
        same("city_1", "city_2").mean().alias("same_city_after"),
        same("state_1", "state_2").mean().alias("same_state_after"),
        (same("city_1", "city_2") | same("state_1", "state_2")).mean().alias("same_city_or_state_after"),
        pl.col("city_2").is_null().mean().alias("match_city_missing_after"),
        pl.col("state_2").is_null().mean().alias("match_state_missing_after"),
    ]
    res = pl.concat(
        [after.group_by("country_1", "msrc").agg(aggs).sort("country_1", "msrc"),
         after.select(pl.lit("ALL").alias("country_1"), pl.lit("ALL").alias("msrc"), *aggs)],
        how="vertical_relaxed",
    )
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    res.write_csv(a.out)
    with pl.Config(tbl_cols=12, tbl_width_chars=220):
        print(res)
    miss = after.filter(~same("city_1", "city_2") & pl.col("city_1").is_not_null())
    with pl.Config(tbl_rows=15, fmt_str_lengths=60, tbl_width_chars=220):
        print("sample of remaining city mismatches:")
        print(miss.sample(n=min(15, miss.height), seed=0).select("country_1", "msrc", "addr_1", "addr_2", "city_1", "city_2"))
    print(f"total {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
