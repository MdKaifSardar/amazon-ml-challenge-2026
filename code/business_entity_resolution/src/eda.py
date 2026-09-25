"""Phase 0 EDA: basic facts, match structure, blocking design, noise, hard cases, test vs train, split.

Usage: python src/eda.py --data-dir data_parquet --out-dir outputs/eda
Reads the seven Parquet files (found by name anywhere under --data-dir), writes one CSV per table,
eda_summary.md, and g25_split.parquet (the S1-grouped train/val split, IDs only) into --out-dir.
Counts use the full data; per-pair string statistics use a random sample of true pairs.
"""
import argparse
import re
import resource
import time
import warnings
from pathlib import Path

import numpy as np
import polars as pl
from anyascii import anyascii
from rapidfuzz import fuzz, process
from rapidfuzz.distance import Levenshtein

warnings.filterwarnings("ignore", category=DeprecationWarning, module="polars")
warnings.filterwarnings("ignore", message=".*empty_as_null.*")

SEED = 42
PAIR_SAMPLE = 300_000  # true pairs used for string-similarity and noise statistics
HARD_NEG_QUERIES = 30  # S1 records searched against the full same-country pool (item 21)
VAL_S1, TRAIN_S1 = 100_000, 300_000  # proposed validation / training S1 sample sizes (item 25)

NON_LATIN = r"[\p{L}&&[^\p{Latin}]]"
OTHER_INDIC = r"[\p{Bengali}\p{Gurmukhi}\p{Gujarati}\p{Oriya}\p{Telugu}\p{Kannada}\p{Malayalam}]"
LANDMARK = r"\b(near|nr|opp|opposite|behind|beside|next to|adjacent|in front of)\b"

OUT: Path
SUMMARY: list[str] = []
KEY: dict[str, str] = {}
T0 = time.time()


# ---------------------------------------------------------------- helpers
def log(msg: str) -> None:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20  # Linux: KiB -> GiB
    print(f"[{time.time() - T0:7.1f}s peak {peak:5.1f} GB] {msg}", flush=True)


def save(df: pl.DataFrame, name: str) -> pl.DataFrame:
    df.write_csv(OUT / f"{name}.csv")
    return df


def md_table(df: pl.DataFrame, max_rows: int = 15) -> str:
    df = df.head(max_rows)
    fmt = lambda v: f"{v:.4g}" if isinstance(v, float) else str(v).replace("|", "/").replace("\n", " ")
    rows = [" | ".join(df.columns), " | ".join("---" for _ in df.columns)]
    rows += [" | ".join(fmt(v) for v in r) for r in df.iter_rows()]
    return "\n".join(f"| {r} |" for r in rows)


def section(title: str) -> None:
    SUMMARY.append(f"\n## {title}\n")


def note(text: str) -> None:
    SUMMARY.append(text + "\n")


def table(df: pl.DataFrame, max_rows: int = 15) -> None:
    SUMMARY.append(md_table(df, max_rows) + "\n")


def pct(x: float) -> str:
    return f"{100 * x:.2f}%"


def norm(e: pl.Expr) -> pl.Expr:
    """Vectorised normalisation: strip Latin accents, lowercase, & -> and, punctuation -> space.
    Non-Latin scripts are kept as they are (transliteration is done by translit() on samples)."""
    return (
        e.str.normalize("NFKD")
        .str.replace_all(r"(\p{Latin})\p{M}+", "$1")
        .str.to_lowercase()
        .str.replace_all("&", " and ")
        .str.replace_all(r"[^\p{L}\p{M}\p{N}]+", " ")
        .str.strip_chars()
    )


def translit(s: str) -> str:
    """Full normalisation incl. transliteration of any script to ASCII (anyascii, ISC licence)."""
    s = anyascii(s).lower().replace("&", " and ")
    return " ".join(re.findall(r"[a-z0-9]+", s))


def find(data_dir: Path, name: str) -> Path:
    hits = sorted(data_dir.rglob(name))
    if not hits:
        raise FileNotFoundError(f"{name} not found under {data_dir}")
    return hits[0]


def qstats(df: pl.DataFrame, by: list[str], cols: list[str]) -> pl.DataFrame:
    aggs = []
    for c in cols:
        aggs += [pl.col(c).mean().round(1).alias(f"{c}_mean")]
        aggs += [pl.col(c).quantile(q).alias(f"{c}_p{int(q * 100)}") for q in (0.05, 0.5, 0.95)]
    return df.group_by(by).agg(aggs).sort(by)


def jaccard3(a: str, b: str) -> float:
    ga = {a[i : i + 3] for i in range(max(len(a) - 2, 1))}
    gb = {b[i : i + 3] for i in range(max(len(b) - 2, 1))}
    return len(ga & gb) / len(ga | gb) if ga | gb else 0.0


# ---------------------------------------------------------------- loading
def load(data_dir: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    frames = []
    for split in ("train", "test"):
        for k in (1, 2, 3):
            df = pl.read_parquet(find(data_dir, f"{split}_source{k}.parquet"))
            frames.append(
                df.with_columns(
                    pl.lit(split).cast(pl.Enum(["train", "test"])).alias("split"),
                    pl.lit(f"S{k}").cast(pl.Enum(["S1", "S2", "S3"])).alias("src"),
                )
            )
    recs = pl.concat(frames).rename({"business_name": "name", "business_address": "addr"})
    recs = recs.with_columns(norm(pl.col("name")).alias("name_n"), norm(pl.col("addr")).alias("addr_n"))
    gt = pl.read_parquet(find(data_dir, "train_ground_truth.parquet"))
    return recs, gt


def add_keys(recs: pl.DataFrame) -> pl.DataFrame:
    """Per record: city_key, postcode, first and rarest name token (all heuristic, country-agnostic).

    - region: an address component that is the last component of >= 0.1% of the country's S1
      addresses and is last in >= 60% of its S1 occurrences (states / regions in practice);
      learned from data, no hard-coded lists.
    - city_key: among non-region components without digits, the one most frequent in the country.
    - postcode: last 5-6 digit token that is not the very first token of the address.
    - rarest token: the name token with the lowest frequency across all names in split+country.
    """
    # Called once per split. Lazy + streaming, joined on a u32 row index, so the exploded
    # component / token tables are never fully materialised.
    recs = recs.with_row_index("rid")
    base = recs.lazy().select("rid", "country", "src", "addr")

    def components(lf: pl.LazyFrame) -> pl.LazyFrame:
        return (
            lf.with_columns(pl.col("addr").str.split(",").alias("c"))
            .with_columns(pl.col("c").list.len().alias("n_c"), pl.int_ranges(pl.col("c").list.len()).alias("pos"))
            .drop("addr")
            .explode("c", "pos")
            .with_columns(norm(pl.col("c")).alias("c"))
            .filter(pl.col("c").is_not_null() & (pl.col("c") != ""))
        )

    s1 = base.filter(pl.col("src") == "S1")
    s1_n = s1.group_by("country").agg(pl.len().alias("n_s1"))
    regions = (
        components(s1)
        .group_by("country", "c")
        .agg((pl.col("pos") == pl.col("n_c") - 1).sum().alias("n_last"), pl.len().alias("n_all"))
        .join(s1_n, on="country")
        # frequent as a last component AND mostly last (cities like "Bordeaux" end some addresses but
        # usually sit in the middle, so they are not regions)
        .filter((pl.col("n_last") >= 0.001 * pl.col("n_s1")) & (pl.col("n_last") >= 0.6 * pl.col("n_all")))
        .select("country", "c")
        .collect(engine="streaming")
    )
    cand = components(base).join(regions.lazy(), on=["country", "c"], how="anti").filter(~pl.col("c").str.contains(r"\d"))
    cfreq = cand.group_by("country", "c").agg(pl.len().alias("cf"))
    city = (
        cand.join(cfreq, on=["country", "c"])
        .group_by("rid")
        .agg(pl.col("c").sort_by("cf", descending=True).first().alias("city_key"))
        .collect(engine="streaming")
    )
    ntok = (
        recs.lazy()
        .select("rid", "country", pl.col("name_n").str.split(" ").alias("t"))
        .explode("t")
        .filter(pl.col("t").is_not_null() & (pl.col("t") != ""))
    )
    tfreq = ntok.group_by("country", "t").agg(pl.len().alias("tf"))
    rare = (
        ntok.join(tfreq, on=["country", "t"])
        .group_by("rid")
        .agg(pl.col("t").sort_by("tf").first().alias("rare_tok"))
        .collect(engine="streaming")
    )
    out = recs.with_columns(
        pl.col("name_n").str.split(" ").list.first().alias("first_tok"),
        pl.col("addr_n").str.replace(r"^\S+\s*", "").str.extract_all(r"\b\d{5,6}\b").list.last().alias("postcode"),
    )
    return out.join(city, on="rid", how="left").join(rare, on="rid", how="left").drop("rid")


# ---------------------------------------------------------------- A. basic facts
def part_a(recs: pl.DataFrame) -> None:
    section("A. Basic facts")
    g = ["split", "src", "country"]
    rows = save(recs.group_by(g).len("rows").sort(g), "a1_rows")
    note("**1. Rows per source and country** (`a1_rows.csv`)")
    table(rows, 30)

    empty = save(
        recs.group_by(g)
        .agg(
            (pl.col("name").str.strip_chars() == "").mean().alias("name_empty"),
            (pl.col("addr").str.strip_chars() == "").mean().alias("addr_empty"),
        )
        .sort(g),
        "a2_empty",
    )
    note("**2. Empty rates** (`a2_empty.csv`)")
    table(empty, 30)

    lens = recs.select(
        *g,
        pl.col("name").str.len_chars().alias("name_chars"),
        pl.col("name_n").str.split(" ").list.len().alias("name_toks"),
        pl.col("addr").str.len_chars().alias("addr_chars"),
        pl.col("addr_n").str.split(" ").list.len().alias("addr_toks"),
    )
    lstats = save(qstats(lens, g, ["name_chars", "name_toks", "addr_chars", "addr_toks"]), "a3_lengths")
    note("**3. Length distributions** (`a3_lengths.csv`; medians shown)")
    table(lstats.select(*g, "name_chars_p50", "name_toks_p50", "addr_chars_p50", "addr_toks_p50"), 30)

    def scripts(col: str) -> list[pl.Expr]:
        c = pl.col(col)
        dev, tam, ind = c.str.contains(r"\p{Devanagari}"), c.str.contains(r"\p{Tamil}"), c.str.contains(OTHER_INDIC)
        nl = c.str.contains(NON_LATIN)
        return [
            nl.mean().alias(f"{col}_nonlatin"),
            dev.mean().alias(f"{col}_devanagari"),
            tam.mean().alias(f"{col}_tamil"),
            ind.mean().alias(f"{col}_other_indic"),
            (nl & ~dev & ~tam & ~ind).mean().alias(f"{col}_other_script"),
        ]

    sc = save(recs.group_by(g).agg(*scripts("name"), *scripts("addr")).sort(g), "a4_scripts")
    note("**4. Non-Latin script shares** (`a4_scripts.csv`)")
    table(sc.select(*g, "name_nonlatin", "name_devanagari", "name_tamil", "name_other_indic", "addr_nonlatin"), 30)


# ---------------------------------------------------------------- B. match structure
def part_b(recs: pl.DataFrame, gt: pl.DataFrame) -> pl.DataFrame:
    section("B. Match structure (train)")
    s1 = recs.filter((pl.col("split") == "train") & (pl.col("src") == "S1")).select("entity_id", "country")
    pairs = (
        gt.select(
            pl.col("source1_entity_id").alias("s1"),
            pl.col("matched_entity_ids").str.split(",").alias("m"),
        )
        .explode("m")
        .filter(pl.col("m").is_not_null() & (pl.col("m") != ""))
        .with_columns(pl.col("m").str.slice(0, 2).alias("msrc"))
    )
    per = (
        gt.select(pl.col("source1_entity_id").alias("s1"))
        .join(
            pairs.group_by("s1").agg(
                pl.len().alias("n"),
                (pl.col("msrc") == "S2").sum().alias("n_s2"),
                (pl.col("msrc") == "S3").sum().alias("n_s3"),
            ),
            on="s1",
            how="left",
        )
        .fill_null(0)
        .join(s1.rename({"entity_id": "s1"}), on="s1", how="left")
    )
    n_missing_country = per["country"].null_count()
    bucket = pl.when(pl.col("n") >= 4).then(pl.lit("4+")).otherwise(pl.col("n").cast(pl.String))
    per = per.with_columns(bucket.alias("bucket"))

    def by_country(df: pl.DataFrame, aggs: list[pl.Expr]) -> pl.DataFrame:
        return pl.concat(
            [df.group_by("country").agg(aggs), df.select(pl.lit("ALL").alias("country"), *aggs)],
            how="vertical_relaxed",
        ).sort("country")

    b5 = (
        pl.concat([per, per.with_columns(pl.lit("ALL").alias("country"))])
        .group_by("country", "bucket")
        .len("s1_count")
        .with_columns((pl.col("s1_count") / pl.col("s1_count").sum().over("country")).alias("share"))
        .sort("country", "bucket")
    )
    save(b5, "b5_match_count")
    sing = per["n"].eq(0).mean()
    KEY["singleton_share"] = pct(sing)
    note(f"**5. Matches per S1** (`b5_match_count.csv`). Singleton share overall: **{pct(sing)}**.")
    table(b5, 20)

    mix = pl.when(pl.col("n") == 0).then(pl.lit("none")).when(pl.col("n_s3") == 0).then(pl.lit("only_S2"))
    mix = mix.when(pl.col("n_s2") == 0).then(pl.lit("only_S3")).otherwise(pl.lit("both"))
    per = per.with_columns(mix.alias("mix"))
    b6 = by_country(
        per,
        [
            *[(pl.col("mix") == m).mean().alias(m) for m in ("none", "only_S2", "only_S3", "both")],
            pl.col("n_s2").mean().alias("avg_s2"),
            pl.col("n_s3").mean().alias("avg_s3"),
            pl.col("n").mean().alias("avg_total"),
        ],
    )
    save(b6, "b6_source_mix")
    note("**6. Source mix per S1** (`b6_source_mix.csv`)")
    table(b6)

    b7 = by_country(
        per,
        [
            (pl.col("n_s2") >= 2).mean().alias("share_2plus_S2"),
            (pl.col("n_s3") >= 2).mean().alias("share_2plus_S3"),
            ((pl.col("n_s2") >= 2) | (pl.col("n_s3") >= 2)).mean().alias("share_2plus_same_source"),
            pl.col("n_s2").max().alias("max_S2"),
            pl.col("n_s3").max().alias("max_S3"),
        ],
    )
    save(b7, "b7_same_source")
    same = per.select(((pl.col("n_s2") >= 2) | (pl.col("n_s3") >= 2)).mean()).item()
    KEY["same_source_multi"] = pct(same)
    note(f"**7. Several matches from the same source**: {pct(same)} of S1 (`b7_same_source.csv`)")
    table(b7)

    reuse = pairs.group_by("m").agg(pl.len().alias("n_s1"), pl.col("s1").alias("s1_list"))
    shared = reuse.filter(pl.col("n_s1") > 1)
    KEY["shared_ids"] = f"{shared.height:,}"
    save(
        shared.head(50).with_columns(pl.col("s1_list").list.join(",")),
        "b8_shared_id_examples",
    )
    note(
        f"**8. S2/S3 IDs in more than one S1 list**: {shared.height:,} of {reuse.height:,} matched IDs "
        f"(max {reuse['n_s1'].max()} S1 per ID). Examples in `b8_shared_id_examples.csv`."
    )

    pool = recs.filter((pl.col("split") == "train") & (pl.col("src") != "S1")).select("entity_id", "src", "country")
    matched_ids = reuse.select(pl.col("m").alias("entity_id"), pl.lit(True).alias("matched"))
    b9 = (
        pool.join(matched_ids, on="entity_id", how="left")
        .group_by("src", "country")
        .agg(pl.len().alias("records"), pl.col("matched").is_null().mean().alias("distractor_share"))
        .sort("src", "country")
    )
    save(b9, "b9_distractors")
    unknown = matched_ids.join(pool, on="entity_id", how="anti").height
    note(f"**9. S2/S3 records that match nothing** (`b9_distractors.csv`). Matched IDs missing from the source files: {unknown}.")
    table(b9)

    b10 = by_country(per, [pl.col("n").eq(0).mean().alias("all_empty_macro_f05"), pl.len().alias("s1")])
    save(b10, "b10_baseline")
    KEY["baseline"] = pct(sing)
    note(f"**10. All-empty baseline** = singleton share (`b10_baseline.csv`). S1 with unknown country: {n_missing_country}.")
    table(b10)
    return per.select("s1", "country", "n", "n_s2", "n_s3", "bucket").join(
        pairs.select("s1", "m", "msrc"), on="s1", how="left"
    )


# ---------------------------------------------------------------- C. blocking design
def part_c(recs: pl.DataFrame, per_pairs: pl.DataFrame) -> pl.DataFrame:
    section("C. Blocking design")
    tr = recs.filter(pl.col("split") == "train")
    cols = ["entity_id", "country", "name", "addr", "name_n", "addr_n", "city_key", "postcode", "first_tok", "rare_tok"]
    a = tr.filter(pl.col("src") == "S1").select(cols).rename(lambda c: f"{c}_1")
    b = tr.filter(pl.col("src") != "S1").select(cols).rename(lambda c: f"{c}_2")
    pairs = (
        per_pairs.filter(pl.col("m").is_not_null())
        .select("s1", "m", "msrc")
        .join(a, left_on="s1", right_on="entity_id_1")
        .join(b, left_on="m", right_on="entity_id_2")
    )
    cross = pairs.filter(pl.col("country_1") != pl.col("country_2"))
    c11 = save(
        pairs.group_by("country_1", "country_2").len("pairs").sort("country_1", "country_2"),
        "c11_cross_country",
    )
    save(cross.select("s1", "m", "name_1", "addr_1", "country_1", "name_2", "addr_2", "country_2").head(50), "c11_cross_examples")
    KEY["cross_country"] = f"{cross.height:,} of {pairs.height:,} pairs ({pct(cross.height / max(pairs.height, 1))})"
    note(f"**11. Cross-country matched pairs**: {KEY['cross_country']} (`c11_cross_country.csv`)")
    table(c11)

    def same(k: str) -> pl.Expr:
        return pl.col(f"{k}_1").is_not_null() & (pl.col(f"{k}_1") == pl.col(f"{k}_2"))

    flags = pairs.with_columns(
        same("city_key").alias("same_city"),
        same("postcode").alias("same_postcode"),
        same("first_tok").alias("same_first_tok"),
        same("rare_tok").alias("same_rare_tok"),
    ).with_columns(
        (~(pl.col("same_city") | pl.col("same_postcode") | pl.col("same_first_tok") | pl.col("same_rare_tok"))).alias("none"),
        (pl.col("postcode_1").is_null() | pl.col("postcode_2").is_null()).alias("postcode_missing"),
        (pl.col("city_key_1").is_null() | pl.col("city_key_2").is_null()).alias("city_missing"),
    )
    fl = ["same_city", "same_postcode", "same_first_tok", "same_rare_tok", "none", "city_missing", "postcode_missing"]
    c12 = pl.concat(
        [
            flags.group_by("country_1", "msrc").agg([pl.col(f).mean() for f in fl] + [pl.len().alias("pairs")]),
            flags.select(pl.lit("ALL").alias("country_1"), pl.lit("ALL").alias("msrc"), *[pl.col(f).mean() for f in fl], pl.len().alias("pairs")),
        ],
        how="vertical_relaxed",
    ).sort("country_1", "msrc")
    save(c12, "c12_partition_keys")
    KEY["none_of_keys"] = pct(flags["none"].mean())
    note(
        "**12. Shared keys in true pairs** (`c12_partition_keys.csv`). Keys are heuristic: city = most frequent "
        "non-region, digit-free address component; postcode = 5-6 digit token not first in the address."
    )
    table(c12)

    c13 = []
    for key in ("city_key", "postcode"):
        g = (
            recs.group_by("split", "country", key)
            .agg((pl.col("src") == "S1").sum().alias("n_s1"), (pl.col("src") != "S1").sum().alias("n_s23"))
            .with_columns(pl.lit(key).alias("key_type"), pl.col(key).alias("key"))
            .drop(key)
        )
        c13.append(g)
    groups = pl.concat(c13)
    top = (
        groups.filter(pl.col("key").is_not_null())
        .sort("n_s23", descending=True)
        .group_by("key_type", "split", "country", maintain_order=True)
        .head(20)
    )
    save(top, "c13_top_groups")
    tot = recs.group_by("split", "country").agg(
        (pl.col("src") == "S1").sum().alias("S1"), (pl.col("src") != "S1").sum().alias("S23")
    )
    cost = (
        groups.filter(pl.col("key").is_not_null())
        .group_by("key_type", "split", "country")
        .agg(
            (pl.col("n_s1").cast(pl.Float64) * pl.col("n_s23")).sum().alias("pairs_in_groups"),
            pl.col("n_s23").max().alias("largest_group_s23"),
            pl.len().alias("groups"),
            pl.col("n_s1").sum().alias("s1_with_key"),
        )
        .join(tot, on=["split", "country"])
        .with_columns(
            (pl.col("pairs_in_groups") / (pl.col("S1").cast(pl.Float64) * pl.col("S23"))).alias("share_of_full_cross"),
            (pl.col("s1_with_key") / pl.col("S1")).alias("s1_key_coverage"),
        )
        .sort("key_type", "split", "country")
    )
    save(cost, "c13_partition_cost")
    note("**13. Partition cost** (`c13_partition_cost.csv`, top groups in `c13_top_groups.csv`)")
    table(cost.select("key_type", "split", "country", "groups", "largest_group_s23", "s1_key_coverage", "share_of_full_cross"), 20)

    # 14: name similarity on a random sample of true pairs (with transliteration)
    smp = pairs.sample(n=min(PAIR_SAMPLE, pairs.height), seed=SEED)
    smp = smp.with_columns(
        pl.col("name_1").map_elements(translit, return_dtype=pl.String).alias("nt_1"),
        pl.col("name_2").map_elements(translit, return_dtype=pl.String).alias("nt_2"),
        pl.col("addr_1").map_elements(translit, return_dtype=pl.String).alias("at_1"),
        pl.col("addr_2").map_elements(translit, return_dtype=pl.String).alias("at_2"),
    )
    smp = smp.with_columns(
        pl.Series("name_tsr", process.cpdist(smp["nt_1"].to_list(), smp["nt_2"].to_list(), scorer=fuzz.token_set_ratio, workers=-1)),
        pl.Series("addr_tsr", process.cpdist(smp["at_1"].to_list(), smp["at_2"].to_list(), scorer=fuzz.token_set_ratio, workers=-1)),
        pl.Series("name_jac3", [jaccard3(x, y) for x, y in zip(smp["nt_1"], smp["nt_2"])]),
    )
    sim_aggs = [
        pl.len().alias("pairs"),
        pl.col("name_tsr").median().alias("tsr_median"),
        *[(pl.col("name_tsr") < t).mean().alias(f"tsr_below_{t}") for t in (50, 70, 90)],
        pl.col("name_jac3").median().alias("jac3_median"),
        *[(pl.col("name_jac3") < t).mean().alias(f"jac3_below_{t}") for t in (0.3, 0.5)],
        pl.col("addr_tsr").median().alias("addr_tsr_median"),
    ]
    c14 = pl.concat(
        [smp.group_by("country_1", "msrc").agg(sim_aggs), smp.select(pl.lit("ALL").alias("country_1"), pl.lit("ALL").alias("msrc"), *sim_aggs)],
        how="vertical_relaxed",
    ).sort("country_1", "msrc")
    save(c14, "c14_name_similarity")
    KEY["tsr_below_70"] = pct(smp["name_tsr"].lt(70).mean())
    note(f"**14. Name similarity of true pairs** (sample of {smp.height:,} pairs, transliterated; `c14_name_similarity.csv`)")
    table(c14)
    return smp


# ---------------------------------------------------------------- D. noise patterns
def token_diffs(smp: pl.DataFrame, a: str, b: str, name: str, top: int = 30) -> pl.DataFrame:
    rows = []
    for c, x, y in zip(smp["country_1"], smp[a], smp[b]):
        tx, ty = set(x.split()), set(y.split())
        rx, ay = tx - ty, ty - tx
        rows += [(c, "only_in_S1", t) for t in rx] + [(c, "only_in_match", t) for t in ay]
        if len(rx) == 1 and len(ay) == 1:
            rows.append((c, "swap", f"{next(iter(rx))} -> {next(iter(ay))}"))
    df = pl.DataFrame(rows, schema=["country", "kind", "token"], orient="row")
    out = (
        df.group_by("country", "kind", "token")
        .len("count")
        .sort("count", descending=True)
        .group_by("country", "kind", maintain_order=True)
        .head(top)
        .sort("country", "kind", "count", descending=[False, False, True])
    )
    return save(out, name)


def part_d(smp: pl.DataFrame) -> None:
    section("D. Noise patterns (sample of true pairs)")
    nd = token_diffs(smp, "nt_1", "nt_2", "d15_name_token_diffs")
    note("**15. Name tokens that differ between S1 and match** (`d15_name_token_diffs.csv`; top swaps shown)")
    table(nd.filter(pl.col("kind") == "swap").group_by("country", maintain_order=True).head(10), 20)
    ad = token_diffs(smp, "at_1", "at_2", "d16_addr_token_diffs")
    note("**16. Address tokens that differ** (`d16_addr_token_diffs.csv`; top swaps shown)")
    table(ad.filter(pl.col("kind") == "swap").group_by("country", maintain_order=True).head(10), 20)

    def typo(x: str, y: str) -> bool:
        tx, ty = set(x.split()), set(y.split())
        return any(len(t) >= 4 and any(0 < Levenshtein.distance(t, u) <= 2 for u in ty if len(u) >= 4) for t in tx - ty)

    s = smp.with_columns(
        pl.Series("typo", [typo(x, y) for x, y in zip(smp["nt_1"], smp["nt_2"])]),
        (
            (pl.col("nt_1") != pl.col("nt_2"))
            & (pl.col("nt_1").str.split(" ").list.sort() == pl.col("nt_2").str.split(" ").list.sort())
        ).alias("word_reorder"),
        ((pl.col("name_tsr") < 50) & (pl.col("addr_tsr") >= 80)).alias("trade_name_like"),
        pl.col("addr_n_2").str.contains(LANDMARK).alias("landmark_in_match"),
        pl.col("addr_n_1").str.contains(LANDMARK).alias("landmark_in_S1"),
        (pl.col("postcode_1").is_not_null() & pl.col("postcode_2").is_null()).alias("postcode_dropped"),
        pl.col("name_2").str.contains(NON_LATIN).alias("match_name_nonlatin"),
        (pl.col("name_1").str.to_lowercase() == pl.col("name_2").str.to_lowercase()).alias("same_ignoring_case"),
        (pl.col("name_1") == pl.col("name_2")).alias("identical_name"),
    )
    kinds = ["typo", "word_reorder", "trade_name_like", "landmark_in_match", "landmark_in_S1", "postcode_dropped",
             "match_name_nonlatin", "same_ignoring_case", "identical_name"]
    d17 = pl.concat(
        [s.group_by("country_1").agg([pl.col(k).mean() for k in kinds]), s.select(pl.lit("ALL").alias("country_1"), *[pl.col(k).mean() for k in kinds])],
        how="vertical_relaxed",
    ).sort("country_1")
    save(d17, "d17_noise_rates")
    ex = pl.concat(
        [
            s.filter(pl.col(k)).sample(n=min(5, s.filter(pl.col(k)).height), seed=SEED)
            .select(pl.lit(k).alias("noise"), "country_1", "name_1", "name_2", "addr_1", "addr_2")
            for k in kinds
        ]
    )
    save(ex, "d17_noise_examples")
    note("**17. Noise rates in true pairs** (`d17_noise_rates.csv`, examples in `d17_noise_examples.csv`)")
    table(d17)

    d18 = s.with_columns(
        (pl.col("addr_2").str.strip_chars() == "").alias("match_addr_empty"),
        (pl.col("addr_2").str.len_chars() < 0.5 * pl.col("addr_1").str.len_chars()).alias("match_addr_under_half"),
    )
    d18 = pl.concat(
        [
            d18.group_by("country_1", "msrc").agg(pl.col("match_addr_empty").mean(), pl.col("match_addr_under_half").mean()),
            d18.select(pl.lit("ALL").alias("country_1"), pl.lit("ALL").alias("msrc"), pl.col("match_addr_empty").mean(), pl.col("match_addr_under_half").mean()),
        ],
        how="vertical_relaxed",
    ).sort("country_1", "msrc")
    save(d18, "d18_addr_short")
    note("**18. Matched address empty or under half the S1 length** (`d18_addr_short.csv`)")
    table(d18)


# ---------------------------------------------------------------- E. hard cases
def part_e(recs: pl.DataFrame, per_pairs: pl.DataFrame, smp: pl.DataFrame) -> None:
    section("E. Hard cases")
    e19 = (
        recs.filter(pl.col("name_n") != "")
        .group_by("split", "src", "country", "name_n")
        .agg(pl.len().alias("records"), pl.col("addr_n").n_unique().alias("distinct_addresses"))
        .sort("records", descending=True)
        .group_by("split", "src", "country", maintain_order=True)
        .head(30)
    )
    save(e19, "e19_frequent_names")
    note("**19. Most frequent normalised names** (`e19_frequent_names.csv`; top train S1 shown)")
    table(e19.filter((pl.col("split") == "train") & (pl.col("src") == "S1")).group_by("country", maintain_order=True).head(8), 16)

    e20 = smp.group_by("country_1", "msrc").agg(pl.all().sample(n=7, seed=SEED, with_replacement=True)).explode(pl.all().exclude("country_1", "msrc"))
    e20 = e20.unique(subset=["s1", "m"]).head(25).select("country_1", "msrc", "s1", "name_1", "addr_1", "m", "name_2", "addr_2", "name_tsr")
    save(e20, "e20_true_pairs")
    note("**20. Random true pairs** (`e20_true_pairs.csv`)")
    table(e20.select("country_1", "msrc", "name_1", "name_2", "addr_1", "addr_2"), 25)

    # 21: for random train S1, the most name-similar S2/S3 record in the same country that is NOT a match
    tr = recs.filter(pl.col("split") == "train")
    truth = per_pairs.group_by("s1").agg(pl.col("m").drop_nulls())
    truth_map = dict(zip(truth["s1"], truth["m"].to_list()))
    q = tr.filter(pl.col("src") == "S1").group_by("country").agg(pl.all().sample(n=HARD_NEG_QUERIES // 2, seed=SEED, with_replacement=True)).explode(pl.all().exclude("country"))
    rows = []
    for country in q["country"].unique().sort():
        pool = tr.filter((pl.col("src") != "S1") & (pl.col("country") == country))
        choices, ids = pool["name_n"].to_list(), pool["entity_id"].to_list()
        for r in q.filter(pl.col("country") == country).unique("entity_id").iter_rows(named=True):
            true = set(truth_map.get(r["entity_id"], []))
            scores = process.cdist([r["name_n"]], choices, scorer=fuzz.ratio, workers=-1)[0]
            top = np.argsort(-scores)[: len(true) + 5]
            best_true = max((float(scores[i]) for i in top if ids[i] in true), default=None)
            neg = next((i for i in top if ids[i] not in true), None)
            if neg is not None:
                nb = pool.row(int(neg), named=True)
                rows.append({"country": country, "s1": r["entity_id"], "s1_name": r["name"], "s1_addr": r["addr"],
                             "neg_id": nb["entity_id"], "neg_name": nb["name"], "neg_addr": nb["addr"],
                             "neg_ratio": round(float(scores[neg]), 1), "best_true_ratio_in_top": best_true, "n_true": len(true)})
    e21 = save(pl.DataFrame(rows).head(25), "e21_hard_negatives")
    note("**21. Hard non-matches**: most name-similar non-matching S2/S3 in the same country (`e21_hard_negatives.csv`)")
    table(e21.select("country", "s1_name", "neg_name", "neg_ratio", "s1_addr", "neg_addr"), 25)

    e22 = smp.sort("name_tsr").head(15).select("country_1", "msrc", "name_1", "name_2", "addr_1", "addr_2", "name_tsr", "addr_tsr")
    save(e22, "e22_low_similarity_pairs")
    note("**22. True pairs with the lowest name similarity** (`e22_low_similarity_pairs.csv`)")
    table(e22, 15)


# ---------------------------------------------------------------- F. test vs train
def top_tokens(recs: pl.DataFrame, col: str, n: int) -> pl.DataFrame:
    return (
        # up to 500k records per split+country: top-token ranks are stable and memory stays small
        recs.select("split", "country", col)
        .filter(pl.int_range(pl.len()).shuffle(seed=SEED).over("split", "country") < 500_000)
        .select("split", "country", pl.col(col).str.split(" ").alias("t"))
        .explode("t")
        .filter(pl.col("t").is_not_null() & (pl.col("t") != ""))
        .group_by("split", "country", "t")
        .len("count")
        .sort("count", descending=True)
        .group_by("split", "country", maintain_order=True)
        .head(n)
    )


def part_f(recs: pl.DataFrame) -> None:
    section("F. Test vs train")
    rows = []
    for col in ("name_n", "addr_n"):
        tt = top_tokens(recs, col, 200)
        save(tt.group_by("split", "country", maintain_order=True).head(30), f"f23_top_{col}_tokens")
        for country in tt.filter(pl.col("split") == "test")["country"].unique().sort():
            a = set(tt.filter((pl.col("split") == "train") & (pl.col("country") == country))["t"])
            b = set(tt.filter((pl.col("split") == "test") & (pl.col("country") == country))["t"])
            rows.append({"field": col, "country": country, "top200_overlap_jaccard": len(a & b) / len(a | b) if a else 0.0})
    f23 = save(pl.DataFrame(rows), "f23_token_overlap")
    note("**23. Train vs test**: length, script and empty-rate comparisons are the train/test rows of the A tables. "
         "Top-200 token overlap per country (`f23_token_overlap.csv`, top tokens in `f23_top_*_tokens.csv`):")
    table(f23)

    fr = recs.filter(pl.col("country") == "France")
    section("F24. France profile (test only)")
    if fr.height == 0:
        note("No France records in this data.")
        return
    last = fr.select("src", pl.col("name_n").str.split(" ").list.last().alias("last_tok"))
    suf = last.group_by("src", "last_tok").len("count").sort("count", descending=True).group_by("src", maintain_order=True).head(30)
    save(suf, "f24_france_last_name_tokens")
    save(top_tokens(fr, "name_n", 50), "f24_france_name_tokens")
    comps = fr.select("src", pl.col("addr").str.split(",").alias("c")).with_columns(pl.col("c").list.len().alias("n_c"))
    pos = (
        comps.with_columns(pl.int_ranges(pl.col("n_c")).alias("pos"))
        .explode("c", "pos")
        .filter(pl.col("c").str.contains(r"\b\d{5}\b"))
        .with_columns(
            pl.when(pl.col("pos") == 0).then(pl.lit("first")).when(pl.col("pos") == pl.col("n_c") - 1).then(pl.lit("last")).otherwise(pl.lit("middle")).alias("postcode_component"),
            pl.col("c").str.strip_chars().str.contains(r"^\d{5}\b").alias("postcode_leads_component"),
        )
    )
    fa = [
        pl.len().alias("records"),
        pl.col("addr").str.contains(r"\b\d{5}\b").mean().alias("has_5digit"),
        *[norm(pl.col("addr")).str.contains(rf"\b{w}\b").mean().alias(f"has_{w}") for w in
          ("rue", "avenue", "av", "boulevard", "bd", "place", "chemin", "route", "allee", "impasse", "quai", "st", "ste", "saint")],
        pl.col("addr").str.contains(r"[A-Za-z]'").mean().alias("addr_apostrophe"),
        norm(pl.col("name")).str.contains(r"\b(sa|sas|sarl|eurl|sasu|sci|snc)\b").mean().alias("name_has_fr_legal"),
        norm(pl.col("name")).str.contains(r"\bet\b").mean().alias("name_has_et"),
        norm(pl.col("name")).str.contains(r"\b(de|du|des|la|le|les)\b").mean().alias("name_has_article"),
        pl.col("name").str.contains(r"[\p{Latin}&&[^a-zA-Z]]").mean().alias("name_has_accent"),
    ]
    f24 = fr.group_by("src").agg(fa).sort("src")
    save(f24, "f24_france_format")
    save(pos.group_by("src", "postcode_component", "postcode_leads_component").len("count").sort("src", "count", descending=[False, True]), "f24_france_postcode_position")
    cities = fr.filter(pl.col("city_key").is_not_null()).group_by("src", "city_key").len("count").sort("count", descending=True).group_by("src", maintain_order=True).head(30)
    save(cities, "f24_france_cities")
    samples = fr.group_by("src").agg(pl.all().sample(n=7, seed=SEED, with_replacement=True)).explode(pl.all().exclude("src")).unique("entity_id").head(20)
    save(samples.select("src", "entity_id", "name", "addr"), "f24_france_samples")
    note("France address and name format (`f24_france_*.csv`):")
    table(f24.select("src", "records", "has_5digit", "has_rue", "has_avenue", "has_st", "has_saint", "name_has_fr_legal", "name_has_et", "name_has_article", "name_has_accent"))
    note("Most frequent last name tokens (legal suffixes) per source:")
    table(suf.group_by("src", maintain_order=True).head(8), 24)
    note("Sample France records:")
    table(samples.select("src", "name", "addr"), 20)


# ---------------------------------------------------------------- G. validation split
def part_g(per_pairs: pl.DataFrame) -> None:
    section("G. Validation design")
    s1 = per_pairs.select("s1", "country", "n", "bucket").unique("s1").sort("s1")
    rng = np.random.default_rng(SEED)
    s1 = s1.with_columns(pl.Series("r", rng.random(s1.height)))
    # stratified by country x match-count bucket: rank within stratum, fold = rank mod 5
    s1 = s1.with_columns(pl.col("r").rank("ordinal").over("country", "bucket").alias("rk")).with_columns(
        ((pl.col("rk") - 1) % 5).cast(pl.Int8).alias("fold")
    )
    frac_val = VAL_S1 / s1.filter(pl.col("fold") == 0).height
    frac_tr = TRAIN_S1 / s1.filter(pl.col("fold") != 0).height
    role = (
        pl.when((pl.col("fold") == 0) & (pl.col("r") < min(frac_val, 1.0))).then(pl.lit("val"))
        .when((pl.col("fold") != 0) & (pl.col("r") < min(frac_tr, 1.0))).then(pl.lit("train"))
        .otherwise(pl.lit("unused"))
    )
    s1 = s1.with_columns(role.alias("role"))
    split = s1.select(pl.col("s1").alias("source1_entity_id"), "country", pl.col("n").alias("n_matches"), "fold", "role")
    split.write_parquet(OUT / "g25_split.parquet")
    g25 = save(
        split.group_by("role", "country").agg(pl.len().alias("s1"), (pl.col("n_matches") == 0).mean().alias("singleton_share"), pl.col("n_matches").mean().alias("avg_matches")).sort("role", "country"),
        "g25_split_summary",
    )
    note(
        f"Split saved to `g25_split.parquet` (all train S1 IDs; 5 folds stratified by country x match-count bucket; "
        f"role = val: ~{VAL_S1:,} S1 from fold 0, train: ~{TRAIN_S1:,} S1 from folds 1-4, rest unused). "
        "Rule: sample S1 only; always keep the FULL S2/S3 pool of each country as candidates."
    )
    table(g25)


# ---------------------------------------------------------------- implications
def implications(per_pairs: pl.DataFrame) -> None:
    section("Implications for the pipeline (auto-generated from the numbers above; refine by hand)")
    note(f"- **Blocking**: cross-country pairs = {KEY.get('cross_country')}. If ~0, partition the search by country "
         f"(a compute split, not a label filter). {KEY.get('none_of_keys')} of true pairs share none of city/postcode/"
         "first token/rarest token, so key-only blocking caps recall; use TF-IDF top-k within country as the main method.")
    note(f"- **Selection**: S2/S3 IDs shared by several S1 = {KEY.get('shared_ids')}. If 0, each S2/S3 record matches at most "
         f"one S1, so one-to-one assignment (greedy by score) is valid. {KEY.get('same_source_multi')} of S1 have 2+ matches "
         "from the same source, so never cap matches per source at 1.")
    note(f"- **Baseline**: all-empty macro F0.5 = {KEY.get('baseline')}; every model must beat it.")
    note(f"- **Names**: {KEY.get('tsr_below_70')} of true pairs have token_set_ratio < 70 (after transliteration). "
         "Normalise with transliteration; use the D15 swap list for abbreviation maps; add address and postcode features.")
    note("- **France**: check F24 before writing normalisation rules (e.g. 'st' = saint vs street, 'sa' legal form, "
         "'et' vs 'and', articles de/du/la, postcode position).")


def main() -> None:
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data_parquet")
    ap.add_argument("--out-dir", default="outputs/eda")
    args = ap.parse_args()
    OUT = Path(args.out_dir)
    OUT.mkdir(parents=True, exist_ok=True)

    recs, gt = load(Path(args.data_dir))
    log(f"loaded {recs.height:,} records, {gt.height:,} ground-truth rows")
    # one split at a time keeps the exploded component/token tables to half the size
    recs = pl.concat([add_keys(recs.filter(pl.col("split") == s)) for s in ("train", "test")])
    log("keys added")
    part_a(recs); log("A done")
    per_pairs = part_b(recs, gt); log("B done")
    smp = part_c(recs, per_pairs); log("C done")
    part_d(smp); log("D done")
    part_e(recs, per_pairs, smp); log("E done")
    part_f(recs); log("F done")
    part_g(per_pairs); log("G done")
    implications(per_pairs)

    head = ["# Phase 0 EDA summary\n", f"Data: `{args.data_dir}`. Runtime: {time.time() - T0:.0f}s.\n", "## Key numbers\n"]
    head += [f"- **{k}**: {v}" for k, v in KEY.items()]
    (OUT / "eda_summary.md").write_text("\n".join(head) + "\n" + "\n".join(SUMMARY))
    log(f"wrote {OUT / 'eda_summary.md'}")


if __name__ == "__main__":
    main()
