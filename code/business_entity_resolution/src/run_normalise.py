"""Full-data normalisation run: learn state aliases (train labels, validation S1 excluded), fit the city
vocabulary (train + test, no labels), normalise every source file, and write a quality report.

Usage: python src/run_normalise.py --data-dir data_parquet --split outputs/eda/g25_split.parquet --out-dir artifacts/normalised
Outputs (in --out-dir):
  normalised/{split}_source{k}.parquet   raw columns + normalised fields + norm_version (one per source file)
  state_aliases_<version>.json           learned aliases with provenance, loaded at inference
  city_vocab_<version>.parquet           (ckey, cand, freq)
  alias_diagnostics.csv, quality.csv, agreement.csv, top_core_tokens.csv, top_legal.csv, examples.csv,
  normalisation_report.md, manifest.json
"""
import argparse
import json
import resource
import sys
import time
from pathlib import Path

import polars as pl

try:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
except NameError:  # inside the Kaggle notebook the modules are written to /tmp/src
    sys.path.insert(0, "/tmp/src")
from normalise import (NORM_VERSION, Normaliser, alias_diagnostics, learn_state_aliases,  # noqa: E402
                       merge_city_counts, normalise_records, save_state_aliases)

SEED = 42
MAX_LEARN_PAIRS = 3_000_000  # aliases occur thousands of times; a seeded subsample keeps memory low
NON_LATIN = r"[\p{L}&&[^\p{Latin}]]"
T0 = time.time()


def log(msg: str) -> None:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    print(f"[{time.time() - T0:7.1f}s peak {peak:5.1f} GB] {msg}", flush=True)


def find(root: Path, name: str) -> Path:
    hits = sorted(root.rglob(name))
    if not hits:
        raise FileNotFoundError(f"{name} not found under {root}")
    return hits[0]


def md(df: pl.DataFrame, max_rows: int = 40) -> str:
    df = df.head(max_rows)
    fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else str(v).replace("|", "/")
    rows = [" | ".join(df.columns), " | ".join("---" for _ in df.columns)] + [" | ".join(fmt(v) for v in r) for r in df.iter_rows()]
    return "\n".join(f"| {r} |" for r in rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data_parquet")
    ap.add_argument("--split", default="outputs/eda/g25_split.parquet", help="split file or folder containing g25_split.parquet")
    ap.add_argument("--out-dir", default="artifacts/normalised")
    a = ap.parse_args()
    data, out = Path(a.data_dir), Path(a.out_dir)
    (out / "normalised").mkdir(parents=True, exist_ok=True)
    files = {(sp, k): find(data, f"{sp}_source{k}.parquet") for sp in ("train", "test") for k in (1, 2, 3)}
    sp_arg = Path(a.split)  # the split file itself, or a folder to search (e.g. /kaggle/input)
    split = pl.read_parquet(sp_arg if sp_arg.is_file() else find(sp_arg, "g25_split.parquet"))
    report: list[str] = [f"# Normalisation report ({NORM_VERSION})\n"]
    flags: list[str] = []

    # ---- 1. state aliases from TRAIN pairs, validation S1 excluded (labels) ----
    gt = pl.read_parquet(find(data, "train_ground_truth.parquet"))
    val_ids = split.filter(pl.col("role") == "val").select(pl.col("source1_entity_id").alias("s1"))
    pairs = (
        gt.select(pl.col("source1_entity_id").alias("s1"), pl.col("matched_entity_ids").str.split(",").alias("m"))
        .explode("m", empty_as_null=True).filter(pl.col("m").is_not_null() & (pl.col("m") != ""))
    )
    n_all = pairs.height
    pairs = pairs.join(val_ids, on="s1", how="anti")
    n_learn = pairs.height
    if pairs.height > MAX_LEARN_PAIRS:
        pairs = pairs.sample(MAX_LEARN_PAIRS, seed=SEED)
    cols = ["entity_id", "business_address", "country"]
    s1 = pl.read_parquet(files[("train", 1)], columns=cols).rename({"entity_id": "s1", "business_address": "addr_1"})
    mt = pl.concat([pl.read_parquet(files[("train", k)], columns=cols[:2]) for k in (2, 3)]).rename(
        {"entity_id": "m", "business_address": "addr_2"})
    lp = pairs.join(s1, on="s1").join(mt, on="m").select("country", "addr_1", "addr_2")
    del s1, mt
    log(f"learning aliases from {lp.height:,} pairs ({n_learn:,} non-validation of {n_all:,} train pairs)")
    aliases = learn_state_aliases(lp)
    alias_path = out / f"state_aliases_{NORM_VERSION}.json"
    save_state_aliases(aliases, alias_path, {
        "learned_from": "train true pairs, S1 role != 'val' in g25_split.parquet",
        "pairs_used": lp.height, "non_validation_pairs": n_learn, "all_train_pairs": n_all, "seed": SEED})
    diag = alias_diagnostics(lp, aliases)
    diag = diag.with_columns(
        pl.when(pl.col("purity") < 0.95).then(pl.lit("low purity"))
        .when(pl.col("pairs") < 100).then(pl.lit("few pairs"))
        .when(pl.col("alias").str.len_chars() <= 3).then(pl.lit("short (abbreviation?)"))
        .otherwise(pl.lit("")).alias("flag"))
    diag.write_csv(out / "alias_diagnostics.csv")
    del lp
    log(f"aliases: { {c: len(m) for c, m in aliases.items()} }")
    report += ["## 1. Learned state aliases (train pairs, validation S1 excluded)\n",
               f"Learned from {n_learn:,} non-validation train pairs (seeded subsample of {min(n_learn, MAX_LEARN_PAIRS):,}).\n",
               md(diag, 200) + "\n"]
    if diag.filter(pl.col("flag") != "").height:
        flags.append(f"{diag.filter(pl.col('flag') != '').height} aliases flagged for review (see section 1).")
    norm = Normaliser(aliases)

    # ---- 2. city vocabulary from train + test records (no labels) ----
    counts = []
    for key, f in files.items():
        df = pl.read_parquet(f, columns=["business_address", "country"])
        counts.append(norm.city_counts(df["business_address"], df["country"]))
    vocab = merge_city_counts(counts)
    vocab.write_parquet(out / f"city_vocab_{NORM_VERSION}.parquet")
    del counts
    log(f"city vocabulary: {vocab.height:,} (country, candidate) entries")

    # ---- 3. normalise each source file ----
    manifest = {"norm_version": NORM_VERSION, "files": {}}
    for (sp, k), f in files.items():
        df = pl.read_parquet(f)
        res = normalise_records(df, norm, vocab)
        dst = out / "normalised" / f"{sp}_source{k}.parquet"
        res.write_parquet(dst)
        manifest["files"][dst.name] = {"rows": res.height, "source": f.name}
        log(f"normalised {sp} S{k}: {res.height:,} rows -> {dst.name}")
        del df, res
    manifest["state_aliases"] = alias_path.name
    manifest["city_vocab"] = f"city_vocab_{NORM_VERSION}.parquet"
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))

    def load(sp: str, k: int, columns: list[str] | None = None) -> pl.DataFrame:
        return pl.read_parquet(out / "normalised" / f"{sp}_source{k}.parquet", columns=columns).with_columns(
            pl.lit(sp).alias("split"), pl.lit(f"S{k}").alias("src"))

    # ---- 4. quality per split, country, source ----
    qcols = ["business_name", "business_address", "country", "core_name", "legal", "city", "state", "dept"]
    q = pl.concat([
        load(sp, k, qcols).group_by("split", "src", "country").agg(
            pl.len().alias("rows"),
            (pl.col("core_name") == "").mean().alias("core_empty"),
            (pl.col("legal") != "").mean().alias("legal_found"),
            pl.col("city").is_not_null().mean().alias("city_found"),
            pl.col("state").is_not_null().mean().alias("state_found"),
            pl.col("dept").is_not_null().mean().alias("dept_found"),
            pl.col("business_name").str.contains(NON_LATIN).mean().alias("name_nonlatin_raw"),
            pl.col("business_address").str.contains(NON_LATIN).mean().alias("addr_nonlatin_raw"),
        ) for sp, k in files
    ]).sort("country", "src", "split")
    q.write_csv(out / "quality.csv")
    report += ["## 2. Quality per country and source (train vs test)\n", md(q, 60) + "\n"]
    metrics = ["core_empty", "legal_found", "city_found", "state_found", "dept_found", "name_nonlatin_raw", "addr_nonlatin_raw"]
    wide = q.filter(pl.col("split") == "train").join(q.filter(pl.col("split") == "test"), on=["country", "src"], suffix="_test")
    for r in wide.iter_rows(named=True):
        for m in metrics:
            if abs(r[m] - r[f"{m}_test"]) > 0.10:
                flags.append(f"{r['country']} {r['src']}: {m} train {r[m]:.1%} vs test {r[m + '_test']:.1%}")
    for r in q.filter(pl.col("country") == "France").iter_rows(named=True):
        if not 0.50 <= r["legal_found"] <= 0.72:
            flags.append(f"France {r['src']}: legal suffix found in {r['legal_found']:.1%}, EDA expects ~55-67%")
    log("quality done")

    # ---- 5. agreement on true pairs (all train S1; validation S1 shown separately) ----
    kc = ["entity_id", "country", "city", "state"]
    s1n = load("train", 1, kc).select(pl.col("entity_id").alias("s1"), "country", "city", "state")
    mn = pl.concat([load("train", k, kc[:1] + kc[2:]) for k in (2, 3)]).rename({"entity_id": "m", "city": "city_2", "state": "state_2"})
    allp = (
        gt.select(pl.col("source1_entity_id").alias("s1"), pl.col("matched_entity_ids").str.split(",").alias("m"))
        .explode("m", empty_as_null=True).filter(pl.col("m").is_not_null() & (pl.col("m") != ""))
        .with_columns(pl.col("m").str.slice(0, 2).alias("msrc"))
        .join(s1n, on="s1").join(mn, on="m")
        .join(val_ids.with_columns(pl.lit(True).alias("is_val")), on="s1", how="left")
        .with_columns(pl.col("is_val").fill_null(False))
    )
    same = lambda x, y: (pl.col(x).is_not_null() & (pl.col(x) == pl.col(y)))
    aggs = [pl.len().alias("pairs"), same("city", "city_2").mean().alias("same_city"), same("state", "state_2").mean().alias("same_state")]
    agr = pl.concat([
        allp.group_by("country", "msrc").agg(aggs).with_columns(pl.lit("all train").alias("subset")),
        allp.filter("is_val").group_by("country", "msrc").agg(aggs).with_columns(pl.lit("validation S1 only").alias("subset")),
    ]).select("subset", "country", "msrc", "pairs", "same_city", "same_state").sort("subset", "country", "msrc")
    agr.write_csv(out / "agreement.csv")
    report += ["## 3. Same-city / same-state agreement on true pairs\n",
               "Validation S1 were not used to learn aliases; similar numbers there mean no leakage effect.\n", md(agr) + "\n"]
    del allp, s1n, mn
    log("agreement done")

    # ---- 6. top core tokens and legal suffixes per split and country ----
    tok, leg = [], []
    for sp, k in files:
        d = load(sp, k, ["country", "core_name", "legal"])
        tok.append(d.select("split", "country", pl.col("core_name").str.split(" ").alias("t")).explode("t", empty_as_null=True)
                   .filter(pl.col("t") != "").group_by("split", "country", "t").len("count"))
        leg.append(d.filter(pl.col("legal") != "").group_by("split", "country", "legal").len("count"))
    top = lambda parts, c: (pl.concat(parts).group_by("split", "country", c).agg(pl.col("count").sum())
                            .sort("count", descending=True).group_by("split", "country", maintain_order=True).head(30)
                            .sort("country", "split", "count", descending=[False, False, True]))
    tt, tl = top(tok, "t"), top(leg, "legal")
    tt.write_csv(out / "top_core_tokens.csv")
    tl.write_csv(out / "top_legal.csv")
    comp = lambda d, c: d.group_by("split", "country", maintain_order=True).agg(pl.col(c).head(30).str.join(", ").alias(f"top_{c}"))
    report += ["## 4. Top 30 core-name tokens and legal suffixes\n", md(comp(tt, "t")) + "\n", md(comp(tl, "legal")) + "\n"]
    log("top tokens done")

    # ---- 7. before/after examples ----
    ex_cols = ["split", "src", "country", "business_name", "full_name", "core_name", "legal",
               "business_address", "addr_norm", "city", "state", "dept"]
    ex = []
    for sp, k in files:
        d = load(sp, k).with_columns(pl.col("numbers").list.join(" ").alias("numbers"))
        ex.append(d.filter(pl.int_range(pl.len()).shuffle(seed=SEED).over("country") < 10).select(ex_cols + ["numbers"]))
        ind = d.filter((pl.col("country") == "India") & pl.col("business_name").str.contains(NON_LATIN))
        if ind.height:
            ex.append(ind.sample(min(10, ind.height), seed=SEED).select(ex_cols + ["numbers"]))
    ex = pl.concat(ex).unique(maintain_order=True)
    ex.write_csv(out / "examples.csv")
    report += ["## 5. Before/after examples (10 per split, country, source + 10 Indian-script per source)\n",
               "Full list in `examples.csv`; France and Indian-script rows shown here.\n",
               md(ex.filter((pl.col("country") == "France") | pl.col("business_name").str.contains(NON_LATIN))
                  .select("split", "src", "country", "business_name", "core_name", "legal", "business_address", "city", "state", "dept"), 80) + "\n"]

    report.insert(1, "## Flags\n" + ("\n".join(f"- {f}" for f in flags) if flags else "- none") + "\n")
    (out / "normalisation_report.md").write_text("\n".join(report))
    log(f"done; report at {out / 'normalisation_report.md'}")


if __name__ == "__main__":
    main()
