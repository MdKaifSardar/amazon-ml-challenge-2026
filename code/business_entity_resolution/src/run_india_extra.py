"""Extra candidate source for India: transliteration-robust search on a consonant skeleton of the core name.

The same Indian business name is transliterated many ways (Devanagari/Tamil -> Latin via anyascii, or typed by
hand): "sharmaa tredars", "sharma traders", "sarma tradars". The skeleton removes most of that variation:
aspirates are merged (bh->b, th->t, sh->s, ...), w->v, z->s, vowels after the first letter of each token are
dropped, and repeated letters are collapsed ("sharma traders" -> "srm trdrs"). A TF-IDF char 2-3-gram search on
the skeleton (within state buckets, as blocking v3) proposes candidates; only pairs NOT already in the v3
candidate set are kept, at most N per S1 with skeleton cosine >= t. (N, t) are chosen on validation: the largest
recall gain with at most MAX_EXTRA extra candidates per India S1.

Blocking v3, its pruner and the v3 candidates are unchanged. The new pairs go through the unchanged features-v2
code (run_features.py) together with the other candidates of the same S1 (list features need whole lists), and
only those S1 lists are written; the model step replaces those S1's rows.

Outputs (--out-dir): extra_pairs_{split}.parquet (s1, cand, skel_score, skel_rank), aug/{split}_candidates.parquet
(affected S1 lists: v3 rows + new rows), features_extra/features_{split}.parquet, india_extra_report.md.
"""
import argparse
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    SRC = Path(__file__).resolve().parent
except NameError:  # Kaggle notebook: modules are written to /tmp/src
    sys.path.insert(0, "/tmp/src")
    SRC = Path("/tmp/src")
from blocking import NONE, fill_states, forward, state_maps  # noqa: E402
from candidates import prune, rule  # noqa: E402

SEED = 42
COUNTRY = "India"
K = 10
MAX_EXTRA = 1.5
NON_LATIN = r"[\p{L}&&[^\p{Latin}]]"
T0 = time.time()
ASPIRATES = [("ph", "f"), ("bh", "b"), ("dh", "d"), ("th", "t"), ("kh", "k"), ("gh", "g"), ("ch", "c"), ("sh", "s"),
             ("jh", "j"), ("ck", "k"), ("w", "v"), ("z", "s"), ("q", "k"), ("x", "ks")]


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s peak {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20:5.1f} GB] {msg}", flush=True)


def find(root: Path, name: str, parent: str | None = None) -> Path:
    hits = sorted(p for p in root.rglob(name) if parent is None or p.parent.name == parent)
    if not hits:
        raise FileNotFoundError(f"{name} ({parent}) not found under {root}")
    return hits[0]


def skeleton(names: pl.Expr) -> pl.Expr:
    """core_name -> consonant skeleton (see module docstring)."""
    e = names.fill_null("").str.to_lowercase()
    for a, b in ASPIRATES:
        e = e.str.replace_all(a, b, literal=True)
    e = e.str.replace_all(r"[^a-z0-9 ]", "").str.split(" ").list.eval(
        pl.when(pl.element().str.len_chars() > 1)
        .then(pl.element().str.slice(0, 1) + pl.element().str.slice(1).str.replace_all("[aeiouy]", ""))
        .otherwise(pl.element())).list.join(" ")
    for c in "abcdefghijklmnopqrstuvwxyz":
        e = e.str.replace_all(c + c + "+", c)
    return e.str.replace_all(r"\s+", " ").str.strip_chars()


def md(df: pl.DataFrame, n: int = 60) -> str:
    df = df.head(n)
    f = lambda v: f"{v:.4f}" if isinstance(v, float) else str(v)
    rows = [" | ".join(df.columns), " | ".join("---" for _ in df.columns)] + [" | ".join(f(v) for v in r) for r in df.iter_rows()]
    return "\n".join(f"| {r} |" for r in rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--skip-features", action="store_true")
    a = ap.parse_args()
    out = a.out_dir
    out.mkdir(parents=True, exist_ok=True)
    norm = {(sd, k): find(a.input, f"{sd}_source{k}.parquet", "normalised") for sd in ("train", "test") for k in (1, 2, 3)}
    cols = ["entity_id", "country", "core_name", "city", "state", "dept", "business_name"]
    rec = {key: pl.scan_parquet(p).select(cols).filter(pl.col("country") == COUNTRY).collect() for key, p in norm.items()}
    maps = state_maps(pl.concat(list(rec.values())))
    for key in rec:
        rec[key] = fill_states(rec[key], maps).with_columns(skeleton(pl.col("core_name")).alias("skel"))
    log(f"records: { {f'{k[0]}{k[1]}': v.height for k, v in rec.items()} }")

    # existing v3 candidate sets (pruned) and the queried S1
    bd = find(a.input, "test_candidates.parquet").parent
    existing = {"train": pl.read_parquet(bd / "train_candidates.parquet"),
                "test": pl.read_parquet(bd / "test_candidates.parquet")}
    vc = pl.read_parquet(bd / "val_candidates.parquet")
    cfg = json.loads((bd / "config.json").read_text())
    op = json.loads(next(o for o in cfg["operating_points"] if o["target"] == cfg.get("default_target", 7))["config"])
    vu = vc.filter((pl.col("scope") == "state") & rule(**cfg["unions"][op["union"]]))
    existing["val"] = prune(vu, vu["p_u50"].fill_null(0).to_numpy(), op["tau"], op.get("cap")).drop("p_keep", strict=False)
    del vc, vu
    split = pl.read_parquet(find(a.input, "g25_split.parquet"))
    india_tr = rec[("train", 1)]["entity_id"]
    queries = {"train": existing["train"]["s1"].unique().filter(existing["train"]["s1"].unique().is_in(india_tr.implode())),
               "val": split.filter((pl.col("role") == "val") & pl.col("source1_entity_id").is_in(india_tr.implode()))["source1_entity_id"],
               "test": rec[("test", 1)]["entity_id"]}
    log(f"India queries: { {k: v.len() for k, v in queries.items()} }")

    # skeleton TF-IDF fitted on train + test India records (no labels)
    corpus = pl.concat([r["skel"] for r in rec.values()])
    corpus = corpus.filter(corpus != "")
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3), min_df=2, max_df=0.05, sublinear_tf=True, dtype=np.float32)
    vec.fit(corpus.sample(min(3_000_000, corpus.len()), seed=SEED).to_list())
    log(f"skeleton vocabulary {len(vec.vocabulary_):,}")

    found = {}
    for sp in ("train", "val", "test"):
        side = "test" if sp == "test" else "train"
        s1 = rec[(side, 1)].filter(pl.col("entity_id").is_in(queries[sp].implode()))
        pool = pl.concat([rec[(side, 2)], rec[(side, 3)]])
        Q, P = vec.transform(s1["skel"].to_list()), vec.transform(pool["skel"].to_list())
        f = forward(Q, s1["state"].fill_null(NONE).to_numpy(), P, pool["state"].fill_null(NONE).to_numpy(), K)
        f = f.with_columns(pl.Series("s1", s1["entity_id"].to_numpy()[f["qi"].to_numpy()]),
                           pl.Series("cand", pool["entity_id"].to_numpy()[f["pi"].to_numpy()])).select("s1", "cand", pl.col("score").alias("skel_score"))
        f = f.join(existing[sp].select("s1", "cand"), on=["s1", "cand"], how="anti")
        found[sp] = f.with_columns(pl.col("skel_score").rank("ordinal", descending=True).over("s1").alias("skel_rank"))
        log(f"{sp}: {found[sp].height:,} new pairs (before the cut) for {s1.height:,} S1")

    # choose (N, t) on validation: largest India recall gain with <= MAX_EXTRA extra candidates per India S1
    gt = pl.read_parquet(find(a.input, "train_ground_truth.parquet"))
    truth = (gt.join(pl.DataFrame({"s1": queries["val"]}), left_on="source1_entity_id", right_on="s1", how="semi")
             .select(pl.col("source1_entity_id").alias("s1"), pl.col("matched_entity_ids").str.split(",").alias("cand"))
             .explode("cand", empty_as_null=True).filter(pl.col("cand").is_not_null() & (pl.col("cand") != "")))
    script = pl.concat([rec[("train", 2)], rec[("train", 3)]]).select(pl.col("entity_id").alias("cand"),
                                                                        pl.col("business_name").str.contains(NON_LATIN).alias("script"))
    truth = truth.join(script, on="cand", how="left")
    base_hit = truth.join(existing["val"].select("s1", "cand"), on=["s1", "cand"], how="semi")
    n_val = queries["val"].len()
    rows = []
    for n in (1, 2, 3):
        for t in (0.5, 0.6, 0.7, 0.8, 0.9):
            sel = found["val"].filter((pl.col("skel_rank") <= n) & (pl.col("skel_score") >= t))
            gain = truth.join(sel.select("s1", "cand"), on=["s1", "cand"], how="semi")
            rows.append({"N": n, "t": t, "extra_per_s1": sel.height / n_val, "new_true": gain.height,
                         "recall_before": base_hit.height / truth.height, "recall_after": (base_hit.height + gain.height) / truth.height,
                         "script_before": base_hit["script"].sum() / max(truth["script"].sum(), 1),
                         "script_after": (base_hit["script"].sum() + gain["script"].sum()) / max(truth["script"].sum(), 1),
                         "precision_new": gain.height / max(sel.height, 1)})
    grid = pl.DataFrame(rows)
    ok = grid.filter(pl.col("extra_per_s1") <= MAX_EXTRA).sort("new_true", "extra_per_s1", descending=[True, False])
    best = ok.row(0, named=True)
    log(f"chosen N={best['N']} t={best['t']}: India val recall {best['recall_before']:.4f} -> {best['recall_after']:.4f}, "
        f"Indian-script {best['script_before']:.4f} -> {best['script_after']:.4f}, +{best['extra_per_s1']:.2f} per India S1")
    for sp in found:
        found[sp] = found[sp].filter((pl.col("skel_rank") <= best["N"]) & (pl.col("skel_score") >= best["t"]))
        found[sp].write_parquet(out / f"extra_pairs_{sp}.parquet")

    # augmented candidate tables for the affected S1 lists only (v3 rows + new rows), then features-v2 on them
    aug = out / "aug"
    aug.mkdir(exist_ok=True)
    stats = {}
    for sp in ("train", "val", "test"):
        ex = existing[sp]
        aff = found[sp].select("s1").unique()
        base = ex.join(aff, on="s1", how="semi")
        new = found[sp].select("s1", "cand")
        if "is_match" in base.columns:
            lab = gt.select(pl.col("source1_entity_id").alias("s1"), pl.col("matched_entity_ids").str.split(",").alias("cand")).explode("cand")
            new = new.join(lab.with_columns(pl.lit(True).alias("is_match")), on=["s1", "cand"], how="left").with_columns(pl.col("is_match").fill_null(False))
        best_cols = [c for c in base.columns if c.startswith("best_")]
        s1best = base.group_by("s1").agg(*[pl.col(c).first() for c in best_cols]) if best_cols else None
        if s1best is not None:
            new = new.join(s1best, on="s1", how="left")
        new = new.with_columns(*[pl.lit(v, base.schema[c]).alias(c) for c, v in (("scope", "state"), ("country", COUNTRY)) if c in base.columns])
        table = pl.concat([base, new], how="diagonal_relaxed").sort("s1", "cand")
        table.write_parquet(aug / f"{sp}_candidates.parquet")
        stats[sp] = {"affected_s1": aff.height, "rows": table.height, "new_rows": new.height}
        log(f"{sp}: {stats[sp]}")
    for name in ("config.json",):
        (aug / name).write_text((bd / name).read_text())
    nl = aug / "normalised"  # file-level links (directory links are not followed by rglob on every Python version)
    nl.mkdir(exist_ok=True)
    for p in norm.values():
        if not (nl / p.name).exists():
            os.symlink(p, nl / p.name)
    if not a.skip_features:
        cmd = [sys.executable, str(SRC / "run_features.py"), "--input", str(aug), "--out-dir", str(out / "features_extra"),
               "--splits", "train", "val", "test"]
        log("features-v2 on the affected lists: " + " ".join(cmd))
        subprocess.run(cmd, check=True, cwd=str(SRC), env={**os.environ, "PYTHONPATH": str(SRC)})
    (out / "india_extra_report.md").write_text("\n".join([
        "# India extra candidates: consonant-skeleton search\n",
        f"Chosen N = {best['N']}, t = {best['t']} (validation; <= {MAX_EXTRA} extra per India S1).\n",
        md(grid) + "\n", "```\n" + json.dumps(stats, indent=1) + "\n```\n"]))
    (out / "india_extra_config.json").write_text(json.dumps({"N": best["N"], "t": best["t"], "grid": grid.to_dicts(), "stats": stats}, indent=1))
    log("done")


if __name__ == "__main__":
    main()
