"""Build pair features (table 6) for the train and val candidate tables, with a quality report.

  python src/run_features.py --input /kaggle/input --out-dir /kaggle/working [--sample-s1 3000] [--splits train val]

Inputs (searched under --input): {split}_candidates.parquet (blocking v3), normalised/train_source{1,2,3}.parquet
(norm-v3). Both candidate tables hold TRAIN-side S1 (train / val roles of g25_split), so records and side
statistics come from the train files.
Outputs: features_{split}.parquet (s1, cand, f_*), features_report.md (list sizes, NaN share and AUC per feature;
the label is read from the candidate table and never written), features_config.json.
--sample-s1 N keeps N random S1 per split with ALL their candidates (lists stay whole), for test runs.
--prune SPLIT ...: cut those candidate tables to the submitted set (table 5) first. val_candidates.parquet from
blocking-eval-v3 is the whole starting list (u20 + u50 unions, 117 per S1); train_candidates.parquet (Job A) is
already pruned. The rule is read from the blocking config.json (default operating point: union u50, then
candidates.prune with tau 0.01, cap 10), the same steps as run_blocking_job_a.py, so every split has the same lists.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

from candidates import prune, rule
from features import FEATURE_VERSION, build, feature_names, pool_stats, prepare_records

SEED = 42
T0 = time.time()
NORM_COLS = ["entity_id", "country", "full_name", "core_name", "legal", "addr_norm", "city", "numbers"]
P_NOTE = ("f_p_u50, f_p_gap and f_list_rank come from the blocking pruner, which was trained on 100k train-split S1: "
          "for those S1 they are in-sample (too confident). Train the model on other train S1 or drop these columns for them.")


def log(msg: str) -> None:
    try:
        import resource
        peak = f"peak {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20:5.1f} GB"
    except ImportError:  # Windows
        peak = ""
    print(f"[{time.time() - T0:7.1f}s {peak}] {msg}", flush=True)


def find(root: Path, name: str, parent: str | None = None) -> Path:
    hits = sorted(p for p in root.rglob(name) if parent is None or p.parent.name == parent)
    if not hits:
        raise FileNotFoundError(f"{name} (parent {parent}) not found under {root}")
    return hits[0]


def md(df: pl.DataFrame) -> str:
    rows = [[f"{v:.4f}" if isinstance(v, float) else str(v) for v in r] for r in df.iter_rows()]
    return "\n".join(["| " + " | ".join(df.columns) + " |", "|" + "---|" * df.width, *["| " + " | ".join(r) + " |" for r in rows]])


def prune_standard(c: pl.DataFrame, cfg: dict) -> pl.DataFrame:
    """Starting union + pruner at the default operating point, exactly as run_blocking_job_a.py: rule(union) on
    rows sorted by (scope, s1, cand) (to_wide order, so ties at the cap break the same way), then prune()."""
    op = next(o for o in cfg["operating_points"] if o["target"] == cfg["default_target"])
    oc = json.loads(op["config"])
    x = c.filter(rule(**cfg["unions"][oc["union"]])).sort("scope", "s1", "cand")
    return prune(x, x[f"p_{oc['union']}"].to_numpy(), tau=oc["tau"], cap=oc.get("cap")).drop("p_keep")


def list_sizes(c: pl.DataFrame, split: str) -> dict:
    n = c.group_by("s1").len()["len"]
    d = {"split": split, "pairs": c.height, "s1": n.len(), "avg": n.mean(), "median": n.median(),
         "p95": n.quantile(0.95, "nearest"), "max": n.max()}
    if "p_u50" in c.columns:
        d["share_p_below_0.01"] = c.select((pl.col("p_u50") < 0.01).mean()).item()
    if "is_match" in c.columns:
        d["positives"] = int(c["is_match"].sum())
    return d


def auc_table(f: pl.DataFrame, y: np.ndarray) -> pl.DataFrame:
    """Per feature: NaN share, AUC on rows where it is present (0.5 = useless, <0.5 = inverse), and |AUC-0.5|."""
    rows = []
    for c in feature_names(f):
        v = f[c].to_numpy()
        ok = ~np.isnan(v)
        auc = roc_auc_score(y[ok], v[ok]) if ok.sum() and 0 < y[ok].sum() < ok.sum() else float("nan")
        rows.append({"feature": c, "nan_share": float(1 - ok.mean()), "auc": float(auc), "strength": abs(auc - 0.5)})
    return pl.DataFrame(rows).with_columns(pl.col("strength").fill_nan(None)).sort("strength", descending=True, nulls_last=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--splits", nargs="*", default=["train", "val"])
    ap.add_argument("--sample-s1", type=int, default=0, help="0 = all S1")
    ap.add_argument("--chunk", type=int, default=1_000_000)
    ap.add_argument("--prune", nargs="*", default=[], help="splits to cut to the submitted candidate set first")
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    cands = {}
    for split in a.splits:
        path = find(a.input, f"{split}_candidates.parquet")
        c = pl.read_parquet(path)
        log(f"{split}: {path} {c.shape}\n  columns: {c.columns}")
        if "scope" in c.columns and c["scope"].n_unique() > 1:
            raise ValueError(f"{split}: several scopes {c['scope'].unique().to_list()}")
        if split in a.prune:
            cfg = json.loads(find(path.parent, "config.json").read_text())
            before = (c.height, int(c["is_match"].sum()) if "is_match" in c.columns else None)
            c = prune_standard(c, cfg)
            log(f"{split}: pruned {before[0]:,} -> {c.height:,} pairs; positives {before[1]} -> "
                f"{int(c['is_match'].sum()) if 'is_match' in c.columns else None}")
        if a.sample_s1:
            keep = c.select("s1").unique().sort("s1").sample(min(a.sample_s1, c["s1"].n_unique()), seed=SEED)
            c = c.join(keep, on="s1", how="semi")
        cands[split] = c
    sizes = [list_sizes(c, s) for s, c in cands.items()]
    log(f"list sizes: {sizes}")

    ids_s1 = pl.concat([c.select(pl.col("s1").alias("entity_id")) for c in cands.values()]).unique()
    ids_pool = pl.concat([c.select(pl.col("cand").alias("entity_id")) for c in cands.values()]).unique()
    src = {k: find(a.input, f"train_source{k}.parquet", parent="normalised") for k in (1, 2, 3)}
    log(f"normalised files: {src}")
    stats = pool_stats(pl.read_parquet(src[1], columns=["core_name"])["core_name"],
                       pl.concat([pl.read_parquet(src[k], columns=["core_name"]) for k in (2, 3)])["core_name"])
    log(f"side stats: pool {stats.n_pool:,} records, {len(stats.token_df):,} tokens")
    rec = lambda paths, ids: prepare_records(pl.concat([pl.scan_parquet(p).select(NORM_COLS).join(ids.lazy(), on="entity_id", how="semi")
                                                        for p in paths]).collect())
    s1_rec, pool_rec = rec([src[1]], ids_s1), rec([src[2], src[3]], ids_pool)
    log(f"records: s1 {s1_rec.height:,}, pool {pool_rec.height:,}")

    report = [f"# Pair features {FEATURE_VERSION}", "", f"Sample: {a.sample_s1 or 'all'} S1 per split. "
              f"Pruned to the submitted set here: {a.prune or 'none'}.", "",
              "## Candidate lists", "", md(pl.DataFrame(sizes)), "", f"**Note for stream B:** {P_NOTE}", ""]
    timing = {}
    for split, c in cands.items():
        t = time.time()
        f = build(c, s1_rec, pool_rec, stats, chunk=a.chunk)
        timing[split] = {"rows": f.height, "seconds": round(time.time() - t, 1),
                         "rows_per_s": round(f.height / max(time.time() - t, 1e-9))}
        f.write_parquet(a.out_dir / f"features_{split}.parquet")
        log(f"{split}: {f.shape} in {timing[split]['seconds']} s -> features_{split}.parquet")
        if "is_match" in c.columns:
            assert f.select("s1", "cand").equals(c.select("s1", "cand"))
            tab = auc_table(f, c["is_match"].cast(pl.Int8).to_numpy())
            tab.write_csv(a.out_dir / f"features_auc_{split}.csv")
            report += [f"## {split}: NaN share and AUC per feature ({f.height:,} pairs)", "", md(tab), ""]
    report += ["## Timing", "", md(pl.DataFrame([{"split": s, **v} for s, v in timing.items()])), ""]
    (a.out_dir / "features_report.md").write_text("\n".join(report))
    (a.out_dir / "features_config.json").write_text(json.dumps(
        {"feature_version": FEATURE_VERSION, "sample_s1": a.sample_s1, "splits": a.splits, "pruned": a.prune, "list_sizes": sizes,
         "timing": timing, "features": feature_names(f), "note": P_NOTE}, indent=1, default=str))
    log("done")


if __name__ == "__main__":
    main()
