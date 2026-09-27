"""Test inference with the model trained by train_model.py; writes both official output files.

  python src/predict.py --input DIR --model-dir artifacts/model_final --out-dir ../../output \
      [--validator ../../utils/validate_submission.py] [--fallback ../../output/fallback_v2/matching_results.tsv]

Per country (candidate lists never cross countries): stage 1 (fold-model average) -> set features -> B features
-> stage 2 (LightGBM seeds [+ CatBoost]) -> blend. Then the selection from training (threshold, margin) with ONE
OWNER per S2/S3 over all test S1. candidate_pairs.tsv = the blocking v3 test candidates, exactly the pairs scored;
matches are a subset of it. Also writes predict_report.md: matches per S1 by country, share without a match,
10 random French S1 with their matches, and the change against a previous matching file (--fallback).
"""
import argparse
import json
import resource
import sys
import tempfile
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

try:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
except NameError:
    sys.path.insert(0, "/tmp/src")
from pipeline import blend, counts, normalised_files, raw_file, records, stage2_frame  # noqa: E402
from selection import select  # noqa: E402
from submission import write_submission  # noqa: E402

SEED = 42
T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s peak {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20:5.1f} GB] {msg}", flush=True)


def md(df: pl.DataFrame, n: int = 80) -> str:
    df = df.head(n)
    f = lambda v: f"{v:.4f}" if isinstance(v, float) else str(v)
    rows = [" | ".join(df.columns), " | ".join("---" for _ in df.columns)] + [" | ".join(f(v) for v in r) for r in df.iter_rows()]
    return "\n".join(f"| {r} |" for r in rows)


def ids_sets(path: Path) -> pl.DataFrame:
    return (pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False)
            .select(pl.col("source1_entity_id").alias("s1"),
                    pl.col("matched_entity_ids").fill_null("").str.split(",").list.eval(pl.element().filter(pl.element() != "")).list.sort().alias("m")))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--model-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--validator", default=None)
    ap.add_argument("--fallback", default=None, type=Path)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    md_dir = a.model_dir if (a.model_dir / "model_config.json").exists() else next(a.model_dir.rglob("model_config.json")).parent
    cfg = json.loads((md_dir / "model_config.json").read_text())
    s1m = [lgb.Booster(model_file=str(p)) for p in sorted(md_dir.glob("stage1_fold*.txt"))]
    s2m = [lgb.Booster(model_file=str(p)) for p in sorted(md_dir.glob("stage2_lgb_seed*.txt"))]
    cb = None
    if cfg["blend"] != "lgb":
        from catboost import CatBoostClassifier
        cb = CatBoostClassifier()
        cb.load_model(str(md_dir / "catboost.cbm"))
    c_feats, fs = cfg["stage1_features"], cfg["stage2_features"]
    log(f"model: {len(s1m)} stage-1 folds, {len(s2m)} stage-2 seeds, blend {cfg['blend']}, selection {cfg['selection']}")

    s1_all = pl.read_parquet(raw_file(a.input, "test_source1.parquet"), columns=["entity_id", "country"]).rename({"entity_id": "s1"})
    norm = normalised_files(a.input)
    cnt = counts(norm)
    fpath = raw_file(a.input, "features_test.parquet")
    parts = []
    for country in sorted(s1_all["country"].unique().to_list()):
        fx = pl.scan_parquet(fpath).join(s1_all.filter(pl.col("country") == country).lazy().select("s1"), on="s1", how="semi").collect().sort("s1", "cand")
        if not fx.height:
            continue
        p1 = np.mean([b.predict(fx.select(c_feats).to_numpy()) for b in s1m], axis=0)
        s1_rec, c_rec = records(norm, "test", fx["s1"].unique(), fx["cand"].unique())
        X = stage2_frame(fx, p1, s1_rec, c_rec, cnt, c_feats)
        m = X.select(fs).to_numpy()
        p_lgb = np.mean([b.predict(m) for b in s2m], axis=0)
        p_cb = cb.predict_proba(m)[:, 1] if cb is not None else None
        parts.append(fx.select("s1", "cand").with_columns(pl.Series("p_lgb", p_lgb), pl.Series("p_cb", p_cb if p_cb is not None else p_lgb)))
        log(f"{country}: {fx.height:,} pairs scored")
        del fx, X, m, s1_rec, c_rec
    st = pl.concat(parts)
    st = st.with_columns(pl.Series("p", blend(st["p_lgb"].to_numpy(), st["p_cb"].to_numpy() if cb is not None else None, cfg["blend"])))
    sel = {**cfg["selection"], "one_owner": True}  # an S2/S3 matches at most one S1; on test all S1 compete
    matched = select(st.select("s1", "cand", "p"), **sel)
    log(f"{matched.height:,} matches")
    st.select("s1", "cand", "p").write_parquet(a.out_dir.parent / "test_scores.parquet")  # pair probabilities (for blends)

    cand = pl.read_parquet(raw_file(a.input, "test_candidates.parquet"), columns=["s1", "cand", "p_u50"])
    assert cand.select("s1", "cand").sort("s1", "cand").equals(st.select("s1", "cand").sort("s1", "cand")), "scored pairs != candidate set"
    lists = cand.sort(["s1", "p_u50", "cand"], descending=[False, True, False], nulls_last=True).group_by("s1", maintain_order=True).agg(pl.col("cand"))
    ml = matched.sort(["s1", "p"], descending=[False, True]).group_by("s1", maintain_order=True).agg(pl.col("cand"))
    write_submission(a.out_dir, s1_all["s1"].to_list(), dict(zip(ml["s1"].to_list(), ml["cand"].to_list())),
                     dict(zip(lists["s1"].to_list(), lists["cand"].to_list())))
    log(f"wrote {a.out_dir}/matching_results.tsv and candidate_pairs.tsv")

    report = [f"# Test predictions ({cfg['setup']})\n", f"Matches: {matched.height:,} for {s1_all.height:,} S1; selection {sel}.\n"]
    if a.validator:
        import importlib.util
        spec = importlib.util.spec_from_file_location("vs", a.validator); vs = importlib.util.module_from_spec(spec); spec.loader.exec_module(vs)
        with tempfile.TemporaryDirectory() as td:
            for k in (1, 2, 3):
                pl.read_parquet(raw_file(a.input, f"test_source{k}.parquet"), columns=["entity_id"]).write_csv(Path(td) / f"test_source{k}.tsv", separator="\t")
            e, w = vs.validate(str(a.out_dir / "matching_results.tsv"), str(a.out_dir / "candidate_pairs.tsv"), td, check_ids=True)
        report.append(f"Validator (--check-ids): {len(e)} errors, {len(w)} warnings {e[:3]}\n")
        log(f"validator: {len(e)} errors, {len(w)} warnings")
    n = s1_all.join(matched.group_by("s1").len("n"), on="s1", how="left").with_columns(pl.col("n").fill_null(0))
    pc = n.group_by("country").agg(pl.len().alias("s1"), pl.col("n").mean().alias("matches_per_s1"),
                                   (pl.col("n") == 0).mean().alias("share_no_match")).sort("country")
    report += ["## Matches per S1\n", md(pc) + "\n"]
    fr = s1_all.filter(pl.col("country") == "France").sample(min(10, int((s1_all["country"] == "France").sum())), seed=SEED)
    if fr.height:
        names1 = pl.read_parquet(raw_file(a.input, "test_source1.parquet"), columns=["entity_id", "business_name", "business_address"]).rename({"entity_id": "s1"})
        ex = matched.join(fr, on="s1", how="semi")
        cn = pl.concat([pl.read_parquet(raw_file(a.input, f"test_source{k}.parquet"), columns=["entity_id", "business_name", "business_address"])
                        .join(ex.select(pl.col("cand").alias("entity_id")), on="entity_id", how="semi") for k in (2, 3)])
        ex = (fr.join(names1, on="s1").join(ex, on="s1", how="left")
              .join(cn.rename({"entity_id": "cand", "business_name": "c_name", "business_address": "c_addr"}), on="cand", how="left")
              .select("s1", "business_name", "business_address", "p", "c_name", "c_addr"))
        report += ["## 10 random French S1\n", md(ex) + "\n"]
    if a.fallback and a.fallback.exists():
        x = ids_sets(a.out_dir / "matching_results.tsv").join(ids_sets(a.fallback), on="s1", suffix="_fb").join(s1_all, on="s1")
        ch = x.with_columns((pl.col("m") != pl.col("m_fb")).alias("changed")).group_by("country").agg(
            pl.len().alias("s1"), pl.col("changed").sum().alias("s1_changed"), pl.col("changed").mean().alias("share")).sort("country")
        report += [f"## Change vs {a.fallback.name}\n", md(ch) + "\n"]
    (a.out_dir / "predict_report.md").write_text("\n".join(report))
    log("done")


if __name__ == "__main__":
    main()
