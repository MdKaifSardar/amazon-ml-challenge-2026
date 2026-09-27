"""Train the final matching model (two-stage LightGBM [+ CatBoost], see pipeline.py) and check it on validation.

  python src/train_model.py --input DIR --config best_config.json --out-dir artifacts/model_final [--final]

--input is searched for: g25_split.parquet, train_source1.parquet and train_ground_truth.parquet (organiser data),
train_candidates.parquet (blocking v3), features_train.parquet / features_val.parquet (features-v2) and
normalised/{train,test}_source{1,2,3}.parquet (norm-v3). --config is the chosen setup from run_model_v3.py
(stage-2 features, LightGBM parameters, seeds, blend, selection).
Training S1: the train split without the blocking pruner's 100k S1 (their pruner-derived inputs are in-sample);
with --final also the validation tune half (even S1 id). The validation report half (odd id) is never trained on
and is the final check: macro F0.5 overall, per country, singletons / non-singletons, precision, recall.
Writes stage1_fold*.txt, stage2_lgb_seed*.txt, [catboost.cbm], model_config.json, train_report.md.
"""
import argparse
import json
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
from model import feature_sets  # noqa: E402
from pipeline import blend, counts, find, normalised_files, raw_file, records, stage2_frame  # noqa: E402
from selection import f05_table, select, summary  # noqa: E402
from two_stage import PARAMS, fit, fold_of, predict, stage1_oof, stage1_predict  # noqa: E402

SEED = 42
BLEND = {"CatBoost": "cb", "blend: mean prob": "mean", "blend: rank mean": "rank"}
CB_PARAMS = dict(iterations=3000, learning_rate=0.08, depth=8, loss_function="Logloss", od_type="Iter", od_wait=100,
                 thread_count=-1, verbose=0)
T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s peak {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20:5.1f} GB] {msg}", flush=True)


def md(df: pl.DataFrame) -> str:
    f = lambda v: f"{v:.4f}" if isinstance(v, float) else str(v)
    rows = [" | ".join(df.columns), " | ".join("---" for _ in df.columns)] + [" | ".join(f(v) for v in r) for r in df.iter_rows()]
    return "\n".join(f"| {r} |" for r in rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--config", required=True, type=Path, help="best_config.json from run_model_v3.py (a folder is searched)")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--final", action="store_true", help="also train on the validation tune half")
    ap.add_argument("--pruner-s1", type=int, default=100_000)
    a = ap.parse_args()
    out = a.out_dir
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((find(a.config, "best_config.json") if a.config.is_dir() else a.config).read_text())
    how = BLEND.get(cfg["setup"], "lgb")
    rd = lambda name, cols=None: pl.read_parquet(raw_file(a.input, name), columns=cols)

    split = rd("g25_split.parquet")
    trn = split.filter(pl.col("role") == "train")["source1_entity_id"]
    pruner_s1 = trn.sample(min(a.pruner_s1, trn.len()), seed=SEED)
    val_s1 = split.filter(pl.col("role") == "val").select(pl.col("source1_entity_id").alias("s1")).join(
        rd("train_source1.parquet", ["entity_id", "country"]).rename({"entity_id": "s1"}), on="s1", how="left")
    gt = rd("train_ground_truth.parquet")
    truth = (gt.join(val_s1, left_on="source1_entity_id", right_on="s1", how="semi")
             .select(pl.col("source1_entity_id").alias("s1"), pl.col("matched_entity_ids").str.split(",").alias("cand"))
             .explode("cand", empty_as_null=True).filter(pl.col("cand").is_not_null() & (pl.col("cand") != "")))
    half = pl.col("s1").str.extract(r"(\d+)$").cast(pl.Int64) % 2
    s1_tune, s1_rep = val_s1.filter(half == 0), val_s1.filter(half == 1)

    ftr = rd("features_train.parquet")
    c_feats = feature_sets(ftr.columns)["C"]
    ftr = ftr.join(rd("train_candidates.parquet", ["s1", "cand", "is_match"]), on=["s1", "cand"], how="left")
    ftr = ftr.filter(~pl.col("s1").is_in(pruner_s1.implode()))
    fva = rd("features_val.parquet").join(truth.with_columns(pl.lit(True).alias("is_match")), on=["s1", "cand"], how="left") \
        .with_columns(pl.col("is_match").fill_null(False))
    if a.final:
        ftr = pl.concat([ftr, fva.join(s1_tune, on="s1", how="semi").select(ftr.columns)])
    ftr = ftr.sort("s1", "cand")
    rep = fva.join(s1_rep, on="s1", how="semi").sort("s1", "cand")
    y = ftr["is_match"].cast(pl.Int8).to_numpy()
    log(f"training {ftr.height:,} pairs / {ftr['s1'].n_unique():,} S1 (final={a.final}); report half {rep.height:,} pairs")

    # stage 1
    oof, models1 = stage1_oof(ftr.select("s1", *c_feats), y, c_feats, log=log)
    for i, b in enumerate(models1):
        b.save_model(str(out / f"stage1_fold{i}.txt"), num_iteration=b.best_iteration)
    # stage-2 features
    norm = normalised_files(a.input)
    cnt = counts(norm)
    s1_rec, c_rec = records(norm, "train", pl.concat([ftr["s1"], rep["s1"]]).unique(), pl.concat([ftr["cand"], rep["cand"]]).unique())
    X = stage2_frame(ftr, oof, s1_rec, c_rec, cnt, c_feats)
    Xr = stage2_frame(rep, stage1_predict(models1, rep, c_feats), s1_rec, c_rec, cnt, c_feats)
    fs = cfg["features"]
    m, mr = X.select(fs).to_numpy(), Xr.select(fs).to_numpy()
    es = fold_of(ftr["s1"], 10, salt=7) == 0
    params = {k: v for k, v in cfg["lgb_params"].items() if k not in PARAMS or PARAMS[k] != v}
    pl_, pr_cb = [], None
    for s in range(int(cfg["seeds"])):
        b = fit(m, y, es, fs, seed=SEED + 100 + s, params=params)
        b.save_model(str(out / f"stage2_lgb_seed{s}.txt"), num_iteration=b.best_iteration)
        pl_.append(predict(b, mr))
        log(f"stage-2 LightGBM seed {s}: best iteration {b.best_iteration}")
    if how != "lgb":
        from catboost import CatBoostClassifier
        cb = CatBoostClassifier(**CB_PARAMS, random_seed=SEED)
        cb.fit(m[~es], y[~es], eval_set=(m[es], y[es]))
        cb.save_model(str(out / "catboost.cbm"))
        pr_cb = cb.predict_proba(mr)[:, 1]
        log(f"CatBoost best iteration {cb.get_best_iteration()}")
    p = blend(np.mean(pl_, axis=0), pr_cb, how)

    # final check on the report half (fixed selection from the experiments)
    sel_cfg = cfg["selection"]
    sv = rep.select("s1", "cand").with_columns(pl.Series("p", p))
    sm = summary(f05_table(select(sv, **sel_cfg), truth, s1_rep), by="country")
    log("report half:\n" + str(sm))
    sm.write_csv(out / "report_half.csv")
    (out / "model_config.json").write_text(json.dumps({
        "stage1_features": c_feats, "stage2_features": fs, "lgb_params": cfg["lgb_params"], "seeds": int(cfg["seeds"]),
        "blend": how, "selection": sel_cfg, "final": a.final, "setup": cfg["setup"],
        "report_half": {r["group"]: {k: r[k] for k in ("macro_f05", "precision", "recall", "pred_per_s1")} for r in sm.iter_rows(named=True)},
        "trained_s1": ftr["s1"].n_unique()}, indent=1, default=str))
    (out / "train_report.md").write_text(f"# Final model ({cfg['setup']}, final={a.final})\n\n{md(sm)}\n")
    log("done")


if __name__ == "__main__":
    main()
