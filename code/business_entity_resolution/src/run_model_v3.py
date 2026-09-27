"""Model v3 experiments on validation (steps A + B): targeted feature groups, hyperparameter search, seeds and a
CatBoost blend on top of the v2 two-stage model. Blocking, candidates and features-v2 are unchanged.

Protocol (as v2): stage 1 = variant C, 5-fold out-of-fold on the non-pruner train S1; set features from stage-1
probabilities; stage 2 = LightGBM on C + set features (+ B groups). Selection tuned on the validation tune half
(even S1 id), reported on the report half (odd id). A change is kept only if the report half beats the current
best (v2: 0.9742, US 0.9830, India 0.9610) overall AND in both countries, with precision >= 0.985.
Writes results.csv, groups.csv, error_groups.csv, hp_trials.csv, best_config.json, val_scores.parquet, report.md.
"""
import argparse
import json
import random
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
from features import prepare_records  # noqa: E402
from model import feature_sets  # noqa: E402
from pair_extra import B_GROUPS, country_counts, extra_features  # noqa: E402
from selection import f05_table, select, summary  # noqa: E402
from two_stage import PARAMS, SET_FEATURES, fit, fold_of, predict, set_features, stage1_oof, stage1_predict  # noqa: E402

SEED = 42
BEST = {"overall": 0.9742, "country=US": 0.9830, "country=India": 0.9610}  # v2, report half
MIN_PRECISION = 0.985
TAUS_FULL = [round(x, 3) for x in np.arange(0.30, 0.96, 0.025)]
TAUS_FAST = [round(x, 2) for x in np.arange(0.30, 0.96, 0.05)]
REC_COLS = ["entity_id", "country", "full_name", "core_name", "legal", "addr_norm", "city", "state", "numbers"]
T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s peak {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20:5.1f} GB] {msg}", flush=True)


def find(root: Path, name: str, parent: str | None = None, not_parent: tuple = ()) -> Path:
    hits = sorted(p for p in root.rglob(name) if (parent is None or p.parent.name == parent) and p.parent.name not in not_parent)
    if not hits:
        raise FileNotFoundError(f"{name} not found under {root}")
    return hits[0]


def md(df: pl.DataFrame, n: int = 60) -> str:
    df = df.head(n)
    f = lambda v: f"{v:.4f}" if isinstance(v, float) else str(v)
    rows = [" | ".join(df.columns), " | ".join("---" for _ in df.columns)] + [" | ".join(f(v) for v in r) for r in df.iter_rows()]
    return "\n".join(f"| {r} |" for r in rows)


def tune(sv, truth, s1s, taus, margins=(0.0, 0.5, 0.7, 0.8), owners=(True,)):
    best = None
    for tau in taus:
        for m in margins:
            for o in owners:
                f = f05_table(select(sv, tau=tau, margin=m, one_owner=o), truth, s1s)["f05"].mean()
                if best is None or f > best[0]:
                    best = (f, {"tau": tau, "margin": m, "one_owner": o})
    return best


def evaluate(name, p, base, truth, s1_tune, s1_rep, full=True):
    sv = base.with_columns(pl.Series("p", p))
    tf, cfg = tune(sv, truth, s1_tune, TAUS_FULL if full else TAUS_FAST, owners=(True, False) if full else (True,))
    sm = summary(f05_table(select(sv, **cfg), truth, s1_rep), by="country")
    g = {r["group"]: r for r in sm.iter_rows(named=True)}
    res = {"setup": name, **cfg, "tune_f05": tf, **{k: g[k]["macro_f05"] for k in g},
           "precision": g["overall"]["precision"], "recall": g["overall"]["recall"]}
    res["beats_best"] = all(res.get(k, 0) > v for k, v in BEST.items()) and res["precision"] >= MIN_PRECISION
    log(f"{name}: {cfg} tune {tf:.4f} -> report {res['overall']:.4f} (US {res['country=US']:.4f}, India {res['country=India']:.4f}, "
        f"P {res['precision']:.4f}) beats v2: {res['beats_best']}")
    return res, cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--pruner-s1", type=int, default=100_000)
    a = ap.parse_args()
    out = a.out_dir
    out.mkdir(parents=True, exist_ok=True)
    raw = lambda name: find(a.input, name, not_parent=("normalised", "features_extra", "aug"))
    rd = lambda name, cols=None: pl.read_parquet(raw(name), columns=cols)

    # ------------------------------------------------------------------ data (as run_two_stage.py)
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
    feats = feature_sets(ftr.columns)["C"]
    ftr = ftr.join(rd("train_candidates.parquet", ["s1", "cand", "is_match"]), on=["s1", "cand"], how="left")
    ftr = ftr.filter(~pl.col("s1").is_in(pruner_s1.implode())).sort("s1", "cand")
    y = ftr["is_match"].cast(pl.Int8).to_numpy()
    fva = rd("features_val.parquet").sort("s1", "cand")
    fva = fva.join(truth.with_columns(pl.lit(True).alias("is_match")), on=["s1", "cand"], how="left").with_columns(pl.col("is_match").fill_null(False))
    base = fva.select("s1", "cand", "is_match")
    log(f"train {ftr.height:,} pairs; val {fva.height:,} pairs")

    # ------------------------------------------------------------------ records and counts (train + test, no labels)
    norm = {(sd, k): find(a.input, f"{sd}_source{k}.parquet", parent="normalised") for sd in ("train", "test") for k in (1, 2, 3)}
    idf, name_state, addr = country_counts(pl.concat([pl.scan_parquet(p).select("country", "state", "core_name", "addr_norm") for p in norm.values()]))
    s1_ids = pl.concat([ftr["s1"], fva["s1"]]).unique().implode()
    c_ids = pl.concat([ftr["cand"], fva["cand"]]).unique().implode()
    s1_rec = prepare_records(pl.scan_parquet(norm[("train", 1)]).select(REC_COLS).filter(pl.col("entity_id").is_in(s1_ids)).collect())
    c_rec = prepare_records(pl.concat([pl.scan_parquet(norm[("train", k)]).select(REC_COLS).filter(pl.col("entity_id").is_in(c_ids)).collect() for k in (2, 3)]))
    rec_set = c_rec.select("entity_id", "full_name", "addr_norm", "numbers")
    log(f"records: S1 {s1_rec.height:,}, candidates {c_rec.height:,}; counts idf {idf.height:,}, names {name_state.height:,}, addresses {addr.height:,}")

    # ------------------------------------------------------------------ stage 1 + set features (v2)
    oof, models1 = stage1_oof(ftr.select("s1", *feats), y, feats, log=log)
    p1_val = stage1_predict(models1, fva, feats)
    sf_tr, sf_va = set_features(ftr, oof, rec_set), set_features(fva, p1_val, rec_set)
    bx_tr = extra_features(ftr.select("s1", "cand"), s1_rec, c_rec, idf, name_state, addr)
    bx_va = extra_features(fva.select("s1", "cand"), s1_rec, c_rec, idf, name_state, addr)
    X_tr = ftr.select(feats).hstack(sf_tr.select(SET_FEATURES)).hstack(bx_tr.drop("s1", "cand"))
    X_va = fva.select(feats).hstack(sf_va.select(SET_FEATURES)).hstack(bx_va.drop("s1", "cand"))
    es = fold_of(ftr["s1"], 10, salt=7) == 0
    base_f = feats + SET_FEATURES
    log(f"B features built: {bx_tr.width - 2}")

    def run(fs, seed=SEED, params=None):
        b = fit(X_tr.select(fs).to_numpy(), y, es, fs, seed=seed, params=params)
        return b, predict(b, X_va.select(fs).to_numpy())

    # ------------------------------------------------------------------ step B: feature groups (1 seed, v2 params)
    results, preds = [], {}
    _, preds["v2 (C + set)"] = run(base_f)
    r, cfg_v2 = evaluate("v2 (C + set), 1 seed", preds["v2 (C + set)"], base, truth, s1_tune, s1_rep)
    results.append(r)
    for g, cols in B_GROUPS.items():
        _, preds[f"+{g}"] = run(base_f + cols)
        results.append(evaluate(f"+{g}", preds[f"+{g}"], base, truth, s1_tune, s1_rep)[0])
    all_b = [c for cols in B_GROUPS.values() for c in cols]
    _, preds["+all B"] = run(base_f + all_b)
    r_all, cfg_all = evaluate("+all B", preds["+all B"], base, truth, s1_tune, s1_rep)
    results.append(r_all)
    kept = [g for g in B_GROUPS if next(x for x in results if x["setup"] == f"+{g}")["overall"] > results[0]["overall"]]
    fs = base_f + all_b if r_all["overall"] >= max(x["overall"] for x in results) - 1e-4 else base_f + [c for g in kept for c in B_GROUPS[g]]
    log(f"feature set for tuning: {len(fs)} features (B groups kept: {kept}; all B {'used' if fs == base_f + all_b else 'not used'})")

    # error groups on the report half: v2 (1 seed) vs the chosen feature set
    eg_rows = []
    for name, cfg in (("v2 (C + set)", cfg_v2), ("+all B", cfg_all)):
        sv = base.with_columns(pl.Series("p", preds[name])).hstack(bx_va.drop("s1", "cand"))
        sel = select(sv, **cfg).select("s1", "cand", pl.lit(True).alias("sel"))
        x = sv.join(s1_rep, on="s1", how="semi").join(sel, on=["s1", "cand"], how="left").with_columns(pl.col("sel").fill_null(False))
        groups = {"empty address + near-identical name (true pairs, recall)": pl.col("is_match") & (pl.col("b3_empty_near") == 1),
                  "trade name at same address + number (true pairs, recall)": pl.col("is_match") & (pl.col("b4_trade") == 1),
                  "legal form differs, same address (non-matches, FP rate)": ~pl.col("is_match") & (pl.col("b2_legal_diff_addr_same") == 1),
                  "descriptor-word difference, same address (non-matches, FP rate)": ~pl.col("is_match") & (pl.col("b1_n_diff") > 0) & (pl.col("b4_same_addr") == 1)}
        for gname, cond in groups.items():
            d = x.filter(cond)
            eg_rows.append({"model": name, "group": gname, "pairs": d.height, "selected_share": d["sel"].mean() if d.height else None})
    eg = pl.DataFrame(eg_rows).pivot(on="model", index=["group"], values="selected_share")

    # ------------------------------------------------------------------ step A: hyperparameter search (tune half, fast grid)
    rng = random.Random(SEED)
    space = {"num_leaves": [63, 127, 255, 511], "min_data_in_leaf": [20, 50, 100, 200, 400], "learning_rate": [0.03, 0.05, 0.08],
             "feature_fraction": [0.6, 0.8, 1.0], "bagging_fraction": [0.7, 0.8, 1.0], "lambda_l1": [0.0, 0.1, 1.0], "lambda_l2": [0.0, 1.0, 5.0, 10.0]}
    trials = [{}] + [{k: rng.choice(v) for k, v in space.items()} for _ in range(a.trials)]
    hp_rows, best_hp = [], (None, {})
    for i, prm in enumerate(trials):
        b, p = run(fs, params=prm)
        tf, cfg = tune(base.with_columns(pl.Series("p", p)), truth, s1_tune, TAUS_FAST)
        hp_rows.append({"trial": i, **prm, "best_iter": b.best_iteration, "tune_f05": tf, **cfg})
        log(f"  trial {i}: {prm} -> tune {tf:.4f} (iter {b.best_iteration})")
        if best_hp[0] is None or tf > best_hp[0]:
            best_hp = (tf, prm)
    hp = pl.DataFrame(hp_rows, infer_schema_length=None).sort("tune_f05", descending=True)
    params = best_hp[1]

    # ------------------------------------------------------------------ seeds + CatBoost blend
    ps = []
    for s in range(a.seeds):
        ps.append(run(fs, seed=SEED + 100 + s, params=params)[1])
    p_lgb = np.mean(ps, axis=0)
    r_lgb, cfg_lgb = evaluate(f"LightGBM tuned, {a.seeds} seeds", p_lgb, base, truth, s1_tune, s1_rep)
    results.append(r_lgb)
    cands = {f"LightGBM tuned, {a.seeds} seeds": (p_lgb, r_lgb, cfg_lgb)}
    try:
        from catboost import CatBoostClassifier
        import catboost
        log(f"catboost {catboost.__version__}")
        cb = CatBoostClassifier(iterations=3000, learning_rate=0.08, depth=8, loss_function="Logloss", random_seed=SEED,
                                od_type="Iter", od_wait=100, thread_count=-1, verbose=0)
        Xt = X_tr.select(fs).to_numpy()
        cb.fit(Xt[~es], y[~es], eval_set=(Xt[es], y[es]))
        p_cb = cb.predict_proba(X_va.select(fs).to_numpy())[:, 1]
        log(f"catboost best iteration {cb.get_best_iteration()}")
        for name, p in (("CatBoost", p_cb), ("blend: mean prob", (p_lgb + p_cb) / 2),
                        ("blend: rank mean", (pl.Series(p_lgb).rank().to_numpy() + pl.Series(p_cb).rank().to_numpy()) / (2 * len(p_lgb)))):
            r, cfg = evaluate(name, p, base, truth, s1_tune, s1_rep)
            results.append(r)
            cands[name] = (p, r, cfg)
        cb.save_model(str(out / "catboost.cbm"))
    except Exception as e:  # CatBoost missing or failing must not lose the other results
        log(f"CatBoost skipped: {e!r}")

    # choose on the TUNE half (the report half only checks)
    best_name = max(cands, key=lambda k: cands[k][1]["tune_f05"])
    p_best, r_best, cfg_best = cands[best_name]
    res = pl.DataFrame(results, infer_schema_length=None)
    res.write_csv(out / "results.csv"); hp.write_csv(out / "hp_trials.csv"); eg.write_csv(out / "error_groups.csv")
    base.with_columns(pl.Series("p", p_best), pl.Series("p_lgb", p_lgb)).write_parquet(out / "val_scores.parquet")
    (out / "best_config.json").write_text(json.dumps({"setup": best_name, "features": fs, "lgb_params": {**PARAMS, **params},
                                                      "seeds": a.seeds, "selection": cfg_best, "report": r_best,
                                                      "beats_v2": r_best["beats_best"]}, indent=1, default=str))
    (out / "report.md").write_text("\n".join([
        "# Model v3 experiments (validation)\n", md(res) + "\n",
        f"**Chosen on the tune half: {best_name}** {cfg_best} -> report {r_best['overall']:.4f}; beats v2: {r_best['beats_best']}.\n",
        "## Error groups (report half, selected share)\n", md(eg) + "\n",
        "## Hyperparameter trials (tune half, fast grid)\n", md(hp, 25) + "\n"]))
    log("done")


if __name__ == "__main__":
    main()
