"""Model v4 (experiment + test outputs in one job): v3 two-stage model plus
  (1) cross-S1 features: is this candidate a confident match of ANOTHER S1? (46% of v2 false positives were
      records that truly belong to another S1; one owner only resolves the case where both S1 score it high), and
  (2) an unseen-country simulation (leave one country out on validation): train stage 2 on US only and score India
      as if it were new, and the reverse. It measures (a) how the best threshold moves for a country the model never
      saw and (b) whether pseudo-labels of the new country help. Both are applied to test countries that are not in
      train (France) only if they help in BOTH directions; otherwise France is treated like the train countries.

  python src/run_model_v4.py --input DIR --config best_config.json --out-dir DIR [--validator PATH] [--sample N]

Validation (as v3): stage 1 = 5-fold OOF on the train split without the pruner's 100k S1; stage 2 trained on the
same S1; selection tuned on the validation tune half (even S1 id), the report half (odd id) is the check.
Keep rule: the cross-S1 features are used only if the report half beats v3 overall AND for US and India with
precision >= 0.985; otherwise the v3 feature set is used (test outputs are always written).
Approximation in the simulation: stage 1 and its set features are trained on both countries (only stage 2 is
country-held-out), so the measured shift is a lower bound of the real one.
"""
import argparse
import gc
import json
import resource
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import polars as pl

try:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
except NameError:
    sys.path.insert(0, "/tmp/src")
from model import feature_sets  # noqa: E402
from pipeline import counts, find, normalised_files, raw_file, records, stage2_frame  # noqa: E402
from selection import f05_table, select, summary, tune  # noqa: E402
from submission import write_submission  # noqa: E402
from two_stage import PARAMS, fit, fold_of, predict, stage1_oof, stage1_predict  # noqa: E402

SEED = 42
X_FEATURES = ["x_p_other", "x_n_other05", "x_lead"]
PL_POS, PL_NEG = 0.95, 0.05  # pseudo-label cut-offs (stage-2 probability)
TAU_CLIP = (0.3, 0.9)
T0 = time.time()
REPORT: list[str] = []


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s peak {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20:5.1f} GB] {msg}", flush=True)


def note(msg: str) -> None:
    REPORT.append(msg)
    log(msg)


def cross_features(pairs: pl.DataFrame, p1: np.ndarray) -> pl.DataFrame:
    """pairs: s1, cand of ALL S1 that compete for the same records (one side: train+val, or one test country).
    For each pair: the best stage-1 probability of the same candidate with ANOTHER S1 (0 if none), how many other
    S1 score it > 0.5, and this pair's lead over that best other S1. Returns the 3 columns in pairs' order."""
    x = pairs.select("cand").with_columns(pl.Series("p1", p1, dtype=pl.Float64)).with_row_index("_i")
    x = x.with_columns(pl.col("p1").rank("ordinal", descending=True).over("cand").alias("_r"))
    g = (x.sort("p1", descending=True).group_by("cand")
         .agg(pl.col("p1").first().alias("_t1"), pl.col("p1").get(1, null_on_oob=True).alias("_t2"), (pl.col("p1") > 0.5).sum().alias("_n05")))
    x = x.join(g, on="cand", how="left").sort("_i")
    other = pl.when(pl.col("_r") == 1).then(pl.col("_t2")).otherwise(pl.col("_t1")).fill_null(0.0)
    return x.select(other.cast(pl.Float32).alias("x_p_other"),
                    (pl.col("_n05") - (pl.col("p1") > 0.5).cast(pl.UInt32)).cast(pl.Float32).alias("x_n_other05"),
                    (pl.col("p1") - other).cast(pl.Float32).alias("x_lead"))


def fit_seeds(m, y, es, feats, params, seeds, w=None):
    out = []
    for s in seeds:
        out.append(fit(m, y, es, feats, seed=SEED + 100 + s, params=params) if w is None else fit_w(m, y, es, feats, params, SEED + 100 + s, w))
    return out


def fit_w(m, y, es, feats, params, seed, w):
    import lightgbm as lgb
    from two_stage import EARLY_STOP, ROUNDS
    p = {**PARAMS, **(params or {}), "seed": seed}
    dtr = lgb.Dataset(m[~es], y[~es], weight=w[~es], feature_name=feats, free_raw_data=True)
    dva = lgb.Dataset(m[es], y[es], reference=dtr)
    return lgb.train(p, dtr, ROUNDS, valid_sets=[dva], callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])


def avg(models, m):
    return np.mean([predict(b, m) for b in models], axis=0)


def score(p, pairs, truth, s1s, sel):
    tab = f05_table(select(pairs.select("s1", "cand").with_columns(pl.Series("p", p)), **sel), truth, s1s)
    return tab


def best_sel(p, pairs, truth, s1s):
    t = tune(pairs.select("s1", "cand").with_columns(pl.Series("p", p)), truth, s1s, owners=(True,))
    r = t.row(0, named=True)
    return {"tau": r["tau"], "margin": r["margin"], "one_owner": True}, r["macro_f05"]


def md(df: pl.DataFrame) -> str:
    f = lambda v: f"{v:.4f}" if isinstance(v, float) else str(v)
    rows = [" | ".join(df.columns), " | ".join("---" for _ in df.columns)] + [" | ".join(f(v) for v in r) for r in df.iter_rows()]
    return "\n".join(f"| {r} |" for r in rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--validator", default=None)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--sample", type=int, default=0, help="smoke test: this many train S1 / val S1 / test S1 per country")
    ap.add_argument("--pruner-s1", type=int, default=100_000)
    a = ap.parse_args()
    out = a.out_dir
    (out / "model").mkdir(parents=True, exist_ok=True)
    cfg = json.loads((find(a.config, "best_config.json") if a.config.is_dir() else a.config).read_text())
    params = {k: v for k, v in cfg["lgb_params"].items() if k not in PARAMS or PARAMS[k] != v}
    fs_v3 = cfg["features"]
    rd = lambda name, cols=None: pl.read_parquet(raw_file(a.input, name), columns=cols)

    # ---------------- data (train side) ----------------
    s1c = rd("train_source1.parquet", ["entity_id", "country"]).rename({"entity_id": "s1"})
    split = rd("g25_split.parquet")
    trn = split.filter(pl.col("role") == "train")["source1_entity_id"]
    pruner_s1 = trn.sample(min(a.pruner_s1, trn.len()), seed=SEED)
    val_s1 = split.filter(pl.col("role") == "val").select(pl.col("source1_entity_id").alias("s1")).join(s1c, on="s1", how="left")
    gt = rd("train_ground_truth.parquet")
    truth = (gt.join(val_s1, left_on="source1_entity_id", right_on="s1", how="semi")
             .select(pl.col("source1_entity_id").alias("s1"), pl.col("matched_entity_ids").str.split(",").alias("cand"))
             .explode("cand", empty_as_null=True).filter(pl.col("cand").is_not_null() & (pl.col("cand") != "")))
    half = pl.col("s1").str.extract(r"(\d+)$").cast(pl.Int64) % 2
    fall = rd("features_train.parquet")
    c_feats = feature_sets(fall.columns)["C"]
    fall = fall.join(rd("train_candidates.parquet", ["s1", "cand", "is_match"]), on=["s1", "cand"], how="left")
    fva = rd("features_val.parquet").join(truth.with_columns(pl.lit(True).alias("is_match")), on=["s1", "cand"], how="left") \
        .with_columns(pl.col("is_match").fill_null(False)).select(fall.columns)
    if a.sample:
        keep_tr = fall.select("s1").unique().sort("s1").sample(a.sample, seed=SEED)
        keep_va = fva.select("s1").unique().sort("s1").sample(min(a.sample, fva["s1"].n_unique()), seed=SEED)
        fall, fva = fall.join(keep_tr, on="s1", how="semi"), fva.join(keep_va, on="s1", how="semi")
        val_s1 = val_s1.join(keep_va, on="s1", how="semi")
    s1_tune, s1_rep = val_s1.filter(half == 0), val_s1.filter(half == 1)
    is_pr = fall["s1"].is_in(pruner_s1.implode())
    ftr, fpr = fall.filter(~is_pr).sort("s1", "cand"), fall.filter(is_pr)
    fva = fva.sort("s1", "cand")
    del fall
    y = ftr["is_match"].cast(pl.Int8).to_numpy()
    log(f"train {ftr.height:,} pairs / {ftr['s1'].n_unique():,} S1; pruner S1 pairs {fpr.height:,}; val {fva.height:,} pairs")

    # ---------------- stage 1 ----------------
    oof, models1 = stage1_oof(ftr.select("s1", *c_feats), y, c_feats, log=log)
    for i, b in enumerate(models1):
        b.save_model(str(out / "model" / f"stage1_fold{i}.txt"), num_iteration=b.best_iteration)
    p1_va, p1_pr = stage1_predict(models1, fva, c_feats), stage1_predict(models1, fpr, c_feats)

    # cross-S1 features over every train-side S1 (train + pruner + val compete for the same S2/S3 pool)
    side = pl.concat([ftr.select("s1", "cand"), fpr.select("s1", "cand"), fva.select("s1", "cand")])
    xf = cross_features(side, np.concatenate([oof, p1_pr, p1_va]))
    xf_tr, xf_va = xf.slice(0, ftr.height), xf.slice(ftr.height + fpr.height, fva.height)
    del side, xf, fpr
    gc.collect()

    # ---------------- stage-2 frames ----------------
    norm = normalised_files(a.input)
    cnt = counts(norm)
    s1_rec, c_rec = records(norm, "train", pl.concat([ftr["s1"], fva["s1"]]).unique(), pl.concat([ftr["cand"], fva["cand"]]).unique())
    X = stage2_frame(ftr, oof, s1_rec, c_rec, cnt, c_feats).hstack(xf_tr)
    V = stage2_frame(fva, p1_va, s1_rec, c_rec, cnt, c_feats).hstack(xf_va)
    del s1_rec, c_rec
    gc.collect()
    log(f"stage-2 frames: train {X.shape}, val {V.shape}")
    va_pairs = fva.select("s1", "cand")
    s1_rep_c = s1_rep.select("s1", "country")
    es = fold_of(ftr["s1"], 10, salt=7) == 0

    # ---------------- step 1: cross-S1 features vs v3 (1 seed each) ----------------
    fs_x = fs_v3 + X_FEATURES
    res = {}
    for name, fs in (("v3", fs_v3), ("v3+cross", fs_x)):
        m, mv = X.select(fs).to_numpy(), V.select(fs).to_numpy()
        b = fit(m, y, es, fs, seed=SEED + 100, params=params)
        p = predict(b, mv)
        tmask = va_pairs["s1"].is_in(s1_tune["s1"].implode()).to_numpy()
        sel, f_tune = best_sel(p[tmask], va_pairs.filter(pl.Series(tmask)), truth, s1_tune)
        sm = summary(score(p, va_pairs, truth, s1_rep_c, sel), by="country")
        res[name] = {"sel": sel, "tune": f_tune, "summary": sm, "booster": b, "p": p}
        note(f"### {name} (1 seed): selection {sel}, tune half {f_tune:.4f}\n\n{md(sm)}\n")
        del m, mv
    g = lambda sm, grp, col="macro_f05": sm.filter(pl.col("group") == grp)[col][0]
    b3, bx = res["v3"]["summary"], res["v3+cross"]["summary"]
    use_x = (g(bx, "overall") > g(b3, "overall") and all(g(bx, f"country={c}") >= g(b3, f"country={c}") for c in ("US", "India"))
             and g(bx, "overall", "precision") >= 0.985)
    fs = fs_x if use_x else fs_v3
    note(f"cross-S1 features {'KEPT' if use_x else 'NOT kept'}: report half {g(bx, 'overall'):.4f} vs v3 {g(b3, 'overall'):.4f}")
    m, mv = X.select(fs).to_numpy(), V.select(fs).to_numpy()

    # ---------------- step 2: unseen-country simulation (stage 2 only) ----------------
    ctry_tr = ftr.select("s1").join(s1c, on="s1", how="left")["country"].to_numpy()
    ctry_va = va_pairs.join(s1c, on="s1", how="left")["country"].to_numpy()
    tune_va = va_pairs["s1"].is_in(s1_tune["s1"].implode()).to_numpy()
    sim = []
    for src, tgt in (("US", "India"), ("India", "US")):
        on_src, on_tgt = ctry_tr == src, ctry_tr == tgt
        b = fit(m[on_src], y[on_src], es[on_src], fs, seed=SEED + 200, params=params)
        pv = predict(b, mv)
        f = lambda mask: (pv[mask], va_pairs.filter(pl.Series(mask)))
        s_src = s1_tune.filter(pl.col("country") == src)
        t_tune = s1_tune.filter(pl.col("country") == tgt)
        t_rep = s1_rep.filter(pl.col("country") == tgt).select("s1")
        sel_src, _ = best_sel(*f(tune_va & (ctry_va == src)), truth, s_src)
        sel_tgt, _ = best_sel(*f(tune_va & (ctry_va == tgt)), truth, t_tune)
        rep_mask = (~tune_va) & (ctry_va == tgt)
        f_src = score(pv[rep_mask], va_pairs.filter(pl.Series(rep_mask)), truth, t_rep, sel_src)["f05"].mean()
        f_tgt = score(pv[rep_mask], va_pairs.filter(pl.Series(rep_mask)), truth, t_rep, sel_tgt)["f05"].mean()
        # pseudo-labels: target-country train pairs scored by the source model, confident ones only
        pt = predict(b, m[on_tgt])
        conf = (pt >= PL_POS) | (pt <= PL_NEG)
        idx = np.concatenate([np.where(on_src)[0], np.where(on_tgt)[0][conf]])
        yl = y.copy()
        yl[np.where(on_tgt)[0][conf]] = (pt[conf] >= PL_POS).astype(np.int8)
        pl_acc = float((yl[np.where(on_tgt)[0][conf]] == y[np.where(on_tgt)[0][conf]]).mean()) if conf.any() else float("nan")
        b_pl = fit(m[idx], yl[idx], es[idx], fs, seed=SEED + 200, params=params)
        pv_pl = predict(b_pl, mv)
        f_pl = score(pv_pl[rep_mask], va_pairs.filter(pl.Series(rep_mask)), truth, t_rep, sel_src)["f05"].mean()
        sim.append({"src": src, "tgt": tgt, "tau_src": sel_src["tau"], "margin_src": sel_src["margin"], "tau_tgt": sel_tgt["tau"],
                    "margin_tgt": sel_tgt["margin"], "f_at_src_sel": f_src, "f_at_tgt_sel": f_tgt, "f_pseudo_at_src_sel": f_pl,
                    "pseudo_pairs": int(conf.sum()), "pseudo_label_acc": pl_acc})
        note(f"simulation {src} -> {tgt}: {sim[-1]}")
        del b, b_pl
    d_tau = [s["tau_tgt"] - s["tau_src"] for s in sim]
    same_sign = all(d > 0 for d in d_tau) or all(d < 0 for d in d_tau)
    use_shift = same_sign and all(s["f_at_tgt_sel"] > s["f_at_src_sel"] for s in sim)
    shift = float(np.mean(d_tau)) if use_shift else 0.0
    use_pl = all(s["f_pseudo_at_src_sel"] > s["f_at_src_sel"] for s in sim)
    note(f"unseen-country threshold shift: {shift:+.3f} ({'applied' if use_shift else 'not applied'}; per direction {d_tau}); "
         f"pseudo-labelling {'APPLIED' if use_pl else 'not applied'}")

    # ---------------- final stage 2 ----------------
    seeds = range(a.seeds)
    models2 = [res["v3+cross" if use_x else "v3"]["booster"]] + fit_seeds(m, y, es, fs, params, list(seeds)[1:])
    for i, b in enumerate(models2):
        b.save_model(str(out / "model" / f"stage2_lgb_seed{i}.txt"), num_iteration=b.best_iteration)
    pv = avg(models2, mv)
    sel, f_tune = best_sel(pv[tune_va], va_pairs.filter(pl.Series(tune_va)), truth, s1_tune)
    sm = summary(score(pv, va_pairs, truth, s1_rep_c, sel), by="country")
    note(f"### final stage 2 ({len(models2)} seeds, {'v3+cross' if use_x else 'v3'} features): selection {sel}, tune {f_tune:.4f}\n\n{md(sm)}\n")
    pl.DataFrame({"s1": va_pairs["s1"], "cand": va_pairs["cand"], "p": pv}).write_parquet(out / "val_scores.parquet")
    train_countries = set(s1c["country"].unique().to_list())
    cfg_out = {"features": fs, "use_cross": use_x, "selection": sel, "unseen_tau_shift": shift, "pseudo_label": use_pl,
               "train_countries": sorted(train_countries), "simulation": sim, "lgb_params": cfg["lgb_params"], "seeds": len(models2)}
    (out / "model" / "model_config.json").write_text(json.dumps(cfg_out, indent=1, default=str))
    del X, V, mv
    gc.collect()

    # ---------------- test ----------------
    t1 = rd("test_source1.parquet", ["entity_id", "country"]).rename({"entity_id": "s1"})
    fpath = raw_file(a.input, "features_test.parquet")
    parts = []
    for country in sorted(t1["country"].unique().to_list()):
        s1_ct = t1.filter(pl.col("country") == country).select("s1")
        if a.sample:
            s1_ct = s1_ct.sort("s1").sample(min(a.sample, s1_ct.height), seed=SEED)
        fx = pl.scan_parquet(fpath).join(s1_ct.lazy(), on="s1", how="semi").collect().sort("s1", "cand")
        if not fx.height:
            continue
        p1 = np.mean([b.predict(fx.select(c_feats).to_numpy()) for b in models1], axis=0)
        xt = cross_features(fx.select("s1", "cand"), p1)  # candidate lists never cross countries
        s1_rec, c_rec = records(norm, "test", fx["s1"].unique(), fx["cand"].unique())
        T = stage2_frame(fx, p1, s1_rec, c_rec, cnt, c_feats).hstack(xt)
        mt = T.select(fs).to_numpy()
        p = avg(models2, mt)
        unseen = country not in train_countries
        if unseen and use_pl:
            conf = (p >= PL_POS) | (p <= PL_NEG)
            m2 = np.vstack([m, mt[conf]])
            y2 = np.concatenate([y, (p[conf] >= PL_POS).astype(np.int8)])
            es2 = np.concatenate([es, np.zeros(int(conf.sum()), dtype=bool)])
            mpl = fit_seeds(m2, y2, es2, fs, params, list(seeds))
            p = avg(mpl, mt)
            note(f"{country}: pseudo-labelled {int(conf.sum()):,} pairs ({(p[conf] >= 0.5).mean():.3f} positive after retraining)")
            del m2, y2, es2, mpl
        tau_c = min(max(sel["tau"] + (shift if unseen else 0.0), TAU_CLIP[0]), TAU_CLIP[1])
        parts.append(fx.select("s1", "cand").with_columns(pl.Series("p", p), pl.lit(tau_c).alias("tau")))
        log(f"{country}: {fx.height:,} pairs scored, tau {tau_c:.3f}")
        del fx, T, mt, s1_rec, c_rec, xt
        gc.collect()
    st = pl.concat(parts)
    st.select("s1", "cand", "p").write_parquet(out / "test_scores.parquet")
    # per-country threshold, then the usual one owner + margin over ALL test S1
    matched = select(st.filter(pl.col("p") >= pl.col("tau")).select("s1", "cand", "p"), tau=0.0, margin=sel["margin"], one_owner=True)
    note(f"test: {matched.height:,} matches")
    if a.sample:
        (out / "report.md").write_text("# Model v4 (SAMPLE run)\n\n" + "\n".join(REPORT))
        log("sample run: no submission files")
        return

    sub = out / "output"
    sub.mkdir(exist_ok=True)
    cand = rd("test_candidates.parquet", ["s1", "cand", "p_u50"])
    assert cand.select("s1", "cand").sort("s1", "cand").equals(st.select("s1", "cand").sort("s1", "cand")), "scored pairs != candidate set"
    lists = cand.sort(["s1", "p_u50", "cand"], descending=[False, True, False], nulls_last=True).group_by("s1", maintain_order=True).agg(pl.col("cand"))
    ml = matched.sort(["s1", "p"], descending=[False, True]).group_by("s1", maintain_order=True).agg(pl.col("cand"))
    write_submission(sub, t1["s1"].to_list(), dict(zip(ml["s1"].to_list(), ml["cand"].to_list())),
                     dict(zip(lists["s1"].to_list(), lists["cand"].to_list())))
    if a.validator:
        import importlib.util
        spec = importlib.util.spec_from_file_location("vs", a.validator); vs = importlib.util.module_from_spec(spec); spec.loader.exec_module(vs)
        with tempfile.TemporaryDirectory() as td:
            for k in (1, 2, 3):
                rd(f"test_source{k}.parquet", ["entity_id"]).write_csv(Path(td) / f"test_source{k}.tsv", separator="\t")
            e, w = vs.validate(str(sub / "matching_results.tsv"), str(sub / "candidate_pairs.tsv"), td, check_ids=True)
        note(f"validator (--check-ids): {len(e)} errors, {len(w)} warnings {e[:3]}")
    n = t1.join(matched.group_by("s1").len("n"), on="s1", how="left").with_columns(pl.col("n").fill_null(0))
    pc = n.group_by("country").agg(pl.len().alias("s1"), pl.col("n").mean().alias("matches_per_s1"),
                                   (pl.col("n") == 0).mean().alias("share_no_match")).sort("country")
    note("## test matches per S1\n\n" + md(pc) + "\n")
    (out / "report.md").write_text("# Model v4\n\n" + "\n".join(REPORT))
    log("done")


if __name__ == "__main__":
    main()
