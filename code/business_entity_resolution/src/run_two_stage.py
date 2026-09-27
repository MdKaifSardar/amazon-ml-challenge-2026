"""Model v2 experiments (blocking, candidates and features-v2 unchanged): error analysis, selection rules,
two-stage model with set features, seed averaging; then test outputs with the best setup.

Validation protocol as in run_model.py: selection tuned on the tune half (even S1 id), reported on the report
half (odd id); every validation S1 counts. A setup is kept only if report-half F0.5 beats the v1 reference
overall AND for US and India. Training S1 exclude the pruner's 100k (in-sample pruner features).

Usage: python src/run_two_stage.py --input DIR --out-dir DIR --validator utils/validate_submission.py [--seeds 3]
"""
import argparse
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
except NameError:  # Kaggle notebook: modules are written to /tmp/src
    sys.path.insert(0, "/tmp/src")
from model import feature_sets  # noqa: E402
from selection import f05_table, select, summary  # noqa: E402
from submission import write_submission  # noqa: E402
from two_stage import SET_FEATURES, fit, fold_of, predict, set_features, stage1_oof, stage1_predict  # noqa: E402

SEED = 42
REF = {"overall": 0.9717121937196413, "country=US": 0.9804, "country=India": 0.9586}  # model-v1 variant C, report half
TAUS = [round(x, 3) for x in np.arange(0.30, 0.96, 0.025)]
T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s peak {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20:5.1f} GB] {msg}", flush=True)


def find(root: Path, name: str, parent: str | None = None) -> Path:
    hits = sorted(p for p in root.rglob(name) if parent is None or p.parent.name == parent)
    if not hits:
        raise FileNotFoundError(f"{name} not found under {root}")
    return hits[0]


def md(df: pl.DataFrame, n: int = 60) -> str:
    df = df.head(n)
    f = lambda v: f"{v:.4f}" if isinstance(v, float) else str(v)
    rows = [" | ".join(df.columns), " | ".join("---" for _ in df.columns)] + [" | ".join(f(v) for v in r) for r in df.iter_rows()]
    return "\n".join(f"| {r} |" for r in rows)


def tune_all(sv: pl.DataFrame, truth, s1s, rules: bool) -> pl.DataFrame:
    """Grid over tau, margin, one_owner (+ tau2 secondary rule when set features exist)."""
    rows = []
    for tau in TAUS:
        for margin in (0.0, 0.5, 0.7, 0.8):
            for owner in (False, True):
                for tau2 in ([None] + ([round(tau - d, 3) for d in (0.1, 0.2, 0.3) if tau - d > 0.05] if rules else [])):
                    cfg = dict(tau=tau, margin=margin, one_owner=owner, tau2=tau2)
                    t = f05_table(select(sv, **cfg), truth, s1s)
                    rows.append({**cfg, "macro_f05": t["f05"].mean()})
    return pl.DataFrame(rows).sort("macro_f05", descending=True)


def evaluate(name: str, sv: pl.DataFrame, truth, s1_tune, s1_rep, rules: bool) -> dict:
    grid = tune_all(sv, truth, s1_tune, rules)
    best = {k: grid.row(0, named=True)[k] for k in ("tau", "margin", "one_owner", "tau2")}
    sm = summary(f05_table(select(sv, **best), truth, s1_rep), by="country")
    g = {r["group"]: r for r in sm.iter_rows(named=True)}
    res = {"setup": name, **best, "tune_f05": grid["macro_f05"][0],
           **{k: g[k]["macro_f05"] for k in g}, "precision": g["overall"]["precision"], "recall": g["overall"]["recall"]}
    res["kept_vs_v1"] = all(res.get(k, 0) > v for k, v in REF.items())
    log(f"{name}: {best} -> report F0.5 {res['overall']:.4f} (US {res['country=US']:.4f}, India {res['country=India']:.4f})")
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--validator", default=None)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--skip-test", action="store_true")
    ap.add_argument("--pruner-s1", type=int, default=100_000)
    a = ap.parse_args()
    out = a.out_dir
    out.mkdir(parents=True, exist_ok=True)
    rd = lambda name, cols=None, parent=None: pl.read_parquet(find(a.input, name, parent), columns=cols)
    raw = lambda name: sorted(p for p in a.input.rglob(name) if p.parent.name != "normalised")[0]  # organiser copy, not norm-v3

    # ------------------------------------------------------------------ data
    split = rd("g25_split.parquet")
    trn = split.filter(pl.col("role") == "train")["source1_entity_id"]
    pruner_s1 = trn.sample(min(a.pruner_s1, trn.len()), seed=SEED)
    val_s1 = split.filter(pl.col("role") == "val").select(pl.col("source1_entity_id").alias("s1")).join(
        pl.read_parquet(raw("train_source1.parquet"), columns=["entity_id", "country"]).rename({"entity_id": "s1"}), on="s1", how="left")
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
    log(f"train {ftr.height:,} pairs / {ftr['s1'].n_unique():,} S1; val {fva.height:,} pairs; {len(feats)} features (set C)")

    # candidate records (S2/S3) for set features and examples
    cols = ["entity_id", "business_name", "business_address", "full_name", "addr_norm", "numbers"]
    ids = pl.concat([ftr["cand"], fva["cand"]]).unique().implode()
    rec_tr = pl.concat([pl.scan_parquet(find(a.input, f"train_source{k}.parquet", "normalised")).select(cols)
                        .filter(pl.col("entity_id").is_in(ids)).collect() for k in (2, 3)])
    s1_rec = pl.scan_parquet(find(a.input, "train_source1.parquet", "normalised")).select("entity_id", "business_name", "business_address") \
        .filter(pl.col("entity_id").is_in(val_s1["s1"].implode())).collect()

    # ------------------------------------------------------------------ stage 1 (OOF on train)
    oof, models1 = stage1_oof(ftr.select("s1", *feats), y, feats, log=log)
    p1_val = stage1_predict(models1, fva, feats)
    sv1 = fva.select("s1", "cand", "is_match").with_columns(pl.Series("p", p1_val))
    results = [evaluate("stage1 (C, 5-fold avg)", sv1, truth, s1_tune, s1_rep, rules=False)]

    # ------------------------------------------------------------------ step 1: error analysis (report half)
    cfg1 = {k: results[0][k] for k in ("tau", "margin", "one_owner", "tau2")}
    sel = select(sv1, **cfg1).select("s1", "cand", pl.lit(True).alias("sel"))
    ea = (sv1.join(s1_rep, on="s1", how="semi").join(sel, on=["s1", "cand"], how="left").with_columns(pl.col("sel").fill_null(False))
          .join(val_s1, on="s1").join(fva.select("s1", "cand", "f_num_jacc", "f_addr_tsr", "f_name_full_tsort", "f_name_nolegal_ratio",
                                                  "f_legal_same", "f_addr_missing", "f_num_only_s1", "f_num_only_cand"), on=["s1", "cand"]))
    fn = ea.filter(pl.col("is_match") & ~pl.col("sel"))
    fp = ea.filter(~pl.col("is_match") & pl.col("sel"))
    band = (pl.when(pl.col("p") < 0.1).then(pl.lit("<0.1")).when(pl.col("p") < 0.3).then(pl.lit("0.1-0.3"))
            .when(pl.col("p") < 0.5).then(pl.lit("0.3-0.5")).when(pl.col("p") < cfg1["tau"]).then(pl.lit(f"0.5-{cfg1['tau']}"))
            .otherwise(pl.lit(">=tau (lost to margin / owner)")))
    fn_tab = fn.with_columns(band.alias("band"), pl.col("cand").str.slice(0, 2).alias("source")).group_by("country", "source", "band").len().sort("country", "source", "band")
    names = lambda d: (d.join(s1_rec.rename({"entity_id": "s1", "business_name": "s1_name", "business_address": "s1_addr"}), on="s1", how="left")
                       .join(rec_tr.select(pl.col("entity_id").alias("cand"), pl.col("business_name").alias("c_name"),
                                           pl.col("business_address").alias("c_addr")), on="cand", how="left"))
    fn_ex = names(fn.sort("p", descending=True).sample(min(30, fn.height), seed=SEED)).select(
        "country", "p", "s1_name", "c_name", "s1_addr", "c_addr", "f_num_jacc", "f_addr_tsr", "f_name_full_tsort", "f_legal_same")
    fp_ex = names(fp.sort("p", descending=True).head(30)).select(
        "country", "p", "s1_name", "c_name", "s1_addr", "c_addr", "f_num_jacc", "f_addr_tsr", "f_name_full_tsort", "f_legal_same")
    fn_tab.write_csv(out / "ea_missed_by_band.csv"); fn_ex.write_csv(out / "ea_missed_examples.csv"); fp_ex.write_csv(out / "ea_false_pos_examples.csv")
    ea_summary = {"missed_true_candidates": fn.height, "false_positives": fp.height, "report_true_pairs": int(truth.join(s1_rep, on="s1", how="semi").height),
                  "missed_by_country": dict(fn.group_by("country").len().iter_rows()), "fp_by_country": dict(fp.group_by("country").len().iter_rows())}
    log(f"error analysis: {ea_summary}")

    # ------------------------------------------------------------------ step 3: set features + stage 2
    sf_tr = set_features(ftr, oof, rec_tr)
    sf_va = set_features(fva, p1_val, rec_tr)
    x2 = ftr.select("s1", *feats).hstack(sf_tr.select(SET_FEATURES))
    f2 = feats + SET_FEATURES
    m2 = x2.select(f2).to_numpy()
    es = fold_of(ftr["s1"], 10, salt=7) == 0
    mv = fva.select(feats).hstack(sf_va.select(SET_FEATURES)).select(f2).to_numpy()
    models2, pv = [], []
    for s in range(a.seeds):
        b = fit(m2, y, es, f2, seed=SEED + s)
        models2.append(b)
        pv.append(predict(b, mv))
        log(f"  stage-2 seed {s}: best iteration {b.best_iteration}")
        if s == 0:
            sv2a = fva.select("s1", "cand", "is_match").hstack(sf_va.select("s_sim_name", "s_num_eq")).with_columns(pl.Series("p", pv[0]))
            results.append(evaluate("stage2 (1 seed)", sv2a, truth, s1_tune, s1_rep, rules=True))
    sv2 = fva.select("s1", "cand", "is_match").hstack(sf_va.select("s_sim_name", "s_num_eq")).with_columns(pl.Series("p", np.mean(pv, axis=0)))
    results.append(evaluate(f"stage2 ({a.seeds} seeds avg)", sv2, truth, s1_tune, s1_rep, rules=True))
    res = pl.DataFrame(results, infer_schema_length=None)
    res.write_csv(out / "results.csv")
    imp = (pl.DataFrame({"feature": f2, "gain": models2[0].feature_importance("gain").astype(float)})
           .with_columns((pl.col("gain") / pl.col("gain").sum()).alias("share")).sort("gain", descending=True))
    imp.write_csv(out / "importance_stage2.csv")

    # calibration (tune half) of the best stage-2 probabilities
    cal = (sv2.join(s1_tune, on="s1", how="semi").with_columns((pl.col("p") * 10).floor().clip(0, 9).alias("bin"))
           .group_by("bin").agg(pl.len().alias("pairs"), pl.col("p").mean().alias("mean_p"), pl.col("is_match").mean().alias("match_rate")).sort("bin"))
    cal.write_csv(out / "calibration.csv")

    best_row = res.sort("overall", descending=True).row(0, named=True)
    chosen = best_row["setup"]
    cfg = {k: best_row[k] for k in ("tau", "margin", "tau2")}
    report = ["# Model v2: two-stage + selection rules\n", md(res) + "\n",
              f"Chosen: **{chosen}** {cfg} (one owner is always applied on test: an S2/S3 matches at most one S1, and on "
              "test all S1 compete, unlike validation).\n",
              "## Error analysis (stage 1, report half)\n", "```\n" + json.dumps(ea_summary, indent=1) + "\n```\n",
              md(fn_tab, 80) + "\n", "### Missed true pairs (30 random)\n", md(fn_ex, 30) + "\n",
              "### Highest-probability false positives\n", md(fp_ex, 30) + "\n",
              "## Stage-2 importance (top 25)\n", md(imp, 25) + "\n", "## Calibration (tune half)\n", md(cal) + "\n"]

    # ------------------------------------------------------------------ test
    if not a.skip_test:
        fte = rd("features_test.parquet").sort("s1", "cand")
        tids = fte["cand"].unique().implode()
        rec_te = pl.concat([pl.scan_parquet(find(a.input, f"test_source{k}.parquet", "normalised")).select(cols)
                            .filter(pl.col("entity_id").is_in(tids)).collect() for k in (2, 3)])
        p1_te = stage1_predict(models1, fte, feats)
        if chosen.startswith("stage1"):
            st = fte.select("s1", "cand").with_columns(pl.Series("p", p1_te), pl.lit(None, pl.Float32).alias("s_sim_name"), pl.lit(None, pl.Float32).alias("s_num_eq"))
        else:
            sf_te = set_features(fte, p1_te, rec_te)
            mt = fte.select(feats).hstack(sf_te.select(SET_FEATURES)).select(f2).to_numpy()
            nm = 1 if "1 seed" in chosen else len(models2)
            st = fte.select("s1", "cand").hstack(sf_te.select("s_sim_name", "s_num_eq")).with_columns(
                pl.Series("p", np.mean([predict(b, mt) for b in models2[:nm]], axis=0)))
        matched = select(st, tau=cfg["tau"], margin=cfg["margin"], one_owner=True, tau2=cfg["tau2"])
        s1_all = pl.read_parquet(raw("test_source1.parquet"), columns=["entity_id", "country"]).rename({"entity_id": "s1"})
        cand = rd("test_candidates.parquet", ["s1", "cand", "p_u50"])
        lists = cand.sort(["s1", "p_u50", "cand"], descending=[False, True, False]).group_by("s1", maintain_order=True).agg(pl.col("cand"))
        ml = matched.sort(["s1", "p"], descending=[False, True]).group_by("s1", maintain_order=True).agg(pl.col("cand"))
        write_submission(out / "output", s1_all["s1"].to_list(), dict(zip(ml["s1"].to_list(), ml["cand"].to_list())),
                         dict(zip(lists["s1"].to_list(), lists["cand"].to_list())))
        vm = None
        if a.validator:
            import importlib.util
            spec = importlib.util.spec_from_file_location("vs", a.validator); vs = importlib.util.module_from_spec(spec); spec.loader.exec_module(vs)
            with tempfile.TemporaryDirectory() as td:
                for k in (1, 2, 3):
                    pl.read_parquet(raw(f"test_source{k}.parquet"), columns=["entity_id"]).write_csv(Path(td) / f"test_source{k}.tsv", separator="\t")
                e, w = vs.validate(str(out / "output/matching_results.tsv"), str(out / "output/candidate_pairs.tsv"), td, check_ids=True)
            vm = {"errors": e, "warnings": w}
            log(f"validator: {len(e)} errors, {len(w)} warnings")
        n = s1_all.join(matched.group_by("s1").len("n"), on="s1", how="left").with_columns(pl.col("n").fill_null(0))
        pc = n.group_by("country").agg(pl.col("n").mean().alias("matches_per_s1"), (pl.col("n") == 0).mean().alias("share_no_match")).sort("country")
        fr = s1_all.filter(pl.col("country") == "France").sample(10, seed=SEED)
        s1te = pl.scan_parquet(find(a.input, "test_source1.parquet", "normalised")).select("entity_id", "business_name", "business_address") \
            .filter(pl.col("entity_id").is_in(fr["s1"].implode())).collect()
        ex = (matched.join(fr, on="s1", how="semi").join(rec_te.select(pl.col("entity_id").alias("cand"), pl.col("business_name").alias("c_name"),
                                                                        pl.col("business_address").alias("c_addr")), on="cand", how="left")
              .join(s1te.rename({"entity_id": "s1"}), on="s1", how="right").select("s1", "business_name", "business_address", "p", "c_name", "c_addr"))
        pc.write_csv(out / "test_matches_per_country.csv"); ex.write_csv(out / "test_france_examples.csv")
        report += ["## Test\n", f"Matches {matched.height:,}; validator {vm}.\n", md(pc) + "\n", "### 10 random French S1\n", md(ex, 60) + "\n"]
        (out / "test_check.json").write_text(json.dumps({"validator": vm, "matches": matched.height}, default=str))
    (out / "report.md").write_text("\n".join(report))
    (out / "config.json").write_text(json.dumps({"chosen": chosen, **cfg, "one_owner_test": True, "features": f2,
                                                 "results": results}, indent=1, default=str))
    for i, b in enumerate(models1):
        b.save_model(str(out / f"stage1_fold{i}.txt"), num_iteration=b.best_iteration)
    for i, b in enumerate(models2):
        b.save_model(str(out / f"stage2_seed{i}.txt"), num_iteration=b.best_iteration)
    sv2.join(sv1.select("s1", "cand", pl.col("p").alias("p1")), on=["s1", "cand"]).write_parquet(out / "val_scores.parquet")
    log("done")


if __name__ == "__main__":
    main()
