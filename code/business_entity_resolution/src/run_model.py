"""Model + selection, end to end on Kaggle: train, validate, choose, score test, write and validate the outputs.

1. Train LightGBM variants on features-v2 train (see model.py): A (all features), B (no f_n_methods),
   C (B without the pruner-derived features), each on the train S1 the blocking pruner never saw, plus A_all
   (A on all train S1, to measure the pruner-overlap effect).
2. Validation (99,994 S1, never used for training): the selection (threshold, one owner per S2/S3, margin) is
   tuned on the tune half (even S1 id) and reported on the report half (odd id): macro F0.5 overall, per country,
   singletons vs non-singletons, precision, recall. Every validation S1 counts, including those with no
   candidates.
3. France-like stress test on validation: true pairs found by only 1-2 search methods (French test candidates:
   median 2 methods vs 4 in train), mean probability and recall per variant; and how the blocking pruner treated
   them (kept share by method count, from the unpruned validation list).
4. Choice: the most France-robust variant (C, then B, then A) whose report-half F0.5 is within MAX_LOSS of the best.
5. Test: score the test features with the chosen model, select, write output/matching_results.tsv and
   output/candidate_pairs.tsv (candidates = the blocking v3 set), run the organiser validator (--check-ids), and
   the France checks: predicted matches per S1 by country, and where French matches sit in capped lists.

Usage: python src/run_model.py --input DIR --out-dir DIR --validator utils/validate_submission.py
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
from metric import per_entity  # noqa: E402
from model import feature_sets, importance, predict, train  # noqa: E402
from selection import f05_table, select, summary, tune  # noqa: E402
from submission import write_submission  # noqa: E402

SEED = 42
PRUNER_S1 = 100_000  # run_blocking_eval.py: split role == "train" .sample(100_000, seed=42)
MAX_LOSS = 0.005  # prefer a more France-robust variant if it loses less than 0.5 F0.5 points
ROBUST_ORDER = ["C", "B", "A"]
T0 = time.time()


def log(msg: str) -> None:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    print(f"[{time.time() - T0:7.1f}s peak {peak:5.1f} GB] {msg}", flush=True)


def find(root: Path, name: str) -> Path:
    hits = sorted(root.rglob(name))
    if not hits:
        raise FileNotFoundError(f"{name} not found under {root}")
    return hits[0]


def md(df: pl.DataFrame, max_rows: int = 60) -> str:
    df = df.head(max_rows)
    fmt = lambda v: f"{v:.4f}" if isinstance(v, float) else str(v)
    rows = [" | ".join(df.columns), " | ".join("---" for _ in df.columns)] + [" | ".join(fmt(v) for v in r) for r in df.iter_rows()]
    return "\n".join(f"| {r} |" for r in rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path, help="folder searched for every input file")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--validator", default=None)
    ap.add_argument("--variants", nargs="*", default=["A", "B", "C", "A_all"])
    ap.add_argument("--skip-test", action="store_true")
    ap.add_argument("--pruner-s1", type=int, default=PRUNER_S1, help="size of the pruner's train sample (100k; smaller only for tests)")
    a = ap.parse_args()
    out = a.out_dir
    out.mkdir(parents=True, exist_ok=True)
    rd = lambda name, cols=None: pl.read_parquet(find(a.input, name), columns=cols)

    # ---------------------------------------------------------------- data
    split = rd("g25_split.parquet")
    trn = split.filter(pl.col("role") == "train")["source1_entity_id"]
    pruner_s1 = trn.sample(min(a.pruner_s1, trn.len()), seed=SEED)
    val_s1 = split.filter(pl.col("role") == "val").select(pl.col("source1_entity_id").alias("s1"))
    country1 = rd("train_source1.parquet", ["entity_id", "country"]).rename({"entity_id": "s1"})
    val_s1 = val_s1.join(country1, on="s1", how="left")
    gt = rd("train_ground_truth.parquet")
    truth_val = (gt.join(val_s1, left_on="source1_entity_id", right_on="s1", how="semi")
                 .select(pl.col("source1_entity_id").alias("s1"), pl.col("matched_entity_ids").str.split(",").alias("cand"))
                 .explode("cand", empty_as_null=True).filter(pl.col("cand").is_not_null() & (pl.col("cand") != "")))
    half = pl.col("s1").str.extract(r"(\d+)$").cast(pl.Int64) % 2
    s1_tune, s1_rep = val_s1.filter(half == 0), val_s1.filter(half == 1)

    ftr = rd("features_train.parquet")
    labels = rd("train_candidates.parquet", ["s1", "cand", "is_match"])
    ftr = ftr.join(labels, on=["s1", "cand"], how="left")
    assert ftr["is_match"].null_count() == 0, "train features without a label row"
    in_pruner = ftr["s1"].is_in(pruner_s1.implode())
    fva = rd("features_val.parquet")
    fva = fva.join(truth_val.with_columns(pl.lit(True).alias("is_match")), on=["s1", "cand"], how="left").with_columns(
        pl.col("is_match").fill_null(False))
    log(f"train {ftr.height:,} pairs ({int(in_pruner.sum()):,} from the pruner's 100k S1); val {fva.height:,} pairs, "
        f"{val_s1.height:,} S1 (tune {s1_tune.height:,} / report {s1_rep.height:,}), {truth_val.height:,} true pairs")
    blocking_recall = fva["is_match"].sum() / truth_val.height
    sets = feature_sets(ftr.columns)

    # ---------------------------------------------------------------- train + validate every variant
    results, imps, scored_val, boosters, chosen_cfg = [], [], {}, {}, {}
    for v in a.variants:
        fs = sets[v.split("_")[0]]
        rows = ftr if v.endswith("_all") else ftr.filter(~in_pruner)
        log(f"variant {v}: training on {rows['s1'].n_unique():,} S1")
        t0 = time.time()
        b = train(rows.select("s1", *fs), rows["is_match"].cast(pl.Int8).to_numpy(), fs, log=log)
        boosters[v] = b
        b.save_model(str(out / f"model_{v}.txt"), num_iteration=b.best_iteration)
        imps.append(importance(b, v))
        sv = fva.select("s1", "cand", "is_match").with_columns(pl.Series("p", predict(b, fva)))
        scored_val[v] = sv
        grid = tune(sv, truth_val, s1_tune)
        best = grid.row(0, named=True)
        chosen_cfg[v] = {k: best[k] for k in ("tau", "margin", "one_owner")}
        grid.head(10).write_csv(out / f"tune_top_{v}.csv")
        rep = f05_table(select(sv, **chosen_cfg[v]), truth_val, s1_rep)
        sm = summary(rep, by="country")
        results.append({"variant": v, "features": len(fs), "train_s1": rows["s1"].n_unique(), "best_iter": b.best_iteration,
                        "train_s": time.time() - t0, **chosen_cfg[v], "tune_f05": best["macro_f05"],
                        **{r["group"]: r["macro_f05"] for r in sm.iter_rows(named=True)},
                        "precision": sm.filter(pl.col("group") == "overall")["precision"].item(),
                        "recall": sm.filter(pl.col("group") == "overall")["recall"].item(),
                        "pred_per_s1": sm.filter(pl.col("group") == "overall")["pred_per_s1"].item()})
        sm.write_csv(out / f"val_report_{v}.csv")
        log(f"variant {v}: tau {best['tau']}, margin {best['margin']}, one owner {best['one_owner']}; "
            f"report-half macro F0.5 {results[-1]['overall']:.4f}")
    res = pl.DataFrame(results)
    res.write_csv(out / "variants.csv")
    pl.concat(imps).write_csv(out / "importance.csv")

    # exact check of the vectorised F0.5 against metric.py on the report half (first variant)
    v0 = a.variants[0]
    pred0 = select(scored_val[v0], **chosen_cfg[v0])
    ids = s1_rep["s1"].to_list()
    pm = {s: [] for s in ids}
    for s, c in pred0.join(s1_rep, on="s1", how="semi").select("s1", "cand").iter_rows():
        pm[s].append(c)
    tm = {s: [] for s in ids}
    for s, c in truth_val.join(s1_rep, on="s1", how="semi").iter_rows():
        tm[s].append(c)
    exact = per_entity(pm, tm)["f05"].mean()
    fast = f05_table(pred0, truth_val, s1_rep)["f05"].mean()
    assert abs(exact - fast) < 1e-9, f"F0.5 mismatch: metric.py {exact} vs vectorised {fast}"
    log(f"F0.5 check vs metric.py: {exact:.6f} == {fast:.6f}")

    # ---------------------------------------------------------------- France-like stress test (validation)
    nm = fva.select("s1", "cand", pl.col("f_n_methods").cast(pl.Int32).alias("n_methods"), "is_match")
    nm = nm.with_columns(pl.when(pl.col("n_methods") >= 3).then(pl.lit("3+")).otherwise(pl.col("n_methods").cast(pl.String)).alias("methods"))
    stress = []
    for v, sv in scored_val.items():
        kept = select(sv, **chosen_cfg[v]).select("s1", "cand", pl.lit(True).alias("kept"))
        x = nm.join(sv.select("s1", "cand", "p"), on=["s1", "cand"]).join(kept, on=["s1", "cand"], how="left").with_columns(
            pl.col("kept").fill_null(False))
        stress.append(x.group_by("methods", "is_match").agg(pl.len().alias("pairs"), pl.col("p").mean().alias("mean_p"),
                                                            pl.col("p").median().alias("median_p"), pl.col("kept").mean().alias("selected_share"))
                      .with_columns(pl.lit(v).alias("variant")))
    stress = pl.concat(stress).sort("is_match", "methods", "variant", descending=[True, False, False])
    stress.write_csv(out / "stress_methods.csv")
    # the pruner: kept share of TRUE pairs by method count, from the unpruned validation starting list (u50)
    vc = rd("val_candidates.parquet", ["scope", "s1", "cand", "rank_name", "rank_name_city", "rank_name_addr",
                                       "rank_reverse", "rank_rare", "p_u50", "is_match"])
    vc = vc.filter((pl.col("scope") == "state") & pl.col("p_u50").is_not_null())
    vc = vc.with_columns(pl.sum_horizontal(*[pl.col(c).is_not_null() for c in vc.columns if c.startswith("rank_")]).alias("n_methods"),
                         pl.col("p_u50").rank("ordinal", descending=True).over("s1").alias("prank"))
    vc = vc.with_columns(((pl.col("p_u50") >= 0.01) & (pl.col("prank") <= 10)).alias("kept"),
                         pl.when(pl.col("n_methods") >= 3).then(pl.lit("3+")).otherwise(pl.col("n_methods").cast(pl.String)).alias("methods"))
    pruner = vc.group_by("methods", "is_match").agg(pl.len().alias("pairs"), pl.col("kept").mean().alias("pruner_kept_share"),
                                                    pl.col("p_u50").median().alias("median_p_u50")).sort("is_match", "methods", descending=[True, False])
    pruner.write_csv(out / "stress_pruner.csv")
    del vc

    # ---------------------------------------------------------------- choice
    best_f = res["overall"].max()
    ok = {r["variant"]: r["overall"] for r in res.iter_rows(named=True) if r["overall"] >= best_f - MAX_LOSS}
    choice = next((v for v in ROBUST_ORDER if v in ok), res.sort("overall", descending=True)["variant"][0])
    log(f"choice: {choice} (best report-half F0.5 {best_f:.4f}; within {MAX_LOSS}: {ok})")
    cfg = {"variant": choice, **chosen_cfg[choice], "val_report_f05": float(res.filter(pl.col("variant") == choice)["overall"].item()),
           "blocking_recall_val": float(blocking_recall), "max_loss_rule": MAX_LOSS, "features": boosters[choice].feature_name()}
    (out / "selection_config.json").write_text(json.dumps(cfg, indent=1))

    report = [
        "# Model + selection (feat-v2, LightGBM)\n",
        f"Validation: {val_s1.height:,} S1, {truth_val.height:,} true pairs; blocking recall of the candidate set "
        f"{blocking_recall:.4f}. Selection tuned on the tune half ({s1_tune.height:,} S1), reported on the report half "
        f"({s1_rep.height:,} S1). All-empty baseline F0.5 = singleton share. Vectorised F0.5 checked against metric.py "
        f"({exact:.6f}).\n",
        "## Variants (report half)\n", md(res) + "\n",
        f"**Choice: {choice}**: the most France-robust variant within {MAX_LOSS * 100:.1f} F0.5 points of the best. "
        f"Selection: tau {cfg['tau']}, margin {cfg['margin']}, one owner {cfg['one_owner']}.\n",
        f"## Chosen variant {choice}: report half by group\n", md(pl.read_csv(out / f"val_report_{choice}.csv")) + "\n",
        "## France-like stress test: pairs by number of search methods that found them\n", md(stress, 40) + "\n",
        "## Blocking pruner: kept share by number of search methods (unpruned validation list)\n", md(pruner) + "\n",
        "## Feature importance (gain share, top 20 per variant)\n",
        md(pl.concat(imps).group_by("variant", maintain_order=True).head(20).select("variant", "feature", "gain_share"), 100) + "\n",
    ]

    # ---------------------------------------------------------------- test
    if not a.skip_test:
        b = boosters[choice]
        tcols = list(dict.fromkeys(["s1", "cand", *b.feature_name(), "f_list_rank", "f_list_size", "f_n_methods"]))
        fte = pl.read_parquet(find(a.input, "features_test.parquet"), columns=tcols)
        p = predict(b, fte)
        s1_all = rd("test_source1.parquet", ["entity_id", "country"]).rename({"entity_id": "s1"})
        st = fte.select("s1", "cand", "f_list_rank", "f_list_size", "f_n_methods").with_columns(pl.Series("p", p)).join(s1_all, on="s1")
        del fte
        matched = select(st.select("s1", "cand", "p"), **chosen_cfg[choice])
        log(f"test: {st.height:,} pairs scored, {matched.height:,} matches")
        cand = rd("test_candidates.parquet", ["s1", "cand", "p_u50"])
        lists = cand.sort(["s1", "p_u50", "cand"], descending=[False, True, False]).group_by("s1", maintain_order=True).agg(pl.col("cand"))
        mlists = matched.sort(["s1", "p"], descending=[False, True]).group_by("s1", maintain_order=True).agg(pl.col("cand"))
        write_submission(out / "output", s1_all["s1"].to_list(), dict(zip(mlists["s1"].to_list(), mlists["cand"].to_list())),
                         dict(zip(lists["s1"].to_list(), lists["cand"].to_list())))
        del cand, lists
        log("wrote output/matching_results.tsv and output/candidate_pairs.tsv")
        val_msgs = None
        if a.validator:
            import importlib.util
            spec = importlib.util.spec_from_file_location("validate_submission", a.validator)
            vs = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(vs)
            with tempfile.TemporaryDirectory() as td:
                for k in (1, 2, 3):
                    rd(f"test_source{k}.parquet", ["entity_id"]).write_csv(Path(td) / f"test_source{k}.tsv", separator="\t")
                errors, warnings = vs.validate(str(out / "output/matching_results.tsv"), str(out / "output/candidate_pairs.tsv"),
                                               td, check_ids=True)
            val_msgs = {"errors": errors, "warnings": warnings}
            log(f"validator: {len(errors)} error(s), {len(warnings)} warning(s)")
        # France checks
        n = s1_all.join(matched.group_by("s1").len("n"), on="s1", how="left").with_columns(pl.col("n").fill_null(0))
        per_country = n.group_by("country").agg(pl.len().alias("s1"), pl.col("n").mean().alias("matches_per_s1"),
                                                (pl.col("n") > 0).mean().alias("share_with_match"),
                                                *[(pl.col("n") == k).mean().alias(f"share_{k}") for k in (0, 1, 2, 3)],
                                                (pl.col("n") >= 4).mean().alias("share_4plus")).sort("country")
        mk = matched.select("s1", "cand", pl.lit(True).alias("m"))
        sm = st.join(mk, on=["s1", "cand"], how="left").with_columns(pl.col("m").fill_null(False))
        capped = sm.filter(pl.col("f_list_size") >= 10)
        cap_pos = capped.filter(pl.col("m")).group_by("country").agg(
            pl.len().alias("matches_in_capped_lists"), pl.col("f_list_rank").mean().alias("mean_position"),
            (pl.col("f_list_rank") <= 3).mean().alias("share_pos_1_3"), (pl.col("f_list_rank") >= 9).mean().alias("share_pos_9_10")).sort("country")
        cap_rate = capped.group_by("country").agg((pl.col("m") & (pl.col("f_list_rank") >= 9)).sum().alias("matches_at_9_10"),
                                                  pl.col("s1").n_unique().alias("capped_s1")).sort("country")
        cap_pos = cap_pos.join(cap_rate, on="country")
        byp = sm.group_by("country").agg(pl.col("p").median().alias("median_p_all"), pl.col("p").quantile(0.9, "nearest").alias("p90_all"),
                                         pl.col("p").filter(pl.col("m")).median().alias("median_p_matched"),
                                         pl.col("f_n_methods").filter(pl.col("m")).mean().alias("methods_of_matches")).sort("country")
        for name, df in (("test_matches_per_country", per_country), ("test_capped_positions", cap_pos), ("test_scores", byp)):
            df.write_csv(out / f"{name}.csv")
        vtext = "not run" if val_msgs is None else f"{len(val_msgs['errors'])} errors, {len(val_msgs['warnings'])} warnings"
        report += ["## Test (no labels)\n", f"Matches: {matched.height:,} for {s1_all.height:,} test S1. Validator: "
                   f"{vtext}.\n",
                   "### Predicted matches per S1 by country\n", md(per_country) + "\n",
                   "### Capped lists (10 candidates): where the predicted matches sit (position by pruner score)\n", md(cap_pos) + "\n",
                   "### Scores by country\n", md(byp) + "\n"]
        (out / "test_check.json").write_text(json.dumps({"validator": val_msgs, "matches": matched.height}, indent=1, default=str))
        if val_msgs and val_msgs["errors"]:
            (out / "model_report.md").write_text("\n".join(report))
            raise SystemExit(f"VALIDATOR ERRORS: {val_msgs['errors'][:3]}")
    (out / "model_report.md").write_text("\n".join(report))
    log("done")


if __name__ == "__main__":
    main()
