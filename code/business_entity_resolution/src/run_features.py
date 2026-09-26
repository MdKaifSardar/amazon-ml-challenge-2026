"""Build pair features (table 6) for the train, val and test candidate tables, with a quality report.

  python src/run_features.py --input /kaggle/input --out-dir /kaggle/working [--splits train val test] [--prune val]
                             [--sample-s1 3000]

Inputs (searched under --input): {split}_candidates.parquet (blocking v3) and normalised/{train,test}_source{1,2,3}
.parquet (norm-v3). train / val hold TRAIN-side S1 (train / val roles of g25_split): their records come from the
train files. test holds test S1: records from the test files. Counts (name / token frequency, rare words) and the
state maps use ALL train + test records of each country, so every split gets the same definitions (no labels).
Each split is built one country at a time (the S1's country; pairs never cross countries) to bound memory: the
12.7M test pairs are at most ~6M per country. Rows come out grouped by country, in input order within a country.
Outputs: features_{split}.parquet (s1, cand, f_*); features_report.md (list sizes, checks, AUC per feature for the
labelled splits, NaN share and median per split x country, test-vs-train shift flags); features_profile.csv,
features_shift.csv, features_auc_{split}.csv, features_config.json. The label is read from the candidate table
and never written.
--sample-s1 N keeps N random S1 per split with ALL their candidates (lists stay whole), for test runs.
--prune SPLIT ...: cut those candidate tables to the submitted set (table 5) first. val_candidates.parquet from
blocking-eval-v3 is the whole starting list (u20 + u50 unions, 117 per S1); train_candidates.parquet (Job A) and
test_candidates.parquet (Job B) are already pruned. The rule is read from the blocking config.json (default
operating point: union u50, then candidates.prune with tau 0.01, cap 10), the same steps as run_blocking_job_a.py.
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
from features import (FEATURE_VERSION, build, count_stats, country_state_maps, feature_names, fill_record_states,
                      prepare_records)

SEED = 42
T0 = time.time()
NORM_COLS = ["entity_id", "country", "business_address", "full_name", "core_name", "legal", "addr_norm", "city", "state",
             "dept", "numbers"]
DEPT_NOTE = ("f_dept_agree (French departement) is all-NaN in train and val (no France there), so a model trained on "
             "them cannot use it; it only fires on French test pairs. f_state_agree is the useful location-agreement feature.")
SHIFT_NAN, SHIFT_MED = 0.10, 0.25  # flag: NaN share moves >= 0.10, or the median moves >= 0.25 x the reference p10-p90 range
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


def checks(f: pl.DataFrame, ref: pl.Schema | None) -> dict:
    """No labels needed: schema identical to the reference split (names, order, dtype), duplicate pairs, inf values."""
    x = f.select(feature_names(f))
    return {"rows": f.height, "s1": f["s1"].n_unique(), "n_features": x.width,
            "same_schema_as_ref": None if ref is None else f.schema == ref,
            "all_float32": all(t == pl.Float32 for t in x.schema.values()),
            "duplicate_pairs": int(f.select("s1", "cand").is_duplicated().sum()),
            "inf_values": int(x.select(pl.sum_horizontal(pl.all().is_infinite().sum())).item()) if x.width else 0}


def profile(f: pl.DataFrame, split: str, country: str) -> pl.DataFrame:
    """Per feature: NaN share, and median / p10 / p90 of the present values."""
    x = f.select(feature_names(f))
    rows = []
    for c in x.columns:
        v = x[c]
        present = v.filter(~v.is_nan())
        q = (lambda p: present.quantile(p, "nearest")) if present.len() else (lambda p: None)
        rows.append({"split": split, "country": country, "feature": c, "nan_share": float(v.is_nan().mean()) if v.len() else None,
                     "median": q(0.5), "p10": q(0.1), "p90": q(0.9)})
    return pl.DataFrame(rows, schema={"split": pl.String, "country": pl.String, "feature": pl.String,
                                      "nan_share": pl.Float64, "median": pl.Float64, "p10": pl.Float64, "p90": pl.Float64})


def shift_table(prof: pl.DataFrame) -> pl.DataFrame:
    """Test groups vs a train reference: the same country in train if it exists, else all train (France). Flags a
    NaN-share change >= SHIFT_NAN, a median move >= SHIFT_MED x the reference p10-p90 range, or a feature present on
    one side only."""
    ref = prof.filter(pl.col("split") == "train")
    train_countries = set(ref["country"]) - {"all"}
    rows = []
    for (country,), g in prof.filter((pl.col("split") == "test") & (pl.col("country") != "all")).group_by("country"):
        vs = country if country in train_countries else "all"
        j = g.join(ref.filter(pl.col("country") == vs), on="feature", suffix="_ref")
        rng = pl.max_horizontal((pl.col("p90_ref") - pl.col("p10_ref")).fill_null(0), pl.lit(1e-6))
        rows.append(j.select(
            pl.lit(country).alias("test_country"), pl.lit(f"train {vs}").alias("vs"), "feature",
            "nan_share", "nan_share_ref", "median", "median_ref",
            (((pl.col("nan_share") - pl.col("nan_share_ref")).abs() >= SHIFT_NAN)
             | ((pl.col("median") - pl.col("median_ref")).abs().fill_null(0) >= SHIFT_MED * rng)
             | (pl.col("median").is_null() != pl.col("median_ref").is_null())).alias("flag")))
    return pl.concat(rows).sort("test_country", "feature") if rows else pl.DataFrame()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--splits", nargs="*", default=["train", "val", "test"])
    ap.add_argument("--sample-s1", type=int, default=0, help="0 = all S1")
    ap.add_argument("--chunk", type=int, default=1_000_000)
    ap.add_argument("--prune", nargs="*", default=[], help="splits to cut to the submitted candidate set first")
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    side = lambda split: "test" if split == "test" else "train"

    files = {(sd, k): find(a.input, f"{sd}_source{k}.parquet", parent="normalised") for sd in ("train", "test") for k in (1, 2, 3)}
    log(f"normalised files: {files}")
    # Counts and state maps over ALL train + test records of each country (no labels): same meaning for every split.
    stats = count_stats([pl.scan_parquet(p) for (_, k), p in files.items() if k == 1],
                        [pl.scan_parquet(p) for (_, k), p in files.items() if k != 1])
    log(f"counts: records per country {stats.n}; {stats.token_df.height:,} (country, token), {stats.name_n.height:,} (country, name)")
    maps = country_state_maps(pl.concat([pl.read_parquet(p, columns=["country", "city", "dept", "state"]) for p in files.values()], how="vertical_relaxed"))
    log(f"state maps (rows per key): { {c: {k: m.height for k, m in v.items()} for c, v in maps.items()} }")

    def records(sd: str, ks: tuple[int, ...], ids: pl.DataFrame) -> pl.DataFrame:
        raw = pl.concat([pl.scan_parquet(files[(sd, k)]).select(NORM_COLS).join(ids.lazy(), on="entity_id", how="semi")
                         for k in ks], how="vertical_relaxed").collect()
        return prepare_records(fill_record_states(raw, maps))

    report = [f"# Pair features {FEATURE_VERSION}", "", f"Sample: {a.sample_s1 or 'all'} S1 per split. "
              f"Pruned to the submitted set here: {a.prune or 'none'}.", "",
              "Counts (f_freq_*, token df for the idf features, f_rare_*) and the state maps are computed over ALL "
              "train + test records of each country, identical for every split. No feature counts how many S1 lists "
              "a candidate appears in.", "", f"**Note for stream B:** {P_NOTE}", "", f"**Note:** {DEPT_NOTE}", ""]
    body, sizes, timing, check_rows, profiles, ref_schema, names = [], [], {}, [], [], None, None
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
        # The S1's country from its normalised record (pairs never cross countries).
        s1_country = (pl.read_parquet(files[(side(split), 1)], columns=["entity_id", "country"])
                      .select(pl.col("entity_id").alias("s1"), pl.col("country").fill_null("").alias("_country")))
        c = c.join(s1_country, on="s1", how="left", maintain_order="left")
        if c["_country"].null_count():
            raise ValueError(f"{split}: {c['_country'].null_count()} pairs whose S1 has no {side(split)} record")
        sizes.append(list_sizes(c, split))
        log(f"{split}: list sizes {sizes[-1]}")

        t_split, parts = time.time(), []
        for country in sorted(c["_country"].unique().to_list()):
            t = time.time()
            cc = c.filter(pl.col("_country") == country).drop("_country")
            s1_rec = records(side(split), (1,), cc.select(pl.col("s1").unique().alias("entity_id")))
            pool_rec = records(side(split), (2, 3), cc.select(pl.col("cand").unique().alias("entity_id")))
            f = build(cc, s1_rec, pool_rec, stats, chunk=a.chunk)
            del s1_rec, pool_rec
            assert f.select("s1", "cand").equals(cc.select("s1", "cand")), "row order changed"
            parts.append(a.out_dir / f"_part_{split}_{country}.parquet")
            f.write_parquet(parts[-1])
            ref_schema, names = ref_schema or f.schema, names or feature_names(f)
            profiles.append(profile(f, split, country))
            check_rows.append({"split": split, "country": country, **checks(f, ref_schema)})
            secs = time.time() - t
            timing[f"{split}/{country}"] = {"rows": f.height, "seconds": round(secs, 1), "rows_per_s": round(f.height / max(secs, 1e-9))}
            log(f"{split}/{country}: {f.shape} in {secs:.1f} s; checks {check_rows[-1]}")
            del f, cc
        out = a.out_dir / f"features_{split}.parquet"
        pl.scan_parquet(parts).sink_parquet(out)
        for part in parts:
            part.unlink()
        f = pl.read_parquet(out)
        check_rows.append({"split": split, "country": "all", **checks(f, ref_schema)})
        profiles.append(profile(f, split, "all"))
        timing[split] = {"rows": f.height, "seconds": round(time.time() - t_split, 1)}
        log(f"{split}: {f.shape} -> {out.name}; checks {check_rows[-1]}")
        if "is_match" in c.columns:
            y = f.select("s1", "cand").join(c.select("s1", "cand", "is_match"), on=["s1", "cand"], how="left",
                                            maintain_order="left")["is_match"]
            tab = auc_table(f, y.cast(pl.Int8).to_numpy())
            tab.write_csv(a.out_dir / f"features_auc_{split}.csv")
            body += [f"## {split}: NaN share and AUC per feature ({f.height:,} pairs)", "", md(tab), ""]
        del f, c

    prof = pl.concat(profiles)
    prof.write_csv(a.out_dir / "features_profile.csv")
    chk = pl.DataFrame(check_rows)
    ok = chk.select(pl.col("same_schema_as_ref").fill_null(True).all() & (pl.col("duplicate_pairs") == 0).all()
                    & (pl.col("inf_values") == 0).all() & pl.col("all_float32").all()).item()
    report += ["## Candidate lists", "", md(pl.DataFrame(sizes)), "",
               f"## Checks (no labels needed): {'all passed' if ok else 'FAILED'}", "",
               "same_schema_as_ref = same column names, order and dtype as the first part built "
               f"({a.splits[0]}, first country).", "", md(chk), "", *body]
    sh = shift_table(prof)
    if sh.height:
        sh.write_csv(a.out_dir / "features_shift.csv")
        flagged = sh.filter(pl.col("flag")).drop("flag")
        report += [f"## Test vs train shift flags ({flagged.height} of {sh.height} feature x country rows)", "",
                   f"Reference: the same country in train, else all train (France). Flag: NaN share moves by >= "
                   f"{SHIFT_NAN}, or the median moves by >= {SHIFT_MED} x the reference p10-p90 range, or a feature "
                   "is present on one side only.", "", md(flagged) if flagged.height else "No flags.", ""]
    wide = (prof.with_columns((pl.col("split") + " " + pl.col("country")).alias("g"))
            .pivot(on="g", index="feature", values=["nan_share", "median"]))
    report += ["## NaN share and median per split x country", "", md(wide), "",
               "## Timing", "", md(pl.DataFrame([{"part": s, **v} for s, v in timing.items()], strict=False)), ""]
    (a.out_dir / "features_report.md").write_text("\n".join(report))
    (a.out_dir / "features_config.json").write_text(json.dumps(
        {"feature_version": FEATURE_VERSION, "sample_s1": a.sample_s1, "splits": a.splits, "pruned": a.prune,
         "list_sizes": sizes, "checks": check_rows, "checks_ok": ok, "timing": timing, "features": names,
         "notes": [P_NOTE, DEPT_NOTE]}, indent=1, default=str))
    log(f"checks ok: {ok}")
    if not ok:
        raise SystemExit("feature checks failed, see features_report.md")
    log("done")


if __name__ == "__main__":
    main()
