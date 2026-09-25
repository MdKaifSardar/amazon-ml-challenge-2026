"""Blocking + candidate selection evaluation on the validation split against the FULL train S2/S3 pool.

The candidate set size is ranked alongside F0.5, so every design is judged on two axes: recall of true pairs
AND candidates per S1 (average, median, p95). Configs evaluated per scope:
  fixed     forward top-k (all methods) + reverse top-m (no margins), + rare tokens
  reverse   reverse top-1/2/3 with a margin rule, alone or with a small forward top-k
  adaptive  forward within a margin of the S1's best (or above a minimum cosine), up to k per method, + reverse
  pruned    a cheap pruner (candidates.py) on a generous union, keep p >= tau up to a cap; trained on candidates
            of --train-queries TRAIN-split S1 (never validation S1)
Operating points: the best recall at <= 5, <= 10 and <= 20 candidates per S1 on average.

Usage: python src/run_blocking_eval.py --norm-dir artifacts/normalised --data-dir data_parquet \
           --split outputs/eda/g25_split.parquet --out-dir artifacts/blocking_eval
--norm-dir / --split may be folders to search (e.g. /kaggle/input). Outputs in --out-dir:
  curve.csv (every config: family, scope, config, per-country and ALL recall / avg / median / p95 / reduction),
  pareto.csv, operating_points.csv, recall_groups.csv, timing.csv, test_estimate.csv, blocking_curve.png,
  val_candidates.parquet (validation S1 x candidates of the generous union, with pruner probability),
  blocking_report.md, config.json.
Pieces: `--stage search --shard i/n [--countries X]` writes one piece of the search (pieces/piece_*.parquet);
`--stage evaluate --pieces DIR...` merges complete piece sets and evaluates; `--stage all` does both.
Scopes: "state" (searches within state buckets) or "country" (whole country). Country-scope searches whose
projected time would pass --budget-min are skipped and reported.
"""
import argparse
import json
import re
import resource
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process

try:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
except NameError:  # Kaggle notebook: modules are written to /tmp/src
    sys.path.insert(0, "/tmp/src")
import blocking  # noqa: E402
from blocking import FORWARD, NONE, SCOPES, _pairs_evaluated, block_country, buckets, gpu_info, texts, token_df  # noqa: E402
from candidates import METHODS, add_best, features, list_stats, matrix, pareto, prune, rule, select, train_pruner  # noqa: E402

SEED = 42
TARGETS = [5, 10, 20]  # operating points: best recall at <= this many candidates per S1 on average
COLS = ["entity_id", "country", "business_name", "core_name", "city", "addr_norm", "state"]
NON_LATIN = r"[\p{L}&&[^\p{Latin}]]"
T0 = time.time()
UNION = dict(rev_m=3, fwd_k=20, rare_k=10)  # generous union the pruner works on


def configs() -> list[tuple[str, dict, int | None]]:
    """(family, rule kwargs, cap) for every selection config."""
    out = []
    for k in [0, 1, 2, 3, 5, 10, 20, 50]:
        for m in [0, 1, 2, 3, 5]:
            for rare in [0, 10]:
                if k or m or rare:
                    out.append(("fixed", dict(fwd_k=k, rev_m=m, rare_k=rare), None))
    for m in [1, 2, 3]:
        for mg in [0.02, 0.05, 0.1, 0.2, None]:
            if m == 1 and mg is not None:
                continue
            for k in [0, 1, 2]:
                out.append(("reverse", dict(rev_m=m, rev_margin=mg, fwd_k=k), None))
    for k in [3, 5, 10, 20]:
        for mg in [0.02, 0.05, 0.1, 0.2]:
            for mn in [None, 0.9]:
                for rv in [(1, None), (2, 0.05), (2, 0.1), (3, 0.1)]:
                    for cap in [None, 10, 20]:
                        out.append(("adaptive", dict(fwd_k=k, fwd_margin=mg, fwd_min=mn, rev_m=rv[0], rev_margin=rv[1]), cap))
    return out


def log(msg: str) -> None:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    print(f"[{time.time() - T0:7.1f}s peak {peak:5.1f} GB] {msg}", flush=True)


def find(root: Path, name: str, parent: str | None = None) -> Path:
    if root.is_file():
        return root
    hits = sorted(p for p in root.rglob(name) if parent is None or p.parent.name == parent)
    if not hits:
        raise FileNotFoundError(f"{name} (parent {parent}) not found under {root}")
    return hits[0]


def md(df: pl.DataFrame, max_rows: int = 60) -> str:
    df = df.head(max_rows)
    fmt = lambda v: f"{v:.4f}" if isinstance(v, float) else str(v)
    rows = [" | ".join(df.columns), " | ".join("---" for _ in df.columns)] + [" | ".join(fmt(v) for v in r) for r in df.iter_rows()]
    return "\n".join(f"| {r} |" for r in rows)


def test_pairs(te: dict, country: str, scopes: list[str]) -> list[dict]:
    """Similarity evaluations a test run would need for this country, per scope (same cost model as timing)."""
    t1 = te[1].filter(pl.col("country") == country)
    tp = pl.concat([te[2].filter(pl.col("country") == country), te[3].filter(pl.col("country") == country)])
    out = []
    for sc in scopes:
        tsb, tpb = buckets(t1, tp, sc)
        out.append({"scope": sc, "country": country, "test_s1": t1.height, "test_pool": tp.height,
                    "forward_pairs": _pairs_evaluated(tsb, tpb),
                    "reverse_pairs": _pairs_evaluated(tpb[tpb != NONE], tsb, with_none=False)})
    return out


rates: dict[str, float] = {}  # seconds per similarity evaluation, "forward" / "reverse", measured so far
BUDGET_S = 600 * 60.0


def measured_rates(timing: list[dict]) -> dict[str, float]:
    out = {}
    for kind in ("forward", "reverse"):
        rows = [t for t in timing if t.get("pairs_evaluated") and not t.get("skipped")
                and (t["method"] == "reverse") == (kind == "reverse")]
        if rows:
            out[kind] = sum(t["search_s"] for t in rows) / sum(t["pairs_evaluated"] for t in rows)
    return out


def over_budget(tim: list[dict]):
    """skip() for block_country: skip a search if its projected time (pairs x measured rate) would pass the
    budget, keeping 20% of the budget for evaluation. State-scope searches are never skipped."""
    def skip(scope: str, method: str, pairs: float) -> bool:
        if scope == "state":
            return False
        r = {**rates, **measured_rates(tim)}.get("reverse" if method == "reverse" else "forward")
        return r is not None and (time.time() - T0) + pairs * r > 0.8 * BUDGET_S
    return skip


def plot(curve: pl.DataFrame, front: pl.DataFrame, ops: pl.DataFrame, floor: float, path: Path) -> None:
    """Recall vs average candidates per S1 (log x), one panel per scope; Pareto front and operating points."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log("matplotlib not available; skipping the plot (curve.csv has the data)")
        return
    colors = {"fixed": "#2a78d6", "reverse": "#eb6834", "adaptive": "#1baf7a", "pruned": "#eda100"}  # fixed order
    scopes = curve["scope"].unique(maintain_order=True).to_list()
    fig, axes = plt.subplots(1, len(scopes), figsize=(6.5 * len(scopes), 4.8), squeeze=False, facecolor="#fcfcfb")
    for ax, sc in zip(axes[0], scopes):
        ax.set_facecolor("#fcfcfb")
        c = curve.filter((pl.col("scope") == sc) & (pl.col("country") == "ALL"))
        for fam, col in colors.items():
            d = c.filter(pl.col("family") == fam)
            if d.height:
                ax.scatter(d["avg"], d["recall"], s=16, color=col, alpha=0.7, linewidths=0, label=fam)
        f = front.filter(pl.col("scope") == sc).sort("avg")
        ax.plot(f["avg"], f["recall"], color="#3d3d3a", lw=1.5, label="Pareto front")
        ax.axvline(floor, color="#8a8a85", lw=1, ls="--")
        ax.text(floor, 0.02 + ax.get_ylim()[0], f" true matches per S1 = {floor:.2f}", color="#5f5e5a", fontsize=8, va="bottom")
        for r in ops.filter((pl.col("scope") == sc) & (pl.col("country") == "ALL")).iter_rows(named=True):
            ax.scatter([r["avg"]], [r["recall"]], s=64, facecolors="none", edgecolors="#1a1a19", linewidths=1.5, zorder=5)
            ax.annotate(f"<= {r['target']}: {r['recall']:.3f} @ {r['avg']:.1f}", (r["avg"], r["recall"]),
                        textcoords="offset points", xytext=(6, -12), fontsize=8, color="#1a1a19")
        ax.set_xscale("log")
        ax.set_xlabel("average candidates per validation S1 (log scale)", color="#3d3d3a")
        ax.set_ylabel("recall of true pairs", color="#3d3d3a")
        ax.set_title(f"Blocking: recall vs candidate set size ({sc} scope)", color="#1a1a19", fontsize=11, loc="left")
        ax.grid(True, color="#e8e7e1", lw=0.8)
        for s in ax.spines.values():
            s.set_color("#c3c2b7")
        ax.tick_params(colors="#5f5e5a")
        ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor="#fcfcfb")
    plt.close(fig)


def search_piece(a, tr: dict, te: dict, role: pl.DataFrame, country: str, shard: int, n_shards: int, out: Path) -> None:
    """One piece of the search: S1 rows with row % n == shard (forward, rare) and pool rows with row % n == shard
    (reverse). The TF-IDF vocabulary is rebuilt identically in every piece (fixed seed, same corpus), so the union
    of all n pieces equals one full run. Writes piece_<country>_<i>of<n>.parquet and its timing JSON."""
    tag = f"piece_{country}_{shard}of{n_shards}"
    log(f"{tag}: start")
    s1 = tr[1].filter(pl.col("country") == country)
    pool = pl.concat([tr[2].filter(pl.col("country") == country), tr[3].filter(pl.col("country") == country)])
    test_all = pl.concat([t.filter(pl.col("country") == country) for t in te.values()], how="diagonal_relaxed")
    train_all = pl.concat([s1, pool])
    corpus = {rep: pl.concat([texts(train_all)[rep], texts(test_all)[rep]]) for rep in FORWARD}
    df_counts = token_df(pl.concat([train_all["core_name"], test_all["core_name"]]))
    in_shard = lambda n: np.arange(n) % n_shards == shard
    wanted = s1["entity_id"].is_in(role["s1"].implode()).to_numpy()
    qmask = wanted & in_shard(s1.height)
    tim: list[dict] = []
    long, _ = block_country(s1, pool, qmask, corpus, df_counts, a.k_max, a.m_max, scopes=tuple(a.scopes),
                            skip=over_budget(tim), log=log, gpu_check=shard == 0, timing=tim,
                            pool_mask=in_shard(pool.height), reverse_keep=wanted)
    rates.update(measured_rates(tim))
    long = long.with_columns(pl.Series("s1", s1["entity_id"].to_numpy()[long["qi"].to_numpy()]),
                             pl.Series("cand", pool["entity_id"].to_numpy()[long["pi"].to_numpy()])).drop("qi", "pi")
    long.write_parquet(out / f"{tag}.parquet")
    (out / f"{tag}_timing.json").write_text(json.dumps([{"country": country, "shard": f"{shard}/{n_shards}", **t} for t in tim]))
    log(f"{tag}: {long.height:,} rows written")


def load_pieces(dirs: list[Path], countries: list[str]) -> tuple[pl.DataFrame, list[dict]]:
    """Merge piece files; every country needs a complete set 0..n-1 of one n."""
    files = sorted({f for d in dirs for f in d.rglob("piece_*of*.parquet")})
    have: dict[str, dict[int, set[int]]] = {}
    for f in files:
        c, i, n = re.fullmatch(r"piece_(.+)_(\d+)of(\d+)\.parquet", f.name).groups()
        have.setdefault(c, {}).setdefault(int(n), set()).add(int(i))
    for c in countries:
        ok = [n for n, got in have.get(c, {}).items() if got == set(range(n))]
        if len(ok) != 1:
            raise RuntimeError(f"pieces for {c} incomplete or mixed: {have.get(c)} (files under {dirs})")
    long = pl.concat([pl.read_parquet(f) for f in files])
    timing = [t for f in files for t in json.loads(f.with_name(f.stem + "_timing.json").read_text())]
    log(f"merged {len(files)} pieces: {long.height:,} rows")
    return long, timing


def main() -> None:
    global BUDGET_S
    ap = argparse.ArgumentParser()
    ap.add_argument("--norm-dir", default="artifacts/normalised")
    ap.add_argument("--data-dir", default="data_parquet")
    ap.add_argument("--split", default="outputs/eda/g25_split.parquet")
    ap.add_argument("--out-dir", default="artifacts/blocking_eval")
    ap.add_argument("--k-max", type=int, default=50)
    ap.add_argument("--m-max", type=int, default=5)
    ap.add_argument("--train-queries", type=int, default=100_000, help="TRAIN-split S1 queried to fit the pruner (0 = no pruner)")
    ap.add_argument("--scopes", nargs="+", default=list(SCOPES), choices=SCOPES)
    ap.add_argument("--budget-min", type=float, default=600, help="wall-clock budget for the searches (minutes)")
    ap.add_argument("--stage", choices=["all", "search", "evaluate"], default="all",
                    help="search: write result pieces only; evaluate: merge pieces and evaluate; all: both")
    ap.add_argument("--shard", default="0/1", help="search stage: piece i/n (S1 queries and pool rows split n ways)")
    ap.add_argument("--countries", nargs="*", default=None, help="search stage: countries to search (default all)")
    ap.add_argument("--pieces", nargs="*", default=None, help="evaluate stage: folders holding piece files")
    a = ap.parse_args()
    BUDGET_S = a.budget_min * 60
    log(f"search device: {gpu_info()}; scopes {a.scopes}; budget {a.budget_min:.0f} min")
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    norm = lambda sp, k, cols=COLS: pl.read_parquet(find(Path(a.norm_dir), f"{sp}_source{k}.parquet", "normalised"), columns=cols)
    split = pl.read_parquet(find(Path(a.split), "g25_split.parquet"))
    tr = {k: norm("train", k) for k in (1, 2, 3)}
    te = {k: norm("test", k, ["country", "core_name", "city", "addr_norm", "state"]) for k in (1, 2, 3)}
    present = tr[1]["entity_id"]  # the local sample holds only part of the split
    val_ids = split.filter((pl.col("role") == "val") & pl.col("source1_entity_id").is_in(present.implode()))["source1_entity_id"]
    trn = split.filter((pl.col("role") == "train") & pl.col("source1_entity_id").is_in(present.implode()))["source1_entity_id"]
    trn_ids = trn.sample(min(a.train_queries, trn.len()), seed=SEED) if a.train_queries else trn.head(0)
    role = pl.concat([pl.DataFrame({"s1": val_ids, "role": "val"}), pl.DataFrame({"s1": trn_ids, "role": "train"})])
    gt = pl.read_parquet(find(Path(a.data_dir), "train_ground_truth.parquet"))
    truth = (
        gt.filter(pl.col("source1_entity_id").is_in(role["s1"].implode()))
        .select(pl.col("source1_entity_id").alias("s1"), pl.col("matched_entity_ids").str.split(",").alias("cand"))
        .explode("cand", empty_as_null=True).filter(pl.col("cand").is_not_null() & (pl.col("cand") != ""))
    )
    log(f"loaded; {val_ids.len():,} validation S1 + {trn_ids.len():,} train S1 (pruner), {truth.height:,} true pairs")

    countries = tr[1]["country"].unique().sort().to_list()
    shard, n_shards = (int(x) for x in a.shard.split("/"))
    if a.stage in ("all", "search"):
        (out / "pieces").mkdir(exist_ok=True)
        for country in a.countries or countries:
            search_piece(a, tr, te, role, country, shard, n_shards, out / "pieces")
        if a.stage == "search":
            log("search stage done")
            return
    piece_dirs = [Path(d) for d in a.pieces] if a.pieces else [out / "pieces"]
    long, timing = load_pieces(piece_dirs, countries)

    # per-country facts for the evaluation (no search needed)
    test_est, attrs, pool_size = [], [], {}
    for country in countries:
        s1 = tr[1].filter(pl.col("country") == country)
        pool = pl.concat([tr[2].filter(pl.col("country") == country), tr[3].filter(pl.col("country") == country)])
        pool_size[country] = pool.height
        test_core = pl.concat([t.filter(pl.col("country") == country)["core_name"] for t in te.values()])
        cnt = pl.concat([s1["core_name"], pool["core_name"], test_core]).alias("core_name").value_counts(name="n_core")
        attrs.append(s1.join(role, left_on="entity_id", right_on="s1", how="semi")
                     .select(pl.col("entity_id").alias("s1"), "country", "core_name").join(cnt, on="core_name", how="left"))
    for country in sorted(set(te[1]["country"].unique().to_list()) | set(countries)):
        test_est += test_pairs(te, country, a.scopes)  # includes test-only countries (France)

    # ---- one row per (scope, S1, candidate): rank/score per method, reverse best, forward best per S1
    best_rev = (long.filter(pl.col("method") == "reverse").group_by("scope", "s1", "cand")
                .agg(pl.col("best").max().alias("best_reverse")))
    wide = long.pivot(on="method", index=["scope", "s1", "cand"], values=["rank", "score"], aggregate_function="min")
    del long
    for mth in METHODS:
        for c in ("rank", "score"):
            if f"{c}_{mth}" not in wide.columns:
                wide = wide.with_columns(pl.lit(None, dtype=pl.Int64 if c == "rank" else pl.Float32).alias(f"{c}_{mth}"))
    wide = add_best(wide.join(best_rev, on=["scope", "s1", "cand"], how="left"))
    wide = wide.join(truth.select("s1", "cand", pl.lit(True).alias("is_match")), on=["s1", "cand"], how="left").with_columns(
        pl.col("is_match").fill_null(False)).join(role, on="s1").sort("scope", "s1", "cand")  # order-independent of pieces
    log(f"blocking table: {wide.height:,} (scope, S1, cand) rows")

    val_s1 = pl.concat(attrs).join(role.filter(pl.col("role") == "val"), on="s1", how="semi")
    s1s = val_s1.select("s1", "country")
    t = truth.join(s1s, on="s1", how="semi")
    floor = t.height / max(s1s.height, 1)
    wv = wide.filter(pl.col("role") == "val")
    tim_rows = []

    # ---- pruner: trained on TRAIN-split S1 candidates of the generous union, applied to validation S1
    pool_attr = pl.concat([tr[2], tr[3]]).select("entity_id", "core_name", "addr_norm", "city", "state")
    s1_attr = tr[1].select("entity_id", "core_name", "addr_norm", "city", "state")
    urule = rule(**UNION)
    pruned: dict[str, tuple[pl.DataFrame, np.ndarray]] = {}
    if trn_ids.len():
        for sc in a.scopes:
            t0 = time.time()
            xt = features(wide.filter((pl.col("scope") == sc) & (pl.col("role") == "train") & urule), s1_attr, pool_attr)
            xv = features(wv.filter((pl.col("scope") == sc) & urule), s1_attr, pool_attr)
            t1 = time.time()
            model = train_pruner(xt, xt["is_match"].to_numpy())
            t2 = time.time()
            pv = model.predict_proba(matrix(xv))[:, 1]
            t3 = time.time()
            pruned[sc] = (xv.select("scope", "s1", "cand", "is_match"), pv)
            tim_rows.append({"scope": sc, "method": "pruner", "train_rows": xt.height, "val_rows": xv.height,
                             "features_s": t1 - t0, "fit_s": t2 - t1, "predict_s": t3 - t2,
                             "s_per_row": (t1 - t0) / max(xt.height + xv.height, 1) + (t3 - t2) / max(xv.height, 1)})
            log(f"[{sc}] pruner: {xt.height:,} train rows ({xt['is_match'].mean():.3f} positive), fit {t2 - t1:.0f}s; "
                f"{xv.height:,} val rows scored")
            del xt

    # ---- every config: recall and list size on validation S1
    rows = []
    for sc in a.scopes:
        w = wv.filter(pl.col("scope") == sc)
        for fam, kw, cap in configs():
            found = select(w, rule(**kw), cap).select("s1", "cand")
            name = json.dumps({**{k: v for k, v in kw.items() if v not in (0, None)}, **({"cap": cap} if cap else {})})
            rows += [{"family": fam, "scope": sc, "config": name, **r} for r in list_stats(found, t, s1s, pool_size).iter_rows(named=True)]
        if sc in pruned:
            base, pv = pruned[sc]
            for tau in [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]:
                for cap in [None, 3, 5, 10, 20]:
                    found = prune(base, pv, tau, cap).select("s1", "cand")
                    name = json.dumps({"union": UNION, "tau": tau, **({"cap": cap} if cap else {})})
                    rows += [{"family": "pruned", "scope": sc, "config": name, **r}
                             for r in list_stats(found, t, s1s, pool_size).iter_rows(named=True)]
        log(f"[{sc}] configs evaluated")
    curve = pl.DataFrame(rows)
    curve.write_csv(out / "curve.csv")
    allc = curve.filter(pl.col("country") == "ALL")
    front = pl.concat([pareto(allc.filter(pl.col("scope") == sc)) for sc in a.scopes])
    front.write_csv(out / "pareto.csv")

    # ---- operating points: best recall at <= target candidates per S1 (ALL countries), then per country
    ops = []
    for sc in a.scopes:
        for tg in TARGETS:
            c = allc.filter((pl.col("scope") == sc) & (pl.col("avg") <= tg)).sort("recall", "avg", descending=[True, False])
            if c.height:
                r = c.row(0, named=True)
                ops += [{"target": tg, **x} for x in curve.filter((pl.col("scope") == sc) & (pl.col("config") == r["config"])
                                                                  & (pl.col("family") == r["family"])).iter_rows(named=True)]
    ops = pl.DataFrame(ops)
    ops.write_csv(out / "operating_points.csv")
    plot(curve, front, ops, floor, out / "blocking_curve.png")

    # ---- group recall at the operating points
    matches = pl.concat([tr[2].select("entity_id", "core_name", "business_name", "state"),
                         tr[3].select("entity_id", "core_name", "business_name", "state")]).rename(
        {"entity_id": "cand", "core_name": "m_core", "business_name": "m_name", "state": "m_state"})
    s1_state = tr[1].select(pl.col("entity_id").alias("s1"), pl.col("state").alias("s1_state"))
    g = t.join(val_s1.select("s1", "country", "core_name", "n_core"), on="s1").join(matches, on="cand", how="left").join(s1_state, on="s1")
    g = g.with_columns(pl.Series("tsr", process.cpdist(g["core_name"].to_list(), g["m_core"].fill_null("").to_list(),
                                                       scorer=fuzz.token_set_ratio, workers=-1)))
    groups = {
        "all": pl.lit(True),
        "common name (core_name 5+ in country)": pl.col("n_core") >= 5,
        "low name similarity (tsr < 70)": pl.col("tsr") < 70,
        "Indian-script match record": pl.col("m_name").str.contains(NON_LATIN),
        "state differs (both known)": pl.col("m_state").is_not_null() & (pl.col("m_state") != pl.col("s1_state")),
        "match has no state": pl.col("m_state").is_null(),
    }
    rows = []
    for r in ops.filter(pl.col("country") == "ALL").iter_rows(named=True):
        sc, cfg = r["scope"], json.loads(r["config"])
        if r["family"] == "pruned":
            base, pv = pruned[sc]
            found = prune(base, pv, cfg["tau"], cfg.get("cap"))
        else:
            cap = cfg.pop("cap", None)
            found = select(wv.filter(pl.col("scope") == sc), rule(**cfg), cap)
        gg = g.join(found.select("s1", "cand", pl.lit(True).alias("found")), on=["s1", "cand"], how="left").with_columns(
            pl.col("found").fill_null(False))
        for gname, cond in groups.items():
            sub = gg.filter(cond)
            for country, d in [("ALL", sub)] + [(c, x) for (c,), x in sub.group_by("country")]:
                rows.append({"scope": sc, "target": r["target"], "family": r["family"], "config": r["config"], "group": gname,
                             "country": country, "true_pairs": d.height, "recall": d["found"].mean() if d.height else None})
    grp = pl.DataFrame(rows).sort("scope", "target", "group", "country")
    grp.write_csv(out / "recall_groups.csv")

    # ---- validation candidates of the generous union (with pruner probability) for later stages
    vc = wv.filter(urule)
    if pruned:
        vc = vc.join(pl.concat([b.with_columns(pl.Series("p_keep", p)) for b, p in pruned.values()]).select("scope", "s1", "cand", "p_keep"),
                     on=["scope", "s1", "cand"], how="left")
    vc.write_parquet(out / "val_candidates.parquet")

    # ---- timing and test-time estimate
    tim = pl.DataFrame(timing + tim_rows, infer_schema_length=None)
    tim.write_csv(out / "timing.csv")
    rate = measured_rates(timing)
    fit = tim.filter(pl.col("method").str.ends_with("_fit_transform") & pl.col("shard").str.starts_with("0/"))  # once per country
    fit_rate = fit["search_s"].sum() / max(fit["pool"].sum() + tr[1].height, 1)
    prune_rate = float(np.mean([x["s_per_row"] for x in tim_rows])) if tim_rows else 0.0
    union_avg = {sc: list_stats(wv.filter((pl.col("scope") == sc) & urule).select("s1", "cand"), t, s1s, pool_size)
                 .filter(pl.col("country") == "ALL")["avg"].item() for sc in a.scopes}
    est = pl.DataFrame(test_est).with_columns(
        (pl.col("forward_pairs") * rate.get("forward", 0.0) * len(FORWARD) / 60).alias("forward_min"),
        (pl.col("reverse_pairs") * rate.get("reverse", 0.0) / 60).alias("reverse_min"),
        ((pl.col("test_s1") + pl.col("test_pool")) * fit_rate / 60).alias("fit_transform_min"),
        (pl.col("test_s1") * pl.col("scope").replace_strict(union_avg, return_dtype=pl.Float64) * prune_rate / 60).alias("pruner_min"),
    ).with_columns(pl.sum_horizontal("forward_min", "reverse_min", "fit_transform_min", "pruner_min").alias("total_min"))
    est.write_csv(out / "test_estimate.csv")
    est_tot = est.group_by("scope").agg(pl.col("^.*_min$").sum()).sort("scope")
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20

    cols = ["country", "recall", "avg", "median", "p95", "empty_share", "reduction_ratio"]
    report = ["# Blocking evaluation: recall vs candidates per S1 (validation split vs full train pool)\n",
              f"Validation: {s1s.height:,} S1, {t.height:,} true pairs, **{floor:.2f} true matches per S1** (floor for the "
              f"average list size). Pruner trained on {trn_ids.len():,} train-split S1.\n"]
    for sc in a.scopes:
        report += [f"# Scope: {sc}\n", "## Operating points\n"]
        for r in ops.filter((pl.col("scope") == sc) & (pl.col("country") == "ALL")).iter_rows(named=True):
            report += [f"### <= {r['target']} candidates per S1: {r['family']} {r['config']}\n",
                       md(ops.filter((pl.col("scope") == sc) & (pl.col("target") == r["target"])).select(cols)) + "\n",
                       md(grp.filter((pl.col("scope") == sc) & (pl.col("target") == r["target"]) & (pl.col("country") == "ALL"))
                          .select("group", "true_pairs", "recall")) + "\n"]
        report += ["## Pareto front (ALL countries)\n",
                   md(front.filter(pl.col("scope") == sc).select("family", "config", "recall", "avg", "median", "p95", "empty_share"), 40) + "\n",
                   "## Best config per family (ALL countries, <= 20 candidates per S1)\n",
                   md(allc.filter((pl.col("scope") == sc) & (pl.col("avg") <= 20)).sort("recall", descending=True)
                      .group_by("family", maintain_order=True).head(3).select("family", "config", "recall", "avg", "median", "p95"), 20) + "\n"]
    report += ["# Timing on validation (seconds)\n", md(tim, 80) + "\n",
               f"Measured rate: forward {rate.get('forward', 0) * 1e9:.2f} ns, reverse {rate.get('reverse', 0) * 1e9:.2f} ns per "
               f"similarity evaluation; searches ran on {'GPU' if blocking.USE_GPU else 'CPU'} ({gpu_info()}). "
               f"Pruner {prune_rate * 1e6:.2f} us per candidate (features + predict).\n",
               "# Test-time estimate (minutes)\n", md(est) + "\n", md(est_tot) + "\n",
               f"Peak host memory of this run: {peak:.1f} GB.\n"]
    (out / "blocking_report.md").write_text("\n".join(report))
    (out / "config.json").write_text(json.dumps({"operating_points": ops.filter(pl.col("country") == "ALL").select(
        "scope", "target", "family", "config", "recall", "avg").to_dicts(), "union": UNION, "scopes": a.scopes,
        "k_max": a.k_max, "m_max": a.m_max, "device": gpu_info(), "peak_gb": peak}, indent=1))
    log("done: " + "; ".join(f"[{r['scope']}] <= {r['target']}: recall {r['recall']:.4f} at {r['avg']:.2f} cands/S1 ({r['family']})"
                             for r in ops.filter(pl.col("country") == "ALL").iter_rows(named=True)))


if __name__ == "__main__":
    main()
