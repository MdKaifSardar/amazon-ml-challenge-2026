"""Cross-encoder re-scoring of borderline pairs (experiment; branch cross-encoder).

Model: distilbert-base-multilingual-cased (Apache-2.0), fine-tuned by cross_encoder_train.py as a binary pair
classifier. Input: "{business_name} | {business_address} | {country}" of the S1 record [SEP] the same for the
candidate, raw organiser text, max 128 tokens.
Blend (borderline pairs only, BAND[0] < p_model < BAND[1]): p = (1 - w) * p_model + w * p_ce; other pairs keep
p_model. Then the usual selection (threshold, margin, one owner per S2/S3).

Test usage (GPU), after run_model_v4.py and cross_encoder_train.py:
  python src/cross_encoder_score.py --input DIR --ce-dir DIR --v4-dir DIR --out-dir output/ce_blend [--validator PATH]
The blend is re-tuned on model v4's validation scores (w and threshold on the tune half, margin and one owner as
v4; report half = check), so it matches the model whose test scores it re-scores. GO only if the report half beats
w = 0 overall AND for US and India with precision >= 0.99; otherwise w = 0 (the v4 scores unchanged). Countries
not in train get v4's unseen-country threshold shift (0 unless v4's simulation kept it).
"""
from __future__ import annotations

import argparse
import json
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

MODEL = "distilbert-base-multilingual-cased"
MAX_LEN = 128
BAND = (0.02, 0.98)
T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')} +{time.time() - T0:6.0f}s] {msg}", flush=True)


def find(root: Path, name: str, not_parent: tuple = ("normalised", "features_extra", "aug")) -> Path:
    hits = sorted(p for p in root.rglob(name) if p.parent.name not in not_parent)
    if not hits:
        raise FileNotFoundError(f"{name} not found under {root}")
    return hits[0]


def texts(root: Path, side: str, ids: pl.Series) -> dict[str, str]:
    """entity_id -> "name | address | country" from the organiser files of one side (train / test)."""
    out = {}
    for k in (1, 2, 3):
        d = (pl.scan_parquet(find(root, f"{side}_source{k}.parquet")).select("entity_id", "business_name", "business_address", "country")
             .filter(pl.col("entity_id").is_in(ids.implode())).collect())
        t = d.select("entity_id", pl.concat_str([pl.col("business_name").fill_null(""), pl.col("business_address").fill_null(""),
                                                 pl.col("country").fill_null("")], separator=" | ").alias("t"))
        out.update(zip(t["entity_id"].to_list(), t["t"].to_list()))
    return out


def score_pairs(model, tok, pairs: pl.DataFrame, txt: dict, batch: int = 512, log_every: int = 200) -> np.ndarray:
    import torch
    model.eval()
    a, b = [txt.get(s, "") for s in pairs["s1"].to_list()], [txt.get(c, "") for c in pairs["cand"].to_list()]
    out = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for i in range(0, len(a), batch):
            enc = tok(a[i:i + batch], b[i:i + batch], truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt").to("cuda")
            out.append(torch.sigmoid(model(**enc).logits.float().squeeze(-1)).cpu().numpy())
            if log_every and (i // batch) % log_every == 0:
                log(f"  scored {i + len(a[i:i + batch]):,} / {len(a):,}")
    return np.concatenate(out) if out else np.empty(0)


def blend(p_model: np.ndarray, p_ce: np.ndarray, w: float) -> np.ndarray:
    """p_ce: NaN outside the borderline band."""
    border = ~np.isnan(p_ce)
    p = p_model.copy()
    p[border] = (1 - w) * p_model[border] + w * p_ce[border]
    return p


WS = [0.0, 0.35, 0.5, 0.65, 0.8, 0.9, 1.0]


def main() -> None:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from selection import f05_table, select, summary
    from submission import write_submission
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--ce-dir", required=True, type=Path, help="folder holding ce_model/ (searched)")
    ap.add_argument("--v4-dir", required=True, type=Path, help="folder holding model v4's test_scores / val_scores / model_config (searched)")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--validator", default=None)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    mdir = next(p for p in a.ce_dir.rglob("config.json") if p.parent.name == "ce_model").parent
    v4cfg = json.loads(next(a.v4_dir.rglob("model_config.json")).read_text())
    tok = AutoTokenizer.from_pretrained(mdir)
    model = AutoModelForSequenceClassification.from_pretrained(mdir).to("cuda")
    rep = ["# Cross-encoder blend on model v4\n", f"v4: selection {v4cfg['selection']}, unseen shift {v4cfg['unseen_tau_shift']}, "
           f"pseudo-label {v4cfg['pseudo_label']}, cross features {v4cfg['use_cross']}.\n"]

    # ---- validation: re-tune the blend on v4's validation scores
    vs = pl.read_parquet(next(a.v4_dir.rglob("val_scores.parquet"))).select("s1", "cand", "p")
    split = pl.read_parquet(find(a.input, "g25_split.parquet"))
    val_s1 = split.filter(pl.col("role") == "val").select(pl.col("source1_entity_id").alias("s1")).join(
        pl.read_parquet(find(a.input, "train_source1.parquet"), columns=["entity_id", "country"]).rename({"entity_id": "s1"}), on="s1", how="left")
    gt = pl.read_parquet(find(a.input, "train_ground_truth.parquet"))
    truth = (gt.join(val_s1, left_on="source1_entity_id", right_on="s1", how="semi")
             .select(pl.col("source1_entity_id").alias("s1"), pl.col("matched_entity_ids").str.split(",").alias("cand"))
             .explode("cand", empty_as_null=True).filter(pl.col("cand").is_not_null() & (pl.col("cand") != "")))
    half = pl.col("s1").str.extract(r"(\d+)$").cast(pl.Int64) % 2
    s1_tune, s1_rep = val_s1.filter(half == 0), val_s1.filter(half == 1)
    vb = vs.filter((pl.col("p") > BAND[0]) & (pl.col("p") < BAND[1]))
    vtxt = texts(a.input, "train", pl.concat([vb["s1"], vb["cand"]]).unique())
    p_ce_v = score_pairs(model, tok, vb.select("s1", "cand"), vtxt, log_every=0)
    log(f"validation borderline pairs scored: {vb.height:,}")
    full = vs.join(vb.select("s1", "cand").with_columns(pl.Series("p_ce", p_ce_v)), on=["s1", "cand"], how="left")
    pm, pc = full["p"].to_numpy(), full["p_ce"].fill_null(np.nan).to_numpy()
    cur = {"margin": v4cfg["selection"]["margin"], "one_owner": True}
    res = []
    for w in WS:
        sv = full.select("s1", "cand").with_columns(pl.Series("p", blend(pm, pc, w)))
        grid = [(f05_table(select(sv, tau=t, **cur), truth, s1_tune)["f05"].mean(), t) for t in [round(x, 3) for x in np.arange(0.30, 0.91, 0.025)]]
        f_t, tau = max(grid)
        sm = summary(f05_table(select(sv, tau=tau, **cur), truth, s1_rep), by="country")
        gg = {r["group"]: r for r in sm.iter_rows(named=True)}
        res.append({"w": w, "tau": tau, "tune_f05": f_t, **{k: gg[k]["macro_f05"] for k in gg}, "precision": gg["overall"]["precision"],
                    "recall": gg["overall"]["recall"]})
        log(f"w {w}: {res[-1]}")
    res = pl.DataFrame(res)
    base = res.filter(pl.col("w") == 0.0).row(0, named=True)
    best = res.filter(pl.col("w") > 0).sort("tune_f05", descending=True).row(0, named=True)
    go = (best["overall"] > base["overall"] and best["country=US"] > base["country=US"] and best["country=India"] > base["country=India"]
          and best["precision"] >= 0.99)
    ch = best if go else base
    rep += ["## Validation (report half; tau tuned on the tune half)\n", str(res) + "\n",
            f"**GO = {go}**: w = {ch['w']}, tau = {ch['tau']} (report {ch['overall']:.4f} vs w=0 {base['overall']:.4f})\n"]
    log(f"GO = {go}; using w {ch['w']} tau {ch['tau']}")

    # ---- test
    st = pl.read_parquet(next(a.v4_dir.rglob("test_scores.parquet")))
    s1_all = pl.read_parquet(find(a.input, "test_source1.parquet"), columns=["entity_id", "country"]).rename({"entity_id": "s1"})
    if ch["w"] > 0:
        border = st.filter((pl.col("p") > BAND[0]) & (pl.col("p") < BAND[1]))
        txt = texts(a.input, "test", pl.concat([border["s1"], border["cand"]]).unique())
        t0 = time.time()
        p_ce = score_pairs(model, tok, border.select("s1", "cand"), txt)
        log(f"scored {border.height:,} borderline test pairs in {time.time() - t0:.0f}s")
        st = st.join(border.select("s1", "cand").with_columns(pl.Series("p_ce", p_ce)), on=["s1", "cand"], how="left")
    else:
        st = st.with_columns(pl.lit(None, dtype=pl.Float64).alias("p_ce"))
    st = st.with_columns(pl.Series("p_blend", blend(st["p"].to_numpy(), st["p_ce"].fill_null(np.nan).to_numpy(), ch["w"])))
    st.write_parquet(a.out_dir / "test_scores_blend.parquet")
    unseen = ~pl.col("country").is_in(v4cfg["train_countries"])
    tau_row = (pl.lit(ch["tau"]) + pl.when(unseen).then(pl.lit(v4cfg["unseen_tau_shift"])).otherwise(0.0)).clip(0.3, 0.9)
    x = st.join(s1_all, on="s1", how="left").filter(pl.col("p_blend") >= tau_row)
    matched = select(x.select("s1", "cand", pl.col("p_blend").alias("p")), tau=0.0, **cur)
    cand = pl.read_parquet(find(a.input, "test_candidates.parquet"), columns=["s1", "cand", "p_u50"])
    lists = cand.sort(["s1", "p_u50", "cand"], descending=[False, True, False], nulls_last=True).group_by("s1", maintain_order=True).agg(pl.col("cand"))
    ml = matched.sort(["s1", "p"], descending=[False, True]).group_by("s1", maintain_order=True).agg(pl.col("cand"))
    write_submission(a.out_dir, s1_all["s1"].to_list(), dict(zip(ml["s1"].to_list(), ml["cand"].to_list())),
                     dict(zip(lists["s1"].to_list(), lists["cand"].to_list())))
    rep.append(f"Test: {matched.height:,} matches.\n")
    if a.validator:
        import importlib.util
        spec = importlib.util.spec_from_file_location("vs", a.validator); vs_ = importlib.util.module_from_spec(spec); spec.loader.exec_module(vs_)
        with tempfile.TemporaryDirectory() as td:
            for k in (1, 2, 3):
                pl.read_parquet(find(a.input, f"test_source{k}.parquet"), columns=["entity_id"]).write_csv(Path(td) / f"test_source{k}.tsv", separator="\t")
            e, w = vs_.validate(str(a.out_dir / "matching_results.tsv"), str(a.out_dir / "candidate_pairs.tsv"), td, check_ids=True)
        rep.append(f"Validator (--check-ids): {len(e)} errors, {len(w)} warnings {e[:3]}\n")
        log(f"validator: {len(e)} errors, {len(w)} warnings")
    n = s1_all.join(matched.group_by("s1").len("n"), on="s1", how="left").with_columns(pl.col("n").fill_null(0))
    pcn = n.group_by("country").agg(pl.col("n").mean().alias("matches_per_s1"), (pl.col("n") == 0).mean().alias("share_no_match")).sort("country")
    rep += ["## Matches per S1\n", str(pcn) + "\n"]
    (a.out_dir / "ce_test_report.md").write_text("\n".join(rep))
    log("done")


if __name__ == "__main__":
    main()
