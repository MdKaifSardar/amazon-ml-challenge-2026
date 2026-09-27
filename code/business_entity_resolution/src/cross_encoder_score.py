"""Cross-encoder re-scoring of borderline pairs (experiment; branch cross-encoder).

Model: distilbert-base-multilingual-cased (Apache-2.0), fine-tuned by cross_encoder_train.py as a binary pair
classifier. Input: "{business_name} | {business_address} | {country}" of the S1 record [SEP] the same for the
candidate, raw organiser text, max 128 tokens.
Blend (borderline pairs only, BAND[0] < p_model < BAND[1]): p = (1 - w) * p_model + w * p_ce; other pairs keep
p_model. Then the usual selection (threshold, margin, one owner per S2/S3).

Test usage (GPU):
  python src/cross_encoder_score.py --input DIR --ce-dir DIR --scores test_scores.parquet --config ce_config.json \
      --out-dir output/ce_blend [--validator utils/validate_submission.py] [--current output/matching_results.tsv]
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


def main() -> None:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from selection import select
    from submission import write_submission
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--ce-dir", required=True, type=Path, help="folder holding ce_model/ and ce_config.json (searched)")
    ap.add_argument("--scores", required=True, type=Path, help="test pair probabilities of the final model (searched if a folder)")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--validator", default=None)
    ap.add_argument("--current", default=None, type=Path, help="current final matching_results.tsv to compare with")
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = next(a.ce_dir.rglob("ce_config.json"))
    cfg = json.loads(cfg_path.read_text())
    mdir = cfg_path.parent / "ce_model"
    sp = a.scores if a.scores.is_file() else next(a.scores.rglob("test_scores.parquet"))
    st = pl.read_parquet(sp)
    border = st.filter((pl.col("p") > BAND[0]) & (pl.col("p") < BAND[1]))
    log(f"test pairs {st.height:,}; borderline {border.height:,} ({border.height / st.height:.1%}); w = {cfg['w']}, selection {cfg['selection']}")
    tok = AutoTokenizer.from_pretrained(mdir)
    model = AutoModelForSequenceClassification.from_pretrained(mdir).to("cuda")
    txt = texts(a.input, "test", pl.concat([border["s1"], border["cand"]]).unique())
    t0 = time.time()
    p_ce = score_pairs(model, tok, border.select("s1", "cand"), txt)
    log(f"scored {border.height:,} borderline test pairs in {time.time() - t0:.0f}s ({border.height / max(time.time() - t0, 1e-9):.0f} pairs/s)")
    st = st.join(border.select("s1", "cand").with_columns(pl.Series("p_ce", p_ce)), on=["s1", "cand"], how="left")
    st = st.with_columns(pl.Series("p_blend", blend(st["p"].to_numpy(), st["p_ce"].fill_null(np.nan).to_numpy(), cfg["w"])))
    st.write_parquet(a.out_dir / "test_scores_blend.parquet")
    matched = select(st.select("s1", "cand", pl.col("p_blend").alias("p")), **{**cfg["selection"], "one_owner": True})
    s1_all = pl.read_parquet(find(a.input, "test_source1.parquet"), columns=["entity_id", "country"]).rename({"entity_id": "s1"})
    cand = pl.read_parquet(find(a.input, "test_candidates.parquet"), columns=["s1", "cand", "p_u50"])
    lists = cand.sort(["s1", "p_u50", "cand"], descending=[False, True, False], nulls_last=True).group_by("s1", maintain_order=True).agg(pl.col("cand"))
    ml = matched.sort(["s1", "p"], descending=[False, True]).group_by("s1", maintain_order=True).agg(pl.col("cand"))
    write_submission(a.out_dir, s1_all["s1"].to_list(), dict(zip(ml["s1"].to_list(), ml["cand"].to_list())),
                     dict(zip(lists["s1"].to_list(), lists["cand"].to_list())))
    rep = [f"# Cross-encoder blend on test (w = {cfg['w']})\n", f"Borderline pairs scored: {border.height:,}. Matches: {matched.height:,}.\n"]
    if a.validator:
        import importlib.util
        spec = importlib.util.spec_from_file_location("vs", a.validator); vs = importlib.util.module_from_spec(spec); spec.loader.exec_module(vs)
        with tempfile.TemporaryDirectory() as td:
            for k in (1, 2, 3):
                pl.read_parquet(find(a.input, f"test_source{k}.parquet"), columns=["entity_id"]).write_csv(Path(td) / f"test_source{k}.tsv", separator="\t")
            e, w = vs.validate(str(a.out_dir / "matching_results.tsv"), str(a.out_dir / "candidate_pairs.tsv"), td, check_ids=True)
        rep.append(f"Validator (--check-ids): {len(e)} errors, {len(w)} warnings {e[:3]}\n")
        log(f"validator: {len(e)} errors, {len(w)} warnings")
    n = s1_all.join(matched.group_by("s1").len("n"), on="s1", how="left").with_columns(pl.col("n").fill_null(0))
    pc = n.group_by("country").agg(pl.col("n").mean().alias("matches_per_s1"), (pl.col("n") == 0).mean().alias("share_no_match")).sort("country")
    rep += ["## Matches per S1\n", str(pc) + "\n"]
    new = matched.select("s1", "cand").with_columns(pl.lit(True).alias("m"))
    fr = s1_all.filter(pl.col("country") == "France").sample(10, seed=42)
    ex = fr.join(matched, on="s1", how="left").join(st.select("s1", "cand", "p", "p_ce"), on=["s1", "cand"], how="left")
    t_fr = texts(a.input, "test", pl.concat([ex["s1"], ex["cand"].drop_nulls()]).unique())
    ex = ex.with_columns(pl.col("s1").replace_strict(t_fr, default="").alias("s1_text"), pl.col("cand").replace_strict(t_fr, default="").alias("cand_text"))
    rep += ["## 10 random French S1\n", str(ex.select("s1_text", "cand_text", "p", "p_ce")) + "\n"]
    if a.current and a.current.exists():
        cur = pl.read_csv(a.current, separator="\t", quote_char=None, infer_schema=False).select(
            pl.col("source1_entity_id").alias("s1"), pl.col("matched_entity_ids").fill_null("").str.split(",").list.eval(pl.element().filter(pl.element() != "")).list.sort().alias("m0"))
        nm = s1_all.join(matched.group_by("s1").agg(pl.col("cand").sort().alias("m1")), on="s1", how="left").with_columns(pl.col("m1").fill_null([]))
        ch = nm.join(cur, on="s1").with_columns((pl.col("m1") != pl.col("m0")).alias("changed")).group_by("country").agg(
            pl.col("changed").sum().alias("s1_changed"), pl.col("changed").mean().alias("share")).sort("country")
        rep += ["## Change vs the current final\n", str(ch) + "\n"]
    (a.out_dir / "ce_test_report.md").write_text("\n".join(rep))
    log("done")


if __name__ == "__main__":
    main()
