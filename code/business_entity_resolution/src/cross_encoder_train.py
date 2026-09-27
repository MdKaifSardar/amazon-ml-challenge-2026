"""Cross-encoder experiment, steps 1-3 (GPU): smoke test, 1-epoch fine-tune, borderline validation scoring, blend.

Model: distilbert-base-multilingual-cased (Apache-2.0; the licence is read from the model card and logged).
Training pairs: blocking-v3 train candidates (train_candidates.parquet: the natural mix of true pairs and hard
negatives; no random negatives), a sample of whole S1 lists (~--pairs pairs); 5% of those S1 held out for AUC.
Validation: borderline pairs (0.02 < p_model < 0.98) of the current model's validation probabilities
(val_scores.parquet from run_model_v3.py: models trained on train only). Blend p = (1-w) p_model + w p_ce on
borderline pairs; w and the threshold tuned on the tune half (margin and one owner as the current selection),
reported on the report half. GO only if the report half beats the current model overall AND for US and India,
with precision >= 0.99. Outputs: ce_model/, ce_config.json, ce_val_scores.parquet, ce_report.md.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

try:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
except NameError:
    sys.path.insert(0, "/tmp/src")
from cross_encoder_score import BAND, MAX_LEN, MODEL, blend, find, log, score_pairs, texts  # noqa: E402
from selection import f05_table, select, summary  # noqa: E402

SEED = 42
WS = [0.0, 0.2, 0.35, 0.5, 0.65]
NON_LATIN = r"[\p{L}&&[^\p{Latin}]]"


def auc(y: np.ndarray, p: np.ndarray) -> float | None:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else None


def train(model, tok, pairs: pl.DataFrame, txt: dict, batch: int, lr: float, max_steps: int | None = None) -> float:
    """1 epoch (or max_steps), fp16, AdamW + linear warmup 5%. Returns training pairs per second."""
    import torch
    from transformers import get_linear_schedule_with_warmup
    torch.manual_seed(SEED)
    x = pairs.sample(fraction=1.0, shuffle=True, seed=SEED)
    a, b = [txt.get(s, "") for s in x["s1"].to_list()], [txt.get(c, "") for c in x["cand"].to_list()]
    y = x["is_match"].cast(pl.Float32).to_numpy()
    steps = math.ceil(len(a) / batch) if max_steps is None else min(max_steps, math.ceil(len(a) / batch))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sch = get_linear_schedule_with_warmup(opt, max(1, int(0.05 * steps)), steps)
    scaler = torch.cuda.amp.GradScaler()
    lossf = torch.nn.BCEWithLogitsLoss()
    model.train()
    t0 = time.time()
    for s in range(steps):
        i = s * batch
        enc = tok(a[i:i + batch], b[i:i + batch], truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt").to("cuda")
        with torch.autocast("cuda", dtype=torch.float16):
            loss = lossf(model(**enc).logits.float().squeeze(-1), torch.tensor(y[i:i + batch], device="cuda"))
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sch.step()
        if s % 500 == 0:
            log(f"  step {s}/{steps} loss {loss.item():.4f}")
    torch.cuda.synchronize()
    return steps * batch / (time.time() - t0)


def main() -> None:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--pairs", type=int, default=400_000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max-train-min", type=float, default=45.0)
    a = ap.parse_args()
    out = a.out_dir
    out.mkdir(parents=True, exist_ok=True)
    rep = ["# Cross-encoder experiment (steps 1-3)\n"]
    log(f"GPU: {torch.cuda.get_device_name(0)}; torch {torch.__version__}")
    try:
        from huggingface_hub import model_info
        lic = getattr(model_info(MODEL).card_data, "license", None)
    except Exception as e:  # the card check must not stop the run; the licence is also recorded in the docs
        lic = f"card check failed: {e!r}"
    log(f"model {MODEL}: licence on the model card = {lic}")
    rep.append(f"Model `{MODEL}`, licence (model card): **{lic}**.\n")

    # ------------------------------------------------------------------ step 1: data + smoke test
    tc = pl.read_parquet(find(a.input, "train_candidates.parquet"), columns=["s1", "cand", "is_match"])
    s1s = tc.select("s1").unique().sort("s1").sample(fraction=1.0, shuffle=True, seed=SEED)
    per = tc.height / s1s.height
    s1s = s1s.head(int(a.pairs / per))
    hold = s1s.with_columns(((pl.col("s1").hash(seed=SEED) % 100) < 5).alias("hold"))
    pairs = tc.join(hold, on="s1")
    txt = texts(a.input, "train", pl.concat([pairs["s1"], pairs["cand"]]).unique())
    tok = AutoTokenizer.from_pretrained(MODEL)
    smp = pairs.sample(2000, seed=SEED)
    lens = [len(x) for x in tok([txt.get(s, "") for s in smp["s1"]], [txt.get(c, "") for c in smp["cand"]], truncation=False)["input_ids"]]
    trunc = float(np.mean(np.array(lens) > MAX_LEN))
    log(f"pairs {pairs.height:,} ({pairs['is_match'].mean():.2f} positive) from {s1s.height:,} S1; held out {int(hold['hold'].sum()):,} S1; "
        f"truncated at {MAX_LEN} tokens: {trunc:.1%} (median length {int(np.median(lens))})")
    m = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=1).to("cuda")
    tr_rate = train(m, tok, smp, txt, a.batch, a.lr)
    t0 = time.time(); score_pairs(m, tok, smp, txt, log_every=0); sc_rate = 2000 / (time.time() - t0)
    n_tr = int((~pairs["hold"]).sum())
    proj = n_tr / tr_rate / 60
    log(f"smoke: train {tr_rate:.0f} pairs/s, score {sc_rate:.0f} pairs/s; full training projected {proj:.1f} min for {n_tr:,} pairs")
    if proj > a.max_train_min:
        keep_s1 = pairs.filter(~pl.col("hold")).select("s1").unique().head(int(250_000 / per))
        pairs = pl.concat([pairs.filter(pl.col("hold")), pairs.filter(~pl.col("hold")).join(keep_s1, on="s1", how="semi")])
        n_tr = int((~pairs["hold"]).sum())
        log(f"cut to {n_tr:,} training pairs (projected {n_tr / tr_rate / 60:.1f} min)")
    rep.append(f"Step 1: {pairs.height:,} pairs; truncated {trunc:.1%}; smoke train {tr_rate:.0f} pairs/s, score {sc_rate:.0f} pairs/s; "
               f"projected training {n_tr / tr_rate / 60:.1f} min.\n")
    del m
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------ step 2: fine-tune (1 epoch)
    m = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=1).to("cuda")
    t0 = time.time()
    rate = train(m, tok, pairs.filter(~pl.col("hold")), txt, a.batch, a.lr)
    ho = pairs.filter(pl.col("hold"))
    p_ho = score_pairs(m, tok, ho, txt, log_every=0)
    ho_auc = auc(ho["is_match"].cast(pl.Int8).to_numpy(), p_ho)
    log(f"fine-tuned in {(time.time() - t0) / 60:.1f} min ({rate:.0f} pairs/s); held-out AUC {ho_auc:.4f} on {ho.height:,} pairs")
    m.save_pretrained(out / "ce_model"); tok.save_pretrained(out / "ce_model")
    rep.append(f"Step 2: trained on {n_tr:,} pairs in {(time.time() - t0) / 60:.1f} min; held-out AUC **{ho_auc:.4f}** ({ho.height:,} pairs).\n")

    # ------------------------------------------------------------------ step 3: borderline validation + blend
    vs = pl.read_parquet(find(a.input, "val_scores.parquet")).select("s1", "cand", "is_match", "p")
    split = pl.read_parquet(find(a.input, "g25_split.parquet"))
    val_s1 = split.filter(pl.col("role") == "val").select(pl.col("source1_entity_id").alias("s1")).join(
        pl.read_parquet(find(a.input, "train_source1.parquet"), columns=["entity_id", "country"]).rename({"entity_id": "s1"}), on="s1", how="left")
    gt = pl.read_parquet(find(a.input, "train_ground_truth.parquet"))
    truth = (gt.join(val_s1, left_on="source1_entity_id", right_on="s1", how="semi")
             .select(pl.col("source1_entity_id").alias("s1"), pl.col("matched_entity_ids").str.split(",").alias("cand"))
             .explode("cand", empty_as_null=True).filter(pl.col("cand").is_not_null() & (pl.col("cand") != "")))
    half = pl.col("s1").str.extract(r"(\d+)$").cast(pl.Int64) % 2
    s1_tune, s1_rep = val_s1.filter(half == 0), val_s1.filter(half == 1)
    border = vs.filter((pl.col("p") > BAND[0]) & (pl.col("p") < BAND[1]))
    vtxt = texts(a.input, "train", pl.concat([border["s1"], border["cand"]]).unique())
    t0 = time.time()
    p_ce = score_pairs(m, tok, border, vtxt, log_every=0)
    log(f"scored {border.height:,} borderline validation pairs in {time.time() - t0:.0f}s")
    b = border.with_columns(pl.Series("p_ce", p_ce)).join(val_s1, on="s1", how="left")
    names = pl.concat([pl.scan_parquet(find(a.input, f"train_source{k}.parquet")).select("entity_id", "business_name")
                       .filter(pl.col("entity_id").is_in(b["cand"].implode())).collect() for k in (2, 3)])
    s1n = pl.scan_parquet(find(a.input, "train_source1.parquet")).select("entity_id", "business_name").filter(pl.col("entity_id").is_in(b["s1"].implode())).collect()
    from rapidfuzz import fuzz, process
    b = b.join(names.rename({"entity_id": "cand", "business_name": "c_name"}), on="cand", how="left").join(
        s1n.rename({"entity_id": "s1", "business_name": "s_name"}), on="s1", how="left")
    b = b.with_columns(pl.Series("tsr", process.cpdist(b["s_name"].fill_null("").str.to_lowercase().to_list(),
                                                       b["c_name"].fill_null("").str.to_lowercase().to_list(), scorer=fuzz.token_set_ratio, workers=-1)))
    rows = []
    for gname, cond in {"all borderline": pl.lit(True), "US": pl.col("country") == "US", "India": pl.col("country") == "India",
                        "Indian-script candidate": pl.col("c_name").str.contains(NON_LATIN), "low name similarity (tsr<70)": pl.col("tsr") < 70}.items():
        d = b.filter(cond)
        y = d["is_match"].cast(pl.Int8).to_numpy()
        rows.append({"group": gname, "pairs": d.height, "true": int(y.sum()), "auc_ce": auc(y, d["p_ce"].to_numpy()), "auc_model": auc(y, d["p"].to_numpy()),
                     "auc_blend_0.5": auc(y, 0.5 * d["p"].to_numpy() + 0.5 * d["p_ce"].to_numpy())})
    auc_tab = pl.DataFrame(rows)
    log("borderline AUC:\n" + str(auc_tab))

    cur = {"margin": 0.7, "one_owner": True}  # current selection (model v3): tau re-tuned, margin and owner kept
    full = vs.join(b.select("s1", "cand", "p_ce"), on=["s1", "cand"], how="left")
    pm, pc = full["p"].to_numpy(), full["p_ce"].fill_null(np.nan).to_numpy()
    grid = []
    for w in WS:
        sv = full.select("s1", "cand").with_columns(pl.Series("p", blend(pm, pc, w)))
        for tau in [round(x, 3) for x in np.arange(0.30, 0.91, 0.025)]:
            f = f05_table(select(sv, tau=tau, **cur), truth, s1_tune)["f05"].mean()
            grid.append({"w": w, "tau": tau, "tune_f05": f})
    g = pl.DataFrame(grid).sort("tune_f05", descending=True)
    res = []
    for w in WS:
        bw = g.filter(pl.col("w") == w).row(0, named=True)
        sv = full.select("s1", "cand").with_columns(pl.Series("p", blend(pm, pc, w)))
        sm = summary(f05_table(select(sv, tau=bw["tau"], **cur), truth, s1_rep), by="country")
        gg = {r["group"]: r for r in sm.iter_rows(named=True)}
        res.append({"w": w, "tau": bw["tau"], "tune_f05": bw["tune_f05"], **{k: gg[k]["macro_f05"] for k in gg},
                    "precision": gg["overall"]["precision"], "recall": gg["overall"]["recall"]})
    res = pl.DataFrame(res)
    base = res.filter(pl.col("w") == 0.0).row(0, named=True)
    best = res.filter(pl.col("w") > 0).sort("tune_f05", descending=True).row(0, named=True)
    go = (best["overall"] > base["overall"] and best["country=US"] > base["country=US"] and best["country=India"] > base["country=India"]
          and best["precision"] >= 0.99)
    log(f"blend results:\n{res}\nGO = {go} (best w {best['w']}: report {best['overall']:.4f} vs current {base['overall']:.4f})")
    full.write_parquet(out / "ce_val_scores.parquet")
    (out / "ce_config.json").write_text(json.dumps({"model": MODEL, "licence": str(lic), "max_len": MAX_LEN, "band": BAND,
                                                    "w": best["w"], "selection": {"tau": best["tau"], **cur}, "go": go,
                                                    "heldout_auc": ho_auc, "train_pairs": n_tr, "truncated_share": trunc}, indent=1, default=str))
    rep += ["Step 3: borderline validation pairs (0.02 < p < 0.98): " + f"{border.height:,}\n", str(auc_tab) + "\n",
            "Blend (tau tuned on the tune half; report half):\n", str(res) + "\n",
            f"**GO = {go}** (rule: beat w = 0 overall, US and India on the report half, precision >= 0.99).\n"]
    (out / "ce_report.md").write_text("\n".join(rep))
    log("done")


if __name__ == "__main__":
    main()
