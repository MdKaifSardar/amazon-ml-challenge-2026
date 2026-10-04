"""Multilingual Cross-Encoder Pipeline for Business Entity Resolution.

Hardware: 1x Nvidia T4 GPU (FP16 enabled)
Inputs (found under /kaggle/input):
  - normalised: train_source{1,2,3}.parquet, test_source{1,2,3}.parquet
  - blocking: train_candidates.parquet, val_candidates.parquet, test_candidates.parquet
Outputs (written to /kaggle/working):
  - ce_scores_val.parquet (s1, cand, ce_score)
  - ce_scores_test.parquet (s1, cand, ce_score)
  - ce_summary.json
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

MODEL_NAME = "microsoft/Multilingual-MiniLM-L12-H384"
BATCH_SIZE = 256
MAX_LEN = 96
LR = 2e-5
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def find_file(root: Path, filename: str) -> Path:
    matches = list(root.rglob(filename))
    if not matches:
        raise FileNotFoundError(f"Could not find {filename} under {root}")
    return matches[0]


class PairDataset(Dataset):
    def __init__(self, texts_a: list[str], texts_b: list[str], labels: list[int] | None = None):
        self.texts_a = texts_a
        self.texts_b = texts_b
        self.labels = labels

    def __len__(self) -> int:
        return len(self.texts_a)

    def __getitem__(self, idx: int) -> dict:
        item = {"a": self.texts_a[idx], "b": self.texts_b[idx]}
        if self.labels is not None:
            item["label"] = self.labels[idx]
        return item


def make_collate_fn(tokenizer):
    def collate_fn(batch: list[dict]) -> dict:
        a = [x["a"] for x in batch]
        b = [x["b"] for x in batch]
        enc = tokenizer(
            a,
            b,
            max_length=MAX_LEN,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        if "label" in batch[0]:
            enc["labels"] = torch.tensor([x["label"] for x in batch], dtype=torch.long)
        return enc
    return collate_fn


def train_fold(
    train_a: list[str],
    train_b: list[str],
    train_y: list[int],
    tokenizer,
    fold_idx: int,
) -> nn.Module:
    log(f"--- Training Fold {fold_idx} ({len(train_y):,} pairs) ---")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.to(DEVICE)
    model.train()

    dataset = PairDataset(train_a, train_b, train_y)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=make_collate_fn(tokenizer),
        num_workers=2,
        pin_memory=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(loader)
    warmup_steps = int(total_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = torch.cuda.amp.GradScaler()

    total_loss = 0.0
    for step, batch in enumerate(loader):
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
        attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
        labels = batch["labels"].to(DEVICE, non_blocking=True)

        with torch.cuda.amp.autocast():
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total_loss += loss.item()

        if (step + 1) % 250 == 0 or (step + 1) == total_steps:
            log(f"  Fold {fold_idx} step {step + 1}/{total_steps} loss: {total_loss / (step + 1):.4f}")

    return model


@torch.no_grad()
def predict_loader(model: nn.Module, loader: DataLoader) -> np.ndarray:
    model.eval()
    preds = []
    for batch in loader:
        input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
        attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
        with torch.cuda.amp.autocast():
            outputs = model(input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=-1)[:, 1].cpu().numpy()
            preds.append(probs)
    return np.concatenate(preds) if preds else np.empty(0, dtype=np.float32)


def make_text_mapping(df: pl.DataFrame) -> dict[str, str]:
    """Map entity_id -> 'name | address | country'."""
    name_col = "full_name" if "full_name" in df.columns else "business_name"
    addr_col = "addr_norm" if "addr_norm" in df.columns else "business_address"
    expr = (
        pl.col(name_col).fill_null("").cast(pl.Utf8)
        + " | "
        + pl.col(addr_col).fill_null("").cast(pl.Utf8)
        + " | "
        + pl.col("country").fill_null("").cast(pl.Utf8)
    )
    temp = df.select(["entity_id", expr.alias("text")])
    return dict(zip(temp["entity_id"].to_list(), temp["text"].to_list()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="/kaggle/input", help="Input root directory")
    parser.add_argument("--out-dir", default="/kaggle/working", help="Output directory")
    parser.add_argument("--sample-train", type=int, default=150000, help="Train pairs per fold")
    args = parser.parse_args()

    input_dir = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log(f"Starting Cross-Encoder Pipeline on {DEVICE} (CUDA available: {torch.cuda.is_available()})")
    if torch.cuda.is_available():
        log(f"GPU: {torch.cuda.get_device_name(0)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 1. Load Normalized Tables
    log("Loading normalized record tables...")
    tr_s1 = pl.read_parquet(find_file(input_dir, "train_source1.parquet"))
    tr_s2 = pl.read_parquet(find_file(input_dir, "train_source2.parquet"))
    tr_s3 = pl.read_parquet(find_file(input_dir, "train_source3.parquet"))

    te_s1 = pl.read_parquet(find_file(input_dir, "test_source1.parquet"))
    te_s2 = pl.read_parquet(find_file(input_dir, "test_source2.parquet"))
    te_s3 = pl.read_parquet(find_file(input_dir, "test_source3.parquet"))

    log("Building text string lookup dictionaries...")
    text_s1 = make_text_mapping(tr_s1)
    text_s1.update(make_text_mapping(te_s1))

    text_cand = make_text_mapping(tr_s2)
    text_cand.update(make_text_mapping(tr_s3))
    text_cand.update(make_text_mapping(te_s2))
    text_cand.update(make_text_mapping(te_s3))
    del tr_s1, tr_s2, tr_s3, te_s1, te_s2, te_s3
    gc.collect()
    log(f"Entity lookup prepared: {len(text_s1):,} S1, {len(text_cand):,} Candidates.")

    # 2. Load Train Candidates for 2-Fold OOF Training
    log("Loading train candidates...")
    train_cands = pl.read_parquet(find_file(input_dir, "train_candidates.parquet"))
    label_col = "is_match" if "is_match" in train_cands.columns else ("label" if "label" in train_cands.columns else None)
    if label_col:
        train_cands = train_cands.with_columns(pl.col(label_col).cast(pl.Int32).alias("label"))
    else:
        log("Attaching ground truth labels from train_ground_truth.parquet...")
        truth = pl.read_parquet(find_file(input_dir, "train_ground_truth.parquet"))
        train_cands = train_cands.join(
            truth.select(["s1", "cand"]).with_columns(pl.lit(1).alias("label")),
            on=["s1", "cand"],
            how="left",
        ).with_columns(pl.col("label").fill_null(0).cast(pl.Int32))

    pos = train_cands.filter(pl.col("label") == 1)
    neg = train_cands.filter(pl.col("label") == 0).sample(n=min(len(pos) * 3, args.sample_train * 2), seed=SEED)
    train_df = pl.concat([pos, neg]).sample(fraction=1.0, shuffle=True, seed=SEED)
    log(f"Sampled train set: {len(train_df):,} pairs ({pos.height:,} pos, {neg.height:,} neg).")

    # Grouped 2-Fold split by S1
    train_df = train_df.with_columns((pl.col("s1").hash(seed=SEED) % 2).alias("fold"))

    fold0_df = train_df.filter(pl.col("fold") == 0)
    fold1_df = train_df.filter(pl.col("fold") == 1)

    t0_a = [text_s1.get(s, "") for s in fold0_df["s1"].to_list()]
    t0_b = [text_cand.get(c, "") for c in fold0_df["cand"].to_list()]
    y0 = fold0_df["label"].to_list()

    t1_a = [text_s1.get(s, "") for s in fold1_df["s1"].to_list()]
    t1_b = [text_cand.get(c, "") for c in fold1_df["cand"].to_list()]
    y1 = fold1_df["label"].to_list()

    # Train Fold 0 -> Model 0
    model_0 = train_fold(t0_a, t0_b, y0, tokenizer, fold_idx=0)
    # Train Fold 1 -> Model 1
    model_1 = train_fold(t1_a, t1_b, y1, tokenizer, fold_idx=1)

    # 3. Score Validation Candidates
    log("Loading and scoring validation candidates...")
    val_cands = pl.read_parquet(find_file(input_dir, "val_candidates.parquet"))
    val_a = [text_s1.get(s, "") for s in val_cands["s1"].to_list()]
    val_b = [text_cand.get(c, "") for c in val_cands["cand"].to_list()]

    val_dataset = PairDataset(val_a, val_b)
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=make_collate_fn(tokenizer),
        num_workers=2,
        pin_memory=True,
    )

    log("Scoring validation set with ensemble of Model 0 & Model 1...")
    p_val_0 = predict_loader(model_0, val_loader)
    p_val_1 = predict_loader(model_1, val_loader)
    p_val = (p_val_0 + p_val_1) / 2.0

    val_out = val_cands.select(["s1", "cand"]).with_columns(pl.Series("ce_score", p_val))
    val_out_path = out_dir / "ce_scores_val.parquet"
    val_out.write_parquet(val_out_path)
    log(f"Wrote validation cross-encoder scores to {val_out_path} ({val_out.height:,} rows).")

    # 4. Score Test Candidates with 2-Tier Smart Sieve
    log("Loading test candidates...")
    test_cands = pl.read_parquet(find_file(input_dir, "test_candidates.parquet"))
    n_test_total = test_cands.height
    log(f"Total test candidate pairs: {n_test_total:,}")

    # 2-Tier Sieve: High-priority candidates (top 3 per S1 or high initial match)
    # Ranks are present or computed over score
    score_col = "score_name" if "score_name" in test_cands.columns else ("p_u50" if "p_u50" in test_cands.columns else None)
    if score_col:
        test_cands = test_cands.with_columns(
            pl.col(score_col).rank(descending=True).over("s1").alias("cand_rank")
        )
        tier2_mask = (pl.col("cand_rank") <= 3) | (pl.col(score_col) >= 0.40)
    else:
        # Fallback if no score col: top 3 by order
        test_cands = test_cands.with_columns(pl.int_range(0, pl.len()).over("s1").alias("cand_rank"))
        tier2_mask = pl.col("cand_rank") < 3

    tier2_df = test_cands.filter(tier2_mask)
    log(f"Tier 2 High-Stakes Ambiguity Zone: {tier2_df.height:,} pairs ({tier2_df.height / n_test_total:.1%} of total).")
    log(f"Tier 1 Instant Rejects: {n_test_total - tier2_df.height:,} pairs assigned ce_score = 0.0.")

    test_a = [text_s1.get(s, "") for s in tier2_df["s1"].to_list()]
    test_b = [text_cand.get(c, "") for c in tier2_df["cand"].to_list()]

    test_dataset = PairDataset(test_a, test_b)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=make_collate_fn(tokenizer),
        num_workers=2,
        pin_memory=True,
    )

    log(f"Running inference on {len(test_dataset):,} Tier-2 pairs...")
    p_test_0 = predict_loader(model_0, test_loader)
    p_test_1 = predict_loader(model_1, test_loader)
    p_test_tier2 = (p_test_0 + p_test_1) / 2.0

    tier2_scored = tier2_df.select(["s1", "cand"]).with_columns(pl.Series("ce_score", p_test_tier2))

    log("Joining Tier 2 predictions back to full candidate table (filling Tier 1 with 0.0)...")
    final_test = (
        test_cands.select(["s1", "cand"])
        .join(tier2_scored, on=["s1", "cand"], how="left")
        .with_columns(pl.col("ce_score").fill_null(0.0).cast(pl.Float32))
    )

    test_out_path = out_dir / "ce_scores_test.parquet"
    final_test.write_parquet(test_out_path)
    log(f"Wrote full test cross-encoder scores to {test_out_path} ({final_test.height:,} rows).")

    summary = {
        "model": MODEL_NAME,
        "n_train_pairs": len(train_df),
        "n_val_scored": val_out.height,
        "n_test_total": n_test_total,
        "n_test_tier2_scored": tier2_df.height,
        "total_elapsed_sec": time.time() - T0,
    }
    with open(out_dir / "ce_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    log(f"=== Cross-Encoder Pipeline Successfully Completed in {(time.time() - T0)/60:.1f} minutes ===")


if __name__ == "__main__":
    main()
