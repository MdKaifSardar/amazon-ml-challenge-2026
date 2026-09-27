"""Shared pieces of the final matching model (train_model.py, predict.py).

Design (model v3):
- stage 1: LightGBM on feature set C (features-v2 without the method count and pruner-derived features), 5 folds
  grouped by S1; out-of-fold probabilities on training pairs, the fold-model average elsewhere;
- stage 2 features: C + set features (two_stage.py: the pair's stage-1 probability relative to the other candidates
  of its S1 list) + targeted B features (pair_extra.py);
- stage 2: LightGBM (several seeds averaged) and optionally CatBoost, blended as configured;
- selection (selection.py): threshold, margin to the S1's best, one owner per S2/S3 (always on test).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from features import prepare_records
from pair_extra import country_counts, extra_features
from two_stage import SET_FEATURES, set_features

REC_COLS = ["entity_id", "country", "full_name", "core_name", "legal", "addr_norm", "city", "state", "numbers"]


def find(root: Path, name: str, parent: str | None = None, not_parent: tuple = ()) -> Path:
    hits = sorted(p for p in root.rglob(name) if (parent is None or p.parent.name == parent) and p.parent.name not in not_parent)
    if not hits:
        raise FileNotFoundError(f"{name} not found under {root}")
    return hits[0]


def raw_file(root: Path, name: str) -> Path:
    """The organiser / blocking / features copy of a file (never the normalised or experiment copies)."""
    return find(root, name, not_parent=("normalised", "features_extra", "aug"))


def normalised_files(root: Path) -> dict[tuple[str, int], Path]:
    return {(sd, k): find(root, f"{sd}_source{k}.parquet", parent="normalised") for sd in ("train", "test") for k in (1, 2, 3)}


def counts(norm: dict) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """IDF, name-per-state and address counts over ALL train + test records (no labels)."""
    return country_counts(pl.concat([pl.scan_parquet(p).select("country", "state", "core_name", "addr_norm") for p in norm.values()]))


def records(norm: dict, side: str, s1_ids: pl.Series, cand_ids: pl.Series) -> tuple[pl.DataFrame, pl.DataFrame]:
    s1 = prepare_records(pl.scan_parquet(norm[(side, 1)]).select(REC_COLS).filter(pl.col("entity_id").is_in(s1_ids.implode())).collect())
    c = prepare_records(pl.concat([pl.scan_parquet(norm[(side, k)]).select(REC_COLS).filter(pl.col("entity_id").is_in(cand_ids.implode())).collect()
                                   for k in (2, 3)]))
    return s1, c


def stage2_frame(fx: pl.DataFrame, p1: np.ndarray, s1_rec: pl.DataFrame, c_rec: pl.DataFrame, cnt, stage1_feats: list[str]) -> pl.DataFrame:
    """fx: s1, cand + features-v2 columns (whole S1 lists, sorted by s1, cand). Returns s1, cand + C + set + B."""
    sf = set_features(fx, p1, c_rec.select("entity_id", "full_name", "addr_norm", "numbers"))
    bx = extra_features(fx.select("s1", "cand"), s1_rec, c_rec, *cnt)
    return fx.select("s1", "cand", *stage1_feats).hstack(sf.select(SET_FEATURES)).hstack(bx.drop("s1", "cand"))


def blend(p_lgb: np.ndarray, p_cb: np.ndarray | None, how: str) -> np.ndarray:
    if p_cb is None or how == "lgb":
        return p_lgb
    if how == "cb":
        return p_cb
    if how == "mean":
        return (p_lgb + p_cb) / 2
    if how == "rank":
        n = len(p_lgb)
        return (pl.Series(p_lgb).rank().to_numpy() + pl.Series(p_cb).rank().to_numpy()) / (2 * n)
    raise ValueError(how)
