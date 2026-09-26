"""Match model: LightGBM binary classifier on candidate pairs (features feat-v2, table 6).

Feature sets (the France question: French test candidates are found by fewer search methods, median 2 vs 4):
  A  all 86 features;
  B  A without f_n_methods (how many search methods found the pair);
  C  B without the pruner-derived features (f_p_u50, f_p_gap, f_list_rank): the blocking pruner itself uses the
     method count and the per-method ranks, so its score can carry the same bias.
Per-method scores and ranks stay in every set (they are similarities; "not found" is NaN).

Training S1 exclude the 100k train-split S1 the blocking pruner was trained on (their pruner scores and lists
are in-sample); `A_all` keeps them, to measure that effect. Early stopping uses a held-out tenth of the training
S1 (by S1, never by pair); validation S1 are never used for training or early stopping.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import polars as pl

SEED = 42
PRUNER_DERIVED = ("f_p_u50", "f_p_gap", "f_list_rank")
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=100, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, seed=SEED, deterministic=True, verbose=-1,
              num_threads=0)
ROUNDS, EARLY_STOP = 3000, 100


def feature_sets(cols: list[str]) -> dict[str, list[str]]:
    f = [c for c in cols if c.startswith("f_")]
    b = [c for c in f if c != "f_n_methods"]
    return {"A": f, "B": b, "C": [c for c in b if c not in PRUNER_DERIVED]}


def early_stop_mask(s1: pl.Series, share: float = 0.1) -> np.ndarray:
    """True for rows whose S1 falls in the held-out share (stable hash of the id, same S1 -> same side)."""
    return ((s1.hash(seed=SEED) % 1000) < int(share * 1000)).to_numpy()


def train(x: pl.DataFrame, y: np.ndarray, feats: list[str], log=print) -> lgb.Booster:
    """x: s1 + feature columns; y: 0/1. Early stopping on a held-out tenth of the S1."""
    hold = early_stop_mask(x["s1"])
    m = x.select(feats).to_numpy()
    dtr = lgb.Dataset(m[~hold], y[~hold], feature_name=feats, free_raw_data=True)
    dva = lgb.Dataset(m[hold], y[hold], reference=dtr)
    booster = lgb.train(PARAMS, dtr, ROUNDS, valid_sets=[dva], valid_names=["early_stop"],
                        callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False), lgb.log_evaluation(0)])
    log(f"    {len(feats)} features, {int((~hold).sum()):,} fit rows, {int(hold.sum()):,} early-stop rows, "
        f"best iteration {booster.best_iteration}, early-stop logloss {booster.best_score['early_stop']['binary_logloss']:.4f}")
    return booster


def predict(booster: lgb.Booster, x: pl.DataFrame, chunk: int = 2_000_000) -> np.ndarray:
    feats = booster.feature_name()
    return np.concatenate([booster.predict(x[lo:lo + chunk].select(feats).to_numpy(), num_iteration=booster.best_iteration)
                           for lo in range(0, x.height, chunk)] or [np.empty(0)])


def importance(booster: lgb.Booster, name: str) -> pl.DataFrame:
    g = booster.feature_importance("gain")
    return (pl.DataFrame({"variant": name, "feature": booster.feature_name(), "gain": g.astype(float)})
            .with_columns((pl.col("gain") / pl.col("gain").sum()).alias("gain_share")).sort("gain", descending=True))
