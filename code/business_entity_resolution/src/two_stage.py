"""Two-stage match model: stage-1 LightGBM (feature set C) -> set-level features -> stage-2 LightGBM.

Stage 1 gives every candidate pair a probability p1: out-of-fold on train (5 folds grouped by S1), the average
of the fold models on validation and test. Set features describe a pair relative to the other candidates of the
same S1 list (list-local, so they mean the same on train, validation and test):
  s_p1           stage-1 probability
  s_rank / s_gap rank of p1 in the S1's list, and the gap to the S1's best p1
  s_n05 / s_n09  number of candidates of the S1 with p1 > 0.5 / > 0.9
  s_sim_name / s_sim_addr / s_num_eq
                 the pair's candidate vs the S1's OTHER confident candidates (p1 > HIGH): max name token-set
                 ratio, max address token-set ratio, and whether any shares the first house number. A second
                 record of the same business looks like the first one; a decoy differs in number or legal form.
Cross-S1 counts (how many S1 lists a record is in) are deliberately NOT used: they depend on which S1 were
queried (300k of 2.2M train S1 vs all test S1), so they would shift between train and test. The reverse-search
rank / gap features already compare S1 for a record with all S1 competing.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import polars as pl
from rapidfuzz import fuzz, process

SEED = 42
HIGH = 0.8
N_FOLDS = 5
SET_FEATURES = ["s_p1", "s_rank", "s_gap", "s_n05", "s_n09", "s_sim_name", "s_sim_addr", "s_num_eq"]
PARAMS = dict(objective="binary", learning_rate=0.08, num_leaves=127, min_data_in_leaf=100, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=0)
ROUNDS, EARLY_STOP = 4000, 100


def fold_of(s1: pl.Series, n: int = N_FOLDS, salt: int = 0) -> np.ndarray:
    return (s1.hash(seed=SEED + salt) % n).cast(pl.Int64).to_numpy()


def fit(m: np.ndarray, y: np.ndarray, hold: np.ndarray, feats: list[str], seed: int = SEED, params: dict | None = None) -> lgb.Booster:
    p = {**PARAMS, **(params or {}), "seed": seed}
    dtr = lgb.Dataset(m[~hold], y[~hold], feature_name=feats, free_raw_data=True)
    dva = lgb.Dataset(m[hold], y[hold], reference=dtr)
    return lgb.train(p, dtr, ROUNDS, valid_sets=[dva], callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])


def predict(b: lgb.Booster, m: np.ndarray, chunk: int = 2_000_000) -> np.ndarray:
    return np.concatenate([b.predict(m[lo:lo + chunk], num_iteration=b.best_iteration) for lo in range(0, len(m), chunk)]
                          or [np.empty(0)])


def stage1_oof(x: pl.DataFrame, y: np.ndarray, feats: list[str], log=print) -> tuple[np.ndarray, list[lgb.Booster]]:
    """5-fold out-of-fold p1 on train (folds by S1); each fold model early-stops on a tenth of its own S1."""
    fold = fold_of(x["s1"])
    es = fold_of(x["s1"], 10, salt=1) == 0
    m = x.select(feats).to_numpy()
    oof, models = np.zeros(len(y)), []
    for k in range(N_FOLDS):
        tr = fold != k
        b = fit(m[tr], y[tr], es[tr], feats)
        oof[~tr] = predict(b, m[~tr])
        models.append(b)
        log(f"  stage-1 fold {k}: best iteration {b.best_iteration}")
    return oof, models


def stage1_predict(models: list[lgb.Booster], x: pl.DataFrame, feats: list[str]) -> np.ndarray:
    m = x.select(feats).to_numpy()
    return np.mean([predict(b, m) for b in models], axis=0)


def set_features(pairs: pl.DataFrame, p1: np.ndarray, rec: pl.DataFrame) -> pl.DataFrame:
    """pairs: s1, cand (all candidates of each S1 present); rec: entity_id, full_name, addr_norm, numbers (S2/S3
    records of the candidates). Returns s1, cand + SET_FEATURES, in the order of pairs."""
    x = pairs.select("s1", "cand").with_row_index("_i").with_columns(pl.Series("s_p1", p1, dtype=pl.Float64))
    x = x.with_columns(
        pl.col("s_p1").rank("ordinal", descending=True).over("s1").cast(pl.Float32).alias("s_rank"),
        (pl.col("s_p1").max().over("s1") - pl.col("s_p1")).alias("s_gap"),
        (pl.col("s_p1") > 0.5).sum().over("s1").cast(pl.Float32).alias("s_n05"),
        (pl.col("s_p1") > 0.9).sum().over("s1").cast(pl.Float32).alias("s_n09"),
    )
    r = rec.select(pl.col("entity_id").alias("cand"), pl.col("full_name").fill_null(""), pl.col("addr_norm").fill_null(""),
                   pl.col("numbers").list.first().alias("num1"))
    a = x.select("_i", "s1", "cand").join(r, on="cand", how="left")
    hi = x.filter(pl.col("s_p1") > HIGH).select("s1", pl.col("cand").alias("cand_o")).join(
        r.rename({"cand": "cand_o", "full_name": "name_o", "addr_norm": "addr_o", "num1": "num1_o"}), on="cand_o", how="left")
    pr = a.join(hi, on="s1", how="inner").filter(pl.col("cand") != pl.col("cand_o"))
    if pr.height:
        pr = pr.with_columns(
            pl.Series("sn", process.cpdist(pr["full_name"].to_list(), pr["name_o"].to_list(), scorer=fuzz.token_set_ratio, workers=-1) / 100.0),
            pl.Series("sa", process.cpdist(pr["addr_norm"].to_list(), pr["addr_o"].to_list(), scorer=fuzz.token_set_ratio, workers=-1) / 100.0),
            (pl.col("num1").is_not_null() & (pl.col("num1") == pl.col("num1_o"))).cast(pl.Float32).alias("ne"))
        agg = pr.group_by("_i").agg(pl.col("sn").max().alias("s_sim_name"), pl.col("sa").max().alias("s_sim_addr"),
                                    pl.col("ne").max().alias("s_num_eq"))
    else:
        agg = pl.DataFrame(schema={"_i": pl.UInt32, "s_sim_name": pl.Float64, "s_sim_addr": pl.Float64, "s_num_eq": pl.Float32})
    out = x.join(agg, on="_i", how="left").sort("_i")
    return out.select("s1", "cand", *[pl.col(c).cast(pl.Float32) for c in SET_FEATURES])
