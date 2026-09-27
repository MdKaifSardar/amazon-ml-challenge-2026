"""Targeted pair features for the error patterns of model v2 (step B). Country-agnostic, no labels.

Groups (column prefixes):
  b1_  descriptor-word difference: words in one name but not the other after removing legal words; how many are
       generic descriptors (directions, "services", "group", "holdings", ... in English and French); the
       IDF-weighted size of the difference (a rare extra word signals a different business).
  b2_  legal form different AND address the same / nearly the same (the "PLLC vs LLC at the same address" decoy).
  b3_  one address empty AND names (legal words removed) identical / near-identical, and how common the S1's core
       name is in its country / state (a rare name with an empty address is likely a match, a common one is not).
  b4_  trade-name case: same normalised address and same first house number but low name similarity; how rare the
       shared address is in the country (a rare shared address is strong evidence).
Counts (IDF, name frequency per country / state, address frequency per country) come from ALL train + test
records of each country, the same for every split.
"""
from __future__ import annotations

import math

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process

B_GROUPS = {
    "b1": ["b1_n_diff", "b1_n_desc_diff", "b1_n_other_diff", "b1_idf_diff", "b1_idf_diff_max"],
    "b2": ["b2_legal_diff", "b2_legal_diff_addr_same", "b2_addr_sort"],
    "b3": ["b3_addr_empty_one", "b3_name_near", "b3_empty_near", "b3_name_freq_state", "b3_empty_near_rare"],
    "b4": ["b4_same_addr", "b4_same_num", "b4_trade", "b4_addr_freq", "b4_trade_rare_addr"],
}
DESCRIPTORS = {
    # directions / places
    "north", "south", "east", "west", "central", "northern", "southern", "eastern", "western", "upper", "lower",
    "nord", "sud", "est", "ouest", "grand", "greater", "new", "old", "city", "metro", "national", "international",
    "global", "india", "indian", "america", "american", "usa", "us", "france", "francais", "francaise",
    # generic business words
    "services", "service", "group", "groupe", "holdings", "holding", "center", "centre", "solutions", "solution",
    "enterprises", "enterprise", "industries", "industry", "associates", "associate", "partners", "brothers", "bros",
    "sons", "and", "company", "co", "trading", "traders", "systems", "technologies", "technology", "tech", "consulting",
    "consultants", "management", "ventures", "international", "the", "of", "de", "du", "des", "la", "le", "les", "et",
    "societe", "society", "agency", "agence", "store", "shop", "clinic", "hospital", "school", "ecole", "office",
    "branch", "unit", "division", "dept", "department", "main", "head", "hq",
}


def _tok(s: str | None) -> set[str]:
    return set(s.split()) if s else set()


def country_counts(records: pl.LazyFrame) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """records: ALL train + test records (S1, S2, S3) with country, state, core_name, addr_norm.
    Returns (token idf per country, core_name count per country + state, addr_norm count per country)."""
    r = records.select(pl.col("country").fill_null(""), pl.col("state").fill_null(""), pl.col("core_name").fill_null(""),
                       pl.col("addr_norm").fill_null(""))
    n = r.group_by("country").len("n_rec")
    idf = (r.select("country", pl.col("core_name").str.split(" ").list.unique().alias("t")).explode("t")
           .filter(pl.col("t").is_not_null() & (pl.col("t") != "")).group_by("country", "t").len("df")
           .join(n, on="country").select("country", "t", (pl.col("n_rec") / pl.col("df")).log().alias("idf")).collect())
    name_state = r.filter(pl.col("core_name") != "").group_by("country", "state", "core_name").len("n_name").collect()
    addr = r.filter(pl.col("addr_norm") != "").group_by("country", "addr_norm").len("n_addr").collect()
    return idf, name_state, addr


def extra_features(pairs: pl.DataFrame, s1_rec: pl.DataFrame, pool_rec: pl.DataFrame, idf: pl.DataFrame,
                   name_state: pl.DataFrame, addr: pl.DataFrame) -> pl.DataFrame:
    """pairs: s1, cand. *_rec: entity_id, country, state, core_name, nolegal, legal, addr_norm, numbers (prepared
    records, e.g. features.prepare_records). Returns s1, cand + every B_GROUPS column (float32), in pairs' order."""
    cols = ("country", "state", "core_name", "nolegal", "legal", "addr_norm", "numbers")
    x = (pairs.select("s1", "cand").with_row_index("_i")
         .join(s1_rec.select(pl.col("entity_id").alias("s1"), *[pl.col(c).alias(f"q_{c}") for c in cols]), on="s1", how="left")
         .join(pool_rec.select(pl.col("entity_id").alias("cand"), *[pl.col(c).alias(f"p_{c}") for c in cols]), on="cand", how="left")
         .sort("_i"))
    qn, pn = x["q_nolegal"].fill_null("").to_list(), x["p_nolegal"].fill_null("").to_list()
    country = x["q_country"].fill_null("").to_list()
    idf_map = {(c, t): v for c, t, v in idf.iter_rows()}
    idf_max = {c: g["idf"].max() for (c,), g in idf.group_by("country")}
    n_diff, n_desc, n_other, idf_sum, idf_mx = [], [], [], [], []
    for a, b, c in zip(qn, pn, country):
        d = _tok(a) ^ _tok(b)
        desc = [t for t in d if t in DESCRIPTORS]
        other = [t for t in d if t not in DESCRIPTORS]
        w = [idf_map.get((c, t), idf_max.get(c, 10.0)) for t in other]
        n_diff.append(len(d)); n_desc.append(len(desc)); n_other.append(len(other))
        idf_sum.append(sum(w)); idf_mx.append(max(w) if w else 0.0)
    near = process.cpdist(qn, pn, scorer=fuzz.token_sort_ratio, workers=-1) / 100.0
    qa, pa = x["q_addr_norm"].fill_null("").to_list(), x["p_addr_norm"].fill_null("").to_list()
    asort = process.cpdist(qa, pa, scorer=fuzz.token_sort_ratio, workers=-1) / 100.0
    empty_a = (x["q_addr_norm"].fill_null("") == "") | (x["p_addr_norm"].fill_null("") == "")
    ql, pl_ = x["q_legal"].fill_null(""), x["p_legal"].fill_null("")
    x = x.with_columns(
        pl.Series("b1_n_diff", n_diff), pl.Series("b1_n_desc_diff", n_desc), pl.Series("b1_n_other_diff", n_other),
        pl.Series("b1_idf_diff", idf_sum, dtype=pl.Float64), pl.Series("b1_idf_diff_max", idf_mx, dtype=pl.Float64),
        pl.Series("_near", near), pl.Series("_asort", np.where(empty_a.to_numpy(), np.nan, asort)),
        empty_a.alias("_empty"), ((ql != "") & (pl_ != "") & (ql != pl_)).alias("_legal_diff"),
        pl.col("q_numbers").list.first().alias("_qn1"), pl.col("p_numbers").list.first().alias("_pn1"),
    )
    x = (x.join(name_state.rename({"country": "q_country", "state": "q_state", "core_name": "q_core_name"}),
                on=["q_country", "q_state", "q_core_name"], how="left", maintain_order="left")
         .join(addr.rename({"country": "q_country", "addr_norm": "q_addr_norm"}), on=["q_country", "q_addr_norm"],
               how="left", maintain_order="left"))
    same_addr = (~pl.col("_empty")) & (pl.col("_asort") >= 0.95)
    same_num = (pl.col("_qn1").is_not_null() & (pl.col("_qn1") == pl.col("_pn1"))).fill_null(False)
    name_freq = pl.col("n_name").fill_null(0).cast(pl.Float64).log1p()
    addr_freq = pl.col("n_addr").fill_null(0).cast(pl.Float64).log1p()
    out = x.with_columns(
        pl.col("_legal_diff").cast(pl.Float32).alias("b2_legal_diff"),
        (pl.col("_legal_diff") & same_addr.fill_null(False)).cast(pl.Float32).alias("b2_legal_diff_addr_same"),
        pl.col("_asort").alias("b2_addr_sort"),
        pl.col("_empty").cast(pl.Float32).alias("b3_addr_empty_one"),
        (pl.col("_near") >= 0.95).cast(pl.Float32).alias("b3_name_near"),
        (pl.col("_empty") & (pl.col("_near") >= 0.95)).cast(pl.Float32).alias("b3_empty_near"),
        name_freq.alias("b3_name_freq_state"),
        (pl.col("_empty") & (pl.col("_near") >= 0.95) & (pl.col("n_name").fill_null(0) <= 3)).cast(pl.Float32).alias("b3_empty_near_rare"),
        same_addr.fill_null(False).cast(pl.Float32).alias("b4_same_addr"),
        same_num.cast(pl.Float32).alias("b4_same_num"),
        (same_addr.fill_null(False) & same_num & (pl.col("_near") < 0.5)).cast(pl.Float32).alias("b4_trade"),
        addr_freq.alias("b4_addr_freq"),
        (same_addr.fill_null(False) & same_num & (pl.col("_near") < 0.5) & (pl.col("n_addr").fill_null(0) <= 5)).cast(pl.Float32).alias("b4_trade_rare_addr"),
    )
    feats = [c for g in B_GROUPS.values() for c in g]
    return out.sort("_i").select("s1", "cand", *[pl.col(c).cast(pl.Float32) for c in feats])
