"""Unit tests for pair features (src/features.py) on small hand-made normalised records."""
import math
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from features import build, count_stats, feature_names, legal_words, prepare_records  # noqa: E402

REC_SCHEMA = {"entity_id": pl.Utf8, "country": pl.Utf8, "business_address": pl.Utf8, "full_name": pl.Utf8, "core_name": pl.Utf8, "legal": pl.Utf8,
              "addr_norm": pl.Utf8, "city": pl.Utf8, "state": pl.Utf8, "dept": pl.Utf8, "numbers": pl.List(pl.Utf8)}


def recs(rows):
    return prepare_records(pl.DataFrame(rows, schema=REC_SCHEMA, orient="row"))


S1 = recs([
    ("S1-1", "US", "24 Greenwood Ln, Austin, TX 78701", "willow llc", "willow", "llc", "24 greenwood lane, austin, tx 78701", "austin", "tx", None, ["24", "78701"]),
    ("S1-2", "India", "12 MG Road, Bengaluru, Karnataka 560001", "royal food pvt ltd", "royal food", "pvt ltd", "12 mg road, bengaluru, karnataka 560001",
     "bengaluru", "karnataka", None, ["12", "560001"]),
    ("S1-3", "France", "12 Rue Saint-Honore, 75001 Paris", "boulangerie martin sarl", "boulangerie martin", "sarl", "12 rue saint honore, 75001 paris",
     "paris", "ile de france", None, ["12", "75001"]),
])
POOL = recs([
    ("S2-a", "US", "24 Greenwood Lane, Austin TX 78701", "willow llc center", "willow llc center", "", "24 greenwood lane, austin, tx 78701", "austin", "tx", None, ["24", "78701"]),
    ("S3-b", "US", "25 Greenwood Lane, Austin, TX 78701", "willow inc", "willow", "inc", "25 greenwood lane, austin, tx 78701", "austin", "tx", None, ["25", "78701"]),
    ("S2-c", "India", "", "royal food private limited", "royal food", "pvt ltd", "", None, None, None, []),
    ("S3-d", "France", "12 r Saint Honore, Paris", "boulangerie martin sarl", "boulangerie martin", "sarl", "12 r saint honore, paris", None, None, None, ["12"]),
])
STATS = count_stats([S1], [POOL])
PAIRS = pl.DataFrame({
    "s1": ["S1-1", "S1-1", "S1-2", "S1-3"], "cand": ["S2-a", "S3-b", "S2-c", "S3-d"],
    "score_name": [0.9, 0.95, 0.8, None], "rank_name": [2, 1, 1, None],
    "score_reverse": [0.7, None, 0.6, 0.9], "rank_reverse": [1, None, 1, 1], "best_reverse": [0.7, None, 0.6, 0.9],
    "p_u50": [0.9, 0.3, 0.8, 0.95],
})


@pytest.fixture(scope="module")
def feats():
    stats = STATS
    return build(PAIRS, S1, POOL, stats, chunk=3)  # chunk smaller than the table: S1-1's list spans two chunks


def row(x, cand):
    return x.filter(pl.col("cand") == cand).row(0, named=True)


def test_schema_order_and_dtype(feats):
    assert feats.columns[:2] == ["s1", "cand"]
    assert feats.select("s1", "cand").equals(PAIRS.select("s1", "cand"))
    names = feature_names(feats)
    assert names and all(feats.schema[c] == pl.Float32 for c in names)
    assert not any(w in c for c in feats.columns for w in ("country", "is_match", "lang"))


def test_nolegal_handles_mid_name_legal_word(feats):
    assert POOL.filter(pl.col("entity_id") == "S2-a")["nolegal"][0] == "willow center"
    r = row(feats, "S2-a")
    assert r["f_name_nolegal_tok_jacc"] > r["f_name_core_tok_jacc"]
    assert r["f_name_core_tsr"] == pytest.approx(1.0)  # token set: {willow} is a subset


def test_legal_agreement(feats):
    a, b = row(feats, "S2-a"), row(feats, "S3-b")
    assert a["f_legal_one_missing"] == 1 and math.isnan(a["f_legal_same"])  # edge legal form missing on S2-a
    assert a["f_legal_any_same"] == 1  # but "llc" is found mid-name
    assert b["f_legal_same"] == 0 and b["f_legal_one_missing"] == 0  # llc vs inc: the decoy


def test_house_number_decoy(feats):
    a, b = row(feats, "S2-a"), row(feats, "S3-b")
    assert a["f_num_first_eq"] == 1 and b["f_num_first_eq"] == 0
    assert a["f_postcode_agree"] == 1 and a["f_s1_has_postcode"] == 1 and a["f_cand_has_postcode"] == 1
    assert a["f_city_same"] == 1
    assert a["f_ctx_num_jacc"] > 0 and b["f_ctx_num_jacc"] < 0  # better / worse than the other candidate


def test_empty_address_gives_nan_not_zero(feats):
    r = row(feats, "S2-c")
    assert r["f_addr_missing"] == 1
    for c in ("f_addr_ratio", "f_addr_tsr", "f_city_same", "f_num_jacc", "f_num_first_eq"):
        assert math.isnan(r[c]), c
    assert r["f_num_n_cand"] == 0
    assert r["f_name_core_ratio"] == pytest.approx(1.0) and r["f_legal_same"] == 1  # private limited = pvt ltd
    assert math.isnan(r["f_ctx_name_core_tsr"])  # only candidate of its S1


def test_france_works_without_labels_or_country_rules(feats):
    r = row(feats, "S3-d")
    assert r["f_name_full_ratio"] == pytest.approx(1.0) and r["f_legal_same"] == 1
    assert math.isnan(r["f_city_same"]) and r["f_city_in_addr"] == 1  # "paris" appears in the candidate's address
    assert r["f_num_first_eq"] == 1 and math.isnan(r["f_postcode_agree"]) and r["f_cand_has_postcode"] == 0  # candidate has no postcode


def test_blocking_features(feats):
    a, b, d = row(feats, "S2-a"), row(feats, "S3-b"), row(feats, "S3-d")
    assert a["f_gap_name"] == pytest.approx(0.05) and b["f_gap_name"] == pytest.approx(0.0)
    assert math.isnan(b["f_score_reverse"]) and math.isnan(d["f_rank_name"])
    assert a["f_n_methods"] == 2 and b["f_n_methods"] == 1
    assert a["f_list_rank"] == 1 and b["f_list_rank"] == 2 and a["f_list_size"] == 2
    assert b["f_p_gap"] == pytest.approx(0.6)
    assert math.isnan(a["f_score_name_city"])  # absent column -> NaN, same schema for every split
    assert "f_score_rare" not in feats.columns and a["f_cand_is_s3"] == 0 and b["f_cand_is_s3"] == 1


def test_same_result_regardless_of_chunk(feats):
    one = build(PAIRS, S1, POOL, STATS, chunk=10)
    assert np.allclose(one.drop("s1", "cand").to_numpy(), feats.drop("s1", "cand").to_numpy(), equal_nan=True)


def test_legal_words_open_set():
    assert "sarl" in legal_words("France") and "sarl" in legal_words("Germany") and "sarl" not in legal_words("US")
    assert "pra" not in legal_words("India") and "praivet" in legal_words("India")


def test_missing_record_raises():
    with pytest.raises(ValueError):
        build(PAIRS.with_columns(pl.lit("S2-zz").alias("cand")).head(1), S1, POOL, STATS)


def test_prune_standard_matches_default_operating_point():
    from run_features import prune_standard
    cfg = {"default_target": 7, "unions": {"u50": {"rev_m": 5, "fwd_k": 50, "rare_k": 20}},
           "operating_points": [{"target": 7, "config": '{"union": "u50", "tau": 0.01, "cap": 10}'}]}
    cols = {"scope": pl.Utf8, "s1": pl.Utf8, "cand": pl.Utf8, "p_u50": pl.Float64,
            **{f"rank_{m}": pl.Int64 for m in ("name", "name_city", "name_addr", "reverse", "rare")}}
    rows = [("state", "a", f"S2-{i:02d}", 0.5 + i / 100, i + 1, None, None, None, None) for i in range(12)]  # 12 above tau
    rows += [("state", "a", "S2-low", 0.005, 1, None, None, None, None),  # below tau
             ("state", "a", "S2-far", None, 60, None, None, None, None),  # outside u50 (fwd rank 60), no p_u50
             ("state", "b", "S3-rev", 0.02, None, None, None, 5, None)]  # reverse rank 5 is in u50
    c = pl.DataFrame(rows, schema=cols, orient="row")
    out = prune_standard(c, cfg)
    assert set(out.filter(pl.col("s1") == "a")["cand"]) == {f"S2-{i:02d}" for i in range(2, 12)}  # top 10 by p_u50
    assert out.filter(pl.col("s1") == "b")["cand"].to_list() == ["S3-rev"]
    assert "p_keep" not in out.columns
