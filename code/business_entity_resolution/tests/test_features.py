"""Unit tests for pair features (src/features.py) on small hand-made normalised records."""
import math
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from features import (build, count_stats, country_state_maps, feature_names, fill_record_states,  # noqa: E402
                      find_postcode, legal_words, prepare_records)

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


# ---------------------------------------------------------------------------- feat-v2

def raw(eid, country, addr, core, city=None, state=None, dept=None, legal="", numbers=()):
    """One normalised record (before prepare_records); full_name = core + legal."""
    return (eid, country, addr, f"{core} {legal}".strip(), core, legal, addr.lower(), city, state, dept, list(numbers))


def frame(rows):
    return pl.DataFrame(rows, schema=REC_SCHEMA, orient="row")


@pytest.mark.parametrize("components, city, expected", [
    (["24 greenwood ln", "austin", "tx 78701"], "austin", "78701"),  # ends a component after a state word
    (["24 greenwood ln austin tx 78701"], "austin", "78701"),  # no commas
    (["12345 main st", "austin", "tx 78701 1234"], "austin", "78701"),  # ZIP+4; the house number is not a ZIP
    (["12345 main st austin tx"], "austin", None),  # house number only: first token of the address
    (["12345 broadway"], None, None),
    (["suite 12345", "100 main st", "austin tx"], "austin", None),  # after a unit word
    (["po box 12345", "austin"], "austin", None),
    (["plot no 123456", "bengaluru"], "bengaluru", None),
    (["unit 5", "12345 main st", "austin"], "austin", None),  # starts a later component, but a street follows
    (["no 12 mg road", "bengaluru", "karnataka 560001", "india"], "bengaluru", "560001"),  # trailing country ignored
    (["12 rue x", "02100 saint quentin", "france"], None, "02100"),  # starts a place-name component; zero kept
    (["12 rue saint honore 02100 saint quentin"], "saint quentin", "02100"),  # right before the city
    (["12 r saint honore", "paris"], "paris", None),
    ([], None, None),
])
def test_find_postcode_by_position(components, city, expected):
    assert find_postcode(components, city) == expected


def test_prepare_records_postcode_from_raw_address_keeps_leading_zero():
    rec = prepare_records(frame([raw("S2-x", "France", "3 Rue Pasteur, 02100 Saint-Quentin", "cafe", city="saint quentin"),
                                 raw("S2-y", "US", "12345 Main St, Austin, TX", "cafe", city="austin"),
                                 raw("S2-z", "India", "Shop 4, MG Road, Pune - 411001", "cafe", city="pune")]))
    assert dict(zip(rec["entity_id"], rec["postcode"])) == {"S2-x": "02100", "S2-y": None, "S2-z": "411001"}
    assert "business_address" not in rec.columns
    assert S1.filter(pl.col("entity_id") == "S1-1")["postcode"][0] == "78701"


def test_postcode_features(feats):
    a, c, d = row(feats, "S2-a"), row(feats, "S2-c"), row(feats, "S3-d")
    assert (a["f_postcode_agree"], a["f_s1_has_postcode"], a["f_cand_has_postcode"]) == (1, 1, 1)
    assert math.isnan(c["f_postcode_agree"]) and c["f_s1_has_postcode"] == 1 and c["f_cand_has_postcode"] == 0
    assert d["f_s1_has_postcode"] == 1 and d["f_cand_has_postcode"] == 0  # "75001 paris" starts a place component
    assert not any(k.startswith("f_num_long") for k in feats.columns)


def test_state_filled_like_blocking_and_dept_agreement():
    # 25 records with city + state teach "saint quentin" -> hauts de france (blocking: >= 20 records, >= 90%).
    loc = pl.DataFrame({"country": ["France"] * 25 + ["US"] * 25, "city": ["saint quentin"] * 25 + ["austin"] * 25,
                        "dept": ["aisne"] * 25 + [None] * 25, "state": ["hauts de france"] * 25 + ["texas"] * 25})
    maps = country_state_maps(loc)
    assert set(maps) == {"France", "US"}
    s1 = prepare_records(fill_record_states(frame([
        raw("S1-f", "France", "1 rue x, 02100 saint quentin", "cafe du nord", "saint quentin", "hauts de france", "aisne"),
        raw("S1-u", "US", "1 main st, austin, tx 78701", "cafe du nord", "austin", "texas"),
    ]), maps))
    pool = prepare_records(fill_record_states(frame([
        raw("S2-f1", "France", "1 rue x, saint quentin", "cafe du nord", "saint quentin", None, "aisne"),  # state from city
        raw("S3-f2", "France", "1 rue x, lille", "cafe du nord", "lille", "hauts de france", "nord"),  # other dept
        raw("S2-f3", "France", "1 rue x", "cafe du nord"),  # no city, state or dept
        raw("S3-u1", "US", "1 main st, austin", "cafe du nord", "austin", None),  # state from city
        raw("S2-u2", "US", "1 main st, dallas, tx", "cafe du nord", "dallas", "oklahoma"),  # different state
    ]), maps))
    assert pool.filter(pl.col("entity_id") == "S2-f1")["state"][0] == "hauts de france"
    assert "state_src" not in pool.columns
    pairs = pl.DataFrame({"s1": ["S1-f"] * 3 + ["S1-u"] * 2, "cand": ["S2-f1", "S3-f2", "S2-f3", "S3-u1", "S2-u2"]})
    x = build(pairs, s1, pool, count_stats([s1], [pool]))
    st = dict(zip(x["cand"], x["f_state_agree"].to_list()))
    dp = dict(zip(x["cand"], x["f_dept_agree"].to_list()))
    assert st["S2-f1"] == 1 and st["S3-f2"] == 1 and math.isnan(st["S2-f3"]) and st["S3-u1"] == 1 and st["S2-u2"] == 0
    assert dp["S2-f1"] == 1 and dp["S3-f2"] == 0 and math.isnan(dp["S2-f3"])
    assert math.isnan(dp["S3-u1"]) and math.isnan(dp["S2-u2"])  # no departement outside France: all-NaN in train
    assert dict(zip(x["cand"], x["f_cand_is_s3"].to_list())) == {"S2-f1": 0, "S3-f2": 1, "S2-f3": 0, "S3-u1": 1, "S2-u2": 0}


def test_rare_word_features_use_blocking_definition():
    s1 = prepare_records(frame([raw("S1-r", "India", "x", "zorbex food ab")]))
    pool = prepare_records(frame([raw("S2-r1", "India", "x", "zorbex food ab"), raw("S3-r2", "India", "x", "qwilly food ab")]))
    # "food" and "ab" are common in the country (df > 20); "zorbex" and "qwilly" are rare.
    filler = frame([raw(f"S2-f{i}", "India", "x", "food ab") for i in range(30)])
    x = build(pl.DataFrame({"s1": ["S1-r", "S1-r"], "cand": ["S2-r1", "S3-r2"]}), s1, pool, count_stats([s1], [pool, filler]))
    a, b = row(x, "S2-r1"), row(x, "S3-r2")
    assert (a["f_rare_shared"], a["f_rare_only_s1"], a["f_rare_only_cand"]) == (1, 0, 0) and a["f_rare_shared_idf"] > 0
    assert (b["f_rare_shared"], b["f_rare_only_s1"], b["f_rare_only_cand"]) == (0, 1, 1) and b["f_rare_shared_idf"] == 0
    assert not any(k in x.columns for k in ("f_score_rare", "f_rank_rare"))
    # a token shorter than 3 characters is never rare, however few records have it
    s1b = prepare_records(frame([raw("S1-s", "India", "x", "ab food")]))
    pb = prepare_records(frame([raw("S2-s", "India", "x", "ab food")]))
    y = build(pl.DataFrame({"s1": ["S1-s"], "cand": ["S2-s"]}), s1b, pb, count_stats([s1b], [pb]))
    assert row(y, "S2-s")["f_rare_shared"] == 1  # "food" (df 2) is rare here, "ab" is not


def test_counts_are_per_country_over_all_given_records():
    us = frame([raw(f"S2-u{i}", "US", "x", "royal food") for i in range(4)])
    india = frame([raw(f"S3-i{i}", "India", "x", "royal food") for i in range(2)])
    st = count_stats([frame([raw("S1-1", "US", "x", "royal food")]).lazy()], [us.lazy(), india])  # lazy scans work too
    assert st.n == {"US": 5, "India": 2}
    n = {(c, k): (a, b) for c, k, a, b in st.name_n.select("country", "core_name", "n_s1", "n_pool").iter_rows()}
    assert n[("US", "royal food")] == (1, 4) and n[("India", "royal food")] == (0, 2)
    df = {(c, t): d for c, t, d in st.token_df.select("country", "t", "df").iter_rows()}
    assert df[("US", "royal")] == 5 and df[("India", "food")] == 2


def test_features_do_not_depend_on_other_s1_lists(feats):
    """Counts come from the record files, not from the pairs: adding another S1 whose list shares S2-a and S3-b (as a
    bigger query set would) changes nothing for S1-1, i.e. there is no candidate-popularity feature."""
    extra = prepare_records(frame([raw("S1-9", "US", "24 Greenwood Ln, Austin, TX 78701", "willow", "austin", "tx",
                                       legal="llc", numbers=["24", "78701"])]))
    pairs = pl.concat([PAIRS, pl.DataFrame({"s1": ["S1-9", "S1-9"], "cand": ["S2-a", "S3-b"]})], how="diagonal_relaxed")
    bigger = build(pairs, pl.concat([S1, extra]), POOL, STATS)
    pick = lambda x, s1: x.filter(pl.col("s1") == s1).sort("cand").drop("s1", "cand").to_numpy()
    assert np.allclose(pick(feats, "S1-1"), pick(bigger, "S1-1"), equal_nan=True)
    alone = build(PAIRS.filter(pl.col("s1") == "S1-2"), S1, POOL, STATS)
    assert np.allclose(pick(alone, "S1-2"), pick(feats, "S1-2"), equal_nan=True)


def test_feature_count_and_names(feats):
    names = feature_names(feats)
    assert len(names) == 90 and len(set(names)) == 90
    for f in ("f_state_agree", "f_dept_agree", "f_cand_is_s3", "f_postcode_agree", "f_s1_has_postcode",
              "f_cand_has_postcode", "f_rare_shared", "f_rare_only_s1", "f_rare_only_cand", "f_rare_shared_idf"):
        assert f in names, f


def test_runner_checks_and_shift_flags(feats):
    from run_features import checks, profile, shift_table
    ok = checks(feats, feats.schema)
    assert ok["same_schema_as_ref"] and ok["all_float32"] and ok["duplicate_pairs"] == 0 and ok["inf_values"] == 0
    bad = pl.concat([feats, feats.head(1)]).with_columns(pl.lit(float("inf"), pl.Float32).alias("f_name_full_ratio"))
    c = checks(bad, feats.schema)
    assert c["duplicate_pairs"] == 2 and c["inf_values"] == bad.height
    assert not checks(feats.select("s1", "cand", *feature_names(feats)[::-1]), feats.schema)["same_schema_as_ref"]
    assert not checks(feats.with_columns(pl.col("f_p_u50").cast(pl.Float64)), feats.schema)["same_schema_as_ref"]
    tr = feats.filter(pl.col("s1") != "S1-3")
    prof = pl.concat([profile(tr, "train", "all"), profile(tr.filter(pl.col("s1") == "S1-1"), "train", "US"),
                      profile(feats.filter(pl.col("s1") == "S1-3"), "test", "France"),
                      profile(feats.filter(pl.col("s1") == "S1-1"), "test", "US")])
    sh = shift_table(prof)
    assert set(sh.filter(pl.col("test_country") == "France")["vs"]) == {"train all"}  # France: no train rows
    assert set(sh.filter(pl.col("test_country") == "US")["vs"]) == {"train US"}
    assert not sh.filter(pl.col("test_country") == "US")["flag"].any()  # identical data: no flag
