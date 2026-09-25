import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from metric import f05, parse_ids, per_entity, report  # noqa: E402


def test_problem_statement_example():
    # PS: predict 3, 2 correct out of 2 true -> P=2/3, R=1 -> 0.714
    pred = ["S2-00047", "S2-00193", "S3-00812"]
    truth = ["S2-00047", "S3-00812"]
    assert f05(pred, truth) == pytest.approx(0.714, abs=5e-4)


def test_singleton_empty_prediction_scores_one():
    assert f05([], []) == 1.0


def test_singleton_any_prediction_scores_zero():
    assert f05(["S2-1"], []) == 0.0


def test_empty_prediction_with_true_matches_scores_zero():
    assert f05([], ["S2-1"]) == 0.0


def test_no_overlap_scores_zero():
    assert f05(["S2-9"], ["S2-1"]) == 0.0


def test_perfect_match():
    assert f05(["S2-1", "S3-2"], ["S3-2", "S2-1"]) == 1.0


def test_precision_weighted_over_recall():
    # half the true matches, all correct (P=1, R=0.5) beats all true matches plus as many wrong (P=0.5, R=1)
    assert f05(["a"], ["a", "b"]) > f05(["a", "b", "x", "y"], ["a", "b"])


def test_duplicates_in_prediction_ignored():
    assert f05(["S2-1", "S2-1"], ["S2-1"]) == 1.0


def test_parse_ids():
    assert parse_ids("S2-1,S3-2") == {"S2-1", "S3-2"}
    assert parse_ids("") == set()
    assert parse_ids(None) == set()


def test_macro_average_and_missing_prediction():
    truth = {"A": {"x"}, "B": set(), "C": {"y", "z"}}
    pred = {"A": {"x"}}  # B missing -> empty -> 1.0; C missing -> 0.0
    df = per_entity(pred, truth)
    assert df["f05"].to_list() == [1.0, 1.0, 0.0]
    r = report(pred, truth, {"A": "US", "B": "US", "C": "India"})
    overall = r.filter(r["group"] == "overall")["macro_f05"].item()
    assert overall == pytest.approx(2 / 3)
    assert r.filter(r["group"] == "country=India")["macro_f05"].item() == 0.0


def test_all_empty_equals_singleton_fraction():
    truth = {"A": set(), "B": {"x"}, "C": set(), "D": {"y"}}
    r = report({}, truth)
    assert r.filter(r["group"] == "overall")["macro_f05"].item() == 0.5
