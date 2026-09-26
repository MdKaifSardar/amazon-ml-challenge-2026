"""Unit tests for match selection and the vectorised F0.5 (src/selection.py), checked against metric.py."""
import sys
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from metric import per_entity  # noqa: E402
from selection import f05_table, select, summary, tune  # noqa: E402


def scored(rows):
    return pl.DataFrame(rows, schema={"s1": pl.String, "cand": pl.String, "p": pl.Float64}, orient="row")


def test_threshold_owner_margin():
    x = scored([("a", "x", 0.9), ("b", "x", 0.8), ("a", "y", 0.4), ("b", "z", 0.95), ("b", "w", 0.3)])
    assert set(select(x, 0.5, one_owner=False).rows()) == {("a", "x", 0.9), ("b", "x", 0.8), ("b", "z", 0.95)}
    # one owner: x goes to a (0.9 > 0.8)
    assert set(map(tuple, select(x, 0.5).select("s1", "cand").rows())) == {("a", "x"), ("b", "z")}
    # margin 0.5: b keeps z (0.95) and drops w (0.3 < 0.475)
    assert set(map(tuple, select(x, 0.2, margin=0.5).select("s1", "cand").rows())) == {("a", "x"), ("b", "z")}
    assert set(map(tuple, select(x, 0.2, margin=0.0).select("s1", "cand").rows())) == {("a", "x"), ("a", "y"), ("b", "z"), ("b", "w")}


def test_owner_tie_is_deterministic():
    x = scored([("b", "x", 0.7), ("a", "x", 0.7)])
    assert select(x, 0.5)["s1"].to_list() == ["a"]


def test_f05_matches_metric_py():
    pred = pl.DataFrame({"s1": ["a", "a", "b", "d"], "cand": ["x", "y", "z", "q"]})
    truth = pl.DataFrame({"s1": ["a", "b", "b", "c"], "cand": ["x", "z", "w", "u"]})
    s1s = pl.DataFrame({"s1": ["a", "b", "c", "d", "e"]})  # c: missed; d: singleton with a prediction; e: empty singleton
    tab = f05_table(pred, truth, s1s).sort("s1")
    pm = {s: pred.filter(pl.col("s1") == s)["cand"].to_list() for s in "abcde"}
    tm = {s: truth.filter(pl.col("s1") == s)["cand"].to_list() for s in "abcde"}
    ref = per_entity(pm, tm).sort("s1")
    assert tab["f05"].to_list() == pytest.approx(ref["f05"].to_list())
    assert tab["f05"].to_list() == pytest.approx([1 / 1.0 * (1.25 * 0.5 * 1) / (0.25 * 0.5 + 1), 1.25 * 1 * 0.5 / (0.25 + 0.5), 0.0, 0.0, 1.0])


def test_f05_ignores_rows_outside_the_evaluated_s1():
    pred = pl.DataFrame({"s1": ["zz"], "cand": ["x"]})
    truth = pl.DataFrame({"s1": ["zz"], "cand": ["x"]})
    assert f05_table(pred, truth, pl.DataFrame({"s1": ["a"]}))["f05"].to_list() == [1.0]


def test_summary_and_tune():
    x = scored([("a", "x", 0.9), ("b", "y", 0.2)])
    truth = pl.DataFrame({"s1": ["a"], "cand": ["x"]})
    s1s = pl.DataFrame({"s1": ["a", "b"], "country": ["US", "India"]})
    best = tune(x, truth, s1s, taus=[0.1, 0.5], margins=[0.0], owners=(True,)).row(0, named=True)
    assert best["tau"] == 0.5 and best["macro_f05"] == 1.0  # tau 0.5 drops b's wrong match
    sm = summary(f05_table(select(x, 0.5), truth, s1s), by="country")
    assert sm.filter(pl.col("group") == "overall")["macro_f05"].item() == 1.0
    assert set(sm["group"]) == {"overall", "country=India", "country=US", "kind=non_singleton", "kind=singleton"}
