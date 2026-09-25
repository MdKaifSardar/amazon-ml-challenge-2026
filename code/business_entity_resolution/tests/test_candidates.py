"""Unit tests for candidate selection rules and list statistics (src/candidates.py)."""
import sys
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from candidates import add_best, list_stats, pareto, rule, select  # noqa: E402


def table(rows: list[dict]) -> pl.DataFrame:
    """Blocking table with every rank/score column (null = method did not find the pair)."""
    cols = {"scope": pl.Utf8, "s1": pl.Utf8, "cand": pl.Utf8, "best_reverse": pl.Float32}
    for m in ("name", "name_city", "name_addr", "reverse", "rare"):
        cols[f"rank_{m}"] = pl.Int64
        cols[f"score_{m}"] = pl.Float32
    full = [{c: r.get(c) for c in cols} | {"scope": r.get("scope", "state")} for r in rows]
    return add_best(pl.DataFrame(full, schema=cols))


def picked(df: pl.DataFrame, expr: pl.Expr, cap: int | None = None) -> set[str]:
    return set(select(df, expr, cap)["cand"].to_list())


def test_reverse_margin_keeps_rank1_and_close_rank2():
    df = table([
        {"s1": "a", "cand": "x", "rank_reverse": 1, "score_reverse": 0.9, "best_reverse": 0.9},
        {"s1": "a", "cand": "y", "rank_reverse": 2, "score_reverse": 0.88, "best_reverse": 0.9},  # within 0.05
        {"s1": "a", "cand": "z", "rank_reverse": 2, "score_reverse": 0.70, "best_reverse": 0.9},  # too far
        {"s1": "a", "cand": "w", "rank_reverse": 3, "score_reverse": 0.89, "best_reverse": 0.9},  # rank > m
    ])
    assert picked(df, rule(rev_m=2, rev_margin=0.05)) == {"x", "y"}
    assert picked(df, rule(rev_m=2)) == {"x", "y", "z"}  # no margin: plain top-2
    assert picked(df, rule(rev_m=1)) == {"x"}


def test_reverse_rank2_uses_the_pool_records_own_best():
    # y is rank 2 for its pool record, whose best (0.95, another S1) is far: dropped even though it is a's best
    df = table([{"s1": "a", "cand": "y", "rank_reverse": 2, "score_reverse": 0.80, "best_reverse": 0.95}])
    assert picked(df, rule(rev_m=2, rev_margin=0.1)) == set()


def test_forward_margin_min_and_k():
    df = table([
        {"s1": "a", "cand": "x", "rank_name": 1, "score_name": 0.95},
        {"s1": "a", "cand": "y", "rank_name": 2, "score_name": 0.91},  # within 0.05 of 0.95
        {"s1": "a", "cand": "z", "rank_name": 3, "score_name": 0.80},  # 0.15 below
        {"s1": "b", "cand": "u", "rank_name": 1, "score_name": 0.60},  # b's own best
        {"s1": "b", "cand": "v", "rank_name": 2, "score_name": 0.58},
    ])
    assert picked(df, rule(fwd_k=5, fwd_margin=0.05)) == {"x", "y", "u", "v"}
    assert picked(df, rule(fwd_k=5, fwd_margin=0.01, fwd_min=0.9)) == {"x", "y", "u"}  # y kept by the minimum
    assert picked(df, rule(fwd_k=1, fwd_margin=0.5)) == {"x", "u"}  # k caps the margin


def test_rare_and_empty_rule():
    df = table([{"s1": "a", "cand": "x", "rank_rare": 3, "score_rare": 1.0}])
    assert picked(df, rule(rare_k=3)) == {"x"}
    assert picked(df, rule(rare_k=2)) == set()
    assert picked(df, rule()) == set()


def test_cap_keeps_best_by_cosine_per_s1_and_scope():
    df = table([
        {"s1": "a", "cand": "x", "rank_name": 1, "score_name": 0.9},
        {"s1": "a", "cand": "y", "rank_name": 2, "score_name": 0.8},
        {"s1": "a", "cand": "z", "rank_reverse": 1, "score_reverse": 0.85, "best_reverse": 0.85},
        {"s1": "a", "cand": "x", "rank_name": 1, "score_name": 0.9, "scope": "country"},
    ])
    out = select(df, rule(fwd_k=5, rev_m=1), cap=2)
    assert set(out.filter(pl.col("scope") == "state")["cand"]) == {"x", "z"}
    assert out.filter(pl.col("scope") == "country").height == 1


def test_list_stats_recall_sizes_and_reduction():
    found = pl.DataFrame({"s1": ["a", "a", "b"], "cand": ["x", "y", "z"]})
    truth = pl.DataFrame({"s1": ["a", "b", "b"], "cand": ["x", "z", "w"]})
    s1s = pl.DataFrame({"s1": ["a", "b", "c"], "country": ["US", "US", "India"]})
    st = {r["country"]: r for r in list_stats(found, truth, s1s, {"US": 100, "India": 50}).iter_rows(named=True)}
    assert st["US"]["recall"] == pytest.approx(2 / 3)
    assert st["US"]["avg"] == 1.5 and st["US"]["reduction_ratio"] == pytest.approx(1 - 3 / 200)
    assert st["India"]["avg"] == 0 and st["India"]["empty_share"] == 1.0  # S1 with no candidates count as 0
    assert st["ALL"]["avg"] == 1.0 and st["ALL"]["reduction_ratio"] == pytest.approx(1 - 3 / 250)


def test_list_stats_ignores_s1_outside_the_evaluated_set():
    found = pl.DataFrame({"s1": ["a", "zz"], "cand": ["x", "q"]})
    truth = pl.DataFrame({"s1": ["a", "zz"], "cand": ["x", "q"]})
    s1s = pl.DataFrame({"s1": ["a"], "country": ["US"]})
    st = list_stats(found, truth, s1s, {"US": 10}).filter(pl.col("country") == "ALL").row(0, named=True)
    assert st["recall"] == 1.0 and st["cands"] == 1 and st["true"] == 1


def test_pareto_drops_dominated_points():
    pts = pl.DataFrame({"avg": [1.0, 2.0, 3.0, 4.0, 4.0], "recall": [0.5, 0.4, 0.8, 0.7, 0.9]})
    assert pareto(pts).select("avg", "recall").rows() == [(1.0, 0.5), (3.0, 0.8), (4.0, 0.9)]
