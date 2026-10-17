"""
evaluate_duo's closing summary must cope with a dataset that has no complete
30 s chain (chained_drift returns {}), instead of crashing on a None format
after the report was already computed.
"""
from __future__ import annotations

from src.evaluate_duo import print_summary


def test_summary_with_empty_chain(capsys):
    report = {"comma2k19_test_only": {
        "window": {"duo": {"n": 3, "mean": 12.0, "median": 11.0, "pass_pct": 33.3}},
        "chained_30s": {"duo": {}},
    }}
    print_summary(report)
    out = capsys.readouterr().out
    assert "median   11.0%" in out
    assert "no complete chains" in out
