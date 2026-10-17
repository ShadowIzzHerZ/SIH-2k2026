"""
eval.example_checks (the PS's worked examples) are not scored on their own by
src/evaluate.py; its docstring says they are covered by drift_target_pct.
That only holds while every example is at least as loose as the 10% ratio,
so check it here for every config that carries them.
"""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_example_checks_covered_by_drift_target():
    configs = sorted((ROOT / "configs").glob("*.yaml"))
    checked = 0
    for path in configs:
        ev = (yaml.safe_load(path.read_text()) or {}).get("eval") or {}
        for ex in ev.get("example_checks") or []:
            ratio_pct = 100.0 * ex["max_drift_m"] / ex["distance_m"]
            assert ratio_pct >= ev["drift_target_pct"], (path.name, ex)
            checked += 1
    assert checked, "no example_checks found in configs/"


def test_evaluate_does_not_promise_unscored_checks():
    import src.evaluate as ev

    doc = ev.__doc__
    assert "not scored" in doc and "example checks:" not in doc
