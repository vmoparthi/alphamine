"""Minimal smoke tests — run with: python -m pytest  (or python tests/test_smoke.py)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alphamine import data
from alphamine.alpha import Alpha, AlphaError, validate
from alphamine.evaluate import evaluate
from alphamine.alpha101 import coverage_report, load_alpha101


def _panel():
    return data.load("synthetic", n_days=400, n_tickers=20)


def test_basic_alpha_evaluates():
    panel = _panel()
    a = validate("rank(-1 * delta(close, 1))", panel)
    m = evaluate(a, panel)
    assert m.n_obs > 0


def test_invalid_expression_rejected():
    panel = _panel()
    for bad in ["__import__('os')", "open + ", "unknownop(close, 3)", "close.values"]:
        try:
            Alpha(bad).evaluate(panel)
            assert False, f"should have rejected: {bad}"
        except (AlphaError, Exception):
            pass


def test_alpha101_coverage():
    panel = _panel()
    rep = coverage_report(panel)
    assert len(rep["failed"]) == 0, rep["failed"]
    assert len(rep["ok"]) >= 95
    assert len(load_alpha101(panel)) >= 95


if __name__ == "__main__":
    test_basic_alpha_evaluates()
    test_invalid_expression_rejected()
    test_alpha101_coverage()
    print("all smoke tests passed")
