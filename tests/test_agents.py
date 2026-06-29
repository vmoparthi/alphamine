"""Tests for the agentic layer (reflection memory + risk review)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alphamine.agents import ReflectiveMemory, risk_review, expr_tokens
from alphamine.evaluate import Metrics


def _m(rank_ic=0.05, sharpe=1.0, turnover=0.3, n_obs=200):
    return Metrics(rank_ic=rank_ic, icir=0.3, sharpe=sharpe, ann_return=0.1,
                   max_drawdown=-0.1, turnover=turnover, n_obs=n_obs)


def test_expr_tokens_extracts_operators():
    toks = expr_tokens("rank(-1 * delta(close, 1))")
    assert "rank" in toks and "delta" in toks and "close" in toks


def test_risk_flags_implausible_ic():
    assert any("look-ahead" in w for w in risk_review("rank(close)", _m(rank_ic=0.35)))


def test_risk_flags_cost_fragile():
    warns = risk_review("rank(close)", _m(sharpe=6.0, turnover=1.5))
    assert any("turnover" in w for w in warns)


def test_risk_flags_short_sample():
    assert any("60 days" in w for w in risk_review("rank(close)", _m(n_obs=30)))


def test_clean_alpha_has_no_warnings():
    assert risk_review("rank(-1 * delta(close, 1))", _m()) == []


def test_reflection_memory_summarizes():
    mem = ReflectiveMemory()
    verdicts = [
        {"admitted": True, "reason": "admitted", "expr": "rank(delta(close,1))", "metrics": _m()},
        {"admitted": False, "reason": "too_correlated(0.9)", "expr": "rank(returns)", "metrics": _m()},
    ]
    mem.update(verdicts)
    s = mem.summary()
    assert "Round 1" in s and "admitted 1/2" in s


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("all agent tests passed")
