"""Parallel evaluation: results must match sequential exactly."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alphamine import data
from alphamine.alpha import Alpha
from alphamine.evaluate import evaluate, evaluate_many


def _panel():
    return data.load("synthetic", n_days=400, n_tickers=20)


_EXPRS = [
    "rank(-1 * delta(close, 1))",
    "rank(corr(close, volume, 10)) * -1",
    "zscore(ts_mean(returns, 5) - ts_mean(returns, 20))",
    "scale(rank(ts_std(returns, 10)))",
    "rank(-1 * (close - ts_mean(close, 10)))",
    "__bad__ +",                       # unparseable -> metrics None
    "rank(ts_max(high, 10) / close - 1)",
]


def test_evaluate_many_matches_sequential():
    panel = _panel()
    alphas = [Alpha(e) for e in _EXPRS]
    # force the parallel branch regardless of the work-size heuristic
    par = evaluate_many(alphas, panel, n_jobs=4)
    assert len(par) == len(alphas)
    for (a, m), expr in zip(par, _EXPRS):
        assert a.expr == expr  # order preserved
        if "__bad__" in expr:
            assert m is None
            continue
        ref = evaluate(Alpha(expr), panel)
        assert m is not None
        assert m.rank_ic == ref.rank_ic
        assert m.sharpe == ref.sharpe
        assert m.n_obs == ref.n_obs


def test_n_jobs_one_is_sequential_and_equal():
    panel = _panel()
    alphas = [Alpha(e) for e in _EXPRS if "__bad__" not in e]
    seq = evaluate_many(alphas, panel, n_jobs=1)
    assert all(m is not None for _, m in seq)
    assert [a.expr for a, _ in seq] == [a.expr for a in alphas]


if __name__ == "__main__":
    test_evaluate_many_matches_sequential()
    test_n_jobs_one_is_sequential_and_equal()
    print("all parallel tests passed")
