"""Evaluation: turn a signal panel into performance metrics.

Two layers:
  - predictive power:  IC, Rank-IC, ICIR (does the signal forecast next-day returns?)
  - economic value:    long-short decile backtest with transaction costs -> Sharpe etc.

Point-in-time discipline: the signal at day t is used to predict the t -> t+1 return.
We lag the signal by one day before trading so we never use t's close to trade t's move.
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass
class Metrics:
    rank_ic: float          # mean cross-sectional Spearman IC
    icir: float             # mean IC / std IC  (stability)
    sharpe: float           # annualized Sharpe of the long-short book
    ann_return: float       # annualized return
    max_drawdown: float
    turnover: float         # avg daily fraction of book traded
    n_obs: int              # number of days scored
    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


def _forward_returns(panel) -> pd.DataFrame:
    """t -> t+1 simple return of close."""
    close = panel.fields["close"]
    return close.shift(-1) / close - 1.0


def information_coefficient(signal: pd.DataFrame, fwd: pd.DataFrame) -> pd.Series:
    """Daily cross-sectional Spearman correlation between signal and forward return."""
    s_rank = signal.rank(axis=1)
    r_rank = fwd.rank(axis=1)
    # row-wise correlation
    s = s_rank.sub(s_rank.mean(axis=1), axis=0)
    r = r_rank.sub(r_rank.mean(axis=1), axis=0)
    num = (s * r).sum(axis=1)
    den = np.sqrt((s ** 2).sum(axis=1) * (r ** 2).sum(axis=1))
    ic = num / den.replace(0, np.nan)
    return ic.dropna()


def long_short_backtest(signal: pd.DataFrame, fwd: pd.DataFrame,
                        quantile=0.2, cost_bps=5.0) -> Dict[str, object]:
    """Dollar-neutral long-short book.

    Each day: long the top `quantile`, short the bottom `quantile`, equal-weighted,
    gross exposure = 1 per side. Subtract `cost_bps` per side on the traded fraction.
    """
    sig = signal.shift(0)  # signal known at close of day t
    weights = pd.DataFrame(0.0, index=sig.index, columns=sig.columns)

    for dt, row in sig.iterrows():
        r = row.dropna()
        if len(r) < 5:
            continue
        n_side = max(1, int(len(r) * quantile))
        order = r.sort_values()
        shorts = order.index[:n_side]
        longs = order.index[-n_side:]
        weights.loc[dt, longs] = 1.0 / n_side
        weights.loc[dt, shorts] = -1.0 / n_side

    # trade the t->t+1 return with weights set at t
    gross = (weights * fwd).sum(axis=1)
    turnover = weights.diff().abs().sum(axis=1).fillna(0.0)
    cost = turnover * (cost_bps / 1e4)
    net = (gross - cost).dropna()

    return {"pnl": net, "turnover": turnover.reindex(net.index).fillna(0.0)}


def _sharpe(pnl: pd.Series) -> float:
    if pnl.std() == 0 or len(pnl) < 2:
        return 0.0
    return float(np.sqrt(TRADING_DAYS) * pnl.mean() / pnl.std())


def _max_drawdown(pnl: pd.Series) -> float:
    curve = (1 + pnl).cumprod()
    peak = curve.cummax()
    dd = curve / peak - 1.0
    return float(dd.min())


def evaluate(alpha, panel, cost_bps=5.0) -> Metrics:
    """Full evaluation of an Alpha on a panel."""
    signal = alpha.evaluate(panel)
    fwd = _forward_returns(panel)
    # align
    signal, fwd = signal.align(fwd, join="inner")

    ic = information_coefficient(signal, fwd)
    bt = long_short_backtest(signal, fwd, cost_bps=cost_bps)
    pnl = bt["pnl"]

    rank_ic = float(ic.mean()) if len(ic) else 0.0
    icir = float(ic.mean() / ic.std()) if len(ic) > 1 and ic.std() else 0.0
    ann_return = float(pnl.mean() * TRADING_DAYS) if len(pnl) else 0.0

    return Metrics(
        rank_ic=round(rank_ic, 4),
        icir=round(icir, 3),
        sharpe=round(_sharpe(pnl), 3),
        ann_return=round(ann_return, 4),
        max_drawdown=round(_max_drawdown(pnl), 4) if len(pnl) else 0.0,
        turnover=round(float(bt["turnover"].mean()), 3) if len(pnl) else 0.0,
        n_obs=int(len(pnl)),
    )


# ---------------------------------------------------------------------------
# Parallel evaluation. Scoring N independent alphas on one panel is
# embarrassingly parallel — the only heavy object is the panel, so we ship it
# to each worker ONCE via the pool initializer and map cheap Alpha (expr string)
# objects across the workers. This is the main lever for large-universe sweeps.
# ---------------------------------------------------------------------------

_WORKER_PANEL = None       # set once per worker process via _init_worker
_WORKER_COST = 5.0


def _init_worker(panel, cost_bps):
    global _WORKER_PANEL, _WORKER_COST
    _WORKER_PANEL = panel
    _WORKER_COST = cost_bps


def _eval_one(alpha) -> Tuple[object, Optional["Metrics"]]:
    try:
        return alpha, evaluate(alpha, _WORKER_PANEL, cost_bps=_WORKER_COST)
    except Exception:
        return alpha, None  # bad expr / scalar result -> caller skips it


def evaluate_many(alphas, panel, cost_bps: float = 5.0,
                  n_jobs: Optional[int] = None) -> List[Tuple[object, Optional["Metrics"]]]:
    """Evaluate many alphas on one panel, in parallel across processes.

    Returns ``[(alpha, metrics_or_None), ...]`` in the same order as ``alphas``
    (None for any alpha that failed to evaluate). ``n_jobs`` defaults to all CPU
    cores; ``n_jobs=1`` forces the sequential path. Small batches stay sequential
    to avoid process-pool startup cost dominating.
    """
    alphas = list(alphas)
    if n_jobs is None:
        n_jobs = os.cpu_count() or 1

    # Only fan out when the work justifies the process-pool + panel-pickling
    # overhead. Tiny universes (the offline demo) stay sequential automatically;
    # large-universe sweeps parallelize. Roughly: cells-per-eval * #alphas.
    cells = len(panel.index) * max(1, len(panel.tickers))
    work = len(alphas) * cells
    if n_jobs <= 1 or len(alphas) < 4 or work < 5_000_000:
        out = []
        for a in alphas:
            try:
                out.append((a, evaluate(a, panel, cost_bps=cost_bps)))
            except Exception:
                out.append((a, None))
        return out

    chunk = max(1, len(alphas) // (n_jobs * 4))
    with ProcessPoolExecutor(max_workers=n_jobs, initializer=_init_worker,
                             initargs=(panel, cost_bps)) as ex:
        return list(ex.map(_eval_one, alphas, chunksize=chunk))


def signal_correlation(a_sig: pd.DataFrame, b_sig: pd.DataFrame) -> float:
    """Average daily cross-sectional rank correlation between two signals.

    Used by the library to reject near-duplicate alphas (diversity gate).
    """
    a, b = a_sig.align(b_sig, join="inner")
    ar, br = a.rank(axis=1), b.rank(axis=1)
    a_c = ar.sub(ar.mean(axis=1), axis=0)
    b_c = br.sub(br.mean(axis=1), axis=0)
    num = (a_c * b_c).sum(axis=1)
    den = np.sqrt((a_c ** 2).sum(axis=1) * (b_c ** 2).sum(axis=1))
    corr = (num / den.replace(0, np.nan)).dropna()
    return float(corr.mean()) if len(corr) else 0.0


def deflated_sharpe(sharpe: float, n_trials: int, n_obs: int) -> float:
    """Rough Deflated-Sharpe haircut for multiple testing (Bailey & Lopez de Prado).

    Returns an approximate probability that the TRUE Sharpe is > 0 after accounting
    for `n_trials` independent strategies tried. Use it as a sanity gate, not gospel.
    """
    from math import log, sqrt, erf
    if n_obs < 2 or n_trials < 1:
        return 0.0
    # expected max of n_trials standard normals (approx)
    emax = sqrt(2 * log(max(n_trials, 2)))
    sr_std = 1.0 / sqrt(n_obs - 1)            # std error of Sharpe (simplified)
    z = (sharpe - emax * sr_std) / sr_std
    # standard normal CDF
    return 0.5 * (1 + erf(z / sqrt(2)))
