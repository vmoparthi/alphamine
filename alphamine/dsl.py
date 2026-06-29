"""The alpha DSL.

Operators act on panels (DataFrame, index=dates, columns=tickers).
  - Cross-sectional ops work per row (across tickers): rank, scale, zscore, sign...
  - Time-series ops work per column (along time): delay, delta, ts_mean, ts_std,
    ts_rank, corr, decay_linear...

Every time-series op uses ONLY past data (no look-ahead). Outputs are aligned to the
same index/columns as the inputs.

The grammar an LLM is allowed to use:
  fields:    open, high, low, close, volume, returns, vwap
  unary:     rank, scale, zscore, sign, abs_, log, delay(x,d), delta(x,d)
  ts:        ts_mean(x,w), ts_std(x,w), ts_sum(x,w), ts_min(x,w), ts_max(x,w),
             ts_rank(x,w), decay_linear(x,w)
  binary:    corr(x,y,w), cov(x,y,w), add, sub, mul, div, pow_
  consts:    plain numbers (e.g. 5, -1, 0.5)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ---- cross-sectional (per row, across tickers) ----

def rank(x: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional rank in [0,1]."""
    return x.rank(axis=1, pct=True)

def scale(x: pd.DataFrame, a: float = 1.0) -> pd.DataFrame:
    """Rescale each row so sum(|x|) == a."""
    denom = x.abs().sum(axis=1).replace(0, np.nan)
    return x.mul(a).div(denom, axis=0)

def zscore(x: pd.DataFrame) -> pd.DataFrame:
    mu = x.mean(axis=1)
    sd = x.std(axis=1).replace(0, np.nan)
    return x.sub(mu, axis=0).div(sd, axis=0)

def sign(x: pd.DataFrame) -> pd.DataFrame:
    return np.sign(x)

def abs_(x: pd.DataFrame) -> pd.DataFrame:
    return x.abs()

def log(x: pd.DataFrame) -> pd.DataFrame:
    return np.log(x.where(x > 0))

# ---- time-series (per column, along time) ----

def delay(x: pd.DataFrame, d: int = 1) -> pd.DataFrame:
    return x.shift(int(d))

def delta(x: pd.DataFrame, d: int = 1) -> pd.DataFrame:
    return x - x.shift(int(d))

def ts_mean(x: pd.DataFrame, w: int = 5) -> pd.DataFrame:
    return x.rolling(int(w), min_periods=int(w)).mean()

def ts_std(x: pd.DataFrame, w: int = 5) -> pd.DataFrame:
    return x.rolling(int(w), min_periods=int(w)).std()

def ts_sum(x: pd.DataFrame, w: int = 5) -> pd.DataFrame:
    return x.rolling(int(w), min_periods=int(w)).sum()

def ts_min(x: pd.DataFrame, w: int = 5) -> pd.DataFrame:
    return x.rolling(int(w), min_periods=int(w)).min()

def ts_max(x: pd.DataFrame, w: int = 5) -> pd.DataFrame:
    return x.rolling(int(w), min_periods=int(w)).max()

def ts_rank(x: pd.DataFrame, w: int = 5) -> pd.DataFrame:
    """Rank of the last value within its trailing window, in [0,1]."""
    return x.rolling(int(w), min_periods=int(w)).apply(
        lambda s: s.argsort().argsort()[-1] / (len(s) - 1) if len(s) > 1 else 0.5,
        raw=True,
    )

def decay_linear(x: pd.DataFrame, w: int = 5) -> pd.DataFrame:
    """Linearly-weighted moving average (more weight on recent)."""
    w = int(w)
    weights = np.arange(1, w + 1, dtype=float)
    weights /= weights.sum()
    return x.rolling(w, min_periods=w).apply(lambda s: np.dot(s, weights), raw=True)

def corr(x: pd.DataFrame, y: pd.DataFrame, w: int = 5) -> pd.DataFrame:
    return x.rolling(int(w), min_periods=int(w)).corr(y)

def cov(x: pd.DataFrame, y: pd.DataFrame, w: int = 5) -> pd.DataFrame:
    return x.rolling(int(w), min_periods=int(w)).cov(y)

# ---- advanced time-series ops (needed for WorldQuant Alpha101) ----

def ts_argmax(x: pd.DataFrame, w: int = 5) -> pd.DataFrame:
    """Index (0..w-1) of the max within the trailing window."""
    return x.rolling(int(w), min_periods=int(w)).apply(
        lambda s: float(np.argmax(s)), raw=True)

def ts_argmin(x: pd.DataFrame, w: int = 5) -> pd.DataFrame:
    """Index (0..w-1) of the min within the trailing window."""
    return x.rolling(int(w), min_periods=int(w)).apply(
        lambda s: float(np.argmin(s)), raw=True)

def ts_product(x: pd.DataFrame, w: int = 5) -> pd.DataFrame:
    return x.rolling(int(w), min_periods=int(w)).apply(np.prod, raw=True)

def signedpower(x, a):
    """sign(x) * |x|**a  (a may be a scalar or a panel)."""
    return np.sign(x) * (np.abs(x) ** a)

def indneutralize(x: pd.DataFrame) -> pd.DataFrame:
    """APPROXIMATION of WorldQuant IndNeutralize.

    The paper neutralizes against an industry/sector grouping. We don't ship sector
    labels, so we neutralize against the whole cross-section (subtract the daily mean).
    Swap in a real grouping by passing sector ids and demeaning within group.
    """
    return x.sub(x.mean(axis=1), axis=0)

# ---- elementwise helpers (Alpha101 uses min/max of two series, and ternaries) ----

def _pair(x, y):
    """Broadcast a (DataFrame|scalar, DataFrame|scalar) pair to two aligned DataFrames."""
    if isinstance(x, pd.DataFrame) and isinstance(y, pd.DataFrame):
        return x.align(y, join="outer")
    if isinstance(x, pd.DataFrame):
        return x, pd.DataFrame(y, index=x.index, columns=x.columns)
    if isinstance(y, pd.DataFrame):
        return pd.DataFrame(x, index=y.index, columns=y.columns), y
    return x, y

def max2(x, y):
    a, b = _pair(x, y)
    return a.where(a > b, b) if isinstance(a, pd.DataFrame) else max(a, b)

def min2(x, y):
    a, b = _pair(x, y)
    return a.where(a < b, b) if isinstance(a, pd.DataFrame) else min(a, b)

def lt(x, y):
    a, b = _pair(x, y)
    return (a < b).astype(float) if isinstance(a, pd.DataFrame) else float(a < b)

def gt(x, y):
    a, b = _pair(x, y)
    return (a > b).astype(float) if isinstance(a, pd.DataFrame) else float(a > b)

def le(x, y):
    a, b = _pair(x, y)
    return (a <= b).astype(float) if isinstance(a, pd.DataFrame) else float(a <= b)

def ge(x, y):
    a, b = _pair(x, y)
    return (a >= b).astype(float) if isinstance(a, pd.DataFrame) else float(a >= b)

def iff(cond, a, b):
    """Ternary: where cond is truthy (>0) use a, else b. cond is usually a 1/0 mask."""
    if isinstance(cond, pd.DataFrame):
        mask = cond > 0
        a_df = a if isinstance(a, pd.DataFrame) else pd.DataFrame(a, index=cond.index, columns=cond.columns)
        b_df = b if isinstance(b, pd.DataFrame) else pd.DataFrame(b, index=cond.index, columns=cond.columns)
        a_df, _ = _pair(a_df, cond)
        b_df, _ = _pair(b_df, cond)
        return a_df.where(mask, b_df)
    return a if cond else b

# ---- binary arithmetic (exposed as named funcs so the AST evaluator is uniform) ----

def add(x, y): return x + y
def sub(x, y): return x - y
def mul(x, y): return x * y
def div(x, y): return x / y.replace(0, np.nan) if hasattr(y, "replace") else x / y
def pow_(x, y): return x ** y

# registry the evaluator is allowed to call
OPERATORS = {
    "rank": rank, "scale": scale, "zscore": zscore, "sign": sign, "abs_": abs_, "log": log,
    "delay": delay, "delta": delta,
    "ts_mean": ts_mean, "ts_std": ts_std, "ts_sum": ts_sum, "ts_min": ts_min,
    "ts_max": ts_max, "ts_rank": ts_rank, "decay_linear": decay_linear,
    "corr": corr, "cov": cov,
    "add": add, "sub": sub, "mul": mul, "div": div, "pow_": pow_,
    # advanced (Alpha101)
    "ts_argmax": ts_argmax, "ts_argmin": ts_argmin, "ts_product": ts_product,
    "signedpower": signedpower, "indneutralize": indneutralize,
    "max2": max2, "min2": min2,
    "lt": lt, "gt": gt, "le": le, "ge": ge, "iff": iff,
}

FIELDS = ["open", "high", "low", "close", "volume", "returns", "vwap"]

# Compact spec string for prompting the LLM.
DSL_SPEC = """\
Fields (each is a dates x tickers matrix): open, high, low, close, volume, returns, vwap
Cross-sectional ops: rank(x), scale(x[,a]), zscore(x), sign(x), abs_(x), log(x)
Time-series ops: delay(x,d), delta(x,d), ts_mean(x,w), ts_std(x,w), ts_sum(x,w),
  ts_min(x,w), ts_max(x,w), ts_rank(x,w), decay_linear(x,w)
Pairwise ops: corr(x,y,w), cov(x,y,w)
Arithmetic: +  -  *  /  and unary minus; numeric constants allowed.
Window args (d,w) must be positive integers, typically 2..60.
Example valid expressions:
  rank(-1 * delta(close, 1))
  rank(corr(close, volume, 10)) * -1
  zscore(ts_mean(returns, 5) - ts_mean(returns, 20))
  scale(rank(ts_std(returns, 10)))
"""
