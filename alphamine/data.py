"""Data layer: load daily OHLCV into aligned panels (index=dates, columns=tickers).

Two sources:
  - "synthetic": deterministic fake market, so the whole system runs offline.
  - "yfinance" : real US-equity daily bars (requires `pip install yfinance` + internet).

A `Panel` is just a dict of DataFrames, one per field (open/high/low/close/volume),
all sharing the same DatetimeIndex and the same set of ticker columns.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

FIELDS = ["open", "high", "low", "close", "volume", "returns", "vwap"]


@dataclass
class Panel:
    fields: Dict[str, pd.DataFrame]      # field name -> (dates x tickers)
    tickers: List[str]

    @property
    def index(self) -> pd.DatetimeIndex:
        return self.fields["close"].index

    def slice(self, start=None, end=None) -> "Panel":
        sl = {k: v.loc[start:end] for k, v in self.fields.items()}
        return Panel(fields=sl, tickers=self.tickers)

    def split(self, train_frac=0.6, valid_frac=0.2):
        """Chronological split into (train, valid, test) panels. No shuffling."""
        n = len(self.index)
        i1 = int(n * train_frac)
        i2 = int(n * (train_frac + valid_frac))
        idx = self.index
        return (
            self.slice(idx[0], idx[i1 - 1]),
            self.slice(idx[i1], idx[i2 - 1]),
            self.slice(idx[i2], idx[-1]),
        )


def _finalize(raw: Dict[str, pd.DataFrame], tickers: List[str]) -> Panel:
    close = raw["close"]
    raw["returns"] = close.pct_change()
    if "vwap" not in raw:
        raw["vwap"] = (raw["high"] + raw["low"] + raw["close"]) / 3.0
    return Panel(fields=raw, tickers=tickers)


def load_synthetic(n_days=750, n_tickers=40, seed=7) -> Panel:
    """Deterministic synthetic market with mild, *recoverable* structure.

    We inject two faint real effects so a good alpha can actually find signal:
      - short-term reversal (yesterday's losers tend to bounce)
      - a volume-confirmation effect
    Returns are otherwise mostly noise, like real life.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-01", periods=n_days)
    tickers = [f"SYN{i:02d}" for i in range(n_tickers)]

    # latent daily returns
    ret = rng.normal(0, 0.015, size=(n_days, n_tickers))
    vol = rng.lognormal(mean=14, sigma=0.5, size=(n_days, n_tickers))

    for t in range(2, n_days):
        prev = ret[t - 1]
        # reversal: push next return against yesterday's move
        ret[t] += -0.06 * prev
        # volume confirmation: high-volume up-days persist a touch
        vnorm = (vol[t - 1] - vol[t - 1].mean()) / (vol[t - 1].std() + 1e-9)
        ret[t] += 0.004 * np.sign(prev) * vnorm

    ret_df = pd.DataFrame(ret, index=dates, columns=tickers)
    price = 100 * (1 + ret_df).cumprod()
    high = price * (1 + np.abs(rng.normal(0, 0.004, price.shape)))
    low = price * (1 - np.abs(rng.normal(0, 0.004, price.shape)))
    open_ = price.shift(1).fillna(price.iloc[0])
    vol_df = pd.DataFrame(vol, index=dates, columns=tickers)

    raw = {"open": open_, "high": high, "low": low, "close": price, "volume": vol_df}
    return _finalize(raw, tickers)


def load_yfinance(tickers: List[str], start="2018-01-01", end=None,
                  min_obs: int = 60) -> Panel:
    """Real daily OHLCV via yfinance. Requires the optional dependency + internet.

    Cleans up the common real-world messes: bad/delisted symbols (all-NaN columns)
    are dropped, tickers with fewer than `min_obs` real bars are dropped, columns are
    returned in the requested order, and an empty download raises a clear error
    (usually a typo'd symbol or a rate-limit, not a code bug).
    """
    if isinstance(tickers, str):
        tickers = [tickers]
    try:
        import yfinance as yf  # local import so the package imports without it
    except ImportError as e:
        raise ImportError(
            "yfinance is required for DATA_SOURCE='yfinance'. "
            "Install it with: pip install 'alphamine[data]'  (or: pip install yfinance)"
        ) from e

    data = yf.download(tickers, start=start, end=end, auto_adjust=True,
                       progress=False, group_by="column")
    if data is None or len(data) == 0:
        raise RuntimeError(
            f"yfinance returned no data for {tickers} ({start}..{end}). "
            "Check the symbols, the date range, and your connection."
        )

    _FIELDS = {"open": "Open", "high": "High", "low": "Low",
               "close": "Close", "volume": "Volume"}

    def _field(name: str) -> pd.DataFrame:
        col = data[name]
        # single ticker comes back as a 1-D Series; normalize to a 1-col frame
        if isinstance(col, pd.Series):
            col = col.to_frame(name=tickers[0])
        return col

    raw = {k: _field(v) for k, v in _FIELDS.items()}

    # drop symbols that never returned usable closes, or have too little history
    close = raw["close"]
    keep = [t for t in close.columns
            if close[t].notna().sum() >= min_obs]
    dropped = [t for t in tickers if t not in keep]
    if dropped:
        print(f"[yfinance] dropped {len(dropped)} ticker(s) with insufficient data: {dropped}")
    if not keep:
        raise RuntimeError(
            f"No ticker had >= {min_obs} bars in {start}..{end}; nothing to mine."
        )
    # preserve the caller's requested order, then forward-fill gaps
    raw = {k: v[keep].dropna(how="all").ffill() for k, v in raw.items()}
    return _finalize(raw, keep)


def load(source="synthetic", **kwargs) -> Panel:
    if source == "synthetic":
        return load_synthetic(**kwargs)
    if source == "yfinance":
        return load_yfinance(**kwargs)
    raise ValueError(f"unknown data source: {source}")
