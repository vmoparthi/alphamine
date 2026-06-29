"""Alpha library: persistent store + diversity (de-duplication) gate.

An alpha is admitted only if:
  - it passes a minimum quality bar (rank_ic / sharpe), AND
  - its signal is not too correlated with any alpha already stored.

The correlation gate is what stops the LLM from "discovering" 30 reskins of momentum.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from .alpha import Alpha
from .evaluate import Metrics, signal_correlation


@dataclass
class Entry:
    alpha: Alpha
    metrics: Metrics
    signal: pd.DataFrame = field(repr=False, default=None)  # in-memory only


class AlphaLibrary:
    def __init__(self, max_corr: float = 0.7,
                 min_rank_ic: float = 0.01, min_sharpe: float = 0.3):
        self.entries: List[Entry] = []
        self.max_corr = max_corr
        self.min_rank_ic = min_rank_ic
        self.min_sharpe = min_sharpe
        self.trials = 0  # total alphas evaluated (for multiple-testing accounting)

    def quality_ok(self, m: Metrics) -> bool:
        return abs(m.rank_ic) >= self.min_rank_ic and abs(m.sharpe) >= self.min_sharpe

    def is_novel(self, signal: pd.DataFrame) -> Optional[float]:
        """Return None if novel, else the max correlation to an existing alpha."""
        worst = 0.0
        for e in self.entries:
            if e.signal is None:
                continue
            c = abs(signal_correlation(signal, e.signal))
            worst = max(worst, c)
            if c >= self.max_corr:
                return c
        return None if worst < self.max_corr else worst

    def consider(self, alpha: Alpha, metrics: Metrics, signal: pd.DataFrame) -> Dict:
        """Try to admit an alpha. Returns a verdict dict for logging/feedback."""
        self.trials += 1
        if not self.quality_ok(metrics):
            return {"admitted": False, "reason": "below_quality_bar", "metrics": metrics}
        dup = self.is_novel(signal)
        if dup is not None:
            return {"admitted": False, "reason": f"too_correlated({dup:.2f})", "metrics": metrics}
        self.entries.append(Entry(alpha=alpha, metrics=metrics, signal=signal))
        return {"admitted": True, "reason": "admitted", "metrics": metrics}

    def top(self, k: int = 10, by: str = "rank_ic") -> List[Entry]:
        return sorted(self.entries, key=lambda e: abs(getattr(e.metrics, by)), reverse=True)[:k]

    def save(self, path: str):
        data = [
            {
                "expr": e.alpha.expr,
                "rationale": e.alpha.rationale,
                "metrics": e.metrics.as_dict(),
            }
            for e in self.entries
        ]
        with open(path, "w") as f:
            json.dump({"trials": self.trials, "alphas": data}, f, indent=2)

    def __len__(self):
        return len(self.entries)
