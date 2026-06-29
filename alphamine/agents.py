"""Agentic layer — ideas borrowed from TradingAgents (TauricResearch, arXiv 2412.20138).

TradingAgents is a multi-agent *decision* framework (analyst -> bull/bear debate ->
trader -> risk manager -> portfolio manager) with a reflection memory that learns from
realized returns. We don't need single-name decisions, but two of its mechanisms make
formulaic alpha mining noticeably better:

  1. ReflectiveMemory  — after each mining round, summarize what worked / what failed and
     inject those lessons into the next generation prompt. (Their "trading_memory.md".)
  2. risk_review       — a critic/risk-manager pass that flags look-ahead and overfitting
     smells on a candidate alpha BEFORE we trust its backtest. (Their risk team / debate.)

Both are deterministic here (no extra LLM calls), so they run offline and are reproducible.
You can later swap `reflect_with_llm` in to get richer, model-written lessons.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List

from . import dsl

# operator/field tokens we recognize when analyzing an expression
_TOKENS = set(dsl.OPERATORS) | set(dsl.FIELDS)


def expr_tokens(expr: str) -> List[str]:
    """Extract the DSL operators/fields used in an expression (for pattern analysis)."""
    names = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", expr)
    return [n for n in names if n in _TOKENS]


# ---------------------------------------------------------------------------
# 1) Reflection memory: learn across rounds, feed lessons forward.
# ---------------------------------------------------------------------------

@dataclass
class ReflectiveMemory:
    lessons: List[str] = field(default_factory=list)

    def update(self, verdicts: List[Dict]):
        """Summarize one round of admit/reject verdicts into a one-line lesson.

        verdicts: list of {"admitted": bool, "reason": str, "expr": str, "metrics": Metrics}
        """
        admitted = [v for v in verdicts if v.get("admitted")]
        rejected = [v for v in verdicts if not v.get("admitted")]

        # which operators show up in winners vs losers
        win_ops = Counter(op for v in admitted for op in expr_tokens(v["expr"]))
        win_ops = {k: c for k, c in win_ops.items() if k in dsl.OPERATORS}
        top_win = ", ".join(k for k, _ in Counter(win_ops).most_common(4)) or "none"

        reasons = Counter(_reason_bucket(v["reason"]) for v in rejected)
        top_reason = ", ".join(f"{r}×{c}" for r, c in reasons.most_common(2)) or "none"

        n = len(self.lessons) + 1
        self.lessons.append(
            f"Round {n}: admitted {len(admitted)}/{len(verdicts)}. "
            f"Productive ops: {top_win}. Top rejection causes: {top_reason}."
        )

    def summary(self, k: int = 3) -> str:
        if not self.lessons:
            return "(no rounds yet)"
        return "\n".join(self.lessons[-k:])


def _reason_bucket(reason: str) -> str:
    if reason.startswith("too_correlated"):
        return "duplicate-of-existing"
    if reason.startswith("below_quality"):
        return "weak-signal"
    if reason.startswith("invalid"):
        return "did-not-parse"
    if reason.startswith("risk:"):
        return "flagged-risky"
    return reason


# ---------------------------------------------------------------------------
# 2) Risk review: a critic that smells out look-ahead / overfitting.
# ---------------------------------------------------------------------------

def risk_review(expr: str, metrics) -> List[str]:
    """Return a list of risk warnings for a candidate alpha. Empty list = clean.

    Heuristics mirror the concerns a risk manager (and the 'Profit Mirage' leakage paper)
    would raise. These are cheap structural checks, not a substitute for real OOS testing.
    """
    warns: List[str] = []

    # implausibly strong predictive power on daily equity -> suspect leakage/overfit
    if abs(metrics.rank_ic) > 0.20:
        warns.append("rank_ic>0.20 is implausibly high for daily equities — check look-ahead")

    # great Sharpe driven by churn that real costs would kill
    if abs(metrics.sharpe) > 4 and metrics.turnover > 1.0:
        warns.append("high Sharpe with high turnover — likely cost-fragile / overfit")

    # very long windows -> few independent observations behind the signal
    for w in re.findall(r"[,(]\s*(\d{3,})\s*\)", expr):
        if int(w) >= 200:
            warns.append(f"window {w} uses long history — few independent samples, fragile")
            break

    # tiny sample actually scored
    if getattr(metrics, "n_obs", 0) < 60:
        warns.append("scored on <60 days — too short to trust")

    return warns


def annotate_risk(verdict: Dict, expr: str) -> Dict:
    """Attach risk warnings to a verdict dict (in place) and return it."""
    warns = risk_review(expr, verdict["metrics"])
    verdict["risk"] = warns
    return verdict


# ---------------------------------------------------------------------------
# Optional: richer, LLM-written reflection (drop-in upgrade over the deterministic one).
# ---------------------------------------------------------------------------

def reflect_with_llm(client, verdicts: List[Dict]) -> str:
    """Ask an LLM to write a short 'lessons learned' note from a round's results.

    `client` is any object with .propose(prompt) -> list (we just want its text); for the
    MockClient this returns canned text. Kept optional so the core loop stays offline.
    """
    lines = [
        f"- {v['expr']}  -> {'ADMIT' if v['admitted'] else 'reject'} "
        f"(ic={v['metrics'].rank_ic}, sharpe={v['metrics'].sharpe}, reason={v['reason']})"
        for v in verdicts
    ]
    prompt = (
        "You are a quant risk reviewer. From these alpha results, write 2-3 concise, "
        "actionable lessons for the next round (what structures to favor/avoid, leakage "
        "risks). Be specific.\n\n" + "\n".join(lines)
    )
    try:
        out = client.propose(prompt, n=1)
        return out[0].get("rationale") or out[0].get("expr") or "(no lesson)"
    except Exception:
        return "(llm reflection unavailable)"
