"""Seed bank — classic formulaic alphas, so you never start from a blank page.

You do NOT need to invent any alpha yourself. Use these in two ways:

  1. WARM-START the library: evaluate them, keep the ones that pass the quality/novelty
     gates, and let the miner mutate from there.
  2. FEW-SHOT the LLM: pass a sample into the prompt as examples to recombine.

These are well-known public ideas (WorldQuant "101 Formulaic Alphas", Kakushadze 2015,
and standard momentum/reversal/vol factors), rewritten into AlphaMine's DSL. They use
only the operators in dsl.py, so they parse out of the box.
"""
from __future__ import annotations

from typing import List

from .alpha import Alpha, AlphaError, validate

# (expression, human rationale)
SEED_ALPHAS = [
    # --- reversal family ---
    ("rank(-1 * delta(close, 1))", "1-day price reversal"),
    ("rank(-1 * returns)", "1-day return reversal"),
    ("-1 * rank((close - ts_min(low, 9)) / (ts_max(high, 9) - ts_min(low, 9)))",
     "stochastic-%K style reversal"),
    ("rank(-1 * (close - ts_mean(close, 10)))", "mean-reversion to 10d MA"),

    # --- momentum / trend family ---
    ("rank(close / delay(close, 20) - 1)", "20-day momentum"),
    ("zscore(ts_mean(returns, 5) - ts_mean(returns, 20))", "fast-minus-slow momentum"),
    ("rank(decay_linear(returns, 10))", "weighted trend over 10d"),

    # --- price-volume family ---
    ("-1 * rank(corr(rank(close), rank(volume), 10))", "price-volume divergence (WQ-style)"),
    ("rank(corr(open, volume, 10))", "open-volume co-movement"),
    ("rank(-1 * delta(log(volume), 2))", "fading volume bursts"),

    # --- volatility family ---
    ("-1 * rank(ts_std(returns, 20))", "low-volatility preference"),
    ("rank(ts_std(returns, 5) / ts_std(returns, 20))", "short-vs-long vol ratio"),

    # --- range / location family ---
    ("rank((ts_max(high, 10) - close) / (ts_max(high, 10) - ts_min(low, 10)))",
     "distance below recent range high"),
    ("rank((close - open) / (high - low))", "intraday close location in range"),
]


def load_seeds(panel) -> List[Alpha]:
    """Return the seed alphas that successfully parse/evaluate on the given panel."""
    out = []
    for expr, why in SEED_ALPHAS:
        try:
            a = validate(expr, panel)
            a.rationale = why
            out.append(a)
        except AlphaError:
            # skip any seed that doesn't fit the current fields (e.g. options panel)
            continue
    return out


def warm_start(library, panel, cost_bps: float = 5.0, verbose: bool = True,
               alphas=None) -> int:
    """Evaluate seed alphas and admit the ones that pass the library's gates.

    Returns the number admitted. Run this BEFORE mine() to give the LLM a base to build on.
    Pass `alphas=` to seed from a custom list (e.g. alpha101.load_alpha101(panel));
    defaults to the small curated bank in this module.
    """
    from .evaluate import evaluate

    if alphas is None:
        alphas = load_seeds(panel)
    admitted = 0
    for alpha in alphas:
        try:
            metrics = evaluate(alpha, panel, cost_bps=cost_bps)
            verdict = library.consider(alpha, metrics, alpha.evaluate(panel))
        except Exception:
            continue
        if verdict["admitted"]:
            admitted += 1
        if verbose:
            tag = "ADMIT" if verdict["admitted"] else "rej  "
            label = alpha.rationale or alpha.expr
            print(f"  [{tag}] {label[:46]:46s} ic={metrics.rank_ic:+.3f} "
                  f"sharpe={metrics.sharpe:+.2f} ({verdict['reason']})")
    return admitted


def sample_for_prompt(k: int = 4) -> str:
    """A few seed expressions to drop into an LLM prompt as few-shot examples."""
    lines = [f'  {{"expr": "{e}", "rationale": "{r}"}}' for e, r in SEED_ALPHAS[:k]]
    return "\n".join(lines)
