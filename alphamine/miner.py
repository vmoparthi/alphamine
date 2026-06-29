"""The mining loop: prompt -> generate -> validate -> evaluate -> admit -> feedback.

Over rounds, the LLM acts as a guided mutation/crossover operator: each prompt includes
the best alphas found so far and the most recent failures (with reasons), so the model
proposes increasingly novel, higher-quality expressions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from . import dsl
from .alpha import Alpha, AlphaError, validate
from .evaluate import evaluate
from .library import AlphaLibrary
from .llm import LLMClient


@dataclass
class MinerConfig:
    rounds: int = 4
    per_round: int = 6
    cost_bps: float = 5.0
    max_corr: float = 0.7
    min_rank_ic: float = 0.01
    min_sharpe: float = 0.3


def build_prompt(library: AlphaLibrary, recent_rejects: List[dict]) -> str:
    top = library.top(k=5)
    best_block = "\n".join(
        f"  {e.alpha.expr}   (rank_ic={e.metrics.rank_ic}, sharpe={e.metrics.sharpe})"
        for e in top
    ) or "  (none yet)"
    rej_block = "\n".join(
        f"  {r['expr']}  -> {r['reason']}" for r in recent_rejects[-6:]
    ) or "  (none yet)"

    return f"""Invent NEW formulaic alpha factors for US equity daily data.

DSL you must use:
{dsl.DSL_SPEC}

Best alphas found so far (try to beat these, and be DIFFERENT from them):
{best_block}

Recently rejected (avoid these / fix the issue):
{rej_block}

Rules:
- Produce economically-motivated signals (momentum, reversal, volume, volatility, price-volume).
- Be diverse: do NOT just rescale the alphas above.
- Use only the listed fields and operators. Integer windows 2..60.
Return a JSON array of objects: {{"expr": "...", "rationale": "..."}}."""


def mine(library: AlphaLibrary, client: LLMClient, train_panel,
         cfg: MinerConfig, verbose: bool = True):
    recent_rejects: List[dict] = []

    for rnd in range(1, cfg.rounds + 1):
        prompt = build_prompt(library, recent_rejects)
        proposals = client.propose(prompt, n=cfg.per_round)
        if verbose:
            print(f"\n=== Round {rnd}: {len(proposals)} proposals ===")

        for p in proposals:
            expr = p["expr"].strip()
            try:
                alpha = validate(expr, train_panel)
                alpha.rationale = p.get("rationale", "")
            except AlphaError as e:
                recent_rejects.append({"expr": expr, "reason": f"invalid: {e}"})
                if verbose:
                    print(f"  [skip ] {expr}  ({e})")
                continue

            metrics = evaluate(alpha, train_panel, cost_bps=cfg.cost_bps)
            verdict = library.consider(alpha, metrics, alpha.evaluate(train_panel))

            if not verdict["admitted"]:
                recent_rejects.append({"expr": expr, "reason": verdict["reason"]})
            if verbose:
                tag = "ADMIT" if verdict["admitted"] else "rej  "
                print(f"  [{tag}] {expr:48s} ic={metrics.rank_ic:+.3f} "
                      f"sharpe={metrics.sharpe:+.2f} ({verdict['reason']})")

    return library


def evaluate_on_test(library: AlphaLibrary, test_panel, cost_bps: float = 5.0):
    """Re-score the final library on the held-out test window. The only honest number."""
    results = []
    for e in library.entries:
        try:
            m = evaluate(e.alpha, test_panel, cost_bps=cost_bps)
            results.append((e.alpha, m))
        except AlphaError:
            continue
    results.sort(key=lambda x: abs(x[1].rank_ic), reverse=True)
    return results
