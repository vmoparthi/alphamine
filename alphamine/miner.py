"""The mining loop: prompt -> generate -> validate -> evaluate -> risk-review -> admit.

Over rounds, the LLM acts as a guided mutation/crossover operator. Two mechanisms borrowed
from TradingAgents make the search smarter (see agents.py):
  - a reflection memory feeds "lessons" from past rounds into each new prompt, and
  - a risk-review critic can veto alphas that smell of look-ahead / overfitting before they
    ever enter the library.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from . import dsl
from .agents import ReflectiveMemory, risk_review
from .alpha import Alpha, AlphaError, validate
from .evaluate import evaluate, evaluate_many
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
    risk_veto: bool = True   # reject alphas the risk-reviewer flags (look-ahead/overfit)


def build_prompt(library: AlphaLibrary, recent_rejects: List[dict], lessons: str = "") -> str:
    top = library.top(k=5)
    best_block = "\n".join(
        f"  {e.alpha.expr}   (rank_ic={e.metrics.rank_ic}, sharpe={e.metrics.sharpe})"
        for e in top
    ) or "  (none yet)"
    rej_block = "\n".join(
        f"  {r['expr']}  -> {r['reason']}" for r in recent_rejects[-6:]
    ) or "  (none yet)"
    lessons_block = lessons or "  (no rounds yet)"

    return f"""Invent NEW formulaic alpha factors for US equity daily data.

DSL you must use:
{dsl.DSL_SPEC}

Lessons from previous rounds (use them):
{lessons_block}

Best alphas found so far (try to beat these, and be DIFFERENT from them):
{best_block}

Recently rejected (avoid these / fix the issue):
{rej_block}

Rules:
- Produce economically-motivated signals (momentum, reversal, volume, volatility, price-volume).
- Be diverse: do NOT just rescale the alphas above.
- Avoid look-ahead and avoid churn-heavy signals that realistic costs would kill.
- Use only the listed fields and operators. Integer windows 2..60.
Return a JSON array of objects: {{"expr": "...", "rationale": "..."}}."""


def mine(library: AlphaLibrary, client: LLMClient, train_panel,
         cfg: MinerConfig, verbose: bool = True, memory: ReflectiveMemory = None):
    recent_rejects: List[dict] = []
    memory = memory or ReflectiveMemory()

    for rnd in range(1, cfg.rounds + 1):
        prompt = build_prompt(library, recent_rejects, memory.summary())
        proposals = client.propose(prompt, n=cfg.per_round)
        if verbose:
            print(f"\n=== Round {rnd}: {len(proposals)} proposals ===")

        round_verdicts: List[dict] = []
        for p in proposals:
            expr = p["expr"].strip()
            try:
                alpha = validate(expr, train_panel)
                alpha.rationale = p.get("rationale", "")
            except AlphaError as e:
                v = {"expr": expr, "admitted": False, "reason": f"invalid: {e}", "metrics": None}
                round_verdicts.append(v)
                recent_rejects.append({"expr": expr, "reason": f"invalid: {e}"})
                if verbose:
                    print(f"  [skip ] {expr}  ({e})")
                continue

            metrics = evaluate(alpha, train_panel, cost_bps=cfg.cost_bps)

            # risk-review critic BEFORE admission (borrowed from TradingAgents' risk team)
            warns = risk_review(expr, metrics)
            if cfg.risk_veto and warns:
                v = {"expr": expr, "admitted": False, "reason": f"risk: {warns[0]}",
                     "metrics": metrics, "risk": warns}
                round_verdicts.append(v)
                recent_rejects.append({"expr": expr, "reason": f"risk: {warns[0]}"})
                if verbose:
                    print(f"  [risk ] {expr:46s} ic={metrics.rank_ic:+.3f} "
                          f"sharpe={metrics.sharpe:+.2f} ({warns[0][:40]})")
                continue

            verdict = library.consider(alpha, metrics, alpha.evaluate(train_panel))
            verdict["expr"] = expr
            verdict["risk"] = warns
            round_verdicts.append(verdict)

            if not verdict["admitted"]:
                recent_rejects.append({"expr": expr, "reason": verdict["reason"]})
            if verbose:
                tag = "ADMIT" if verdict["admitted"] else "rej  "
                print(f"  [{tag}] {expr:48s} ic={metrics.rank_ic:+.3f} "
                      f"sharpe={metrics.sharpe:+.2f} ({verdict['reason']})")

        memory.update(round_verdicts)  # reflect: lessons feed into next round's prompt

    if verbose:
        print("\n=== Reflection memory ===")
        print(memory.summary(k=cfg.rounds))
    return library


def evaluate_on_test(library: AlphaLibrary, test_panel, cost_bps: float = 5.0,
                     n_jobs: int = None):
    """Re-score the final library on the held-out test window. The only honest number.

    Parallelized across CPU cores (`n_jobs`, default all) — re-scoring is pure
    metrics with no shared state, so it's an unconditional win at scale.
    """
    scored = evaluate_many((e.alpha for e in library.entries), test_panel,
                           cost_bps=cost_bps, n_jobs=n_jobs)
    results = [(a, m) for a, m in scored if m is not None]
    results.sort(key=lambda x: abs(x[1].rank_ic), reverse=True)
    return results
