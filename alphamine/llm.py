"""LLM clients that PROPOSE alpha expressions.

Interface: .propose(prompt) -> list[{"expr": str, "rationale": str}]

  - MockClient     : offline. Emits valid expressions from a rotating pool so the whole
                     loop runs with no API key and no internet. Great for testing wiring.
  - AnthropicClient: real Claude API. Requires `pip install anthropic` and ANTHROPIC_API_KEY.

Swapping to a local model (Ollama/vLLM) is just another subclass with the same .propose().
"""
from __future__ import annotations

import json
import os
import re
from typing import Dict, List


class LLMClient:
    def propose(self, prompt: str, n: int = 5) -> List[Dict[str, str]]:
        raise NotImplementedError


def _extract_json_array(text: str) -> List[Dict[str, str]]:
    """Pull the first JSON array of {expr, rationale} objects out of a model reply."""
    # try fenced block first
    m = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    blob = m.group(1) if m else None
    if blob is None:
        m = re.search(r"(\[.*\])", text, re.DOTALL)
        blob = m.group(1) if m else None
    if blob is None:
        return []
    try:
        data = json.loads(blob)
    except json.JSONDecodeError:
        return []
    out = []
    for item in data:
        if isinstance(item, dict) and "expr" in item:
            out.append({"expr": str(item["expr"]), "rationale": str(item.get("rationale", ""))})
    return out


class MockClient(LLMClient):
    """Deterministic-ish offline generator. Cycles through a pool, lightly mutating."""

    POOL = [
        ("rank(-1 * delta(close, 1))", "short-term reversal: fade yesterday's move"),
        ("rank(-1 * returns)", "1-day reversal on raw returns"),
        ("rank(corr(close, volume, 10)) * -1", "price-volume divergence"),
        ("zscore(ts_mean(returns, 5) - ts_mean(returns, 20))", "fast-vs-slow momentum"),
        ("scale(rank(ts_std(returns, 10)))", "low-vol preference"),
        ("rank(-1 * ts_rank(close, 5))", "reversal via short-window ts-rank"),
        ("rank(delta(volume, 1) * sign(-1 * returns))", "volume spike on down days"),
        ("-1 * rank(decay_linear(returns, 5))", "decayed reversal"),
        ("rank(ts_max(high, 10) / close - 1)", "distance below recent high"),
        ("zscore(corr(returns, delay(returns, 1), 15))", "autocorrelation of returns"),
        ("rank(-1 * (close - ts_mean(close, 10)))", "mean-reversion vs 10d MA"),
        ("scale(-1 * delta(rank(volume), 3))", "fading volume-rank changes"),
    ]

    def __init__(self):
        self._i = 0

    def propose(self, prompt: str, n: int = 5) -> List[Dict[str, str]]:
        out = []
        for _ in range(n):
            expr, why = self.POOL[self._i % len(self.POOL)]
            self._i += 1
            out.append({"expr": expr, "rationale": why})
        return out


class AnthropicClient(LLMClient):
    def __init__(self, model: str = "claude-sonnet-4-6", api_key: str | None = None):
        from anthropic import Anthropic  # local import; optional dependency
        self.client = Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self.model = model

    def propose(self, prompt: str, n: int = 5) -> List[Dict[str, str]]:
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=1500,
            system=(
                "You are a quant researcher inventing formulaic alpha factors. "
                "Reply ONLY with a JSON array of objects, each {\"expr\": <DSL expression>, "
                "\"rationale\": <one short sentence>}. No prose outside the JSON."
            ),
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
        return _extract_json_array(text)[:n]


def make_client(provider: str = "mock", **kwargs) -> LLMClient:
    if provider == "mock":
        return MockClient()
    if provider == "anthropic":
        return AnthropicClient(**kwargs)
    raise ValueError(f"unknown llm provider: {provider}")
