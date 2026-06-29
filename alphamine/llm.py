"""LLM clients that PROPOSE alpha expressions.

Interface: .propose(prompt) -> list[{"expr": str, "rationale": str}]

  - MockClient     : offline. Emits valid expressions from a rotating pool so the whole
                     loop runs with no API key and no internet. Great for testing wiring.
  - AnthropicClient: real Claude API. Requires `pip install anthropic` and ANTHROPIC_API_KEY.
  - AnthropicBedrockClient: Claude via Amazon Bedrock (AWS auth + billing). Same Messages API;
                     `pip install 'anthropic[bedrock]'`, AWS creds + AWS_REGION, `anthropic.`-prefixed ids.
  - OpenAICompatClient: any OpenAI-compatible /chat/completions endpoint. Covers OpenAI's own
                     GPT models (preset "openai") plus open-source stacks — Ollama, vLLM,
                     LM Studio, llama.cpp --server, Together, Groq, OpenRouter — by pointing
                     `base_url` at the right server. Needs `pip install openai`.

Every backend implements the same .propose() so they are drop-in swaps.
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
    def __init__(self, model: str = "claude-opus-4-8", api_key: str | None = None):
        from anthropic import Anthropic  # local import; optional dependency
        self.client = Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self.model = model

    def propose(self, prompt: str, n: int = 5) -> List[Dict[str, str]]:
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=1500,
            system=_PROPOSE_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
        return _extract_json_array(text)[:n]


class AnthropicBedrockClient(AnthropicClient):
    """Claude via Amazon Bedrock — same Messages API, AWS-native auth + billing.

    Auth uses the standard AWS credential chain (env vars / shared profile / IAM role),
    not ANTHROPIC_API_KEY. Region comes from `aws_region` or AWS_REGION.

    Bedrock model IDs carry an `anthropic.` provider prefix (e.g.
    `anthropic.claude-opus-4-8`). If you pass a bare first-party id, we add the prefix.
    """

    def __init__(self, model: str = "anthropic.claude-opus-4-8", aws_region: str | None = None):
        from anthropic import AnthropicBedrock  # local import; needs `anthropic[bedrock]`
        if model.startswith("claude-"):          # accept bare ids, add the Bedrock prefix
            model = "anthropic." + model
        self.client = AnthropicBedrock(aws_region=aws_region or os.environ.get("AWS_REGION"))
        self.model = model


_PROPOSE_SYSTEM = (
    "You are a quant researcher inventing formulaic alpha factors. "
    "Reply ONLY with a JSON array of objects, each {\"expr\": <DSL expression>, "
    "\"rationale\": <one short sentence>}. No prose outside the JSON."
)


# Convenience presets for the common open-source / hosted OpenAI-compatible servers.
# Each maps to (base_url, env var holding the API key — None means no key needed).
_OPENAI_COMPAT_PRESETS = {
    "openai":     ("https://api.openai.com/v1",       "OPENAI_API_KEY"),
    "ollama":     ("http://localhost:11434/v1", None),
    "vllm":       ("http://localhost:8000/v1",  None),
    "lmstudio":   ("http://localhost:1234/v1",  None),
    "llamacpp":   ("http://localhost:8080/v1",  None),
    "together":   ("https://api.together.xyz/v1",     "TOGETHER_API_KEY"),
    "groq":       ("https://api.groq.com/openai/v1",  "GROQ_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1",    "OPENROUTER_API_KEY"),
}


class OpenAICompatClient(LLMClient):
    """Any OpenAI-compatible chat endpoint — local open-source models or hosted gateways.

    Examples:
        # local Llama 3.1 8B via Ollama
        OpenAICompatClient(model="llama3.1", base_url="http://localhost:11434/v1")
        # Qwen2.5 served by vLLM
        OpenAICompatClient(model="Qwen/Qwen2.5-7B-Instruct", base_url="http://localhost:8000/v1")
    """

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        preset: str | None = None,
        temperature: float = 0.8,
    ):
        if not model:
            raise ValueError(
                "OpenAI-compatible providers need an explicit model id, e.g. "
                "LLM_KWARGS={\"model\": \"gpt-4o\"} (openai) or {\"model\": \"llama3.1\"} (ollama)."
            )
        from openai import OpenAI  # local import; optional dependency
        if preset:
            if preset not in _OPENAI_COMPAT_PRESETS:
                raise ValueError(f"unknown preset {preset!r}; choose from {sorted(_OPENAI_COMPAT_PRESETS)}")
            preset_url, key_env = _OPENAI_COMPAT_PRESETS[preset]
            base_url = base_url or preset_url
            if api_key is None and key_env:
                api_key = os.environ.get(key_env)
        # local servers usually ignore the key but the SDK requires a non-empty string
        self.client = OpenAI(base_url=base_url, api_key=api_key or os.environ.get("OPENAI_API_KEY") or "not-needed")
        self.model = model
        self.temperature = temperature

    def propose(self, prompt: str, n: int = 5) -> List[Dict[str, str]]:
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=1500,
            messages=[
                {"role": "system", "content": _PROPOSE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
        )
        text = resp.choices[0].message.content or ""
        return _extract_json_array(text)[:n]


def make_client(provider: str = "mock", **kwargs) -> LLMClient:
    if provider == "mock":
        return MockClient()
    if provider == "anthropic":
        return AnthropicClient(**kwargs)
    if provider in ("bedrock", "anthropic-bedrock"):
        return AnthropicBedrockClient(**kwargs)
    # open-source / hosted OpenAI-compatible backends
    if provider in _OPENAI_COMPAT_PRESETS:
        return OpenAICompatClient(preset=provider, **kwargs)
    if provider in ("openai", "openai-compat", "local"):
        return OpenAICompatClient(**kwargs)
    raise ValueError(f"unknown llm provider: {provider}")
