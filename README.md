# AlphaMine — LLM-driven alpha mining (starter)

A small, runnable system that uses an LLM to **generate**, **evaluate**, and **iteratively improve**
formulaic alpha factors on US equities. It runs **offline out of the box** (synthetic data + a mock
"LLM" that emits real alpha expressions), so you can see the full loop before plugging in a real API
key or real data.

---

## 1. Design rationale

**Formulaic alphas on US equity daily OHLCV are the starting point.** Each signal is a short symbolic
expression (e.g. `rank(-1 * corr(close, volume, 5))`):

- Cheap for an LLM to generate, trivial to mutate, and **fully transparent** — no black-box overfitting.
- Evaluation is fast and objective (Information Coefficient + a long-short backtest), enabling a tight
  propose → score → critique → propose loop — the approach used by Chain-of-Alpha / QuantaAlpha.
- It avoids the leakage traps that sink news/agentic approaches (see §5).

**The miner is model-agnostic.** Alpha generation is a *reasoning + code* task run a few hundred times per
run, so generation quality matters more than throughput and token cost is small. A frontier API model gives
the best yield; local open-source models are a drop-in swap for privacy or cost. Every backend sits behind
one `LLMClient` interface, so the provider is a single config line (see §7.1).

**Options support is Phase 2.** Options alpha mining needs a clean historical options chain (IV surface,
greeks) and a more careful backtest (path-dependent P&L, delta hedging, bid/ask). The architecture drives
it with the *same* mining loop — only the data panel and evaluator change (see §6).

---

## 2. System architecture

![AlphaMine architecture: pluggable LLM backends (Mock / Anthropic / OpenAI / Bedrock / local OSS) drive a clockwise mining loop where the LLM client proposes alpha expressions, which are validated and evaluated against a data panel, screened by a risk-review critic, admitted to a correlation-deduplicated alpha library, and distilled into a reflection memory whose lessons feed back into the next prompt; the library is re-scored on a held-out test split to produce ranked alphas.](assets/architecture.svg)

The loop runs clockwise. The **LLM client** (any backend — see §7.1) proposes new alpha expressions; each
is **validated** (safe AST parse, no arbitrary `eval`) and **evaluated** on the data panel (Rank-IC + a
long-short backtest). A **risk-review** critic then vetoes look-ahead-prone or cost-fragile candidates
before they reach the **alpha library**, which admits a survivor only if it also clears the novelty gate
(correlation < 0.7 vs every stored alpha). Each round's outcomes are distilled into a **reflection memory**
whose lessons feed back into the next prompt — so the loop behaves like a guided evolutionary search with
the LLM as the mutation operator (agentic layer in `agents.py`, ideas from TradingAgents).

### 2.1 Components — what each part does

- **`data.py` — market data.** Loads daily OHLCV into a `Panel`: one aligned DataFrame per field
  (open/high/low/close/volume, plus derived returns/vwap), indexed by date × ticker, with a chronological
  `split()` into train/valid/test. Ships a deterministic **synthetic** market (so the whole system runs with
  no internet) and a real **yfinance** loader that cleans up delisted/short-history symbols.
- **`dsl.py` — the alpha language.** Defines the operator vocabulary the LLM may use — cross-sectional
  (`rank`, `scale`, `zscore`), time-series (`delay`, `delta`, `ts_mean`, `ts_std`, `ts_rank`, `corr`,
  `decay_linear`, …), and arithmetic — each implemented as a vectorized function on pandas panels. Also holds
  `DSL_SPEC`, the human-readable operator list injected into the prompt.
- **`alpha.py` — expression → signal.** An `Alpha` wraps an expression string and turns it into a signal
  panel through a **safe AST evaluator**: only whitelisted fields, numeric constants, and DSL operators are
  allowed — never `eval` of arbitrary Python. Invalid or non-panel expressions raise `AlphaError`.
- **`evaluate.py` — scoring.** Turns a signal into performance: IC / Rank-IC / ICIR, and a decile
  **long-short backtest** with transaction costs → Sharpe, annual return, max drawdown, turnover. Includes a
  `deflated_sharpe` helper (multiple-testing haircut) and `evaluate_many`, the process-pool parallel scorer.
- **`library.py` — the alpha store.** `AlphaLibrary` decides what gets kept: an alpha is admitted only if it
  clears the **quality** bar (Rank-IC / Sharpe) *and* the **novelty** gate (signal correlation below a
  threshold vs every stored alpha). Tracks total trials (for the multiple-testing haircut) and persists to JSON.
- **`llm.py` — the proposer.** One `LLMClient` interface with a `.propose(prompt)` method, plus pluggable
  backends behind it (offline mock, Anthropic, Anthropic-on-Bedrock, and an OpenAI-compatible client covering
  OpenAI and local/hosted open-source servers). `make_client(provider)` selects one. See §7.1.
- **`agents.py` — the agentic layer** (ideas from TradingAgents). Two critics that make the search smarter: a
  **reflection memory** that summarizes each round's lessons (which operators worked, why proposals failed)
  and feeds them into the next prompt, and a **risk-review** critic that vetoes look-ahead-prone or
  cost-fragile candidates before admission.
- **`miner.py` — the loop.** Orchestrates a round: build the prompt → ask the proposer → validate → evaluate →
  risk-review → admit/reject → reflect → repeat. `evaluate_on_test` does the final held-out re-score.
- **`seeds.py` — warm start.** `warm_start()` evaluates a set of seed alphas and admits the good, diverse ones
  before mining, so the LLM builds on a base instead of a blank page.
- **`alpha101.py` — seed bank.** All of WorldQuant's 101 Formulaic Alphas translated into the DSL, ready to
  warm-start with (see §8).
- **`run_demo.py` — entry point.** End-to-end run with one config block (data source, provider, cost, parallelism).

---

## 3. The mining loop (what actually happens each round)

1. **Prompt** the LLM with: the DSL spec, the top-N alphas found so far (with their scores), the most
   recent rejects (with *why* they failed), and an instruction to produce *novel, economically-motivated*
   expressions in strict JSON.
2. **Parse** each expression with the safe evaluator. Reject anything that doesn't parse or references
   unknown operators/fields.
3. **Evaluate** on the **train** split only: compute Rank-IC and a long-short backtest.
4. **Novelty gate**: compute the new signal's correlation to every alpha already in the library; reject if
   it's a near-duplicate (|corr| ≥ 0.7).
5. **Admit** survivors to the library; record rejects + reasons.
6. **Feed back** the round's results into the next prompt. Over rounds this behaves like a guided
   evolutionary search (the LLM is the mutation/crossover operator).
7. After mining, **re-evaluate the final library on the held-out test split** — this is the only number you
   should trust.

---

## 4. Metrics

- **IC / Rank-IC**: cross-sectional correlation between today's signal and next-period returns. Rank-IC
  (Spearman) is the robust workhorse. Mean Rank-IC > ~0.03 with positive IR is interesting on daily equity.
- **ICIR**: mean IC / std of IC across days — stability matters more than peak IC.
- **Long-short backtest**: each day go long the top decile / short the bottom decile of the signal,
  rebalance daily, subtract per-side transaction cost (bps). Report annualized Sharpe, return, max
  drawdown, average daily turnover.

---

## 5. Anti-overfitting & anti-leakage (do not skip)

This is where most LLM-alpha projects quietly fail. Built into the design:

- **Strict time splits**: mining sees only `train`. Final numbers come only from `test`. No peeking.
- **Point-in-time operators**: every time-series operator uses only past data (`delay`, trailing windows).
  Returns used for scoring are *forward* returns; signals are lagged so a signal at day *t* trades the
  *t→t+1* move.
- **Multiple-testing reality check**: you will test thousands of alphas. A 2-sigma Sharpe means little after
  10k trials. The library tracks the trial count so you can apply a **Deflated Sharpe Ratio** / Bonferroni
  haircut before believing anything. (Hook provided in `evaluate.py`.)
- **Diversity gate**: correlation-based dedup stops the LLM from "finding" 50 rescalings of momentum.
- **Costs on by default**: turnover-heavy alphas must clear realistic transaction costs to score well.
- **Leakage notes** (see Profit Mirage, arxiv 2510.07920): if you later add news/fundamentals, restrict all
  observations to dates *after* the LLM's training cutoff, and use only as-reported (point-in-time)
  fundamentals — never restated values.

---

## 6. Extending to options (Phase 2)

The loop is unchanged; you swap two things:

1. **Data panel** → historical options chains: per (underlying, expiry, strike) you need close, IV, delta,
   gamma, vega, open interest. (Sources: ORATS, CBOE DataShop, or your broker's history.)
2. **Evaluator** → the long-short-of-stocks backtest is replaced by a **delta-hedged option P&L** engine
   (so the alpha is a view on *vol*, not direction). Market-Bench (arxiv 2512.12264) shows how sensitive
   this P&L is to implementation details — build it against a verified reference.

The DSL gains a few fields (`iv`, `delta`, `gamma`, `vega`, `oi`) and operators stay the same. Typical
option alphas: IV rank, term-structure slope, skew, IV-minus-realized vol.

---

## 7. Usage

```bash
pip install -r requirements.txt

# Offline demo — no API key, no internet. Synthetic data + mock LLM.
python run_demo.py

# Real data (yfinance) — internet, no API key needed (still uses the mock LLM):
pip install 'alphamine[data]'        # or: pip install yfinance
#   in run_demo.py: DATA_SOURCE = "yfinance"  (TICKERS already set)

# Real data + a real model (any provider from §7.1):
#   DATA_SOURCE  = "yfinance"
#   LLM_PROVIDER = "anthropic" | "openai" | "bedrock" | "ollama" | ...
# then set that provider's credential, e.g.:
export ANTHROPIC_API_KEY=...     # or OPENAI_API_KEY, AWS creds, a local server, ...
```

The demo prints each round's admitted/rejected alphas with scores, then the final library re-scored on the
held-out test window, ranked by test Rank-IC.

The `yfinance` loader handles real-world data messes: bad/delisted symbols and tickers with too little
history (`< min_obs` bars, default 60) are dropped with a note, the requested ticker order is preserved, and
an empty download raises a clear error instead of failing deep in the pipeline.

### 7.1 LLM backends

The miner is model-agnostic — every backend implements the same `.propose()` and is selected by setting
`LLM_PROVIDER` (and optionally `LLM_KWARGS`) in `run_demo.py`. Pick one:

| `LLM_PROVIDER` | Backend | Install | Auth |
|----------------|---------|---------|------|
| `mock` | Offline rotating pool (no network) | — | none |
| `anthropic` | Anthropic API | `pip install 'alphamine[llm]'` | `ANTHROPIC_API_KEY` |
| `bedrock` | Anthropic models via **Amazon Bedrock** (AWS auth + billing) | `pip install 'alphamine[bedrock]'` | AWS creds + `AWS_REGION` |
| `openai` | OpenAI API | `pip install 'alphamine[openai]'` | `OPENAI_API_KEY` |
| `groq` / `together` / `openrouter` | Hosted open-source gateways | `pip install 'alphamine[openai]'` | provider key (`GROQ_API_KEY`, …) |
| `ollama` / `vllm` / `lmstudio` / `llamacpp` | **Local** open-source models | `pip install 'alphamine[openai]'` + run the server | none |
| `local` / `openai-compat` | Any custom OpenAI-compatible endpoint (pass `base_url`) | `pip install 'alphamine[openai]'` | optional |

```python
# run_demo.py — examples (model id goes in LLM_KWARGS; each provider has a sensible default)
LLM_PROVIDER = "anthropic"; LLM_KWARGS = {"model": "claude-opus-4-8"}
LLM_PROVIDER = "openai";    LLM_KWARGS = {"model": "gpt-4o"}
LLM_PROVIDER = "bedrock";   LLM_KWARGS = {"model": "anthropic.claude-opus-4-8"}
LLM_PROVIDER = "ollama";    LLM_KWARGS = {"model": "llama3.1"}               # local, free, private
LLM_PROVIDER = "groq";      LLM_KWARGS = {"model": "llama-3.3-70b-versatile"}
```

Open-source stacks (Ollama, vLLM, LM Studio, llama.cpp, Together, Groq, OpenRouter) are all reached through
a single `OpenAICompatClient` — they expose an OpenAI-compatible `/chat/completions` endpoint, so switching
between them is just a provider name (and `base_url` for anything custom). Smaller local models (7–8B) emit
valid DSL/JSON less reliably than frontier models; to cushion this, every API-backed client does a **one-shot
JSON repair** — if a reply can't be parsed into alpha objects, it re-asks once with a stricter instruction
before giving up. Expect a somewhat lower yield of admitted alphas per round on small models regardless.

### 7.2 Running at scale

The system has two independent compute axes that scale very differently:

- **LLM inference (the proposer)** is *not* the bottleneck for serious mining — even hundreds of rounds is a
  few hundred calls. Use a frontier model for quality: an API (`anthropic`/`openai`) or **AWS Bedrock**
  (`LLM_PROVIDER="bedrock"` — managed models on your AWS account, no GPU to run). Stand up a GPU box
  (local, or AWS `g5`/`g6` + vLLM) only if you want fully-private OSS inference at speed.
- **Evaluation (the backtest)** *is* the bottleneck at scale — scoring thousands of candidate alphas over a
  large universe. It's embarrassingly parallel, so it runs across all CPU cores: set `N_JOBS` in
  `run_demo.py` (or pass `n_jobs=` to `warm_start` / `evaluate_on_test`). The panel is shipped to each
  worker once; small universes auto-stay sequential (no overhead), large ones fan out. Put your "power
  machine" budget into a high-core box (e.g. AWS `c7i.8xlarge`) for the evaluation sweep.

For Russell-3000-scale runs you'll also want a bulk OHLCV source + a parquet cache rather than yfinance,
which rate-limits at thousands of tickers (roadmap item).

---

## 8. WorldQuant Alpha101 seed bank

`alphamine/alpha101.py` ships all 101 formulaic alphas from Kakushadze (2015), translated
into this DSL so you have a large, ready-made seed library — you supply nothing.

- **100 of 101 are usable.** #56 needs market cap (`cap`), which isn't in OHLCV data, so it's
  omitted (listed in `SKIPPED`).
- **18 use `IndNeutralize`.** We don't ship sector labels, so `indneutralize()` approximates it
  as a daily cross-sectional demean. Those alphas are flagged `approx=True`; swap in real sector
  grouping for production. Use `load_alpha101(panel, include_approx=False)` to drop them.
- **Non-integer windows** from the paper (e.g. `8.93345`) are rounded to ints (rolling windows
  must be integers) — standard practice, negligible effect.

Use them three ways:

```python
from alphamine.alpha101 import load_alpha101, coverage_report
alphas = load_alpha101(train_panel)        # -> list[Alpha], ready to evaluate
warm_start(library, train, alphas=alphas)  # admit the good, diverse ones as seeds
coverage_report(train_panel)               # diagnostic: which parse/eval, which fail
```

In the demo (`USE_ALPHA101=True`, the default), the 101 are evaluated, ~49 clear the
quality/novelty gates, and the LLM then mines novel alphas on top of that base.

The same bank degrades gracefully on the options panel later: `load_alpha101` silently skips any
alpha whose fields don't exist there.

## 9. Roadmap

Done so far: the offline scaffold, the Alpha101 seed bank, multi-provider LLM access, the real-data
(yfinance) loader, the agentic layer (reflection + risk-review), and parallel evaluation. Open items —
including the Deflated-Sharpe gate, sector/beta neutralization, a meta-signal blend, a big-universe data
loader, the AWS deployment (`infra/`), and Phase 2 options — are tracked in **[BACKLOG.md](BACKLOG.md)**.

> Not investment advice. Backtested edges routinely vanish live — treat every result as a hypothesis until
> validated out-of-sample and forward-tested.
