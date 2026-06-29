# AlphaMine

**LLM-driven discovery of formulaic alpha factors for US equities.**

AlphaMine uses a large language model as the search operator in a closed loop: it proposes alpha
expressions in a small, safe domain-specific language; each is parsed, evaluated (Information Coefficient
and a long-short backtest), screened by a risk-review critic, and admitted to a diversity-gated library. A
reflection memory turns each round's results into guidance for the next, so the system behaves like a
guided evolutionary search with the model as the mutation operator.

The design prioritizes **transparency and statistical discipline** — every signal is a human-readable
expression, evaluation is point-in-time, and the anti-leakage guarantees that sink most LLM-alpha projects
are built into the core (see [§5](#5-anti-overfitting-and-anti-leakage)).

> **Research software.** Backtests are hypothetical and this is not investment advice. Treat every result
> as a hypothesis until validated out-of-sample and forward-tested.

---

## Highlights

- **Transparent signals.** Alphas are short symbolic expressions (e.g. `rank(-1 * corr(close, volume, 5))`),
  not opaque weights — cheap to generate, trivial to mutate, and auditable.
- **Runs offline out of the box.** A deterministic synthetic market and a mock proposer mean the full loop
  runs with no API key and no internet; swap in real data and a real model with one config line each.
- **Provider-agnostic.** One `LLMClient` interface behind which any backend plugs in — Anthropic, OpenAI,
  Amazon Bedrock, or local/self-hosted open-source models (Ollama, vLLM, Groq, …).
- **Statistical rigor first.** Strict train/test splits, point-in-time operators, a correlation-based
  diversity gate, transaction costs on by default, and a Deflated-Sharpe hook for multiple-testing.
- **Agentic search.** A reflection memory and a risk-review critic (ideas from TradingAgents) make each
  round smarter and catch look-ahead and cost-fragile candidates before admission.
- **Scales to large universes.** Evaluation is parallelized across CPU cores; the proposer runs on managed
  or local infrastructure. A headless, environment-driven job runner (`infra/run_job.py`) emits
  machine-readable artifacts and a run report.
- **Batteries included.** Ships all 100 usable WorldQuant Alpha101 factors as a ready-to-use seed library.

---

## Quickstart

```bash
pip install -r requirements.txt        # core: numpy + pandas

python run_demo.py                     # offline end-to-end: synthetic data + mock proposer
python -m pytest -q                    # test suite
```

The demo prints each round's admitted and rejected alphas with scores, then re-scores the final library on
a held-out test window ranked by test Rank-IC. To go live, set `DATA_SOURCE = "yfinance"` and choose an
`LLM_PROVIDER` in `run_demo.py` (see [§7](#7-usage)).

---

## 1. Design rationale

**Formulaic alphas on daily OHLCV are the foundation.** Each signal is a short symbolic expression, which
makes the approach uniquely well-suited to an LLM loop:

- Expressions are cheap for a model to generate, trivial to mutate, and **fully transparent** — no
  black-box overfitting.
- Evaluation is fast and objective (Information Coefficient plus a long-short backtest), enabling a tight
  propose → score → critique → propose loop, as in Chain-of-Alpha / QuantaAlpha.
- It sidesteps the leakage traps that sink news- and agent-driven approaches (see [§5](#5-anti-overfitting-and-anti-leakage)).

**The miner is model-agnostic.** Alpha generation is a *reasoning + code* task run a few hundred times per
run, so generation quality matters more than throughput and token cost is small. A frontier API model
yields the best results; local open-source models are a drop-in swap for privacy or cost. Every backend
sits behind one `LLMClient` interface, so the provider is a single configuration line ([§7.1](#71-llm-backends)).

**Options support is Phase 2.** Options mining requires a clean historical options chain (IV surface,
greeks) and a path-dependent, delta-hedged backtest. The architecture is built so the *same* mining loop
drives it later — only the data panel and evaluator change (see [§6](#6-extending-to-options-phase-2)).

---

## 2. System architecture

![AlphaMine architecture: pluggable LLM backends (Mock / Anthropic / OpenAI / Bedrock / local OSS) drive a clockwise mining loop where the LLM client proposes alpha expressions, which are validated and evaluated against a data panel, screened by a risk-review critic, admitted to a correlation-deduplicated alpha library, and distilled into a reflection memory whose lessons feed back into the next prompt; the library is re-scored on a held-out test split to produce ranked alphas.](assets/architecture.svg)

The loop runs clockwise. The **LLM client** (any backend — see [§7.1](#71-llm-backends)) proposes new alpha
expressions; each is **validated** (safe AST parse, no arbitrary `eval`) and **evaluated** on the data
panel (Rank-IC plus a long-short backtest). A **risk-review** critic then vetoes look-ahead-prone or
cost-fragile candidates before they reach the **alpha library**, which admits a survivor only if it also
clears the novelty gate (signal correlation below threshold against every stored alpha). Each round's
outcomes are distilled into a **reflection memory** whose lessons feed back into the next prompt.

### 2.1 Components

| Module | Responsibility |
|--------|----------------|
| `data.py`     | Loads daily OHLCV into a `Panel` (one aligned DataFrame per field, indexed date × ticker) with a chronological train/valid/test `split()`. Ships a deterministic **synthetic** market and a real **yfinance** loader that cleans up delisted/short-history symbols. |
| `dsl.py`      | The alpha language — cross-sectional (`rank`, `scale`, `zscore`), time-series (`delay`, `delta`, `ts_mean`, `ts_std`, `ts_rank`, `corr`, `decay_linear`, …), and arithmetic operators, each vectorized over pandas panels. Holds `DSL_SPEC`, the operator reference injected into the prompt. |
| `alpha.py`    | The `Alpha` object: turns an expression string into a signal panel through a **safe AST evaluator** that permits only whitelisted fields, numeric constants, and DSL operators — never arbitrary Python. |
| `evaluate.py` | Scoring: IC / Rank-IC / ICIR and a decile **long-short backtest** with costs → Sharpe, annual return, max drawdown, turnover. Includes `deflated_sharpe` (multiple-testing haircut) and `evaluate_many` (parallel scorer). |
| `library.py`  | `AlphaLibrary` — admits an alpha only if it clears the **quality** bar (Rank-IC / Sharpe) *and* the **novelty** gate (low correlation vs every stored alpha). Tracks trial count; persists to JSON. |
| `llm.py`      | The proposer: one `LLMClient` interface with pluggable backends (mock, Anthropic, Anthropic-on-Bedrock, and an OpenAI-compatible client covering OpenAI and local/hosted OSS). `make_client(provider)` selects one. |
| `agents.py`   | The agentic layer (ideas from TradingAgents): a **reflection memory** that feeds per-round lessons forward, and a **risk-review critic** that vetoes look-ahead / cost-fragile alphas before admission. |
| `miner.py`    | The loop — prompt → propose → validate → evaluate → risk-review → admit/reject → reflect → repeat — plus `evaluate_on_test` for the held-out re-score. |
| `seeds.py`    | `warm_start()` evaluates seed alphas and admits the good, diverse ones before mining. |
| `alpha101.py` | All of WorldQuant's 101 Formulaic Alphas translated into the DSL (see [§8](#8-worldquant-alpha101-seed-bank)). |
| `run_demo.py` / `infra/run_job.py` | Interactive demo and headless, environment-driven job runner. |

---

## 3. The mining loop

Each round:

1. **Prompt** the model with the DSL specification, the top-N alphas found so far (with scores), the most
   recent rejects (with *why* they failed), and an instruction to produce novel, economically-motivated
   expressions in strict JSON.
2. **Parse** each expression with the safe evaluator; reject anything that fails to parse or references
   unknown operators or fields.
3. **Evaluate** on the **train** split only — Rank-IC and a long-short backtest.
4. **Gate**: reject candidates that fail the quality bar, are flagged by the risk-review critic, or are
   near-duplicates of an existing alpha (correlation above threshold).
5. **Admit** survivors to the library and record rejects with reasons.
6. **Reflect**: feed the round's outcomes into the next prompt — over rounds this is a guided evolutionary
   search with the model as mutation/crossover operator.

After mining, the final library is **re-evaluated on the held-out test split** — the only number to trust.

---

## 4. Metrics

- **IC / Rank-IC** — cross-sectional correlation between today's signal and next-period returns. Rank-IC
  (Spearman) is the robust workhorse; mean Rank-IC > ~0.03 with positive IR is interesting on daily equity.
- **ICIR** — mean IC over its standard deviation across days; stability matters more than peak IC.
- **Long-short backtest** — each day, long the top decile and short the bottom decile of the signal,
  rebalance daily, net of per-side transaction cost. Reports annualized Sharpe, return, max drawdown, and
  average daily turnover.

---

## 5. Anti-overfitting and anti-leakage

This is where most LLM-alpha projects quietly fail. The safeguards are built into the design, not bolted on:

- **Strict time splits** — mining sees only `train`; reported numbers come only from `test`. No peeking.
- **Point-in-time operators** — every time-series operator uses only past data; returns used for scoring
  are *forward* returns, and signals are lagged so a signal at day *t* trades the *t→t+1* move.
- **Multiple-testing discipline** — across thousands of trials a 2-sigma Sharpe means little. The library
  tracks the trial count so a **Deflated Sharpe Ratio** / Bonferroni haircut can be applied before any
  result is believed (`deflated_sharpe` in `evaluate.py`).
- **Diversity gate** — correlation-based dedup prevents the model from "discovering" fifty rescalings of momentum.
- **Costs on by default** — turnover-heavy alphas must clear realistic transaction costs to score.
- **Leakage guidance** — when extending to news/fundamentals, restrict observations to dates after the
  model's training cutoff and use only as-reported (point-in-time) values (cf. *Profit Mirage*, arXiv:2510.07920).

---

## 6. Extending to options (Phase 2)

The mining loop is unchanged; two components are swapped:

1. **Data panel** → historical options chains: per (underlying, expiry, strike), close, IV, delta, gamma,
   vega, open interest. (Sources: ORATS, CBOE DataShop, or broker history.)
2. **Evaluator** → the long-short-of-stocks backtest is replaced by a **delta-hedged option P&L** engine,
   so the alpha expresses a view on *volatility*, not direction. (cf. Market-Bench, arXiv:2512.12264.)

The DSL gains a few fields (`iv`, `delta`, `gamma`, `vega`, `oi`); operators are unchanged. Typical option
alphas: IV rank, term-structure slope, skew, IV-minus-realized vol.

---

## 7. Usage

```bash
pip install -r requirements.txt

# Offline — no API key, no internet (synthetic data + mock proposer)
python run_demo.py

# Real data (yfinance) — internet only, still using the mock proposer
pip install 'alphamine[data]'                  # or: pip install yfinance
#   in run_demo.py: DATA_SOURCE = "yfinance"

# Real data + a real model (any provider from §7.1)
#   DATA_SOURCE  = "yfinance"
#   LLM_PROVIDER = "anthropic" | "openai" | "bedrock" | "ollama" | ...
export ANTHROPIC_API_KEY=...                   # or OPENAI_API_KEY, AWS creds, a local server, …
```

The `yfinance` loader is built for real-world data: bad/delisted symbols and tickers with too little
history (`< min_obs` bars, default 60) are dropped with a note, the requested ticker order is preserved,
and an empty download raises a clear error rather than failing deep in the pipeline.

For non-interactive runs, `infra/run_job.py` takes all configuration from environment variables and writes
artifacts (config, alpha library, test results, and a run report) to a local directory or to `s3://…`.

### 7.1 LLM backends

Every backend implements the same `.propose()` and is selected with `LLM_PROVIDER` (and an optional
`LLM_KWARGS` for the model id):

| `LLM_PROVIDER` | Backend | Install | Auth |
|----------------|---------|---------|------|
| `mock` | Offline rotating pool (no network) | — | none |
| `anthropic` | Anthropic API | `pip install 'alphamine[llm]'` | `ANTHROPIC_API_KEY` |
| `bedrock` | Anthropic models via **Amazon Bedrock** | `pip install 'alphamine[bedrock]'` | AWS creds + `AWS_REGION` |
| `openai` | OpenAI API | `pip install 'alphamine[openai]'` | `OPENAI_API_KEY` |
| `groq` / `together` / `openrouter` | Hosted open-source gateways | `pip install 'alphamine[openai]'` | provider key |
| `ollama` / `vllm` / `lmstudio` / `llamacpp` | Local open-source models | `pip install 'alphamine[openai]'` + run the server | none |
| `local` / `openai-compat` | Any custom OpenAI-compatible endpoint (`base_url`) | `pip install 'alphamine[openai]'` | optional |

```python
# run_demo.py — model id goes in LLM_KWARGS; each provider has a sensible default
LLM_PROVIDER = "anthropic"; LLM_KWARGS = {"model": "claude-opus-4-8"}
LLM_PROVIDER = "openai";    LLM_KWARGS = {"model": "gpt-4o"}
LLM_PROVIDER = "bedrock";   LLM_KWARGS = {"model": "anthropic.claude-opus-4-8"}
LLM_PROVIDER = "ollama";    LLM_KWARGS = {"model": "llama3.1"}   # local, free, private
```

Open-source stacks (Ollama, vLLM, LM Studio, llama.cpp, Together, Groq, OpenRouter) are all reached through
a single `OpenAICompatClient` — they expose an OpenAI-compatible `/chat/completions` endpoint, so switching
between them is just a provider name (and `base_url` for anything custom). Smaller local models emit valid
DSL/JSON less reliably than frontier models; to cushion this, every API-backed client performs a **one-shot
JSON repair** — on an unparseable reply it re-asks once with a stricter instruction before giving up.

### 7.2 Running at scale

The system has two compute axes that scale very differently:

- **LLM inference (the proposer)** is *not* the bottleneck for serious mining — even hundreds of rounds is a
  few hundred calls. Use a frontier model for quality via an API (`anthropic` / `openai`) or **Amazon
  Bedrock** (`LLM_PROVIDER="bedrock"` — managed models, no GPU). Stand up a GPU host (local, or AWS
  `g5`/`g6` with vLLM) only when fully-private open-source inference at speed is required.
- **Evaluation (the backtest)** *is* the bottleneck at scale — scoring thousands of candidate alphas over a
  large universe. It is embarrassingly parallel and runs across all CPU cores: set `N_JOBS` (or pass
  `n_jobs=` to `warm_start` / `evaluate_on_test`). The panel is shipped to each worker once; small
  universes stay sequential automatically while large ones fan out. Direct compute budget at a high-core
  host (e.g. AWS `c7i.8xlarge`) for the evaluation sweep.

For Russell-3000-scale runs, replace yfinance with a bulk OHLCV source and a parquet cache (see
[BACKLOG.md](BACKLOG.md)).

---

## 8. WorldQuant Alpha101 seed bank

`alphamine/alpha101.py` provides all 101 formulaic alphas from Kakushadze (2015), translated into the DSL
as a ready-made seed library.

- **100 of 101 are usable.** #56 requires market cap (`cap`), which is absent from OHLCV data, and is
  omitted (listed in `SKIPPED`).
- **18 use `IndNeutralize`.** Without shipped sector labels, `indneutralize()` approximates it as a daily
  cross-sectional demean; those alphas are flagged `approx=True`. Pass `include_approx=False` to drop them,
  or supply real sector grouping for production.
- **Non-integer windows** from the paper (e.g. `8.93345`) are rounded to integers (rolling windows must be
  integers) — standard practice, negligible effect.

```python
from alphamine.alpha101 import load_alpha101, coverage_report
alphas = load_alpha101(train_panel)        # -> list[Alpha], ready to evaluate
warm_start(library, train, alphas=alphas)  # admit the good, diverse ones as seeds
coverage_report(train_panel)               # diagnostic: which parse/evaluate, which fail
```

With `USE_ALPHA101=True` (the default), the 101 are evaluated, roughly half clear the quality/novelty gates,
and the model then mines novel alphas on top of that base. The same bank degrades gracefully on the options
panel: `load_alpha101` silently skips any alpha whose fields are unavailable.

---

## 9. Roadmap

Delivered: the offline engine, the Alpha101 seed bank, multi-provider LLM access, the real-data (yfinance)
loader, the agentic layer (reflection + risk-review), parallel evaluation, and a headless job runner. Open
items — the Deflated-Sharpe admission gate, sector/beta neutralization, a meta-signal blend, a big-universe
data loader, the local and AWS deployment tracks (`infra/`), a run dashboard, and Phase 2 options — are
tracked in **[BACKLOG.md](BACKLOG.md)**.

---

<sub>Not investment advice. Backtested edges routinely vanish out-of-sample; treat every result as a
hypothesis until validated and forward-tested.</sub>
