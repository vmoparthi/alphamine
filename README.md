# AlphaMine — LLM-driven alpha mining (starter)

A small, runnable system that uses an LLM to **generate**, **evaluate**, and **iteratively improve**
formulaic alpha factors on US equities. It runs **offline out of the box** (synthetic data + a mock
"LLM" that emits real alpha expressions), so you can see the full loop before plugging in a real API
key or real data.

---

## 1. Recommendation (why this design)

You picked *US equities (daily OHLCV)* **and** *options*, "recommend the alpha type", and "recommend the
LLM backend". Here's the call and the reasoning:

**Start with formulaic alphas on US equity daily OHLCV.** This is the single highest-leverage starting
point because:

- The signal is a short symbolic expression (e.g. `rank(-1 * corr(close, volume, 5))`). It's cheap for an
  LLM to generate, trivial to mutate, and **fully transparent** — no black-box overfitting.
- Evaluation is fast and objective (Information Coefficient + a long-short backtest), so the LLM can run a
  tight propose → score → critique → propose loop. This is exactly what Chain-of-Alpha / QuantaAlpha do.
- It avoids the leakage traps that sink news/agentic approaches (covered in §5).

**Use an API model (Claude or GPT) for the miner.** Alpha generation is a *reasoning + code* task done a
few hundred times per run — quality matters far more than volume, and the token cost is small. A local
model (Ollama/vLLM) is a fine swap later for privacy/cost; the `LLMClient` interface makes it a drop-in.

**Options is Phase 2, not Phase 1.** Options alpha mining needs a clean historical options chain (IV
surface, greeks) and a much more careful backtest (path-dependent P&L, delta hedging, bid/ask). The
architecture below is built so the *same* mining loop drives it later — you only swap the data panel and
the evaluator. See §6.

---

## 2. System architecture

```
                ┌─────────────────────────────────────────────────────┐
                │                     MINER LOOP                        │
                │                                                       │
   ┌─────────┐  │  ┌───────────┐   ┌────────────┐   ┌───────────────┐  │
   │  LLM    │──┼─▶│  Generate │──▶│  Parse +   │──▶│  Evaluate     │  │
   │ client  │  │  │  alphas   │   │  validate  │   │  (IC, backtest)│ │
   └─────────┘  │  └───────────┘   └────────────┘   └───────┬───────┘  │
        ▲       │        ▲                                   │          │
        │       │        │   feedback: scores + critique     │          │
        │       │        └───────────────────────────────────┘          │
        │       │                                            │          │
        │       │                                            ▼          │
        │       │                                   ┌──────────────┐    │
        └───────┼───────────────────────────────────│ Alpha library│    │
   prompt incl. │   novelty check (corr < 0.7 vs     │ (dedup by    │    │
   best + recent│   existing) before admission       │ correlation) │    │
                └───────────────────────────────────└──────────────┘────┘
                                  │
                                  ▼
                       data panel (OHLCV)  ◀── yfinance / Stooq / synthetic
```

**Modules**

| File | Responsibility |
|------|----------------|
| `alphamine/data.py`    | Load daily OHLCV into aligned panels (dates × tickers). yfinance loader + deterministic **synthetic** fallback so everything runs offline. |
| `alphamine/dsl.py`     | The alpha language: cross-sectional + time-series operators (`rank`, `delay`, `delta`, `corr`, `ts_mean`, `ts_std`, `scale`, …) implemented on pandas panels. |
| `alphamine/alpha.py`   | `Alpha` object: holds the expression string, parses it to a signal panel via a **safe AST evaluator** (no `eval` of arbitrary Python). |
| `alphamine/evaluate.py`| Turns a signal into performance: IC, Rank-IC, IR, a decile **long-short backtest** with costs → Sharpe, annual return, max drawdown, turnover. |
| `alphamine/llm.py`     | `LLMClient` interface. `AnthropicClient` (real API) + `MockClient` (offline, emits a rotating set of valid alpha expressions). |
| `alphamine/library.py` | Persistent alpha store (JSON). Admits a new alpha only if its signal correlation to every stored alpha is below a threshold → enforces **diversity**. |
| `alphamine/miner.py`   | The loop: prompt → generate → parse → evaluate → admit/reject → build feedback → repeat. |
| `alphamine/seeds.py`   | Small curated seed bank + `warm_start()` so you never start from a blank page. |
| `alphamine/alpha101.py`| **All of WorldQuant's 101 Formulaic Alphas**, translated into the DSL (see §9). |
| `run_demo.py`          | End-to-end offline demo. |

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

# Real run: set a key and flip the flags in run_demo.py / config.py
export ANTHROPIC_API_KEY=sk-...
#   data.source   = "yfinance"
#   llm.provider  = "anthropic"
```

The demo prints each round's admitted/rejected alphas with scores, then the final library re-scored on the
held-out test window, ranked by test Rank-IC.

---

## 9. WorldQuant Alpha101 seed bank

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

## 8. Roadmap

- [ ] Phase 1: this scaffold → swap in real OHLCV (yfinance) and a real API model.
- [ ] Add neutralization (sector/beta) to operators.
- [ ] Add Deflated Sharpe gate before admission.
- [ ] Combine top-K alphas into a meta-signal (equal-risk or regression blend).
- [ ] Phase 2: options data panel + delta-hedged evaluator.
- [ ] Optional: port the backtest to Qlib for a battle-tested engine.

> Not investment advice. Backtested edges routinely vanish live — treat every result as a hypothesis until
> validated out-of-sample and forward-tested.
