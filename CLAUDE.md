# CLAUDE.md — AlphaMine

Context for Claude Code (and any agent) working in this repo.

## What this is

AlphaMine is an LLM-driven **formulaic alpha mining** system for US equities. An LLM proposes
alpha expressions in a small safe DSL; each is parsed, evaluated (IC + long-short backtest),
risk-reviewed, and admitted to a diversity-gated library. A reflection memory feeds lessons
back into the next round. Ships with all 100 usable WorldQuant Alpha101 factors as seeds.

Research/educational only — not investment advice. Backtests are hypothetical.

## Run it

```bash
pip install -r requirements.txt        # numpy, pandas (yfinance/anthropic optional)
python run_demo.py                     # offline end-to-end: synthetic data + mock LLM
python tests/test_smoke.py             # core smoke tests
python tests/test_agents.py            # agentic-layer tests
python -m pytest -q                    # if pytest installed
```

The demo runs with **no API key and no internet** (synthetic data + MockClient). To go live,
edit the CONFIG block in `run_demo.py`: `DATA_SOURCE="yfinance"`, `LLM_PROVIDER="anthropic"`
(or `openai`/`ollama`/…), and set the relevant API key env var.

## Layout

| Module | Role |
|--------|------|
| `alphamine/dsl.py`     | Operators (rank, delay, corr, ts_*, Alpha101 ops) + `DSL_SPEC` prompt text. |
| `alphamine/alpha.py`   | `Alpha` object + **safe AST evaluator** (no `eval`; whitelisted ops only). |
| `alphamine/data.py`    | `Panel` (dates×tickers) loaders: `synthetic` (offline) + `yfinance`. Has `split()`. |
| `alphamine/evaluate.py`| IC / Rank-IC / ICIR, long-short backtest w/ costs, deflated Sharpe, signal corr. |
| `alphamine/library.py` | `AlphaLibrary`: quality + correlation (diversity) gates; JSON persistence. |
| `alphamine/llm.py`     | `LLMClient` base; Mock / Anthropic / Bedrock / OpenAI-compatible backends; `make_client()`. |
| `alphamine/agents.py`  | `ReflectiveMemory` (lessons across rounds) + `risk_review()` (look-ahead/overfit veto). |
| `alphamine/miner.py`   | The loop: prompt → generate → validate → evaluate → risk-review → admit → reflect. |
| `alphamine/seeds.py`   | Curated seed bank + `warm_start()`. |
| `alphamine/alpha101.py`| All 100 usable WorldQuant Alpha101 factors in the DSL (#56 needs market cap, omitted). |

## Conventions / invariants (please preserve)

- **Never use raw `eval()` on expressions.** All evaluation goes through the AST evaluator in
  `alpha.py`, which only permits fields, numeric constants, and `dsl.OPERATORS`.
- **No look-ahead.** Time-series ops use only past data; signals predict *forward* returns and are
  lagged. Mining sees `train` only; report numbers from the held-out `test` split.
- **New DSL operators** must be added to `dsl.OPERATORS` (and `DSL_SPEC` if LLM-facing).
- **IndNeutralize is approximated** as a cross-sectional demean (no sector labels shipped); the
  18 Alpha101 factors that use it are flagged `approx=True` in `alpha101.py`.
- Single-file, dependency-light: core needs only numpy + pandas. Keep `yfinance`/`anthropic`/
  `openai` as optional imports inside the functions that use them.
- Add a test under `tests/` for any new mechanism; keep `run_demo.py` runnable offline.

## Good next steps

- Real data run (yfinance + a live LLM provider) and commit results.
- LLM-written reflection (hook: `agents.reflect_with_llm`) or a proposer-vs-critic debate round.
- Sector/beta neutralization in operators; Deflated-Sharpe admission gate.
- Phase 2: options data panel + delta-hedged evaluator (DSL gains iv/delta/gamma/vega fields).
