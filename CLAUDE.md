# CLAUDE.md — AlphaMine

Context for Claude Code (and any agent) working in this repo. Local-only handoff note; the
public-facing docs are `README.md` and `BACKLOG.md`.

## What this is

AlphaMine is an LLM-driven **formulaic alpha mining** system for US equities. An LLM proposes
alpha expressions in a small safe DSL; each is parsed, evaluated (IC + long-short backtest),
risk-reviewed, and admitted to a diversity-gated library. A reflection memory feeds lessons
back into the next round. Ships with all 100 usable WorldQuant Alpha101 factors as seeds.

Research/educational only — not investment advice. Backtests are hypothetical.

## Current direction (important)

**Target end-goal is PAPER TRADING** (no live money): wire verified alphas to a paper account
(Alpaca first) and forward-test live-vs-backtest. Path is research → paper → maybe-later live; do
not skip paper. Keep all broker code paper/dry-run by default. The execution epic and all other
planned work live in **`BACKLOG.md`** — read it for the full state and priorities.

## Run it

```bash
pip install -r requirements.txt        # numpy, pandas (others are optional extras)
python run_demo.py                     # offline end-to-end: synthetic data + mock LLM
python -m pytest -q                    # full test suite (smoke, agents, llm, parallel, report)

# headless / deployable run -> writes config/library/test/report(.md+.html) to ./out or s3://
OUTPUT=./out python infra/run_job.py
```

Runs with **no API key and no internet** by default (synthetic + MockClient). Go live by editing the
CONFIG block in `run_demo.py`: `DATA_SOURCE="yfinance"`, `LLM_PROVIDER` ∈ {anthropic, openai, bedrock,
ollama, …}, and set that provider's credential. Local Docker deploy: `cd infra/local && make run`.

## Layout

| Module | Role |
|--------|------|
| `alphamine/dsl.py`     | Operators (rank, delay, corr, ts_*, Alpha101 ops) + `DSL_SPEC` prompt text. |
| `alphamine/alpha.py`   | `Alpha` object + **safe AST evaluator** (no `eval`; whitelisted ops only). |
| `alphamine/data.py`    | `Panel` (dates×tickers) loaders: `synthetic` (offline) + `yfinance`. Has `split()`. |
| `alphamine/evaluate.py`| IC / Rank-IC / ICIR, long-short backtest, deflated Sharpe, `evaluate_many` (parallel). |
| `alphamine/library.py` | `AlphaLibrary`: quality + correlation (diversity) gates; JSON persistence. |
| `alphamine/llm.py`     | `LLMClient` base; Mock / Anthropic / Bedrock / OpenAI-compatible backends; `make_client()`. |
| `alphamine/agents.py`  | `ReflectiveMemory` (lessons across rounds) + `risk_review()` (look-ahead/overfit veto). |
| `alphamine/miner.py`   | The loop: prompt → generate → validate → evaluate → risk-review → admit → reflect. |
| `alphamine/report.py`  | `render_html()` — self-contained run dashboard (sortable alpha table, train+test). |
| `alphamine/seeds.py`   | Curated seed bank + `warm_start()`. |
| `alphamine/alpha101.py`| All 100 usable WorldQuant Alpha101 factors in the DSL (#56 needs market cap, omitted). |
| `infra/run_job.py`     | Env-driven headless runner; artifacts to local dir or `s3://`. |
| `infra/Dockerfile`, `infra/local/` | One image; local deployment track (Compose + Makefile). AWS track is planned. |

## Conventions / invariants (please preserve)

- **Never use raw `eval()` on expressions.** All evaluation goes through the AST evaluator in
  `alpha.py`, which only permits fields, numeric constants, and `dsl.OPERATORS`.
- **No look-ahead.** Time-series ops use only past data; signals predict *forward* returns and are
  lagged. Mining sees `train` only; report numbers from the held-out `test` split.
- **New DSL operators** must be added to `dsl.OPERATORS` (and `DSL_SPEC` if LLM-facing).
- **IndNeutralize is approximated** as a cross-sectional demean (no sector labels shipped); the
  18 Alpha101 factors that use it are flagged `approx=True` in `alpha101.py`.
- Single-file, dependency-light: core needs only numpy + pandas. Keep provider SDKs / yfinance /
  boto3 as optional imports inside the functions that use them.
- Add a test under `tests/` for any new mechanism; keep `run_demo.py` runnable offline.
- **Broker/execution code defaults to paper/dry-run.** No real-money order path without an explicit
  user decision revisiting the paper-trading goal.

## Workflow notes

- Work on a branch, then fast-forward to `main` and push (the user has been merging straight to main;
  `gh` is authenticated for PRs if preferred).
- `CLAUDE.md` is intentionally **gitignored** (local-only); don't commit it unless the user asks.

## Next up

Per `BACKLOG.md`, the next PR is the paper-trading core: `signal.py` (top-K alphas → dollar-neutral
target weights) + a `BrokerClient` interface + a `DryRunBroker` (offline, default) + an Alpaca
**paper** adapter, plus a forward-test harness.
