---
name: project-goal-paper-trading
description: "AlphaMine's target end-goal is paper trading (not live money yet)"
metadata: 
  node_type: memory
  type: project
  originSessionId: e5eb2216-3543-40b6-9327-0bfe9db5cb53
---

The end goal for AlphaMine (as of 2026-06-29) is **paper trading**: wire verified alphas to a paper
account and forward-test live-vs-backtest with **zero real money at risk**. Path is research → paper →
(maybe later) live; do not skip paper.

**Why:** the user wants the honest "are these alphas real?" test before any real capital — backtested
edges routinely vanish live.

**How to apply:** prioritize the execution epic in [[BACKLOG.md]] in this order — `signal.py` (live signal
→ target weights) + a `BrokerClient` interface + an **Alpaca paper** adapter, with a forward-test harness.
Default everything to paper/dry-run mode; no live-money order routing unless the user explicitly revisits
this decision. Alpaca is the chosen first venue (free paper sandbox); Robinhood is a cautioned, non-primary
adapter (no official API).
