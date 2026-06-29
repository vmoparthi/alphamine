"""Env-driven mining job — the entrypoint the container/AWS task runs.

Same pipeline as run_demo.py, but every knob comes from environment variables and
the results (alpha library JSON + a Markdown report) are written to a local dir or
uploaded to S3. Runs offline with the defaults (synthetic data + mock proposer).

Config (all optional; defaults in []):
  LLM_PROVIDER   mock | anthropic | bedrock | openai | ollama | ...   [mock]
  LLM_MODEL      model id for the provider (e.g. claude-opus-4-8, gpt-4o)   [provider default]
  DATA_SOURCE    synthetic | yfinance                                  [synthetic]
  TICKERS        comma-separated symbols (yfinance)                    [16 large caps]
  START / END    yfinance date range (YYYY-MM-DD)                      [2018-01-01 / today]
  N_DAYS/N_TICKERS  synthetic panel size                               [750 / 40]
  ROUNDS         mining rounds                                         [4]
  PER_ROUND      proposals per round                                   [6]
  COST_BPS       per-side transaction cost                             [5]
  N_JOBS         eval parallelism (blank=all cores, 1=sequential)      [all cores]
  USE_ALPHA101   warm-start with the Alpha101 bank (1/0)               [1]
  OUTPUT         local dir or s3://bucket/prefix for artifacts         [./out]
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys

# work both as an installed package (container) and run directly from the repo
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alphamine import data
from alphamine.alpha101 import load_alpha101
from alphamine.library import AlphaLibrary
from alphamine.llm import make_client
from alphamine.miner import MinerConfig, evaluate_on_test, mine
from alphamine.seeds import warm_start

_DEFAULT_TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "JPM", "XOM",
                    "JNJ", "PG", "KO", "PEP", "WMT", "DIS", "INTC", "CSCO"]


def _env(k, d=None):
    v = os.environ.get(k)
    return v if v not in (None, "") else d


def _flag(k, d=True):
    v = _env(k)
    return d if v is None else v.strip().lower() in ("1", "true", "yes", "on")


def _load_panel():
    src = _env("DATA_SOURCE", "synthetic")
    if src == "synthetic":
        return data.load("synthetic",
                         n_days=int(_env("N_DAYS", 750)),
                         n_tickers=int(_env("N_TICKERS", 40)))
    tickers = [t.strip() for t in _env("TICKERS", ",".join(_DEFAULT_TICKERS)).split(",") if t.strip()]
    return data.load("yfinance", tickers=tickers,
                     start=_env("START", "2018-01-01"), end=_env("END"))


def _report(run_id, cfg_info, lib, results) -> str:
    lines = [f"# AlphaMine run {run_id}", ""]
    for k, v in cfg_info.items():
        lines.append(f"- **{k}**: {v}")
    lines += ["", f"## Library: {len(lib)} alphas from {lib.trials} trials", "",
              "## Top alphas on held-out TEST (by |Rank-IC|)", "",
              "| Rank-IC | Sharpe | expression |", "|---|---|---|"]
    for a, m in results[:15]:
        lines.append(f"| {m.rank_ic:+.3f} | {m.sharpe:+.2f} | `{a.expr[:80]}` |")
    lines.append("")
    return "\n".join(lines)


def _write(output: str, files: dict):
    """Write {name: text} either to a local dir or to s3://bucket/prefix."""
    if output.startswith("s3://"):
        import boto3  # optional dep, only needed for S3 output
        bucket, _, prefix = output[5:].partition("/")
        s3 = boto3.client("s3")
        for name, body in files.items():
            key = f"{prefix.rstrip('/')}/{name}" if prefix else name
            s3.put_object(Bucket=bucket, Key=key, Body=body.encode("utf-8"))
            print(f"  wrote s3://{bucket}/{key}")
    else:
        os.makedirs(output, exist_ok=True)
        for name, body in files.items():
            path = os.path.join(output, name)
            with open(path, "w") as f:
                f.write(body)
            print(f"  wrote {path}")


def main():
    run_id = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    provider = _env("LLM_PROVIDER", "mock")
    model = _env("LLM_MODEL")
    n_jobs = int(_env("N_JOBS")) if _env("N_JOBS") else None
    cost = float(_env("COST_BPS", 5.0))
    output = _env("OUTPUT", "./out")

    panel = _load_panel()
    train, valid, test = panel.split()
    print(f"[{run_id}] data={_env('DATA_SOURCE','synthetic')} "
          f"{len(panel.index)}d x {len(panel.tickers)}t "
          f"(train={len(train.index)} test={len(test.index)})")

    lib = AlphaLibrary(max_corr=0.7, min_rank_ic=0.01, min_sharpe=0.3)
    if _flag("USE_ALPHA101", True):
        n = warm_start(lib, train, cost_bps=cost, verbose=False,
                       alphas=load_alpha101(train), n_jobs=n_jobs)
        print(f"warm-start: admitted {n} Alpha101 seeds")

    client = make_client(provider, **({"model": model} if model else {}))
    cfg = MinerConfig(rounds=int(_env("ROUNDS", 4)), per_round=int(_env("PER_ROUND", 6)),
                      cost_bps=cost)
    print(f"mining: provider={provider} model={model or '(default)'} "
          f"rounds={cfg.rounds} per_round={cfg.per_round}")
    mine(lib, client, train, cfg)

    results = evaluate_on_test(lib, test, cost_bps=cost, n_jobs=n_jobs)
    print(f"done: {len(lib)} alphas, {lib.trials} trials")

    cfg_info = {"run_id": run_id, "provider": provider, "model": model or "(default)",
                "data_source": _env("DATA_SOURCE", "synthetic"),
                "rounds": cfg.rounds, "per_round": cfg.per_round, "cost_bps": cost}
    lib_json = json.dumps([
        {"expr": e.alpha.expr, "rationale": e.alpha.rationale,
         "rank_ic": e.metrics.rank_ic, "sharpe": e.metrics.sharpe}
        for e in lib.entries], indent=2)
    test_json = json.dumps([
        {"expr": a.expr, "test_rank_ic": m.rank_ic, "test_sharpe": m.sharpe}
        for a, m in results], indent=2)

    _write(output, {
        "config.json": json.dumps(cfg_info, indent=2),
        "alpha_library.json": lib_json,
        "test_results.json": test_json,
        "report.md": _report(run_id, cfg_info, lib, results),
    })
    print(f"artifacts -> {output}")


if __name__ == "__main__":
    main()
