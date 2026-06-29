# Local deployment track

Run a full mining job on your own machine with Docker — no cloud, no credentials required for the
offline path. The same container image is used by the [AWS track](../README.md); only orchestration and
output location differ. Artifacts (the run report and JSON) land in `./out`.

## Prerequisites

- Docker with Compose v2 (`docker compose`).
- `make` (optional — every target is a short `docker compose` command you can run directly).

## Run it

```bash
cd infra/local

make run                       # offline: synthetic data + mock proposer  ->  ./out
make view                      # open http://localhost:8080 to browse ./out/report.html
```

`make run` writes `report.html` (interactive dashboard), `report.md`, `alpha_library.json`,
`test_results.json`, and `config.json` into `./out`.

## Real data and real models

Pass any [`run_job.py`](../run_job.py) variable as a `make` override:

```bash
# real market data, still the offline proposer
make run DATA_SOURCE=yfinance TICKERS="AAPL,MSFT,NVDA,AMZN,GOOGL" ROUNDS=6

# a hosted model (export the credential first)
export ANTHROPIC_API_KEY=...
make run DATA_SOURCE=yfinance LLM_PROVIDER=anthropic LLM_MODEL=claude-opus-4-8

# OpenAI
export OPENAI_API_KEY=...
make run LLM_PROVIDER=openai LLM_MODEL=gpt-4o
```

## Local open-source model (fully private, free)

```bash
make ollama-up                 # start the Ollama service
make ollama-pull MODEL=qwen2.5:7b
make run-ollama MODEL=qwen2.5:7b
```

`run-ollama` points the job at the Ollama container via `LLM_BASE_URL=http://ollama:11434/v1` (inside
Compose, `localhost` is the job container, not the host — so the service name is used instead).

## Cleanup

```bash
make clean                     # remove ./out and stop the optional services
```

## Without `make`

Everything is plain Compose:

```bash
docker compose run --rm alphamine                       # one run
LLM_PROVIDER=openai LLM_MODEL=gpt-4o docker compose run --rm alphamine
docker compose --profile view up viewer                 # dashboard at :8080
docker compose --profile ollama up -d ollama            # local model server
```
