# Deployment

AlphaMine runs the same job two ways — pick the track that fits.

| Track | Where it runs | Output | Status |
|-------|---------------|--------|--------|
| **Local** ([`local/`](local/)) | Your machine, via Docker Compose | `./out` (open `report.html`) | ✅ available |
| **AWS** (`aws/`) | ECS/Batch + S3 + Bedrock | `s3://…` | 🚧 planned ([BACKLOG.md](../BACKLOG.md)) |

Both use one container image ([`Dockerfile`](Dockerfile)) and one entrypoint ([`run_job.py`](run_job.py)),
which takes all configuration from environment variables and writes artifacts to a local directory **or**
to `s3://…`. Write once, deploy either way.

## Shared core

- **[`run_job.py`](run_job.py)** — headless, environment-driven mining run. Loads data, warm-starts with the
  Alpha101 bank, mines, re-scores on the held-out test split, and writes `config.json`,
  `alpha_library.json`, `test_results.json`, `report.md`, and an interactive `report.html` dashboard. Runs
  locally without Docker too: `OUTPUT=./out python infra/run_job.py`.
- **[`Dockerfile`](Dockerfile)** — packages the project + all extras; entrypoint is `run_job.py`.

Configuration (all optional; sensible defaults): `LLM_PROVIDER`, `LLM_MODEL`, `LLM_BASE_URL`, `DATA_SOURCE`,
`TICKERS`, `START`/`END`, `ROUNDS`, `PER_ROUND`, `N_JOBS`, `USE_ALPHA101`, `OUTPUT`. See the
[`run_job.py`](run_job.py) docstring for the full list.

## Local track

See **[local/README.md](local/README.md)**. In short:

```bash
cd infra/local
make run          # offline run -> ./out
make view         # browse ./out/report.html at http://localhost:8080
```

## AWS track

Planned (`infra/aws/`): one image pushed to ECR, artifacts in S3, the parallel evaluation sweep on
ECS/Batch, the proposer on Amazon Bedrock (no GPU), wrapped in Terraform. Tracked in
[BACKLOG.md](../BACKLOG.md) under the infrastructure epic.
