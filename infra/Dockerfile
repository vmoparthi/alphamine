# One image, two deployment tracks (local Compose and AWS). Runs infra/run_job.py,
# which takes all configuration from environment variables and writes artifacts to
# a local dir or to s3://. Build from the repo root:
#   docker build -f infra/Dockerfile -t alphamine .
FROM python:3.12-slim

WORKDIR /app
COPY . /app

# Core + all provider/data extras + boto3 (only used when OUTPUT is an s3:// URI).
RUN pip install --no-cache-dir '.[all]' boto3

ENV PYTHONUNBUFFERED=1 \
    OUTPUT=/app/out

ENTRYPOINT ["python", "infra/run_job.py"]
