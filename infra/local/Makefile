# Local deployment track — thin wrappers over docker compose.
# Usage examples:
#   make run                                  # offline run (mock proposer) -> ./out
#   make run DATA_SOURCE=yfinance             # real data, mock proposer
#   make run LLM_PROVIDER=anthropic LLM_MODEL=claude-opus-4-8   # real model (export the key first)
#   make ollama-up && make ollama-pull MODEL=qwen2.5:7b        # local OSS model
#   make run-ollama MODEL=qwen2.5:7b
#   make view                                 # browse ./out/report.html at http://localhost:8080
#   make clean

COMPOSE ?= docker compose
MODEL   ?= qwen2.5:7b

.PHONY: build run run-ollama ollama-up ollama-pull ollama-down view clean

build:
	$(COMPOSE) build

# One mining run. Pass overrides on the command line, e.g. ROUNDS=8 DATA_SOURCE=yfinance.
run:
	$(COMPOSE) run --rm alphamine
	@echo "artifacts in ./out  (open ./out/report.html)"

ollama-up:
	$(COMPOSE) --profile ollama up -d ollama

ollama-pull:
	$(COMPOSE) --profile ollama exec ollama ollama pull $(MODEL)

ollama-down:
	$(COMPOSE) --profile ollama down

# Run against the local Ollama service (start it + pull a model first).
run-ollama:
	LLM_PROVIDER=ollama LLM_MODEL=$(MODEL) LLM_BASE_URL=http://ollama:11434/v1 \
		$(COMPOSE) run --rm alphamine
	@echo "artifacts in ./out  (open ./out/report.html)"

# Serve ./out so you can open the dashboard in a browser (Ctrl-C to stop).
view:
	@echo "serving ./out at http://localhost:8080  (Ctrl-C to stop)"
	$(COMPOSE) --profile view up viewer

clean:
	rm -rf ./out
	$(COMPOSE) --profile ollama --profile view down -v 2>/dev/null || true
