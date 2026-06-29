"""End-to-end demo. Runs fully offline: synthetic data + mock LLM.

    python run_demo.py

To go live, change CONFIG below:
    DATA_SOURCE  = "yfinance"   (pip install yfinance, set TICKERS)
    LLM_PROVIDER = "anthropic"  (pip install anthropic, export ANTHROPIC_API_KEY) -> claude-opus-4-8

Claude via Amazon Bedrock (pip install 'anthropic[bedrock]', AWS creds + AWS_REGION):
    LLM_PROVIDER = "bedrock";  LLM_KWARGS = {"model": "anthropic.claude-opus-4-8"}

OpenAI GPT models (pip install openai, export OPENAI_API_KEY):
    LLM_PROVIDER = "openai";   LLM_KWARGS = {"model": "gpt-4o"}

Open-source / local models (pip install openai, run the server, then):
    LLM_PROVIDER = "ollama";   LLM_KWARGS = {"model": "llama3.1"}
    LLM_PROVIDER = "vllm";     LLM_KWARGS = {"model": "Qwen/Qwen2.5-7B-Instruct"}
    LLM_PROVIDER = "lmstudio"; LLM_KWARGS = {"model": "<loaded-model-id>"}
    LLM_PROVIDER = "groq";     LLM_KWARGS = {"model": "llama-3.3-70b-versatile"}  (export GROQ_API_KEY)
"""
from alphamine import data
from alphamine.library import AlphaLibrary
from alphamine.llm import make_client
from alphamine.miner import MinerConfig, mine, evaluate_on_test, evaluate
from alphamine.evaluate import deflated_sharpe
from alphamine.seeds import warm_start
from alphamine.alpha101 import load_alpha101

# ---------------- CONFIG ----------------
DATA_SOURCE = "synthetic"     # or "yfinance"
TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "JPM", "XOM",
           "JNJ", "PG", "KO", "PEP", "WMT", "DIS", "INTC", "CSCO"]
LLM_PROVIDER = "mock"         # "mock" | "anthropic" | "bedrock" | "openai" | open-source: "ollama"/"vllm"/"lmstudio"/"groq"/...
LLM_KWARGS = {}               # extra args for the client, e.g. {"model": "llama3.1"} for ollama
WARM_START = True             # seed the library with classic alphas before mining
USE_ALPHA101 = True           # seed with the full WorldQuant 101 (else small curated bank)
COST_BPS = 5.0
N_JOBS = None                 # eval parallelism: None=all cores, 1=sequential. Auto-stays
                              # sequential on small panels; fans out for large universes.
# ----------------------------------------


def main():
    # 1) data + chronological split
    if DATA_SOURCE == "synthetic":
        panel = data.load("synthetic", n_days=750, n_tickers=40)
    else:
        panel = data.load("yfinance", tickers=TICKERS, start="2018-01-01")
    train, valid, test = panel.split(train_frac=0.6, valid_frac=0.2)
    print(f"Loaded {DATA_SOURCE}: {len(panel.index)} days x {len(panel.tickers)} tickers")
    print(f"  train={len(train.index)}  valid={len(valid.index)}  test={len(test.index)} days")

    # 2) miner setup
    cfg = MinerConfig(rounds=4, per_round=6, cost_bps=COST_BPS,
                      max_corr=0.7, min_rank_ic=0.01, min_sharpe=0.3)
    library = AlphaLibrary(max_corr=cfg.max_corr,
                           min_rank_ic=cfg.min_rank_ic, min_sharpe=cfg.min_sharpe)
    client = make_client(LLM_PROVIDER, **LLM_KWARGS)

    # 2b) OPTIONAL warm-start: admit classic seed alphas so the LLM has a base to build on.
    #     (You don't need to supply any alpha yourself — these ship in alphamine/seeds.py.)
    if WARM_START:
        seeds = load_alpha101(train) if USE_ALPHA101 else None
        label = "WorldQuant Alpha101" if USE_ALPHA101 else "curated classics"
        print(f"\n=== Warm-start: {label} ===")
        n = warm_start(library, train, cost_bps=COST_BPS, verbose=False, alphas=seeds, n_jobs=N_JOBS)
        print(f"  evaluated {len(seeds) if seeds else '~14'} seeds -> admitted {n} "
              f"(rest filtered by quality/novelty gates)")

    # 3) mine on TRAIN only
    mine(library, client, train, cfg, verbose=True)

    # 4) validation pass (optional sanity filter before trusting test)
    print(f"\nLibrary: {len(library)} admitted alphas (from {library.trials} trials)")

    # 5) honest out-of-sample read on TEST
    print("\n=== Held-out TEST performance (ranked by |rank_ic|) ===")
    results = evaluate_on_test(library, test, cost_bps=COST_BPS, n_jobs=N_JOBS)
    print(f"{'expr':50s} {'rank_ic':>8s} {'icir':>6s} {'sharpe':>7s} {'ann_ret':>8s} {'maxDD':>7s}")
    for alpha, m in results:
        dsr = deflated_sharpe(m.sharpe, n_trials=library.trials, n_obs=m.n_obs)
        print(f"{alpha.expr:50s} {m.rank_ic:>8.3f} {m.icir:>6.2f} "
              f"{m.sharpe:>7.2f} {m.ann_return:>8.3f} {m.max_drawdown:>7.3f}  P(SR>0)~{dsr:.2f}")

    library.save("alpha_library.json")
    print("\nSaved -> alpha_library.json")
    print("Reminder: TEST numbers are the only ones to trust. Forward-test before risking capital.")


if __name__ == "__main__":
    main()
