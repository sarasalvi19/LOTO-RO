"""
cost_estimator.py
=================
Automatic cost estimator for the GraphRAG pipeline.
Reads model names directly from config.yaml and counts entities from the
actual Parquet file — the user only needs to set the number of queries
and optionally adjust the token averages if their use case differs.

The only thing to configure is at the top of the file:
    CONFIG_PATH   : path to config.yaml (default: config.yaml)
    N_QUERIES     : total number of queries to run (default: 150)

Everything else is read automatically from the config and the data.

Usage
-----
    python cost_estimator.py
    python cost_estimator.py --queries 150 --config config.yaml
"""

import argparse
from pathlib import Path

import pandas as pd
import yaml

# ==============================================================================
# USER CONFIGURATION — only edit these two values
# ==============================================================================
CONFIG_PATH = "config.yaml"   # path to your config.yaml
N_QUERIES   = 150              # total number of queries you plan to run
# ==============================================================================

# ------------------------------------------------------------------------------
# OpenAI pricing — USD per 1M tokens
# Update if prices change: https://openai.com/pricing
# ------------------------------------------------------------------------------
PRICING = {
    "gpt-5.4": {
        "input":  2.50,
        "output": 15.00,
    },
    "gpt-5.4-mini": {
        "input":  0.75,
        "output": 4.50,
    },
    "gemini-3.5-flash": {
        "input":  1.50,
        "output": 9.00,
    },
    "gemini-3.1-flash-lite": {
        "input":  0.25,
        "output": 1.50,
    },
    "text-embedding-3-small": {
        "input":  0.02,
        "output": 0.00,
    },
    "gemini-embedding-2": {
        "input":  0.20,
        "output": 0.00,
    },
    "text-embedding-3-large": {
        "input":  0.13,
        "output": 0.00,
    },
}

# ------------------------------------------------------------------------------
# Average token counts per call — adjust only if your KG is very large/small
# ------------------------------------------------------------------------------
# Parser call (report_model, Step A):
#   system prompt with graph vocabulary (~600) + query (~50)
PARSER_INPUT_TOKENS  = 650
PARSER_OUTPUT_TOKENS = 200    # structured JSON output

# Answer call (answer_model, Step C):
#   system prompt (~400) + subgraph (~2000) + text sources (~2000) + query (~50)
ANSWER_INPUT_TOKENS  = 4_450
ANSWER_OUTPUT_TOKENS = 800    # answer JSON with citations

# Embedding (one-time indexing phase):
#   title + description per entity
EMBED_TOKENS_PER_ENTITY = 50


# ------------------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def count_entities(config: dict) -> int:
    """Reads entities.parquet and returns the actual node count."""
    path = config["paths"].get("entities")
    if path and Path(path).exists():
        df = pd.read_parquet(path, columns=["id"])
        return len(df)
    return 0


def token_cost(model: str, input_tok: int, output_tok: int) -> float:
    """Computes USD cost for a given model and token counts."""
    p = PRICING.get(model)
    if not p:
        raise ValueError(
            f"No pricing data for model '{model}'.\n"
            f"Add it to the PRICING dict at the top of cost_estimator.py.\n"
            f"Available models: {list(PRICING.keys())}"
        )
    return (input_tok / 1_000_000) * p["input"] + (output_tok / 1_000_000) * p["output"]


def run(config_path: str, n_queries: int) -> None:

    config = load_config(config_path)
    llm    = config["llm"]

    answer_model    = llm["answer_model"]
    report_model    = llm["report_model"]
    embedding_model = llm.get("embedding_model", "text-embedding-3-small")

    # Count entities automatically from the actual Parquet file
    n_entities = count_entities(config)
    entity_source = f"counted from {config['paths']['entities']}" if n_entities else "file not found — set manually"
    if not n_entities:
        n_entities = 500  # fallback default
        entity_source = "default (entities.parquet not found)"

    # ── Cost calculations ─────────────────────────────────────────────────────

    # Per-query costs
    cost_parser_q = token_cost(report_model,    PARSER_INPUT_TOKENS,  PARSER_OUTPUT_TOKENS)
    cost_answer_q = token_cost(answer_model,    ANSWER_INPUT_TOKENS,  ANSWER_OUTPUT_TOKENS)
    cost_per_query = cost_parser_q + cost_answer_q

    # Total query costs
    total_parser = cost_parser_q * n_queries
    total_answer = cost_answer_q * n_queries

    # One-time embedding cost
    total_embed = token_cost(embedding_model, n_entities * EMBED_TOKENS_PER_ENTITY, 0)

    # Grand total
    grand_total = total_parser + total_answer + total_embed

    # ── Output ────────────────────────────────────────────────────────────────
    W = 62
    print()
    print("=" * W)
    print("  GraphRAG Pipeline — Cost Estimate")
    print("=" * W)
    print(f"  Config file      : {config_path}")
    print(f"  Queries          : {n_queries}")
    print(f"  Entities         : {n_entities}  ({entity_source})")
    print("-" * W)
    print(f"  {'Call type':<28} {'Model':<18} {'$/query':>8}")
    print(f"  {'-'*28} {'-'*18} {'-'*8}")
    print(f"  {'Step A — Semantic parser':<28} {report_model:<18} ${cost_parser_q:>7.5f}")
    print(f"  {'Step C — Answer generation':<28} {answer_model:<18} ${cost_answer_q:>7.5f}")
    print(f"  {'Total per query':<28} {'':18} ${cost_per_query:>7.5f}")
    print("-" * W)
    print(f"  {'Cost item':<28} {'Quantity':<18} {'USD':>8}")
    print(f"  {'-'*28} {'-'*18} {'-'*8}")
    print(f"  {'Step A — all queries':<28} {n_queries:<18} ${total_parser:>7.4f}")
    print(f"  {'Step C — all queries':<28} {n_queries:<18} ${total_answer:>7.4f}")
    print(f"  {'Embeddings (one-time)':<28} {n_entities} entities   ${total_embed:>7.4f}")
    print("=" * W)
    print(f"  {'GRAND TOTAL':<28} {'':18} ${grand_total:>7.4f}")
    print("=" * W)
    print()
    print("  Token averages used:")
    print(f"    Parser  : {PARSER_INPUT_TOKENS} input  + {PARSER_OUTPUT_TOKENS} output  tok/query")
    print(f"    Answer  : {ANSWER_INPUT_TOKENS} input + {ANSWER_OUTPUT_TOKENS} output tok/query")
    print(f"    Embed   : {EMBED_TOKENS_PER_ENTITY} tok/entity")
    print()
    print("  To update pricing or token averages, edit the constants")
    print("  at the top of cost_estimator.py.")
    print("=" * W)
    print()


def main():
    parser = argparse.ArgumentParser(description="GraphRAG cost estimator")
    parser.add_argument("--config",  default=CONFIG_PATH,
                        help=f"Path to config.yaml (default: {CONFIG_PATH})")
    parser.add_argument("--queries", type=int, default=N_QUERIES,
                        help=f"Number of queries to run (default: {N_QUERIES})")
    args = parser.parse_args()
    run(args.config, args.queries)


if __name__ == "__main__":
    main()