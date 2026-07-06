"""
main.py
=======
Entry point of the GraphRAG pipeline. Orchestrates two sequential phases:
  (i)  Graph Build — loads the LOTO-RO knowledge graph from Parquet files
       into memory via GraphBuilder; optionally exports an interactive HTML
       visualisation coloured by entity type.
  (ii) Query Processing — reads a query list from an Excel input file,
       submits each query to SearchEngine, and writes results to a
       timestamped Excel output file.

Each output row contains the GraphRAG-generated answer, the retrieval
strategy used, cited source text IDs, cited entity IDs, and a status field.
Partial runs are supported via --max-queries and --task-ids CLI flags.

Inputs
  config.yaml          : all paths, LLM settings, query parameters, ontology
  input Excel file     : columns Task_ID, Variant, Category, query,
                         Gold_Answer, Gold_Entity_Titles
  Parquet files        : entities, relationships, text_units, attributes

Outputs
  graphrag_results_<timestamp>.xlsx : one row per query with columns
    Search_Type, GraphRAG_Answer, Status, Cited_Source_IDs, Cited_Entities_IDs

CLI flags
  --config      : path to config.yaml (default: config.yaml)
  --max-queries : process only the first N queries
  --task-ids    : process only specified Task_IDs (e.g. --task-ids 1 1.1 3.2)
  --visualize   : export interactive graph.html coloured by entity type

── Usage ─────────────────────────────────────────────────────────────────────
  python main.py
  python main.py --config config.yaml --max-queries 10 --visualize
  python main.py --task-ids 1 1.1 2.3
"""
import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm

from graph_builder import GraphBuilder
from llm_client import LLMClient
from search_engine import SearchEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)-8s]  %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

logger = logging.getLogger("main")


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def ensure_dirs(config: dict) -> None:
    Path(config["paths"]["output_excel"]).parent.mkdir(parents=True, exist_ok=True)


def ids_to_str(ids: list) -> str:
    """Converte una lista di ID in stringa CSV, gestendo None e duplicati."""
    if not ids:
        return ""
    return ", ".join(str(i) for i in sorted(set(str(i) for i in ids if i)))


def main() -> None:
    parser = argparse.ArgumentParser(description="GraphRAG pipeline")
    parser.add_argument("--config",      default="config.yaml")
    parser.add_argument("--max-queries", type=int, default=None,
                        help="Processa solo le prime N query (default: tutte)")
    parser.add_argument("--task-ids",    nargs="+", default=None,
                        help="Processa solo le Task_ID specificate (es. --task-ids 1 1.1 3.2)")
    parser.add_argument("--visualize",   action="store_true",
                        help="Genera graph.html colorato per tipo di entità")
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_dirs(config)

    # ── PHASE 1: Graph Build ──────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("PHASE 1: Graph Build")
    logger.info("=" * 60)

    builder = GraphBuilder(config)
    graph, entities_df, text_units_df, attributes_df = builder.load()
    logger.info(f"Graph ready → {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges.")

    if args.visualize:
        builder.visualize(graph)

    # ── PHASE 2: Query Processing ─────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("PHASE 2: Query Processing")
    logger.info("=" * 60)

    llm_client = LLMClient(config)

    engine = SearchEngine(
        config        = config,
        llm_client    = llm_client,
        graph         = graph,
        entities_df   = entities_df,
        text_units_df = text_units_df,
    )

    input_path  = config["paths"]["input_excel"]
    output_path = config["paths"]["output_excel"]

    p           = Path(output_path)
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = str(p.parent / f"graphrag_results_{timestamp}.xlsx")
    logger.info(f"Output file: {output_path}")

    queries_df = pd.read_excel(input_path)

    if args.max_queries:
        queries_df = queries_df.head(args.max_queries)
        logger.info(f"Processing first {args.max_queries} queries only.")

    if args.task_ids:
        queries_df["Task_ID"] = queries_df["Task_ID"].astype(str)
        queries_df = queries_df[
            queries_df["Task_ID"].isin(args.task_ids)
        ].reset_index(drop=True)
        logger.info(f"Processing Task_IDs: {args.task_ids} ({len(queries_df)} rows).")

    expected = ["Task_ID", "Variant", "Category", "query", "Gold_Answer", "Gold_Entity_Titles"]
    missing  = [c for c in expected if c not in queries_df.columns]
    if missing:
        logger.warning(f"Colonne mancanti nell'Excel di input: {missing}")

    for col in ("Search_Type", "GraphRAG_Answer", "Status",
                "Cited_Source_IDs", "Cited_Entities_IDs"):
        queries_df[col] = ""

    for idx, row in tqdm(queries_df.iterrows(), total=len(queries_df), desc="Processing queries"):
        query_text = str(row.get("query", "")).strip()

        if not query_text:
            queries_df.at[idx, "Status"] = "Skipped: empty query"
            continue

        result = engine.query(query_text)

        # cited_text_ids   = subgraph text_unit_ids ∪ LLM-cited texts
        # cited_entity_ids = subgraph nodes ∪ LLM-cited entities
        # (unione già calcolata in search_engine._local_search)
        queries_df.at[idx, "Search_Type"]       = result.get("search_type", "")
        queries_df.at[idx, "GraphRAG_Answer"]    = result.get("answer", "")
        queries_df.at[idx, "Status"]             = result.get("status", "")
        queries_df.at[idx, "Cited_Source_IDs"]   = ids_to_str(result.get("cited_text_ids", []))
        queries_df.at[idx, "Cited_Entities_IDs"] = ids_to_str(result.get("cited_entity_ids", []))

        logger.info(
            f"[{idx+1}/{len(queries_df)}] Task={row.get('Task_ID', idx)} | "
            f"Type={result.get('search_type')} | Status={result.get('status')}"
        )

    queries_df.to_excel(output_path, index=False)
    logger.info(f"Results saved → {output_path}")
    logger.info("Pipeline completed.")


if __name__ == "__main__":
    main()