================================================================================
GraphRAG Pipeline — Custom Knowledge Graph Retrieval and Answer Generation
================================================================================

A domain-agnostic GraphRAG pipeline for structured retrieval and answer
generation over any ontology-grounded knowledge graph. Given a natural
language query, the system retrieves a semantically relevant subgraph and
generates a grounded, fully traceable answer with inline citations to source
documents.

Designed to operationalise any application ontology as a queryable knowledge
infrastructure. The only configuration needed to adapt the pipeline to a
different ontology is updating the entity_types list in config.yaml — no
code changes required.


================================================================================
QUICK START
================================================================================

1. Clone the repository

    git clone https://github.com/your-username/graphrag-pipeline.git
    cd graphrag-pipeline

2. Install dependencies

    pip install -r requirements.txt

3. Set your OpenAI API key in config.yaml

4. Place your input files

    data/
    ├── input/
    │   ├── entities.parquet
    │   ├── relationships.parquet
    │   ├── text_units.parquet
    │   ├── attributes.parquet      (optional)
    │   └── lancedb/                (auto-generated at first run)
    └── Benchmark.xlsx

5. Estimate costs before running

    python cost_estimator.py --queries 150

6. Run the pipeline

    python main.py

    Results are saved to data/output/graphrag_results_<timestamp>.xlsx.

Optional CLI flags:

    --config PATH          Use a different config file (default: config.yaml)
    --max-queries N        Process only the first N queries (for testing)
    --task-ids 1 1.1 2.3   Process only the specified Task_IDs
    --visualize            Export an interactive HTML graph visualisation


================================================================================
PROJECT OVERVIEW
================================================================================

The pipeline executes three sequential steps for each natural language query:

    Natural language query
            │
            ▼
    ┌─────────────────────┐
    │  Step A             │   Semantic parser (report_model)
    │  Query parsing      │   Maps query → entity types, node titles,
    │                     │   attribute filters (structured JSON)
    └──────────┬──────────┘
               │
               ▼
    ┌─────────────────────┐
    │  Step B             │   Four retrieval strategies:
    │  Subgraph           │   type-only / specific / type+specific /
    │  construction       │   vector fallback + hop expansion
    └──────────┬──────────┘
               │
               ▼
    ┌─────────────────────┐
    │  Step C             │   Answer model (answer_model)
    │  Answer             │   Grounded response with inline [tN]/[eN]
    │  generation         │   citations. Provenance completed
    │                     │   architecturally.
    └─────────────────────┘
               │
               ▼
    JSON: answer + cited_entity_ids + cited_text_ids


Key design principles:

  - Domain-agnostic: adapts to any ontology via config.yaml
  - Ontology-grounded: all graph structure follows the input ontology
  - Full provenance: every answer is traceable to source documents
    and entity nodes
  - Nothing hardcoded: all entity types, model names, and parameters
    are read from config.yaml
  - Robust: exponential backoff retry on all LLM calls; automatic
    fallback to vector search if structured retrieval fails
  - Reproducible: temperature 0.0 for deterministic outputs


================================================================================
PROJECT STRUCTURE
================================================================================

    graphrag-pipeline/
    │
    ├── main.py                  Pipeline entry point
    ├── graph_builder.py         Parquet loading, graph assembly, visualisation
    ├── search_engine.py         Retrieval and answer generation engine
    ├── llm_client.py            OpenAI API wrapper (chat + embeddings)
    ├── cost_estimator.py        API cost estimation before running
    │
    ├── config.yaml              Pipeline configuration (do not commit with real key)
    ├── config_template.yaml     Safe template to commit
    ├── requirements.txt         Python dependencies
    │
    ├── data/
    │   ├── input/
    │   │   ├── entities.parquet
    │   │   ├── relationships.parquet
    │   │   ├── text_units.parquet
    │   │   ├── attributes.parquet       (optional)
    │   │   └── lancedb/                 (auto-generated vector index)
    │   ├── output/
    │   │   └── graphrag_results_<timestamp>.xlsx
    │   └── Benchmark.xlsx
    │
    └── README.txt

Module responsibilities:

    ┌──────────────────────┬────────────────────────────────────────────────────┐
    │ Module               │ Responsibility                                     │
    ├──────────────────────┼────────────────────────────────────────────────────┤
    │ main.py              │ CLI, config loading, phase orchestration, Excel I/O│
    │ graph_builder.py     │ Parquet loading, graph assembly, edge              │
    │                      │ deduplication, HTML visualisation                  │
    │ search_engine.py     │ In-memory index building, semantic parsing,        │
    │                      │ subgraph construction, answer generation,          │
    │                      │ provenance completion                              │
    │ llm_client.py        │ OpenAI API calls (chat + embeddings), retry with  │
    │                      │ exponential backoff                                │
    │ cost_estimator.py    │ Token and USD cost estimation from config +        │
    │                      │ actual entity count from Parquet                   │
    └──────────────────────┴────────────────────────────────────────────────────┘


================================================================================
INPUT STRUCTURE
================================================================================

All four Parquet files must be placed in data/input/.
They are read-only — the pipeline never modifies them.

--------------------------------------------------------------------------------
entities.parquet — one row per KG node
--------------------------------------------------------------------------------

    ┌──────────────┬──────────────┬──────────┬─────────────────────────────────┐
    │ Column       │ Type         │ Required │ Description                     │
    ├──────────────┼──────────────┼──────────┼─────────────────────────────────┤
    │ id           │ string       │ YES      │ Unique node identifier           │
    │ title        │ string       │ YES      │ Human-readable node name        │
    │ type         │ string       │ YES      │ Ontological class name          │
    │              │              │          │ (must match ontology.entity_    │
    │              │              │          │ types in config.yaml)           │
    │ description  │ string       │ YES      │ Natural language description    │
    │ text_unit_ids│ list[string] │ YES      │ Source document IDs this entity │
    │              │              │          │ appears in                      │
    │ frequency    │ int          │ YES      │ Number of source records in     │
    │              │              │          │ which the entity appears        │
    │              │              │          │ (pre-computed at indexing time) │
    │ degree       │ int          │ YES      │ Number of edges connected to    │
    │              │              │          │ this node                       │
    └──────────────┴──────────────┴──────────┴─────────────────────────────────┘

--------------------------------------------------------------------------------
relationships.parquet — one row per directed edge (may be exploded)
--------------------------------------------------------------------------------

    ┌─────────────────┬──────────────┬──────────┬─────────────────────────────┐
    │ Column          │ Type         │ Required │ Description                 │
    ├─────────────────┼──────────────┼──────────┼─────────────────────────────┤
    │ id              │ string       │ YES      │ Unique edge identifier      │
    │ source_id       │ string       │ YES      │ ID of the source node       │
    │                 │              │          │ (also: Source_id, source)   │
    │ target_id       │ string       │ YES      │ ID of the target node       │
    │                 │              │          │ (also: Target_id, target)   │
    │ description     │ string       │ YES      │ Relationship description    │
    │ weight          │ float        │ YES      │ Edge weight                 │
    │ combined_degree │ int          │ YES      │ Sum of endpoint node degrees│
    │ text_unit_ids   │ list[string] │ YES      │ Source document IDs         │
    │                 │              │          │ supporting this relationship │
    └─────────────────┴──────────────┴──────────┴─────────────────────────────┘

    NOTE: if the same edge appears on multiple rows (one per text_unit_id),
    the pipeline automatically merges them into a single edge with all
    text_unit_ids aggregated and deduplicated.

--------------------------------------------------------------------------------
text_units.parquet — one row per source document chunk
--------------------------------------------------------------------------------

    ┌──────────┬────────┬──────────┬──────────────────────────────────────────┐
    │ Column   │ Type   │ Required │ Description                              │
    ├──────────┼────────┼──────────┼──────────────────────────────────────────┤
    │ id       │ string │ YES      │ Unique identifier (referenced by nodes   │
    │          │        │          │ and edges via text_unit_ids)             │
    │ text     │ string │ YES      │ Full text of the source document chunk   │
    └──────────┴────────┴──────────┴──────────────────────────────────────────┘

--------------------------------------------------------------------------------
attributes.parquet — structured attributes attached to nodes (OPTIONAL)
--------------------------------------------------------------------------------

    ┌──────────────────┬────────┬──────────┬──────────────────────────────────┐
    │ Column           │ Type   │ Required │ Description                      │
    ├──────────────────┼────────┼──────────┼──────────────────────────────────┤
    │ id               │ string │ YES      │ Unique attribute identifier      │
    │ subject_id       │ string │ YES      │ Title of the entity node this    │
    │                  │        │          │ attribute belongs to             │
    │ type /           │ string │ YES      │ Attribute type label             │
    │ attribute_type   │        │          │                                  │
    │ description      │ string │ YES      │ Attribute value as natural       │
    │                  │        │          │ language string                  │
    └──────────────────┴────────┴──────────┴──────────────────────────────────┘

    If attributes.parquet is absent or its path is not set in config.yaml,
    the pipeline continues without error. Attribute-based filtering and
    inline attribute display will simply be unavailable.

--------------------------------------------------------------------------------
Benchmark.xlsx — input query list
--------------------------------------------------------------------------------

    ┌─────────────────────┬──────────┬──────────────────────────────────────────┐
    │ Column              │ Required │ Description                              │
    ├─────────────────────┼──────────┼──────────────────────────────────────────┤
    │ Task_ID             │ YES      │ Query identifier, supports decimals      │
    │                     │          │ (e.g. 1, 1.1, 2.3)                      │
    │ Variant             │ YES      │ Formulation variant: 0 = original,       │
    │                     │          │ 1/2 = rewordings, 3 = prompt-engineered  │
    │ Category            │ YES      │ Task type: Information Retrieval,        │
    │                     │          │ Summarisation, Classification, Reasoning,│
    │                     │          │ Comparison, Recommendation               │
    │ query               │ YES      │ Natural language query text              │
    │ Gold_Answer         │ NO       │ Expert-validated reference answer        │
    │                     │          │ (used by evaluation)                     │
    │ Gold_Entity_Titles  │ NO       │ Expert-annotated entity titles           │
    │                     │          │ (used for retrieval precision/recall)    │
    └─────────────────────┴──────────┴──────────────────────────────────────────┘


================================================================================
OUTPUT STRUCTURE
================================================================================

The pipeline saves results to:
    data/output/graphrag_results_<YYYYMMDD_HHMMSS>.xlsx

The output file contains all input columns plus the following fields:

    ┌──────────────────────┬──────────────────────────────────────────────────┐
    │ Column               │ Description                                      │
    ├──────────────────────┼──────────────────────────────────────────────────┤
    │ Search_Type          │ Retrieval strategy used (Local)                  │
    │ GraphRAG_Answer      │ Generated answer with inline [tN]/[eN] citations │
    │ Status               │ Success / Skipped: empty query / Error: <msg>    │
    │ Cited_Source_IDs     │ Deduplicated, sorted CSV of all source document  │
    │                      │ IDs used in retrieval and generation             │
    │ Cited_Entities_IDs   │ Deduplicated, sorted CSV of all entity IDs       │
    │                      │ consulted during retrieval and reasoning         │
    └──────────────────────┴──────────────────────────────────────────────────┘

    NOTE: Cited_Source_IDs and Cited_Entities_IDs are the architectural union
    of LLM-reported IDs and all IDs present in the retrieved subgraph.
    This guarantees complete traceability independently of LLM self-reporting:
    no entity involved in retrieval or reasoning can be omitted.

    Output files are timestamped to prevent accidental overwrites and to
    preserve a full run history across multiple executions.


================================================================================
CONFIGURATION
================================================================================

All pipeline behaviour is controlled by config.yaml, structured in five
sections. No values are hardcoded in the pipeline modules.

    paths       input/output file locations
    pipeline    which phases to run on startup
    llm         model names, token limits, retry settings
    embeddings  batch size and vector store backend
    query       retrieval parameters (hops, context size, top-k)
    ontology    entity_types — single source of truth for the entire pipeline

Critical settings:

    ┌───────────────────────────┬──────────────────────────┐
    │ Key                       │ Effect                   │
    ├───────────────────────────┼──────────────────────────┤
    │ llm.answer_model          │ Model for answer         │
    │                           │ generation (Step C)      │
    │ llm.report_model          │ Model for semantic       │
    │                           │ parsing (Step A)         │
    │ llm.temperature           │ Deterministic output for │
    │                           │ reproducible evaluation  │
    │ llm.max_retries           │ Retry attempts on API    │
    │                           │ or JSON errors           │
    │ query.local_search_hops   │ Hop expansion for        │
    │                           │ structured strategies    │
    │ query.vector_fallback_hops│ Hop expansion for        │
    │                           │ vector fallback          │
    │ query.top_k_entities      │ Candidates from vector   │
    │                           │ fallback search          │
    │ ontology.entity_types     │ Single source of truth   │
    │                           │ for all ontology types   │
    └───────────────────────────┴──────────────────────────┘

Adapting to a different ontology:

    1. Replace the Parquet files with your KG data
    2. Update ontology.entity_types in config.yaml with your class names
    3. No code changes required


================================================================================
COST ESTIMATION
================================================================================

Before running the full benchmark, estimate the API cost:

    python cost_estimator.py --queries 150

The estimator automatically:
  - reads model names from config.yaml
  - counts entities from the actual entities.parquet file
  - computes per-query and total USD cost for all three call types:
    Step A (report_model), Step C (answer_model), embeddings (one-time)

To add a new model, update the PRICING dict at the top of cost_estimator.py:

    PRICING = {
        "your-model-name": {
            "input":  X.XX,    # USD per 1M input tokens
            "output": X.XX,    # USD per 1M output tokens
        },
        ...
    }

Current prices at: https://openai.com/pricing

================================================================================
DEPENDENCIES
================================================================================

    ┌─────────────────────────┬───────────────────────────────────────────────┐
    │ Package                 │ Purpose                                       │
    ├─────────────────────────┼───────────────────────────────────────────────┤
    │ pandas>=2.0.0           │ DataFrame operations and Excel I/O            │
    │ pyarrow>=14.0.0         │ Parquet file reading                          │
    │ numpy>=1.24.0           │ Numerical operations                          │
    │ networkx>=3.0           │ In-memory directed graph                      │
    │ pyvis>=0.3.2            │ Interactive HTML graph visualisation          │
    │ lancedb>=0.6.0          │ Vector store for embedding fallback           │
    │ openai>=1.30.0          │ Chat completions and embeddings API           │
    │ pyyaml>=6.0             │ YAML config parsing                           │
    │ openpyxl>=3.1.0         │ Excel read/write                              │
    │ tqdm>=4.66.0            │ Progress bar for query loop                   │
    └─────────────────────────┴───────────────────────────────────────────────┘

Install all dependencies:

    pip install -r requirements.txt


================================================================================
CITATION
================================================================================

If you use this pipeline in your research, please cite:

    @article{your-paper-2025,
      title   = {Your Paper Title},
      author  = {Your Name et al.},
      journal = {Safety Science},
      year    = {2025},
    }

================================================================================