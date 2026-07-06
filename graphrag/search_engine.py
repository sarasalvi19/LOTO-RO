"""
search_engine.py
================
Core retrieval and answer generation engine of the GraphRAG pipeline.
Given a natural language query, executes three sequential steps to produce
a grounded, fully traceable answer from the LOTO-RO knowledge graph.
All vocabulary and prompts are built at init from live graph data — nothing hardcoded.

INITIALISATION (once, reused across all queries):
  - text_units DataFrame re-indexed by id for O(1) lookup during answer generation
  - LanceDB connection opened for vector-search fallback
  - Three in-memory graph indices built in a single graph pass:
      · type → [node_ids]                          (type index)
      · lowercase title → [{node_id, type}]         (title index)
      · node_id → concatenated attribute string     (attribute index, for substring matching)
  - Answer system prompt built once: entity types injected from config.yaml
  - Parser prompt built once: entity types from config.yaml; representative
    titles (max 20/type) and attribute values (max 15/field) sampled directly
    from the graph, excluding null-like values (none, n/a, unknown, nan)

STEP A — Semantic parsing:
  A single LLM call (report_model, max 200 tokens) maps the natural language
  query to the graph vocabulary, returning a structured JSON with three fields:
    · entity_types      → resolved against type index     → type_ids
    · specific_titles   → resolved against title index    → specific_ids
    · attribute_filters → resolved via substring match    → specific_ids
  All lookups are case-insensitive. Parser failure (any cause) returns empty
  lists, silently triggering the vector-search fallback in Step B.

STEP B — Subgraph construction:
  Strategy selected based on Step A output:
    · Type only       : all nodes of that type, no hop expansion
                        (distributional queries require full class population;
                        hop expansion would add irrelevant neighbours)
    · Specific only   : matched nodes as seeds, expanded with local_search_hops
                        (brings related nodes — causes, consequences, actors —
                        into context for relational queries)
    · Type + specific : intersection of type_ids and specific_ids as seeds,
                        expanded with local_search_hops; if intersection is
                        empty, specific nodes used as seeds (type signal
                        used for routing only)
    · Vector fallback : query embedded via OpenAI Embeddings API, top-k nodes
                        retrieved from LanceDB, expanded with vector_fallback_hops
                        (more hops than structured strategies to compensate for
                        lower matching precision)
  After subgraph construction, all text_unit_ids from nodes and edges are
  collected, plus any IDs directly referenced in the query string (\\bt\\d+\\b),
  preserving the full provenance chain:
  source narrative → entity/relationship → subgraph node/edge → LLM context window

STEP C — Answer generation:
  Subgraph serialised to structured plain text (ENTITIES section: id, title,
  type, frequency, source records, inline attributes; RELATIONSHIPS section:
  source → target, source records, description) and passed to the answer model
  with source narrative chunks. The model is instructed to: cite every factual
  claim with inline [tN]/[eN] references; use the pre-computed frequency field
  for all quantitative claims; restrict output strictly to provided context.
  Model returns JSON: answer, cited_text_ids, cited_entity_ids.
  Provenance output:
    · cited_entity_ids = LLM-reported entity IDs only (entities actually
                         consulted to build the answer — counts, comparisons,
                         citations, reasoning — not all subgraph nodes)
    · cited_text_ids   = LLM-cited text IDs ∪ text_unit_ids of cited entities

Inputs
  config dict   : paths, llm settings, query parameters, ontology.entity_types
  nx.DiGraph    : in-memory knowledge graph (from GraphBuilder)
  entities_df   : raw entity DataFrame
  text_units_df : source narrative chunks DataFrame
  LLMClient     : shared API client instance (from llm_client.py)

Outputs
  dict: answer, cited_text_ids, cited_entity_ids, search_type, status
"""

import logging
import re

import lancedb
import networkx as nx
import pandas as pd

from llm_client import LLMClient

logger = logging.getLogger(__name__)

# ── Answer generation system prompt ──────────────────────────────────────────
# Ontological entity types are injected at runtime from config.yaml
# so the prompt always reflects the actual ontology without hardcoded values.

_ANSWER_SYSTEM_TEMPLATE = """\
You are an AI assistant specialised in knowledge graph analysis.

You receive:
  1. A SUBGRAPH — entities and their relationships.
     Each entity: [id] title (type: X, records: t1, t5, ...)
     attributes (structured attributes) are shown inline with each entity.
  2. TEXT SOURCES — original text chunks labelled [tN].

ENTITY TYPE RULES — use ONLY the correct entity type for each question.
Available entity types:
{entity_type_rules}

CRITICAL RULE — ONE ENTITY = ONE RECORD:
  Each entity belongs to specific records listed in its (records: ...) field.
  Two entities from different records are NOT co-occurring.

COUNTING RULES:
  • Each entity has a frequency field — use it directly as the count for that entity.
  • Do NOT count record IDs manually — the frequency is already computed and reliable.
  • Always provide: absolute count (from frequency), percentage over total, list of record IDs.
  • For rankings: full ranked list with frequency and % for every item.
  • For comparisons: structured breakdown per category with frequency and %.
  • Never give a number without its percentage, never a % without its count.

STRICT TRACEABILITY RULES:
  • Answer ONLY from the provided context. Never use external knowledge.
  • Every factual claim MUST end with [tN] or [eN].
  • Deductions not in sources → label [INFERENCE].
  • Insufficient context → state exactly what was NOT found.

CITATION RULES:
  cited_entity_ids MUST include ONLY entities of the ontological type(s)
  explicitly requested by the query or necessary to directly answer it.

  Step 1 — Identify which entity type(s) the query is asking about
           (e.g. EnergySource, Plant, Hazard, Consequence, WorkActivity).
  Step 2 — Include only nodes of those type(s) that appear in your answer.
  Step 3 — Exclude all other nodes, even if they were in the subgraph.

  Symmetry rule: every [eN] tag in your answer MUST appear in
  cited_entity_ids, and every ID in cited_entity_ids MUST have a
  corresponding [eN] tag in your answer. These two sets must be identical.
  
Respond ONLY with valid JSON — no preamble, no markdown fences:
{{
  "answer":            "<answer with counts, %, and inline [SOURCE_ID] citations>",
  "cited_text_ids":    ["<id1>", "<id2>", ...],
  "cited_entity_ids":  ["<every eN consulted>"]
}}"""

_ANSWER_USER = """\
Query: {query}

=== SUBGRAPH ===
{subgraph_context}

=== TEXT SOURCES ===
{text_sources}"""


class SearchEngine:

    def __init__(
        self,
        config        : dict,
        llm_client    : LLMClient,
        graph         : nx.DiGraph,
        entities_df   : pd.DataFrame,
        text_units_df : pd.DataFrame,
    ):
        self.config = config
        self.llm    = llm_client
        self.graph  = graph
        self.qc     = config.get("query", {})

        # Index text_units by id for fast lookup during answer generation
        self.text_units = (
            text_units_df.set_index("id")
            if "id" in text_units_df.columns
            else text_units_df
        )

        # Connect to LanceDB for vector-search fallback
        self.db = lancedb.connect(config["paths"]["vector_db"])

        # Entity types: single source of truth from config.yaml → ontology.entity_types
        ontology_types: list[str] = config.get("ontology", {}).get("entity_types", [])

        # Inject entity types into the answer system prompt once at init
        self._answer_system_prompt = _ANSWER_SYSTEM_TEMPLATE.format(
            entity_type_rules="\n".join(f"  • {t}" for t in ontology_types)
        )

        # ── Build in-memory graph indices ─────────────────────────────────────

        # type → [node_ids]: used to resolve entity_types from parser output
        self._type_index: dict[str, list[str]] = {}
        for nid, data in self.graph.nodes(data=True):
            self._type_index.setdefault(data.get("type", ""), []).append(nid)

        # lowercase type → original type: enables case-insensitive type lookup
        self._type_lower_map: dict[str, str] = {
            t.lower(): t for t in self._type_index
        }

        # lowercase title → [{node_id, type}]: used to resolve specific_titles
        self._title_to_info: dict[str, list[dict]] = {}
        for nid, data in self.graph.nodes(data=True):
            title = data.get("title", "").lower()
            if title:
                self._title_to_info.setdefault(title, []).append({
                    "node_id": nid,
                    "type":    data.get("type", ""),
                })

        # node_id → concatenated attribute values string (lowercase):
        # used for substring matching against attribute_filters from parser
        self._node_cov_str: dict[str, str] = {}
        for nid, data in self.graph.nodes(data=True):
            parts = []
            for cov in (data.get("attributes") or []):
                parts.extend(str(v) for v in cov.values())
            self._node_cov_str[nid] = " ".join(parts).lower()

        # Build the semantic parser prompt once from live graph vocabulary
        self._query_parser_prompt = self._build_parser_prompt(ontology_types)

    # ── Public entry point ────────────────────────────────────────────────────

    def query(self, query_text: str) -> dict:
        """
        Main entry point. Executes the full search pipeline for a single
        natural language query and returns a structured result dict.

        Returns
        -------
        dict with keys:
            answer           : str   — grounded natural language response
            cited_text_ids   : list  — source narrative IDs used
            cited_entity_ids : list  — entity IDs consulted
            search_type      : str   — always 'Local'
            status           : str   — 'Success' or 'Error: <message>'
        """
        try:
            result = self._local_search(query_text)
            result["search_type"] = "Local"
            result["status"]      = "Success"
            return result
        except Exception as exc:
            logger.error(f"Query failed: {exc}", exc_info=True)
            return {
                "search_type":      "Local",
                "answer":           "",
                "cited_text_ids":   [],
                "cited_entity_ids": [],
                "status":           f"Error: {exc}",
            }

    # ── Search pipeline ───────────────────────────────────────────────────────

    def _local_search(self, query_text: str) -> dict:
        """
        Executes the three-step search pipeline:
        Step A — semantic parsing (LLM call via report model)
        Step B — subgraph construction (index lookups + hop expansion)
        Step C — answer generation (LLM call via answer model)
        """
        ctx_size    = self.qc.get("context_size",         10_000)
        src_size    = self.qc.get("text_source_size",     10_000)
        max_tok     = self.qc.get("max_tokens",           self.config["llm"]["max_tokens"])
        hops        = self.qc.get("local_search_hops",    1)
        vector_hops = self.qc.get("vector_fallback_hops", 2)
        top_k       = self.qc.get("top_k_entities",       50)

        # Step A: resolve query to graph vocabulary via semantic parser
        type_ids, specific_ids = self._identify_entities(query_text)
        is_type     = bool(type_ids)
        is_specific = bool(specific_ids)

        # ── Step B: subgraph construction ─────────────────────────────────────
        if is_type and is_specific:
            filtered = set(type_ids) & set(specific_ids)
            if filtered:
                subgraph_nodes = self._expand(filtered, hops)
                logger.info(
                    f"Type+specific: {len(filtered)} intersection nodes → "
                    f"{len(subgraph_nodes)} nodes ({hops}-hop)"
                )
            else:
                subgraph_nodes = self._expand(specific_ids, hops)
                logger.info(
                    f"Type+specific intersection empty → specific fallback: "
                    f"{len(specific_ids)} seed → {len(subgraph_nodes)} nodes ({hops}-hop)"
                )

        elif is_specific:
            subgraph_nodes = self._expand(specific_ids, hops)
            logger.info(f"Specific: {len(specific_ids)} seed → {len(subgraph_nodes)} nodes ({hops}-hop)")

        elif is_type:
            subgraph_nodes = set(type_ids)
            logger.info(f"Type-only: {len(subgraph_nodes)} nodes (no hop)")

        else:
            query_vec  = self.llm.embed([query_text])[0]
            hits       = self.db.open_table("entities").search(query_vec).limit(top_k).to_pandas()
            vector_ids = hits["id"].tolist()
            subgraph_nodes = self._expand(vector_ids, vector_hops)
            logger.info(
                f"Vector fallback: {len(vector_ids)} seed → "
                f"{len(subgraph_nodes)} nodes ({vector_hops}-hop)"
            )

        subgraph = self.graph.subgraph(subgraph_nodes)
        logger.info(f"Subgraph: {subgraph.number_of_nodes()} nodes, {subgraph.number_of_edges()} edges")

        # ── Collect text_unit_ids from all subgraph nodes and edges ───────────
        tu_ids: set[str] = set()
        for _, data in subgraph.nodes(data=True):
            tu_ids.update(str(t) for t in (data.get("text_unit_ids") or []))
        for _, _, data in subgraph.edges(data=True):
            tu_ids.update(str(t) for t in (data.get("text_unit_ids") or []))
        # Also extract any text unit IDs directly referenced in the query string
        tu_ids.update(re.findall(r'\bt\d+\b', query_text))
        logger.info(f"Text units: {sorted(tu_ids)}")

        # ── Step C: answer generation ─────────────────────────────────────────
        result = self.llm.chat(
            system_prompt = self._answer_system_prompt,
            user_prompt   = _ANSWER_USER.format(
                query            = query_text,
                subgraph_context = self._subgraph_to_text(subgraph)[:ctx_size],
                text_sources     = self._get_text_units(list(tu_ids))[:src_size],
            ),
            model_role  = "answer",
            expect_json = True,
            max_tokens  = max_tok,
        )

        # ── Entity and text provenance: LLM-reported citations only ──────────
        # cited_entity_ids: only what the LLM explicitly reported consulting
        # to build the answer (counts, comparisons, citations, reasoning).
        # We do NOT union with all subgraph nodes — that would include entities
        # passed as context but not actually used, polluting retrieval evaluation.
        cited_entity_ids = sorted(set(result.get("cited_entity_ids", [])))

        # cited_text_ids: union of LLM-cited texts and text_unit_ids
        # structurally linked to the cited entities (not all subgraph text units).
        cited_entity_set   = set(cited_entity_ids)
        tu_from_cited: set[str] = set()
        for nid in cited_entity_set:
            if nid in subgraph:
                tu_from_cited.update(
                    str(t) for t in (subgraph.nodes[nid].get("text_unit_ids") or [])
                )
        llm_text_ids   = set(result.get("cited_text_ids", []))
        cited_text_ids = sorted(llm_text_ids | tu_from_cited)

        return {
            "answer":           result.get("answer", ""),
            "cited_text_ids":   cited_text_ids,
            "cited_entity_ids": cited_entity_ids,
        }

    # ── Step A: semantic query parser ─────────────────────────────────────────

    def _build_parser_prompt(self, ontology_types: list[str]) -> str:
        """
        Builds the semantic parser system prompt once at initialisation.
        Vocabulary (entity types, representative titles, attribute values)
        is extracted directly from the loaded graph — nothing is hardcoded.
        Entity types come from config.yaml → ontology.entity_types.
        """
        type_block = "\n".join(f"  - {t}" for t in ontology_types)

        titles_by_type: dict[str, list[str]] = {}
        for _, data in self.graph.nodes(data=True):
            t     = data.get("type", "")
            title = data.get("title", "")
            if t and title and title not in titles_by_type.get(t, []):
                titles_by_type.setdefault(t, []).append(title)
        title_block = "".join(
            f"  {t}: {', '.join(titles[:20])}\n"
            for t, titles in sorted(titles_by_type.items())
        )

        cov_values: dict[str, set[str]] = {}
        for _, data in self.graph.nodes(data=True):
            for cov in (data.get("attributes") or []):
                for k, v in cov.items():
                    sv = str(v).strip()
                    if sv and sv.lower() not in ("none", "n/a", "unknown", "nan", ""):
                        cov_values.setdefault(k, set()).add(sv)
        cov_block = "".join(
            f"  {field}: {', '.join(sorted(vals)[:15])}\n"
            for field, vals in sorted(cov_values.items())
        )

        return f"""\
You are a semantic parser for a knowledge graph.
Given a natural language query, identify what the query is looking for
in terms of the graph's vocabulary and return a structured JSON.

ENTITY TYPES:
{type_block}

ENTITY TITLES (grouped by type):
{title_block}
ATTRIBUTE VALUES (grouped by field):
{cov_block}
Return:
- entity_types:      types conceptually needed to answer the query
- specific_titles:   exact node titles explicitly or implicitly referenced
- attribute_filters: specific data values mentioned (IDs, dates, codes, names, etc.)

Match by meaning, not exact wording. Handle synonyms, plurals, paraphrases.
Convert dates to ISO format (YYYY-MM-DD) if needed.
If nothing matches, return empty lists.

Respond ONLY with valid JSON, no preamble, no markdown:
{{"entity_types": [...], "specific_titles": [...], "attribute_filters": [...]}}"""

    def _identify_entities(self, query_text: str) -> tuple[list[str], list[str]]:
        """
        Executes Step A: single LLM call (report model) that semantically
        matches the query against the graph vocabulary.
        Resolves the three parser output fields against the in-memory indices
        and returns (type_ids, specific_ids) ready for subgraph construction.
        Falls back to empty lists (triggering vector search) if the call fails.
        """
        try:
            parsed = self.llm.chat(
                system_prompt = self._query_parser_prompt,
                user_prompt   = query_text,
                model_role    = "report",
                expect_json   = True,
                max_tokens    = 200,
            )
        except Exception as exc:
            logger.warning(f"Query parser failed: {exc} — falling back to vector search")
            return [], []

        type_ids     : set[str] = set()
        specific_ids : set[str] = set()

        # Resolve entity_types → node IDs via type index (case-insensitive)
        for t in parsed.get("entity_types", []):
            if not isinstance(t, str):
                continue
            real = self._type_lower_map.get(t.lower(), t)
            ids  = self._type_index.get(real, [])
            type_ids.update(ids)
            logger.info(f"Parser type '{t}': {len(ids)} nodes")

        # Resolve specific_titles → node IDs via title index (case-insensitive)
        for title in parsed.get("specific_titles", []):
            if not isinstance(title, str):
                continue
            matches = self._title_to_info.get(title.lower(), [])
            specific_ids.update(m["node_id"] for m in matches)
            logger.info(f"Parser title '{title}': {len(matches)} node(s)")

        # Resolve attribute_filters → node IDs via substring match on attribute strings
        for value in parsed.get("attribute_filters", []):
            if not isinstance(value, str):
                continue
            vl      = value.lower()
            matched = [nid for nid, cs in self._node_cov_str.items() if vl in cs]
            specific_ids.update(matched)
            logger.info(f"Parser attribute '{value}': {len(matched)} node(s)")
 
        logger.info(
            f"Parser → entity_types={parsed.get('entity_types',[])} "
            f"titles={parsed.get('specific_titles',[])} "
            f"filters={parsed.get('attribute_filters',[])} "
            f"| {len(type_ids)} type nodes, {len(specific_ids)} specific nodes"
        )
        return list(type_ids), list(specific_ids)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _expand(self, seed_ids: list[str] | set[str], hops: int) -> set[str]:
        """
        Expands a set of seed node IDs by traversing up to `hops` steps
        in the graph. Returns the union of all reachable nodes within the hop limit.
        """
        nodes = set(seed_ids)
        for nid in seed_ids:
            if nid in self.graph:
                nodes.update(
                    nx.single_source_shortest_path_length(self.graph, nid, cutoff=hops).keys()
                )
        return nodes

    def _subgraph_to_text(self, subgraph: nx.DiGraph) -> str:
        """
        Serialises the subgraph to structured plain text for the LLM context window.
        Format: one line per entity with id, title, type, frequency, source records,
        and inline attributes; followed by one line per relationship.
        """
        lines = ["ENTITIES:"]
        for nid, data in subgraph.nodes(data=True):
            tu_str  = ", ".join(str(t) for t in (data.get("text_unit_ids") or []))
            covs    = data.get("attributes") or []
            cov_str = ""
            if covs:
                cov_items = [
                    f"{c.get('type') or c.get('attribute_type','?')}"
                    f"={c.get('description','')[:80]}"
                    for c in covs
                ]
                cov_str = " | " + "; ".join(cov_items)
            lines.append(
                f"  [{nid}] {data.get('title', nid)}"
                f" (type: {data.get('type','N/A')},"
                f" frequency: {data.get('frequency', 0)},"
                f" records: {tu_str})"
                f"{cov_str}: {data.get('description','N/A')}"
            )
        lines.append("\nRELATIONSHIPS:")
        for u, v, data in subgraph.edges(data=True):
            tu_str = ", ".join(str(t) for t in (data.get("text_unit_ids") or []))
            lines.append(
                f"  [{data.get('id','')}] "
                f"{subgraph.nodes[u].get('title',u)} → {subgraph.nodes[v].get('title',v)}"
                f" (records: {tu_str}): {data.get('description','N/A')}"
            )
        return "\n".join(lines)

    def _get_text_units(self, tu_ids: list[str]) -> str:
        """
        Retrieves and formats source narrative chunks for the LLM context window.
        Each chunk is prefixed with its ID for inline citation traceability.
        Missing IDs are silently skipped with a debug log entry.
        """
        lines = []
        for tid in sorted(tu_ids):
            try:
                lines.append(f"[{tid}]: {self.text_units.loc[tid]['text']}")
            except KeyError:
                logger.debug(f"text_unit_id not found: {tid}")
        return "\n\n".join(lines)