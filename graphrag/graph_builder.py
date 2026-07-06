"""
graph_builder.py
================
Loads the LOTO-RO knowledge graph from four structured Parquet files
(entities, relationships, text_units, attributes) and assembles it as a
NetworkX DiGraph. Duplicate edges sharing the same source/target pair are
merged, with their text_unit_ids aggregated. An optional interactive HTML
visualisation coloured by entity type can be exported via pyvis.

Inputs
------
- entities.parquet       : one row per KG node
- relationships.parquet  : one row per directed edge (may be exploded)
- text_units.parquet     : source narrative chunks with provenance links
- attributes.parquet     : structured attributes attached to nodes (optional)

Outputs
-------
- nx.DiGraph             : in-memory directed weighted graph
- HTML file              : interactive pyvis visualisation (optional)
"""

import logging
from pathlib import Path

import networkx as nx
import pandas as pd

logger = logging.getLogger(__name__)

# Colour palette for entity-type-based node colouring in visualisation
PALETTE = [
    "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F",
    "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC",
    "#FF6347", "#4682B4", "#32CD32", "#FFD700", "#DC143C",
    "#00CED1", "#FF69B4", "#8A2BE2", "#A52A2A", "#5F9EA0",
    "#D2691E", "#FF7F50", "#6495ED", "#00008B", "#008B8B",
    "#B8860B", "#556B2F", "#FF8C00", "#9400D3", "#2E8B57",
    "#FF1493", "#00BFFF", "#696969", "#1E90FF", "#B22222",
    "#FFFAF0", "#228B22", "#FF00FF",
]


def _safe_list(val) -> list:
    """
    Safely converts any value (numpy array, list, None) to a plain Python list.
    Used to normalise the text_unit_ids field, which may arrive in different
    formats depending on how the Parquet file was serialised.
    """
    if val is None:
        return []
    try:
        return list(val)
    except TypeError:
        return []


def _make_network(directed: bool = True) -> "Network":
    """
    Initialises a pyvis Network object with dark background, Barnes-Hut
    physics layout, hover tooltips, and navigation buttons.
    Returns a configured Network instance ready to receive nodes and edges.
    """
    from pyvis.network import Network
    net = Network(
        height="900px", width="100%",
        directed=directed,
        bgcolor="#ffffff",
        font_color="black",
    )
    # Barnes-Hut parameters: gravity spreads nodes apart,
    # spring_length controls default edge length
    net.barnes_hut(gravity=-8000, central_gravity=0.3, spring_length=150)
    net.set_options("""
    {
      "interaction": {
        "hover": true,
        "tooltipDelay": 100,
        "navigationButtons": true
      },
      "physics": {
        "enabled": true,
        "stabilization": {"iterations": 200}
      }
    }
    """)
    return net


class GraphBuilder:
    """
    Builds and visualises the LOTO-RO knowledge graph from Parquet input files.

    Parameters
    ----------
    config : dict
        Configuration dictionary loaded from config.yaml. Must contain:
        - config["paths"]                  : dict of Parquet file paths
        - config["ontology"]["entity_types"]: list of ontological class names
                                             used for colour assignment
    """

    def __init__(self, config: dict):
        self.paths  = config["paths"]
        self.config = config

    def load(self) -> tuple[nx.DiGraph, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Loads all Parquet input files and builds the in-memory directed graph.

        Returns
        -------
        graph           : nx.DiGraph  — the assembled knowledge graph
        entities_df     : pd.DataFrame — raw entity records
        text_units_df   : pd.DataFrame — source narrative chunks
        attributes_df   : pd.DataFrame — node attributes (empty if file absent)
        """
        logger.info("Loading entities.parquet …")
        entities_df = pd.read_parquet(self.paths["entities"])

        logger.info("Loading relationships.parquet …")
        relationships_df = pd.read_parquet(self.paths["relationships"])

        logger.info("Loading text_units.parquet …")
        text_units_df = pd.read_parquet(self.paths["text_units"])

        # attributes are optional: skip gracefully if the file is absent
        attributes_df = pd.DataFrame()
        cov_path = self.paths.get("attributes")
        if cov_path and Path(cov_path).exists():
            logger.info("Loading attributes.parquet …")
            attributes_df = pd.read_parquet(cov_path)
        else:
            logger.info("No attributes file found — skipping.")

        graph = self._build_graph(entities_df, relationships_df, attributes_df)
        return graph, entities_df, text_units_df, attributes_df

    def _build_graph(
        self,
        entities_df     : pd.DataFrame,
        relationships_df: pd.DataFrame,
        attributes_df   : pd.DataFrame,
    ) -> nx.DiGraph:
        """
        Assembles the NetworkX DiGraph from the loaded DataFrames.

        Node attributes stored per node:
            title, type, description, text_unit_ids, frequency, degree, attributes

        Edge attributes stored per edge:
            id, description, weight, combined_degree, text_unit_ids (deduplicated)

        Duplicate edges (same source/target pair) are merged: their
        text_unit_ids are aggregated and deduplicated into a single edge.
        """
        G = nx.DiGraph()

        # Build attribute index: entity_title → list of attribute dicts
        # Allows O(1) lookup of structured attributes when adding nodes
        cov_index: dict[str, list] = {}
        if not attributes_df.empty and "subject_id" in attributes_df.columns:
            for _, row in attributes_df.iterrows():
                cov_index.setdefault(str(row["subject_id"]), []).append(row.to_dict())

        # ── Add nodes ────────────────────────────────────────────────────────
        for _, row in entities_df.iterrows():
            node_id = str(row["id"])
            title   = str(row.get("title", ""))
            G.add_node(
                node_id,
                title         = title,
                type          = str(row.get("type", "")),
                description   = str(row.get("description", "")),
                text_unit_ids = _safe_list(row["text_unit_ids"]),
                frequency     = int(row.get("frequency", 0)),
                degree        = int(row.get("degree", 0)),
                attributes    = cov_index.get(title, []),  # attach structured attributes
            )

        logger.info(f"Nodes added: {G.number_of_nodes()}")

        # ── Add edges — merge duplicate (src, tgt) pairs ─────────────────────
        # The relationships Parquet may be exploded (one row per text_unit),
        # so the same (source, target) pair can appear multiple times.
        # edge_acc accumulates all text_unit_ids before committing a single edge.
        skipped  = 0
        edge_acc: dict[tuple, dict] = {}

        for _, row in relationships_df.iterrows():
            # Support multiple column naming conventions for source/target
            src = str(row.get("Source_id") or row.get("source_id") or row.get("source", ""))
            tgt = str(row.get("Target_id") or row.get("target_id") or row.get("target", ""))

            # Skip edges whose endpoint nodes were not found in the graph
            if src not in G or tgt not in G:
                logger.debug(f"Skipping edge {src} → {tgt}: node(s) missing.")
                skipped += 1
                continue

            key = (src, tgt)
            tu  = _safe_list(row["text_unit_ids"])

            if key not in edge_acc:
                # First occurrence: initialise the accumulator entry
                edge_acc[key] = {
                    "id"             : str(row.get("id", "")),
                    "description"    : str(row.get("description", "")),
                    "weight"         : float(row.get("weight", 1.0)),
                    "combined_degree": int(row.get("combined_degree", 0)),
                    "text_unit_ids"  : [],
                }
            # Accumulate provenance references across duplicate rows
            edge_acc[key]["text_unit_ids"].extend(tu)

        # Commit one edge per (src, tgt) pair with deduplicated text_unit_ids
        for (src, tgt), data in edge_acc.items():
            data["text_unit_ids"] = list(set(data["text_unit_ids"]))
            G.add_edge(src, tgt, **data)

        logger.info(f"Edges added: {G.number_of_edges()}  (skipped: {skipped})")
        return G

    # ── Visualisation ────────────────────────────────────────────────────────

    def visualize(
        self,
        graph      : nx.DiGraph,
        output_path: str = "data/output/graph.html",
        max_nodes  : int = 200,
    ) -> None:
        """
        Exports an interactive HTML visualisation of the graph, with nodes
        coloured by entity type as defined in config.yaml → ontology.entity_types.
        Node size scales with frequency (number of source incidents).
        If the graph exceeds max_nodes, only the highest-degree nodes are shown.

        Parameters
        ----------
        graph       : nx.DiGraph — the knowledge graph to visualise
        output_path : str        — destination path for the HTML file
        max_nodes   : int        — maximum number of nodes to render (default 200)
        """
        try:
            from pyvis.network import Network  # noqa: F401
        except ImportError:
            print("Installa pyvis:  pip install pyvis")
            return

        # Map each entity type to a distinct colour from the palette
        entity_types = self.config.get("ontology", {}).get("entity_types", [])
        type_colors  = {t: PALETTE[i % len(PALETTE)] for i, t in enumerate(entity_types)}

        graph = self._trim(graph, max_nodes)
        net   = _make_network()

        for node_id, data in graph.nodes(data=True):
            entity_type = data.get("type", "")
            tu_str      = ", ".join(str(t) for t in _safe_list(data.get("text_unit_ids")))
            net.add_node(
                node_id,
                label = data.get("title", node_id)[:30],   # truncate long labels
                title = (                                    # HTML tooltip on hover
                    f"<b>{data.get('title', node_id)}</b><br>"
                    f"Type: {entity_type}<br>"
                    f"ID: {node_id}<br>"
                    f"Degree: {data.get('degree', 0)}<br>"
                    f"Frequency: {data.get('frequency', 0)}<br>"
                    f"Text units: {tu_str}<br>"
                ),
                color = type_colors.get(entity_type, "#AAAAAA"),  # grey fallback
                size  = max(10, min(40, data.get("frequency", 1) * 3)),  # clamp [10, 40]
            )

        self._add_edges(net, graph)
        self._save(net, output_path)

    # ── Private helpers ──────────────────────────────────────────────────────

    def _trim(self, graph: nx.DiGraph, max_nodes: int) -> nx.DiGraph:
        """
        Returns a subgraph containing only the top max_nodes nodes by degree.
        Used to keep the visualisation readable for large graphs.
        """
        if graph.number_of_nodes() > max_nodes:
            top_nodes = sorted(
                graph.nodes(), key=lambda n: graph.degree(n), reverse=True
            )[:max_nodes]
            graph = graph.subgraph(top_nodes)
            logger.info(f"Visualising top {max_nodes} nodes by degree.")
        return graph

    def _add_edges(self, net, graph: nx.DiGraph) -> None:
        """
        Adds all edges to the pyvis Network.
        Edge width scales with the ontology-defined relationship weight.
        Tooltip shows the relationship description (truncated to 150 chars).
        """
        for src, tgt, data in graph.edges(data=True):
            net.add_edge(
                src, tgt,
                title = data.get("description", "")[:150],
                color = "#888888",
                width = max(1, float(data.get("weight", 1.0))),
            )

    def _save(self, net, output_path: str) -> None:
        """
        Creates the output directory if needed and saves the pyvis graph
        as a self-contained HTML file openable in any browser.
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        net.save_graph(output_path)
        logger.info(f"Graph saved → {output_path}  (open in browser)")