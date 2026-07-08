# LOTO-RO

LOTO-RO is a formal application ontology that turns Lockout/Tagout (LOTO) incident
reports into structured, reusable safety knowledge. This repository contains the
ontology itself, a knowledge graph grounded on it, a domain-agnostic GraphRAG
pipeline used to query that graph in natural language, and the evaluation
results produced against a benchmark of test queries.

## Repository contents

- `LOTO-RO_ontology.xlsx` — the LOTO-RO application ontology (classes, relations,  attributes).
- `LOTO-RO_schema.pdf` — the LOTO-RO application ontology (classes, relations,  attributes) illustred as a graph.
- `LOTO-RO_ Protege.rdf` — the ontology exported from Protégé (OWL/RDF).
- `LOTO-RO_grounded_KG + GT.xlsx` — the knowledge graph grounded on the ontology,
  with gold-standard ("ground truth") annotations used for evaluation.
- `Benchmark.xlsx` — the benchmark query set used to test retrieval and
  question-answering quality.
- `Appendix_LOTORO.pdf` — supplementary material: ontology entity clusters and
  sample competency-question validation.
- `graphrag_LOTORO/` — the GraphRAG pipeline, evaluation notebooks and results used with the LOTO-RO knowledge graph. See
  [`graphrag_LOTORO/graphrag/readme.txt`](graphrag_LOTORO/graphrag/readme.txt)
  for full documentation of the pipeline (architecture, configuration, input/output
  formats, cost estimation).
- `graphrag/` — standalone copy of the GraphRAG pipeline code (no bundled data).

## GraphRAG pipeline (short version)

Given a natural language query, the pipeline retrieves a semantically relevant
subgraph from the LOTO-RO knowledge graph and generates a grounded answer with
inline citations back to source documents and entities. It is domain-agnostic:
adapting it to a different ontology only requires updating the `entity_types`
list in `config.yaml`, no code changes.

```
cd graphrag_LOTORO/graphrag
pip install -r Requirements.txt
python main.py
```

Set your LLM API key via environment variable (`OPENAI_API_KEY` / `GOOGLE_API_KEY`)
— do not hardcode it in `config.yaml`.

## Evaluation

`graphrag_LOTORO/Evaluationresults/` contains the outputs of the GraphRAG
evaluation (retrieval metrics, judge scores, inter-judge agreement) together
with the notebooks that produced them (`GraphRAGEvaluator.ipynb`,
`Grafici.ipynb`).

## Notes

- The Python virtual environment used for development is not included in this
  repository (see `.gitignore`). Recreate it locally with
  `pip install -r graphrag_LOTORO/graphrag/Requirements.txt`.
- Any API keys must be supplied via environment variables, never committed.
