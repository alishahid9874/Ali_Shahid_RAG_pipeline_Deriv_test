# Mini RAG Pipeline

A replayable, deterministic Retrieval-Augmented Generation pipeline that ingests a product knowledge base, indexes it with BM25, answers user questions with citation-strict extractive responses, and evaluates retrieval quality.

No LLM is used. All stages are deterministic code.

---

## Quick Start

```bash
# 1. Install dependencies
make install

# 2. Run the full pipeline
make run

# 3. Validate all artifacts
make validate
```

Or run all three steps at once:

```bash
make pipeline
```

---

## Pipeline Stages

```
INIT
 -> DOCUMENTS_LOADED
 -> DOCUMENTS_CHUNKED
 -> INDEX_BUILT
 -> RETRIEVAL_COMPLETE
 -> ANSWERS_GENERATED
 -> EVALUATION_COMPLETE
 -> VALIDATION_COMPLETE
 -> RESULTS_FINALISED
```

---

## Input Files

- `kb/` — plain-text knowledge base articles (Title + Section headers)
- `queries.json` — list of queries with expected doc titles and ground-truth answer points

The evaluator may replace these with equivalent fixtures using the same schema. The pipeline does not depend on exact filenames or document order.

---

## Output Artifacts

| File | Description |
|------|-------------|
| `artifacts/chunks.json` | All document chunks (paragraph strategy) |
| `artifacts/retrieval.json` | Top-3 BM25 results per query |
| `artifacts/answers.json` | Extractive answers with citations |
| `artifacts/eval.json` | Per-query retrieval evaluation + aggregate summary |
| `artifacts/grounding_check.json` | Citation validity and overlap checks |
| `artifacts/chunking_comparison.json` | Paragraph vs fixed-size strategy comparison |

---

## No LLM Used

Answer generation is fully extractive: sentences from retrieved chunks are scored by token overlap with the query. No external API calls are made.

---

## Requirements

- Python 3.10+
- `rank-bm25==0.2.2`
