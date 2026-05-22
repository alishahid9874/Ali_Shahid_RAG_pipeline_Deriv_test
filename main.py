"""
Mini RAG Pipeline
-----------------
Stages: INIT -> DOCUMENTS_LOADED -> DOCUMENTS_CHUNKED -> INDEX_BUILT ->
        RETRIEVAL_COMPLETE -> ANSWERS_GENERATED -> EVALUATION_COMPLETE ->
        VALIDATION_COMPLETE -> RESULTS_FINALISED

No LLM used. Answer generation is fully deterministic extractive logic.
"""

import os
import re
import json
import math
import hashlib
from datetime import datetime, timezone
from pathlib import Path

from rank_bm25 import BM25Okapi

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
KB_DIR        = Path("kb")
QUERIES_FILE  = Path("queries.json")
ARTIFACTS_DIR = Path("artifacts")

ANSWER_LABELS     = {"grounded_answer", "insufficient_context", "conflicting_context"}
RETRIEVAL_STATUSES = {"hit", "partial_hit", "miss"}
TOP_K             = 3
OVERLAP_THRESHOLD = 0.15   # min token-overlap ratio to label as grounded

STOPWORDS = {
    "a","an","the","is","it","in","on","of","to","and","or","for",
    "my","i","me","can","will","be","are","was","were","do","does",
    "did","has","have","had","not","this","that","with","from","at",
    "by","as","if","its","into","than","then","so","but","how","what",
    "when","where","who","which","after","before","about","up","out"
}

# ──────────────────────────────────────────────
# STAGE TRACKER
# ──────────────────────────────────────────────
STAGES = [
    "INIT", "DOCUMENTS_LOADED", "DOCUMENTS_CHUNKED", "INDEX_BUILT",
    "RETRIEVAL_COMPLETE", "ANSWERS_GENERATED", "EVALUATION_COMPLETE",
    "VALIDATION_COMPLETE", "RESULTS_FINALISED"
]

stage = "INIT"

def advance_stage(expected: str, next_stage: str):
    global stage
    assert stage == expected, f"Stage mismatch: expected {expected}, got {stage}"
    stage = next_stage
    print(f"  [STAGE] {stage}")

# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────
def tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, remove stopwords."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if t not in STOPWORDS]

def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  [SAVED] {path}")

def split_sentences(text: str) -> list[str]:
    """Split text into sentences on . ! ? newlines."""
    parts = re.split(r"(?<=[.!?])\s+|\n", text)
    return [s.strip() for s in parts if s.strip()]

def token_overlap(a_tokens: list[str], b_tokens: list[str]) -> float:
    """Jaccard-style overlap coefficient."""
    if not a_tokens or not b_tokens:
        return 0.0
    sa, sb = set(a_tokens), set(b_tokens)
    return len(sa & sb) / min(len(sa), len(sb))

# ──────────────────────────────────────────────
# STAGE 1 — DOCUMENT LOADING
# ──────────────────────────────────────────────
def load_documents() -> list[dict]:
    advance_stage("INIT", "DOCUMENTS_LOADED")
    docs = []
    for fpath in sorted(KB_DIR.glob("*.txt")):          # sorted → deterministic order
        raw = fpath.read_text(encoding="utf-8")
        lines = raw.splitlines()

        title, section, body_lines = "unknown", "unknown", []
        body_started = False

        for line in lines:
            if not body_started:
                if line.startswith("Title:"):
                    title = line.split(":", 1)[1].strip()
                elif line.startswith("Section:"):
                    section = line.split(":", 1)[1].strip()
                elif line.strip() == "":
                    if title != "unknown":      # blank line after header → body starts
                        body_started = True
            else:
                body_lines.append(line)

        body = "\n".join(body_lines).strip()
        docs.append({"title": title, "section": section, "body": body, "source": fpath.name})
        print(f"  [LOADED] {fpath.name} → '{title}'")

    assert docs, "No documents found in kb/"
    return docs

# ──────────────────────────────────────────────
# STAGE 2 — CHUNKING (two strategies)
# ──────────────────────────────────────────────
def chunk_paragraph(docs: list[dict], id_offset: int = 0) -> list[dict]:
    """Split on double-newlines or single newlines for tight body text."""
    chunks, counter = [], id_offset
    for doc in docs:
        # These KB files are newline-separated sentences; treat each line as paragraph
        paragraphs = [p.strip() for p in re.split(r"\n+", doc["body"]) if p.strip()]
        body = doc["body"]
        cursor = 0
        for para in paragraphs:
            start = body.find(para, cursor)
            end   = start + len(para)
            counter += 1
            chunks.append({
                "chunk_id":  f"chunk_{counter}",
                "doc_title": doc["title"],
                "section":   doc["section"],
                "text":      para,
                "start_char": start,
                "end_char":   end,
                "strategy":  "paragraph"
            })
            cursor = end
    return chunks

def chunk_fixed(docs: list[dict], max_chars: int = 300, id_offset: int = 0) -> list[dict]:
    """
    Fixed-size chunking that respects sentence boundaries.
    Accumulate sentences until adding the next would exceed max_chars.
    """
    chunks, counter = [], id_offset
    for doc in docs:
        sentences = split_sentences(doc["body"])
        body      = doc["body"]
        buffer, buf_start = [], None
        cursor = 0

        for sent in sentences:
            s_start = body.find(sent, cursor)
            s_end   = s_start + len(sent)

            if buf_start is None:
                buf_start = s_start

            if buffer and (sum(len(s) for s in buffer) + len(sent) + 1) > max_chars:
                # flush current buffer
                counter += 1
                chunk_text = " ".join(buffer)
                c_start = body.find(buffer[0], buf_start)
                chunks.append({
                    "chunk_id":   f"chunk_{counter}",
                    "doc_title":  doc["title"],
                    "section":    doc["section"],
                    "text":       chunk_text,
                    "start_char": c_start,
                    "end_char":   c_start + len(chunk_text),
                    "strategy":   "fixed_size"
                })
                buffer, buf_start = [sent], s_start
            else:
                buffer.append(sent)

            cursor = s_end

        if buffer:
            counter += 1
            chunk_text = " ".join(buffer)
            c_start = body.find(buffer[0], buf_start if buf_start else 0)
            chunks.append({
                "chunk_id":   f"chunk_{counter}",
                "doc_title":  doc["title"],
                "section":    doc["section"],
                "text":       chunk_text,
                "start_char": c_start,
                "end_char":   c_start + len(chunk_text),
                "strategy":   "fixed_size"
            })

    return chunks

def build_chunks(docs: list[dict]) -> tuple[list[dict], list[dict]]:
    advance_stage("DOCUMENTS_LOADED", "DOCUMENTS_CHUNKED")
    para_chunks  = chunk_paragraph(docs, id_offset=0)
    fixed_chunks = chunk_fixed(docs, id_offset=len(para_chunks))

    save_json(ARTIFACTS_DIR / "chunks.json", para_chunks)      # primary artifact
    print(f"  [CHUNKS] paragraph={len(para_chunks)}, fixed={len(fixed_chunks)}")
    return para_chunks, fixed_chunks

# ──────────────────────────────────────────────
# STAGE 3 — INDEX
# ──────────────────────────────────────────────
def build_index(chunks: list[dict]) -> BM25Okapi:
    advance_stage("DOCUMENTS_CHUNKED", "INDEX_BUILT")
    corpus = [tokenize(c["text"]) for c in chunks]
    index  = BM25Okapi(corpus)
    print(f"  [INDEX] BM25 built over {len(chunks)} chunks")
    return index

# ──────────────────────────────────────────────
# STAGE 4 — RETRIEVAL
# ──────────────────────────────────────────────
def retrieve(queries: list[dict], chunks: list[dict], index: BM25Okapi) -> list[dict]:
    advance_stage("INDEX_BUILT", "RETRIEVAL_COMPLETE")
    results = []
    for q in queries:
        q_tokens = tokenize(q["question"])
        scores   = index.get_scores(q_tokens)

        ranked = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True
        )[:TOP_K]

        top_k = []
        for rank, (idx, score) in enumerate(ranked, start=1):
            c = chunks[idx]
            top_k.append({
                "rank":       rank,
                "chunk_id":   c["chunk_id"],
                "doc_title":  c["doc_title"],
                "score":      round(float(score), 4),
                "chunk_text": c["text"]
            })

        results.append({
            "query_id": q["query_id"],
            "question": q["question"],
            "top_k":    top_k
        })

    save_json(ARTIFACTS_DIR / "retrieval.json", results)
    return results

# ──────────────────────────────────────────────
# STAGE 5 — ANSWER GENERATION (extractive, no LLM)
# ──────────────────────────────────────────────
def generate_answers(queries: list[dict], retrieval: list[dict], chunks: list[dict]) -> list[dict]:
    advance_stage("RETRIEVAL_COMPLETE", "ANSWERS_GENERATED")

    chunk_map = {c["chunk_id"]: c for c in chunks}
    ret_map   = {r["query_id"]: r for r in retrieval}
    answers   = []

    for q in queries:
        qid      = q["query_id"]
        q_tokens = tokenize(q["question"])
        top_k    = ret_map[qid]["top_k"]

        best_sentence, best_score, best_chunk_id, best_title = None, -1.0, None, None

        for entry in top_k:
            cid        = entry["chunk_id"]
            chunk_text = entry["chunk_text"]
            sentences  = split_sentences(chunk_text)

            for sent in sentences:
                s_tokens = tokenize(sent)
                score    = token_overlap(q_tokens, s_tokens)
                if score > best_score:
                    best_score, best_sentence = score, sent
                    best_chunk_id, best_title = cid, entry["doc_title"]

        if best_sentence and best_score >= OVERLAP_THRESHOLD:
            citation    = f"[{best_title} §{best_chunk_id}]"
            answer_text = f"{best_sentence} {citation}"
            label       = "grounded_answer"
            used_ids    = [best_chunk_id]
            citations   = [citation]
        else:
            answer_text = "I cannot answer this question with the available context."
            label       = "insufficient_context"
            used_ids    = []
            citations   = []

        answers.append({
            "query_id":      qid,
            "answer_label":  label,
            "answer":        answer_text,
            "citations":     citations,
            "used_chunk_ids": used_ids
        })

    save_json(ARTIFACTS_DIR / "answers.json", answers)
    return answers

# ──────────────────────────────────────────────
# STAGE 6 — DETERMINISTIC EVALUATION
# ──────────────────────────────────────────────
def evaluate(queries: list[dict], retrieval: list[dict]) -> dict:
    advance_stage("ANSWERS_GENERATED", "EVALUATION_COMPLETE")

    ret_map    = {r["query_id"]: r for r in retrieval}
    per_query  = []
    hits = partial = misses = 0

    for q in queries:
        qid      = q["query_id"]
        expected = set(q["expected_doc_titles"])
        top3_titles = [e["doc_title"] for e in ret_map[qid]["top_k"]]
        matched  = expected & set(top3_titles)

        if matched == expected:
            status = "hit";         hits    += 1
        elif matched:
            status = "partial_hit"; partial += 1
        else:
            status = "miss";        misses  += 1

        # Build explanation
        if matched:
            ranks = [str(e["rank"]) for e in ret_map[qid]["top_k"] if e["doc_title"] in matched]
            expl  = f"Expected title(s) found at rank(s): {', '.join(ranks)}"
        else:
            expl = "Expected title not found in top-3 results"

        per_query.append({
            "query_id":                qid,
            "expected_doc_titles":     list(expected),
            "retrieved_doc_titles_top3": top3_titles,
            "retrieval_status":        status,
            "matched_expected_title":  bool(matched),
            "explanation":             expl
        })

    total = len(queries)
    summary = {
        "top3_hit_rate": round(hits / total, 4) if total else 0.0,
        "total_queries": total,
        "hits":          hits,
        "partial_hits":  partial,
        "misses":        misses
    }

    eval_out = {"per_query": per_query, "summary": summary}
    save_json(ARTIFACTS_DIR / "eval.json", eval_out)
    return eval_out

# ──────────────────────────────────────────────
# STAGE 7 — GROUNDING CHECK
# ──────────────────────────────────────────────
def grounding_check(answers: list[dict], retrieval: list[dict], chunks: list[dict]):
    chunk_map = {c["chunk_id"]: c for c in chunks}
    ret_map   = {r["query_id"]: {e["chunk_id"]: e for e in r["top_k"]} for r in retrieval}
    records   = []

    for ans in answers:
        qid   = ans["query_id"]
        label = ans["answer_label"]
        checks = []

        if label == "grounded_answer":
            for citation in ans["citations"]:
                # parse [doc_title §chunk_id]
                m = re.match(r"\[(.+?) §(chunk_\d+)\]", citation)
                if not m:
                    checks.append({"citation": citation, "valid_format": False, "in_retrieval": False, "overlap_ok": False})
                    continue

                cited_title = m.group(1)
                cited_cid   = m.group(2)
                in_ret      = cited_cid in ret_map.get(qid, {})
                chunk_text  = chunk_map.get(cited_cid, {}).get("text", "")

                # strip citation from answer text before overlap check
                ans_text_clean = re.sub(r"\[.+?\]", "", ans["answer"]).strip()
                a_tokens = tokenize(ans_text_clean)
                c_tokens = tokenize(chunk_text)
                overlap  = token_overlap(a_tokens, c_tokens)

                checks.append({
                    "citation":      citation,
                    "valid_format":  True,
                    "in_retrieval":  in_ret,
                    "overlap_ratio": round(overlap, 4),
                    "overlap_ok":    overlap >= OVERLAP_THRESHOLD
                })

        records.append({
            "query_id":     qid,
            "answer_label": label,
            "checks":       checks,
            "all_passed":   all(c.get("in_retrieval") and c.get("overlap_ok") for c in checks) if checks else (label != "grounded_answer")
        })

    save_json(ARTIFACTS_DIR / "grounding_check.json", records)
    return records

# ──────────────────────────────────────────────
# STAGE 8 — CHUNKING COMPARISON
# ──────────────────────────────────────────────
def chunking_comparison(queries: list[dict], docs: list[dict]):
    """Run retrieval for both strategies, compare hit rates."""

    def run_strategy(chunks, strategy_name):
        idx = BM25Okapi([tokenize(c["text"]) for c in chunks])
        hits = 0
        for q in queries:
            q_tokens   = tokenize(q["question"])
            scores     = idx.get_scores(q_tokens)
            ranked     = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:TOP_K]
            top3_titles = [chunks[i]["doc_title"] for i, _ in ranked]
            expected   = set(q["expected_doc_titles"])
            if expected & set(top3_titles):
                hits += 1
        return {
            "strategy":      strategy_name,
            "total_chunks":  len(chunks),
            "hits":          hits,
            "total_queries": len(queries),
            "top3_hit_rate": round(hits / len(queries), 4)
        }

    para_chunks  = chunk_paragraph(docs, id_offset=0)
    fixed_chunks = chunk_fixed(docs, id_offset=0)

    para_result  = run_strategy(para_chunks,  "paragraph")
    fixed_result = run_strategy(fixed_chunks, "fixed_size")

    tradeoff = (
        "Paragraph chunking preserves semantic units (one fact per chunk) which improves "
        "precision for these short KB articles. Fixed-size chunking may split a single fact "
        "across chunk boundaries, diluting BM25 term frequency and reducing recall. "
        "For longer documents, fixed-size with overlap would outperform paragraph chunks "
        "that grow too large to match specific query terms."
    )

    comparison = {
        "results":   [para_result, fixed_result],
        "tradeoff":  tradeoff,
        "winner":    max([para_result, fixed_result], key=lambda x: x["top3_hit_rate"])["strategy"]
    }
    save_json(ARTIFACTS_DIR / "chunking_comparison.json", comparison)
    return comparison

# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    print("\n=== MINI RAG PIPELINE ===\n")
    ARTIFACTS_DIR.mkdir(exist_ok=True)

    # Load queries
    queries = json.loads(QUERIES_FILE.read_text(encoding="utf-8"))

    # Stage 1: Load documents
    docs = load_documents()

    # Stage 2: Chunk (paragraph = primary)
    para_chunks, fixed_chunks = build_chunks(docs)

    # Stage 3: Build BM25 index on paragraph chunks
    index = build_index(para_chunks)

    # Stage 4: Retrieve
    retrieval = retrieve(queries, para_chunks, index)

    # Stage 5: Generate answers
    answers = generate_answers(queries, retrieval, para_chunks)

    # Stage 6: Evaluate
    eval_out = evaluate(queries, retrieval)

    # Stage 7: Grounding check
    advance_stage("EVALUATION_COMPLETE", "VALIDATION_COMPLETE")
    grounding_check(answers, retrieval, para_chunks)

    # Stage 8: Chunking comparison
    chunking_comparison(queries, docs)

    # Finalise
    advance_stage("VALIDATION_COMPLETE", "RESULTS_FINALISED")

    print("\n=== PIPELINE COMPLETE ===")
    print(f"  Hit rate (paragraph): {eval_out['summary']['top3_hit_rate']}")
    print(f"  Hits: {eval_out['summary']['hits']}/{eval_out['summary']['total_queries']}")

if __name__ == "__main__":
    main()
