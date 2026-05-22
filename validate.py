"""
validate.py — Checks all required artifacts against the spec.
Exit 0 = all pass. Exit 1 = failures found.
"""

import json
import re
import sys
from pathlib import Path

ARTIFACTS_DIR = Path("artifacts")
QUERIES_FILE  = Path("queries.json")

ANSWER_LABELS      = {"grounded_answer", "insufficient_context", "conflicting_context"}
RETRIEVAL_STATUSES = {"hit", "partial_hit", "miss"}
CITATION_RE        = re.compile(r"\[.+? §chunk_\d+\]")

errors = []

def err(msg):
    errors.append(msg)
    print(f"  [FAIL] {msg}")

def ok(msg):
    print(f"  [PASS] {msg}")

def load_json(path):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        ok(f"{path} is valid JSON")
        return data
    except Exception as e:
        err(f"{path} — invalid JSON: {e}")
        return None

# ──────────────────────────────────────────────
print("\n=== ARTIFACT VALIDATION ===\n")

# 1. Required files exist
required = [
    ARTIFACTS_DIR / "chunks.json",
    ARTIFACTS_DIR / "retrieval.json",
    ARTIFACTS_DIR / "answers.json",
    ARTIFACTS_DIR / "eval.json",
]
for path in required:
    if path.exists():
        ok(f"{path} exists")
    else:
        err(f"{path} MISSING")

# 2. Load all artifacts
queries    = load_json(QUERIES_FILE)
chunks     = load_json(ARTIFACTS_DIR / "chunks.json")
retrieval  = load_json(ARTIFACTS_DIR / "retrieval.json")
answers    = load_json(ARTIFACTS_DIR / "answers.json")
eval_data  = load_json(ARTIFACTS_DIR / "eval.json")

if not all([queries, chunks, retrieval, answers, eval_data]):
    print("\n[ABORT] Cannot validate further — missing or corrupt artifacts.")
    sys.exit(1)

query_ids = {q["query_id"] for q in queries}

# 3. All queries processed
print()
ret_qids = {r["query_id"] for r in retrieval}
ans_qids = {a["query_id"] for a in answers}

for qid in query_ids:
    if qid not in ret_qids:
        err(f"Query {qid} missing from retrieval.json")
    if qid not in ans_qids:
        err(f"Query {qid} missing from answers.json")
if query_ids == ret_qids and query_ids == ans_qids:
    ok("All queries processed in retrieval and answers")

# 4. Each query has >= 3 retrieved chunks, scores are numeric
print()
for r in retrieval:
    qid = r["query_id"]
    top_k = r.get("top_k", [])
    if len(top_k) < 3:
        err(f"{qid}: only {len(top_k)} chunks retrieved (need >= 3)")
    else:
        ok(f"{qid}: {len(top_k)} chunks retrieved")

    for entry in top_k:
        score = entry.get("score")
        if not isinstance(score, (int, float)):
            err(f"{qid} chunk {entry.get('chunk_id')}: score is not numeric ({score!r})")

# 5. Answer labels use controlled vocabulary
print()
for ans in answers:
    label = ans.get("answer_label")
    if label not in ANSWER_LABELS:
        err(f"{ans['query_id']}: invalid answer_label '{label}'")
    else:
        ok(f"{ans['query_id']}: answer_label='{label}' ✓")

# 6. Grounded answers include at least one citation
print()
for ans in answers:
    if ans.get("answer_label") == "grounded_answer":
        cits = ans.get("citations", [])
        if not cits:
            err(f"{ans['query_id']}: grounded_answer has no citations")
        else:
            for c in cits:
                if not CITATION_RE.fullmatch(c):
                    err(f"{ans['query_id']}: citation format invalid: '{c}'")
                else:
                    ok(f"{ans['query_id']}: citation format valid: '{c}'")

# 7. Citations refer only to retrieved chunks
print()
ret_chunk_map = {
    r["query_id"]: {e["chunk_id"] for e in r["top_k"]}
    for r in retrieval
}
for ans in answers:
    qid = ans["query_id"]
    for cid in ans.get("used_chunk_ids", []):
        if cid not in ret_chunk_map.get(qid, set()):
            err(f"{qid}: used_chunk_id '{cid}' not in retrieved chunks")
        else:
            ok(f"{qid}: used_chunk_id '{cid}' correctly references retrieved chunk")

# 8. Eval statuses use controlled vocab + aggregate summary present
print()
per_query = eval_data.get("per_query", [])
summary   = eval_data.get("summary")

if not summary:
    err("eval.json missing 'summary' block")
else:
    ok("eval.json has aggregate summary")
    for key in ["top3_hit_rate", "total_queries", "hits", "partial_hits", "misses"]:
        if key not in summary:
            err(f"eval.json summary missing key '{key}'")
    if isinstance(summary.get("top3_hit_rate"), float):
        ok(f"top3_hit_rate = {summary['top3_hit_rate']}")

for rec in per_query:
    status = rec.get("retrieval_status")
    if status not in RETRIEVAL_STATUSES:
        err(f"{rec['query_id']}: invalid retrieval_status '{status}'")
    else:
        ok(f"{rec['query_id']}: retrieval_status='{status}' ✓")

# 9. Optional artifacts (warn only)
print()
optional = [
    ARTIFACTS_DIR / "grounding_check.json",
    ARTIFACTS_DIR / "chunking_comparison.json",
]
for path in optional:
    if path.exists():
        ok(f"Optional artifact present: {path}")
    else:
        print(f"  [WARN] Optional artifact not found: {path}")

# ──────────────────────────────────────────────
print("\n" + "="*40)
if errors:
    print(f"VALIDATION FAILED — {len(errors)} error(s)")
    for e in errors:
        print(f"  ✗ {e}")
    sys.exit(1)
else:
    print("VALIDATION PASSED — all checks OK")
    sys.exit(0)
