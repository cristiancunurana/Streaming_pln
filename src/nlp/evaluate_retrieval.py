"""Evaluate Top-K retrieval against manually annotated timestamp ranges."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.nlp.retrieval import rank_chunks

ROOT = Path(__file__).resolve().parents[2]
MEMORY_DIR = ROOT / "data" / "processed" / "memory"
GOLD_FILE = ROOT / "data" / "evaluation" / "gold_questions.jsonl"


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def evaluate(k: int) -> dict:
    questions = read_jsonl(GOLD_FILE)
    memories = []
    for path in sorted(MEMORY_DIR.glob("*.json")):
        try:
            memories.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    chunks, vectors = [], []
    has_vectors = False
    for memory in memories:
        current = memory.get("chunks", [])
        embedded = memory.get("embedding_vectors")
        if embedded and len(embedded) == len(current):
            vectors.extend(embedded)
            has_vectors = True
        else:
            vectors.extend([None] * len(current))
        chunks.extend(current)
    dense_vectors = None
    if has_vectors:
        dimension = next(len(vector) for vector in vectors if vector is not None)
        dense_vectors = [vector if vector is not None else [0.0] * dimension for vector in vectors]
    rows, precisions, recalls = [], [], []
    modes = set()
    for question in questions:
        q_chunks = chunks
        q_vectors = dense_vectors
        stream_ids = set(question.get("stream_ids", []))
        if stream_ids:
            indices = [index for index, chunk in enumerate(chunks) if chunk.get("stream_id") in stream_ids]
            q_chunks = [chunks[index] for index in indices]
            q_vectors = [dense_vectors[index] for index in indices] if dense_vectors else None
        ranked, method = rank_chunks(question["question"], q_chunks, q_vectors, top_k=k)
        modes.add(method)
        relevant = question.get("relevant", [])
        hits, matched_ranges = [], set()
        for result in ranked:
            matched_indices = [index for index, gold in enumerate(relevant) if
                result.get("stream_id") == gold.get("stream_id")
                and float(result.get("end_sec", 0)) >= float(gold.get("start_sec", 0))
                and float(result.get("start_sec", 0)) <= float(gold.get("end_sec", gold.get("start_sec", 0)))
            ]
            hits.append(bool(matched_indices))
            matched_ranges.update(matched_indices)
        relevant_found = sum(hits)
        precision = relevant_found / k
        recall = len(matched_ranges) / len(relevant) if relevant else 0.0
        precisions.append(precision)
        recalls.append(recall)
        rows.append({"question": question["question"], "hits": relevant_found, "returned": len(ranked),
                     "relevant_ranges": len(relevant), "precision_at_k": precision, "recall_at_k": recall,
                     "method": method})
    count = len(rows)
    return {"k": k, "questions": count, "precision_at_k": sum(precisions) / count if count else 0.0,
            "recall_at_k": sum(recalls) / count if count else 0.0, "methods": sorted(modes), "rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    report = evaluate(max(1, min(args.k, 20)))
    if not report["questions"]:
        print(f"No hay preguntas anotadas en {GOLD_FILE}")
        return 0
    print(f"Preguntas evaluadas: {report['questions']}")
    print(f"Precision@{report['k']}: {report['precision_at_k']:.3f}")
    print(f"Recall@{report['k']}: {report['recall_at_k']:.3f}")
    print(f"Recuperadores: {', '.join(report['methods'])}")
    for row in report["rows"]:
        print(f"- {row['question']} | P@{report['k']}={row['precision_at_k']:.2f} | R@{report['k']}={row['recall_at_k']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
