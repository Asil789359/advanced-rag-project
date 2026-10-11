"""Run: python evaluate.py. Labels are per document, not per chunk."""
import json
import math
from pathlib import Path
from rag import HybridRAG, ROOT


def unique_doc_ids(ranked):
    return list(dict.fromkeys(c["doc_id"] for c in ranked))


def recall_at_k(ids, grades, k):
    relevant = {doc for doc, grade in grades.items() if grade > 0}
    return len(set(ids[:k]) & relevant) / len(relevant) if relevant else 0.0


def reciprocal_rank(ids, grades):
    return next((1.0 / rank for rank, doc in enumerate(ids, 1)
                 if grades.get(doc, 0) > 0), 0.0)


def dcg(grades):
    return sum((2 ** grade - 1) / math.log2(i + 2)
               for i, grade in enumerate(grades))


def ndcg_at_k(ids, grades, k):
    actual = dcg([grades.get(doc, 0) for doc in ids[:k]])
    ideal = dcg(sorted(grades.values(), reverse=True)[:k])
    return actual / ideal if ideal else 0.0


def main():
    rag = HybridRAG()
    examples = json.loads((ROOT / "data/eval.json").read_text(encoding="utf-8"))
    totals = [0.0, 0.0, 0.0]
    for ex in examples:
        # Retrieve all chunks to permit evaluating @10 (even on this tiny corpus).
        ranked = rag.retrieve(ex["query"], bm25_k=len(rag.chunks),
                              vector_k=len(rag.chunks), rerank_k=len(rag.chunks))
        ids = unique_doc_ids(ranked)
        grades = ex["relevance"]
        values = [recall_at_k(ids, grades, 5), reciprocal_rank(ids, grades),
                  ndcg_at_k(ids, grades, 10)]
        totals = [a + b for a, b in zip(totals, values)]
        print(f"{ex['query']}\n  Recall@5={values[0]:.3f} MRR contribution={values[1]:.3f} nDCG@10={values[2]:.3f}")
    print("\nAVERAGES")
    for name, value in zip(("Recall@5", "MRR", "nDCG@10"), totals):
        print(f"{name}: {value / len(examples):.3f}")


if __name__ == "__main__":
    main()
