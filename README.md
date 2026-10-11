# Advanced Hybrid RAG — hands-on prototype

## Architecture
Documents → overlapping word chunks → BM25 + dense embedding retrieval → RRF (k=60) → cross-encoder reranking → budgeted top chunks → LLM answer with inline source IDs.

## Setup
Python 3.10+ recommended. First run downloads two embedding/reranking models; internet is needed.

```bash
python -m venv .venv
# macOS/Linux: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python rag.py --question "How can I reduce cloud costs?"
python evaluate.py
```

## Generate an answer with citations
Set your API key securely in your shell (do not commit it). Requires access to the selected OpenAI model and may incur API charges.

```bash
# macOS/Linux
export OPENAI_API_KEY="your-key"
python rag.py --question "How can I reduce cloud costs?" --generate
```

PowerShell:
```powershell
$env:OPENAI_API_KEY="your-key"
python rag.py --question "How can I reduce cloud costs?" --generate
```

## Citation verification
The program checks that generated citation IDs exist in retrieved context; **this is not evidence that a claim is supported**. For semantic verification, split the answer into atomic claims, inspect each cited excerpt, and mark `supported=1` only when the cited excerpt entails the claim. Save your judgments as CSV columns `claim,citation,supported` and run:

```bash
python citation_audit.py data/audit_example.csv
```

Citation accuracy = supported claim–citation pairs / all audited claim–citation pairs. The example CSV is illustrative, not a measured model result. Also audit uncited factual claims separately: citation accuracy alone cannot catch them.

## Evaluation
`data/eval.json` contains graded document relevance labels (0=irrelevant, 1=weak, 2=useful, 3=highly relevant). The script deduplicates chunks to document IDs before computing document-level Recall@5, MRR, and nDCG@10. This tiny toy set is for pipeline validation, not statistically meaningful benchmarking. On larger corpora, evaluate a fixed candidate budget and compare BM25-only, vector-only, RRF, and reranked RRF on the same held-out queries.

## Context-window optimization
`select_context` selects highest-ranked chunks that fit an **approximate word-count budget**. For production, use your LLM's tokenizer, reserve tokens for the answer, deduplicate near-identical chunks, and optionally merge adjacent chunks while retaining source offsets.

## Production gaps
This demo uses an in-memory vector matrix, not a persistent ANN index. For scale, add FAISS/HNSW or a vector DB, caching, ingestion pipelines, metadata filters and ACL enforcement, observability, prompt-injection defenses, semantic citation validation, and continuous evaluation.
