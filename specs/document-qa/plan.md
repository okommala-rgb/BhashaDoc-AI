# Implementation Plan — Document Q&A

## Related Spec

See `specs/document-qa/spec.md`

## Architecture Overview

The document Q&A feature is the core of BhashaDoc AI. It is implemented as a
hybrid retrieval-augmented generation pipeline:

```
Documents → Loader → Cleaning → Chunking → Embeddings
  → FAISS Index + BM25 Index → Hybrid Retrieval (RRF merge)
  → Reranking → Gemini 2.5 Flash → Answer + Citations
```

## Implementation Steps

1. `rag/loader.py` — load PDF/DOCX/TXT files, preserving filename and
   page-number metadata per document.
2. `rag/chunker.py` — split cleaned text using `RecursiveCharacterTextSplitter`
   (chunk size 1000, overlap 200), attaching `filename`, `page_number`, and
   `chunk_id` metadata to every chunk.
3. `rag/embeddings.py` — embed chunks using
   `sentence-transformers/all-MiniLM-L6-v2`.
4. `rag/vectorstore.py` — build, save, and load a FAISS index over the
   embedded chunks.
5. `rag/retriever.py` — run FAISS semantic search and BM25 keyword search in
   parallel, merge results via Reciprocal Rank Fusion, and rerank the
   merged candidates.
6. `rag/llm.py` — call Gemini 2.5 Flash with the reranked context and the
   user's question, in the user's selected UI language.
7. `rag/chains.py` — orchestrate the QA chain (and the related summary,
   insights, and clause-finder chains) end to end.
8. `utils/citations.py` — format the retrieved chunks into human-readable
   citations (filename, page number, chunk id) shown alongside every answer.
9. `app.py` — wire the chain into the Streamlit "Chat" tab, including the
   retrieved-context viewer and chat history/download.

## Affected Files

- `rag/loader.py`, `rag/chunker.py`, `rag/embeddings.py`,
  `rag/vectorstore.py`, `rag/retriever.py`, `rag/llm.py`, `rag/chains.py`
- `utils/citations.py`, `utils/translator.py`
- `app.py`
- `locales/en.json`, `locales/hi.json`, `locales/te.json`

## Testing Strategy

- Unit tests for chunk metadata correctness, FAISS save/load round-trip,
  and BM25/FAISS result merging.
- Manual end-to-end test: upload a multi-page PDF, ask a question, confirm
  the answer includes correct source citations, in all three UI languages.

## Rollback Plan

The FAISS index and BM25 index are rebuilt from uploaded documents on each
session, so no persistent migration is required — reverting to a previous
version of the retrieval code does not require any data cleanup.
