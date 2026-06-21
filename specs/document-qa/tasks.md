# Tasks — Document Q&A

## Related Plan

See `specs/document-qa/plan.md`

## Task Breakdown

| # | Task | Status | Notes |
|---|------|--------|-------|
| 1 | Implement multi-format loader (PDF, DOCX, TXT) with page metadata | ☑ Done | `rag/loader.py` |
| 2 | Implement recursive chunking with filename/page/chunk_id metadata | ☑ Done | `rag/chunker.py` |
| 3 | Implement embeddings via all-MiniLM-L6-v2 | ☑ Done | `rag/embeddings.py` |
| 4 | Implement FAISS index create/save/load | ☑ Done | `rag/vectorstore.py` |
| 5 | Implement BM25 keyword retrieval | ☑ Done | `rag/retriever.py` |
| 6 | Implement hybrid retrieval (RRF merge) + reranking | ☑ Done | `rag/retriever.py` |
| 7 | Integrate Gemini 2.5 Flash for answer generation | ☑ Done | `rag/llm.py` |
| 8 | Implement QA / Summary / Insights / Clause chains | ☑ Done | `rag/chains.py` |
| 9 | Implement citation formatting | ☑ Done | `utils/citations.py` |
| 10 | Wire chat UI, history, and download in Streamlit | ☑ Done | `app.py` |
| 11 | Add full i18n (English, Hindi, Telugu) | ☑ Done | `locales/*.json` |
| 12 | Add automated tests for chunking/retrieval logic | ☐ Todo | unit tests |

## Definition of Done

- [x] Code implemented and passes lint/type-check
- [x] Manual test passes in all supported languages
- [x] Documentation updated (README/CHANGELOG)
- [x] No secrets or API keys committed
