"""
retriever.py - Hybrid retrieval module for BhashaDoc AI

Implements:
  1. FAISS semantic search
  2. BM25 keyword search
  3. Reciprocal Rank Fusion (RRF) merging of both result sets
  4. Cross-encoder reranking of the merged candidates
  5. Returns the top-k context chunks to send to Gemini

Designed to remain fast and accurate even with 1000+ page document
collections by keeping BM25 in-memory over tokenized chunk text and
only reranking a small candidate pool (not the full corpus).
"""

import re
from functools import lru_cache
from typing import List, Tuple

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> List[str]:
    """Simple, language-agnostic tokenizer (lowercase, word characters)."""
    return re.findall(r"\w+", text.lower())


@lru_cache(maxsize=1)
def _get_reranker():
    """Lazily load a cross-encoder reranker. Cached so it loads only once."""
    try:
        from sentence_transformers import CrossEncoder

        return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    except Exception:
        return None


class HybridRetriever:
    """
    Combines FAISS (semantic) and BM25 (keyword) retrieval over the same
    chunk collection, merges results with Reciprocal Rank Fusion, and
    reranks the merged candidates with a cross-encoder when available.
    """

    def __init__(self, faiss_index: FAISS, chunks: List[Document]):
        self.faiss_index = faiss_index
        self.chunks = chunks
        self._tokenized_corpus = [_tokenize(doc.page_content) for doc in chunks]
        self.bm25 = BM25Okapi(self._tokenized_corpus) if chunks else None

    # ---------- Step 1: FAISS semantic search ----------
    def faiss_search(self, query: str, k: int = 10) -> List[Tuple[Document, float]]:
        if self.faiss_index is None:
            return []
        results = self.faiss_index.similarity_search_with_score(query, k=k)
        return results  # list of (Document, distance_score)

    # ---------- Step 2: BM25 keyword search ----------
    def bm25_search(self, query: str, k: int = 10) -> List[Tuple[Document, float]]:
        if self.bm25 is None or not self.chunks:
            return []
        tokenized_query = _tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        ranked_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:k]
        return [(self.chunks[i], float(scores[i])) for i in ranked_indices]

    # ---------- Step 3: Merge results (Reciprocal Rank Fusion) ----------
    @staticmethod
    def _doc_key(doc: Document) -> str:
        meta = doc.metadata
        return f"{meta.get('filename')}::{meta.get('page_number')}::{meta.get('chunk_id')}"

    def reciprocal_rank_fusion(
        self,
        faiss_results: List[Tuple[Document, float]],
        bm25_results: List[Tuple[Document, float]],
        rrf_k: int = 60,
    ) -> List[Document]:
        scores = {}
        doc_lookup = {}

        for rank, (doc, _) in enumerate(faiss_results):
            key = self._doc_key(doc)
            doc_lookup[key] = doc
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)

        for rank, (doc, _) in enumerate(bm25_results):
            key = self._doc_key(doc)
            doc_lookup[key] = doc
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)

        merged_keys = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
        return [doc_lookup[k] for k in merged_keys]

    # ---------- Step 4: Rerank ----------
    def rerank(self, query: str, candidates: List[Document], top_n: int = 5) -> List[Document]:
        if not candidates:
            return []

        reranker = _get_reranker()
        if reranker is None:
            # Graceful fallback: keep RRF order if reranker model unavailable.
            return candidates[:top_n]

        pairs = [(query, doc.page_content) for doc in candidates]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[:top_n]]

    # ---------- Step 5: Full hybrid retrieval pipeline ----------
    def retrieve(
        self,
        query: str,
        faiss_k: int = 10,
        bm25_k: int = 10,
        final_k: int = 5,
    ) -> List[Document]:
        faiss_results = self.faiss_search(query, k=faiss_k)
        bm25_results = self.bm25_search(query, k=bm25_k)
        merged = self.reciprocal_rank_fusion(faiss_results, bm25_results)
        reranked = self.rerank(query, merged, top_n=final_k)
        return reranked
