"""
BhashaDoc AI - RAG Engine Package

Exposes the core RAG pipeline components:
loader -> chunker -> embeddings -> vectorstore -> retriever -> llm -> chains
"""

from .loader import load_documents
from .chunker import chunk_documents
from .embeddings import get_embedding_model
from .vectorstore import build_faiss_index, save_faiss_index, load_faiss_index
from .retriever import HybridRetriever
from .llm import get_llm
from .chains import run_qa_chain, run_summary_chain, run_insights_chain, run_clause_chain

__all__ = [
    "load_documents",
    "chunk_documents",
    "get_embedding_model",
    "build_faiss_index",
    "save_faiss_index",
    "load_faiss_index",
    "HybridRetriever",
    "get_llm",
    "run_qa_chain",
    "run_summary_chain",
    "run_insights_chain",
    "run_clause_chain",
]
