"""
embeddings.py - Embedding model loader for BhashaDoc AI

Uses sentence-transformers/all-MiniLM-L6-v2 via LangChain's
HuggingFaceEmbeddings wrapper. The model is cached so it only
loads once per Streamlit session/process.
"""

from functools import lru_cache

from langchain_community.embeddings import HuggingFaceEmbeddings

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_embedding_model() -> HuggingFaceEmbeddings:
    """Return a cached HuggingFaceEmbeddings instance for all-MiniLM-L6-v2."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
