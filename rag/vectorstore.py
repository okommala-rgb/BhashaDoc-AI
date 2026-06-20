"""
vectorstore.py - FAISS vector store management for BhashaDoc AI

Implements creation, saving, and loading of a FAISS index built
from chunked Documents and an embedding model.
"""

import os
from typing import List, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings


def build_faiss_index(
    chunks: List[Document], embedding_model: HuggingFaceEmbeddings
) -> FAISS:
    """Build a FAISS index in-memory from a list of chunked Documents."""
    if not chunks:
        raise ValueError("Cannot build a FAISS index from an empty document list.")
    return FAISS.from_documents(chunks, embedding_model)


def save_faiss_index(index: FAISS, path: str) -> None:
    """Persist a FAISS index to disk at the given directory path."""
    os.makedirs(path, exist_ok=True)
    index.save_local(path)


def load_faiss_index(
    path: str, embedding_model: HuggingFaceEmbeddings
) -> Optional[FAISS]:
    """Load a previously saved FAISS index from disk, if it exists."""
    if not os.path.isdir(path):
        return None
    try:
        return FAISS.load_local(
            path, embedding_model, allow_dangerous_deserialization=True
        )
    except Exception:
        return None
