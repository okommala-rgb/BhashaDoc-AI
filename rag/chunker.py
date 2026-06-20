"""
chunker.py - Document chunking module for BhashaDoc AI

Splits loaded Documents into overlapping chunks suitable for embedding,
using RecursiveCharacterTextSplitter, while preserving and extending
metadata (filename, page_number, chunk_id) for accurate citations.
"""

import uuid
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def chunk_documents(
    documents: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    """
    Split documents into chunks with RecursiveCharacterTextSplitter.

    Each resulting chunk carries metadata:
      - filename
      - page_number
      - chunk_id (unique short id, stable per chunk)
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunked_documents: List[Document] = []

    for doc in documents:
        splits = splitter.split_text(doc.page_content)
        for split_text in splits:
            if not split_text.strip():
                continue
            chunk_id = uuid.uuid4().hex[:8]
            metadata = dict(doc.metadata)
            metadata["chunk_id"] = chunk_id
            chunked_documents.append(
                Document(page_content=split_text, metadata=metadata)
            )

    return chunked_documents
