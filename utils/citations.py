"""
citations.py - Citation formatting module for BhashaDoc AI

Generates human-readable, consistently formatted citations
(filename, page number, chunk id) for retrieved document chunks,
used in the chat UI's "Sources" section and clause finder output.
"""

from typing import List, Dict, Any

from langchain_core.documents import Document


def format_citation(chunk: Document, labels: Dict[str, str] = None) -> str:
    """
    Format a single chunk's metadata into a readable citation string.

    `labels` may optionally provide translated labels for
    'file_label', 'page_label', and 'chunk_label' so the citation
    text matches the active UI language.
    """
    labels = labels or {}
    file_label = labels.get("file_label", "File")
    page_label = labels.get("page_label", "Page")
    chunk_label = labels.get("chunk_label", "Chunk")

    meta = chunk.metadata
    filename = meta.get("filename", "unknown")
    page_number = meta.get("page_number", "?")
    chunk_id = meta.get("chunk_id", "?")

    return f"{file_label}: {filename} | {page_label}: {page_number} | {chunk_label}: {chunk_id}"


def format_citations_list(
    chunks: List[Document], labels: Dict[str, str] = None
) -> List[Dict[str, Any]]:
    """
    Format a list of chunks into structured citation dicts, suitable
    for rendering in Streamlit (e.g. one expander row per citation).

    Returns a list of:
      {
        "label": "<formatted citation string>",
        "filename": ...,
        "page_number": ...,
        "chunk_id": ...,
        "content": "<chunk text>"
      }
    """
    results = []
    for chunk in chunks:
        meta = chunk.metadata
        results.append(
            {
                "label": format_citation(chunk, labels),
                "filename": meta.get("filename", "unknown"),
                "page_number": meta.get("page_number", "?"),
                "chunk_id": meta.get("chunk_id", "?"),
                "content": chunk.page_content,
            }
        )
    return results
