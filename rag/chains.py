"""
chains.py - LLM chains for BhashaDoc AI

Implements four language-aware chains, all powered by Gemini 2.5 Flash:
  - QA chain: answer a question using retrieved hybrid-search context
  - Summary chain: map-reduce summarization across the full document set
    (so collections of 1000+ pages can be summarized without exceeding
    context limits)
  - Insights chain: map-reduce extraction of key insights
  - Clause chain: find and explain clauses/sections relevant to a query

All chains instruct Gemini to respond in the requested UI language
(English, Hindi, or Telugu), regardless of the source document language.
"""

from typing import List

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

LANGUAGE_NAMES = {
    "en": "English",
    "hi": "Hindi",
    "te": "Telugu",
}

MAP_BATCH_CHAR_BUDGET = 12000  # ~ a few thousand tokens per map step


def _language_name(language_code: str) -> str:
    return LANGUAGE_NAMES.get(language_code, "English")


def _format_context(chunks: List[Document]) -> str:
    """Format retrieved chunks into a numbered context block with citation tags."""
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        meta = chunk.metadata
        citation = (
            f"[Source {i}: {meta.get('filename', 'unknown')}, "
            f"Page {meta.get('page_number', '?')}, "
            f"Chunk {meta.get('chunk_id', '?')}]"
        )
        blocks.append(f"{citation}\n{chunk.page_content}")
    return "\n\n---\n\n".join(blocks)


def _batch_chunks(chunks: List[Document], char_budget: int) -> List[List[Document]]:
    """Group chunks into batches that fit within a character budget, for
    map-reduce style processing of very large document collections."""
    batches: List[List[Document]] = []
    current_batch: List[Document] = []
    current_chars = 0

    for chunk in chunks:
        chunk_len = len(chunk.page_content)
        if current_batch and current_chars + chunk_len > char_budget:
            batches.append(current_batch)
            current_batch = []
            current_chars = 0
        current_batch.append(chunk)
        current_chars += chunk_len

    if current_batch:
        batches.append(current_batch)

    return batches


# ---------------------------------------------------------------------------
# QA Chain
# ---------------------------------------------------------------------------
def run_qa_chain(llm, context_chunks: List[Document], question: str, language_code: str) -> str:
    """Answer a user question using retrieved context, citing sources inline."""
    language = _language_name(language_code)
    context_text = _format_context(context_chunks)

    system_prompt = (
        f"You are BhashaDoc AI, a precise document question-answering assistant. "
        f"Answer ONLY using the provided context. If the answer is not present in "
        f"the context, say so clearly instead of guessing. "
        f"Always respond in {language}, even if the source context is in a "
        f"different language. When you use information from a source, reference "
        f"it using its bracketed citation tag, e.g. [Source 1]. "
        f"Be concise, accurate, and well-structured."
    )

    user_prompt = (
        f"CONTEXT:\n{context_text}\n\n"
        f"QUESTION:\n{question}\n\n"
        f"Provide a clear answer in {language}, citing sources by their tag "
        f"(e.g. [Source 1]) where relevant."
    )

    if not context_chunks:
        user_prompt = (
            f"No relevant context was found in the indexed documents for this "
            f"question. Politely tell the user, in {language}, that no relevant "
            f"information was found and they may want to rephrase the question.\n\n"
            f"QUESTION:\n{question}"
        )

    response = llm.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    )
    return response.content


# ---------------------------------------------------------------------------
# Summary Chain (map-reduce, scales to 1000+ pages)
# ---------------------------------------------------------------------------
def run_summary_chain(llm, all_chunks: List[Document], language_code: str) -> str:
    """Generate a concise summary of the full document collection using a
    map-reduce strategy so very large collections (1000+ pages) don't
    exceed the LLM's context window."""
    language = _language_name(language_code)

    if not all_chunks:
        return ""

    batches = _batch_chunks(all_chunks, MAP_BATCH_CHAR_BUDGET)

    # MAP step: summarize each batch independently
    partial_summaries = []
    for batch in batches:
        batch_text = "\n\n".join(c.page_content for c in batch)
        system_prompt = (
            "You are a document summarization engine. Summarize the given text "
            "concisely, preserving key facts, figures, names, and obligations. "
            "Write in English regardless of source language (this is an "
            "intermediate summary, not the final user-facing output)."
        )
        user_prompt = f"Summarize the following text in 5-8 bullet points:\n\n{batch_text}"
        response = llm.invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        )
        partial_summaries.append(response.content)

    # REDUCE step: combine partial summaries into one final summary
    combined = "\n\n".join(partial_summaries)
    reduce_system_prompt = (
        f"You are a document summarization engine. Combine the following partial "
        f"summaries into a single, well-organized, non-redundant summary. "
        f"Respond entirely in {language}. Use clear headings and bullet points "
        f"where helpful."
    )
    reduce_user_prompt = (
        f"Partial summaries from different sections of the document collection:\n\n"
        f"{combined}\n\n"
        f"Write a final, cohesive summary in {language}."
    )
    final_response = llm.invoke(
        [
            SystemMessage(content=reduce_system_prompt),
            HumanMessage(content=reduce_user_prompt),
        ]
    )
    return final_response.content


# ---------------------------------------------------------------------------
# Insights Chain (map-reduce)
# ---------------------------------------------------------------------------
def run_insights_chain(llm, all_chunks: List[Document], language_code: str) -> str:
    """Extract key insights/important points from the full document
    collection, using the same map-reduce approach as the summary chain."""
    language = _language_name(language_code)

    if not all_chunks:
        return ""

    batches = _batch_chunks(all_chunks, MAP_BATCH_CHAR_BUDGET)

    partial_insights = []
    for batch in batches:
        batch_text = "\n\n".join(c.page_content for c in batch)
        system_prompt = (
            "You extract key insights, risks, obligations, deadlines, and "
            "noteworthy facts from documents. List them as concise bullet points. "
            "Write in English (intermediate output, not final)."
        )
        user_prompt = f"Extract the key insights from this text:\n\n{batch_text}"
        response = llm.invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        )
        partial_insights.append(response.content)

    combined = "\n\n".join(partial_insights)
    reduce_system_prompt = (
        f"You are an insights extraction engine. Combine the following partial "
        f"insight lists into one deduplicated, prioritized list of the most "
        f"important insights. Respond entirely in {language}, as a clean bullet "
        f"list grouped by theme if appropriate."
    )
    reduce_user_prompt = (
        f"Partial insight lists from different sections:\n\n{combined}\n\n"
        f"Produce the final consolidated key insights in {language}."
    )
    final_response = llm.invoke(
        [
            SystemMessage(content=reduce_system_prompt),
            HumanMessage(content=reduce_user_prompt),
        ]
    )
    return final_response.content


# ---------------------------------------------------------------------------
# Clause Finder Chain
# ---------------------------------------------------------------------------
def run_clause_chain(llm, context_chunks: List[Document], query: str, language_code: str) -> str:
    """Find and explain clauses, policy sections, or regulations relevant
    to the user's query, using retrieved context."""
    language = _language_name(language_code)
    context_text = _format_context(context_chunks)

    system_prompt = (
        f"You are BhashaDoc AI's clause finder. You specialize in locating "
        f"policy clauses, legal sections, and regulatory text within documents. "
        f"For each relevant clause found in the context, quote or closely "
        f"paraphrase it, explain its meaning in plain language, and cite its "
        f"source tag (e.g. [Source 1]). If no relevant clause is found, say so "
        f"clearly. Always respond in {language}."
    )

    user_prompt = (
        f"CONTEXT:\n{context_text}\n\n"
        f"CLAUSE/SECTION TO FIND:\n{query}\n\n"
        f"List all relevant clauses or sections found, with explanations, "
        f"in {language}. Cite sources using their bracketed tags."
    )

    if not context_chunks:
        user_prompt = (
            f"No relevant context was found for this clause search. Politely "
            f"inform the user, in {language}, that no matching clauses were "
            f"found in the indexed documents.\n\nSEARCH TERM:\n{query}"
        )

    response = llm.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    )
    return response.content
