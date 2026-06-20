"""
app.py - BhashaDoc AI

A multilingual RAG application for chatting with large document
collections (1000+ pages) in English, Hindi, and Telugu.

Pipeline: Documents -> Loader -> Cleaning -> Chunking -> Embeddings ->
FAISS Index -> BM25 Index -> Hybrid Retrieval -> Reranking -> Gemini ->
Answer + Citations.
"""

import os
import datetime
import traceback

import streamlit as st
from dotenv import load_dotenv

from rag.loader import load_documents
from rag.chunker import chunk_documents
from rag.embeddings import get_embedding_model
from rag.vectorstore import build_faiss_index
from rag.retriever import HybridRetriever
from rag.llm import get_llm, MissingAPIKeyError
from rag.chains import run_qa_chain, run_summary_chain, run_insights_chain, run_clause_chain
from utils.translator import Translator, SUPPORTED_LANGUAGES, DEFAULT_LANGUAGE
from utils.citations import format_citations_list

load_dotenv()

st.set_page_config(
    page_title="BhashaDoc AI",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------
def init_session_state():
    defaults = {
        "language_code": DEFAULT_LANGUAGE,
        "chunks": [],
        "faiss_index": None,
        "retriever": None,
        "chat_history": [],          # list of dicts: question, answer, citations
        "last_context": [],          # last retrieved chunks (for context viewer)
        "doc_count": 0,
        "page_count": 0,
        "summary_text": "",
        "insights_text": "",
        "clause_results": "",
        "indexed_filenames": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()
translator = Translator(st.session_state["language_code"])


def t(key: str) -> str:
    return translator.translate(key)


# ---------------------------------------------------------------------------
# Indexing logic
# ---------------------------------------------------------------------------
def index_uploaded_files(uploaded_files):
    with st.spinner(t("indexing_in_progress")):
        try:
            raw_documents = load_documents(uploaded_files)
            chunks = chunk_documents(raw_documents)

            if not chunks:
                st.error(t("indexing_error"))
                return

            embedding_model = get_embedding_model()
            faiss_index = build_faiss_index(chunks, embedding_model)
            retriever = HybridRetriever(faiss_index, chunks)

            st.session_state["chunks"] = chunks
            st.session_state["faiss_index"] = faiss_index
            st.session_state["retriever"] = retriever
            st.session_state["doc_count"] = len(
                {c.metadata.get("filename") for c in chunks}
            )
            st.session_state["page_count"] = len(
                {
                    (c.metadata.get("filename"), c.metadata.get("page_number"))
                    for c in chunks
                }
            )
            st.session_state["indexed_filenames"] = sorted(
                {c.metadata.get("filename") for c in chunks}
            )
            st.session_state["summary_text"] = ""
            st.session_state["insights_text"] = ""
            st.session_state["clause_results"] = ""
            st.success(t("indexing_complete"))
        except Exception:
            st.error(t("indexing_error"))
            with st.expander("Error details"):
                st.code(traceback.format_exc())


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header(t("sidebar_header"))

    language_options = list(SUPPORTED_LANGUAGES.keys())
    language_labels = [SUPPORTED_LANGUAGES[code] for code in language_options]
    current_index = language_options.index(st.session_state["language_code"])

    selected_label = st.selectbox(
        t("language_label"), language_labels, index=current_index
    )
    selected_code = language_options[language_labels.index(selected_label)]

    if selected_code != st.session_state["language_code"]:
        st.session_state["language_code"] = selected_code
        st.rerun()

    st.divider()

    st.subheader(t("upload_header"))
    st.caption(t("upload_help"))

    uploaded_files = st.file_uploader(
        t("upload_header"),
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if st.button(t("index_button"), use_container_width=True, type="primary"):
        if not uploaded_files:
            st.warning(t("no_files_warning"))
        else:
            index_uploaded_files(uploaded_files)

    if st.session_state["chunks"]:
        st.divider()
        st.subheader(t("stats_header"))
        col1, col2, col3 = st.columns(3)
        col1.metric(t("stats_documents"), st.session_state["doc_count"])
        col2.metric(t("stats_pages"), st.session_state["page_count"])
        col3.metric(t("stats_chunks"), len(st.session_state["chunks"]))
        with st.expander(t("upload_header")):
            for fname in st.session_state["indexed_filenames"]:
                st.caption(f"📄 {fname}")

    st.divider()
    api_key_present = bool(os.getenv("GOOGLE_API_KEY")) and os.getenv("GOOGLE_API_KEY") != "your_api_key_here"
    if not api_key_present:
        st.error(t("error_no_api_key"))

    st.caption(t("footer_text"))


# ---------------------------------------------------------------------------
# Main header
# ---------------------------------------------------------------------------
st.title(f"📄 {t('app_title')}")
st.caption(t("app_subtitle"))

if not st.session_state["chunks"]:
    st.info(t("welcome_message"))

tab_chat, tab_summary, tab_insights, tab_clauses, tab_context = st.tabs(
    [
        f"💬 {t('tab_chat')}",
        f"📝 {t('tab_summary')}",
        f"💡 {t('tab_insights')}",
        f"🔍 {t('tab_clauses')}",
        f"📚 {t('tab_context')}",
    ]
)


def labels_dict():
    return {
        "file_label": t("file_label"),
        "page_label": t("page_label"),
        "chunk_label": t("chunk_label"),
    }


def require_index() -> bool:
    if not st.session_state["retriever"]:
        st.warning(t("no_documents_warning"))
        return False
    return True


def require_api_key() -> bool:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key == "your_api_key_here":
        st.error(t("error_no_api_key"))
        return False
    return True


# ---------------------------------------------------------------------------
# Tab 1: Chat
# ---------------------------------------------------------------------------
with tab_chat:
    st.subheader(t("chat_header"))

    with st.form(key="chat_form", clear_on_submit=True):
        question = st.text_input(
            t("chat_input_placeholder"),
            label_visibility="collapsed",
            placeholder=t("chat_input_placeholder"),
        )
        submitted = st.form_submit_button(t("ask_button"), type="primary")

    if submitted and question.strip():
        if require_index() and require_api_key():
            with st.spinner(t("thinking_message")):
                try:
                    retriever = st.session_state["retriever"]
                    context_chunks = retriever.retrieve(question, final_k=5)
                    llm = get_llm()
                    answer = run_qa_chain(
                        llm, context_chunks, question, st.session_state["language_code"]
                    )
                    citations = format_citations_list(context_chunks, labels_dict())

                    st.session_state["chat_history"].append(
                        {
                            "question": question,
                            "answer": answer,
                            "citations": citations,
                            "timestamp": datetime.datetime.now().strftime(
                                "%Y-%m-%d %H:%M:%S"
                            ),
                        }
                    )
                    st.session_state["last_context"] = context_chunks
                except MissingAPIKeyError:
                    st.error(t("error_no_api_key"))
                except Exception:
                    st.error(t("error_generic"))
                    with st.expander("Error details"):
                        st.code(traceback.format_exc())

    st.divider()
    st.subheader(t("chat_history_header"))

    if not st.session_state["chat_history"]:
        st.caption(t("no_chat_history"))
    else:
        for entry in reversed(st.session_state["chat_history"]):
            with st.chat_message("user"):
                st.markdown(entry["question"])
            with st.chat_message("assistant"):
                st.markdown(f"**{t('answer_label')}:** {entry['answer']}")
                if entry["citations"]:
                    with st.expander(t("sources_label")):
                        for c in entry["citations"]:
                            st.markdown(f"- {c['label']}")

        col_a, col_b = st.columns(2)
        with col_a:
            chat_lines = []
            for entry in st.session_state["chat_history"]:
                chat_lines.append(f"[{entry['timestamp']}]")
                chat_lines.append(f"Q: {entry['question']}")
                chat_lines.append(f"A: {entry['answer']}")
                if entry["citations"]:
                    chat_lines.append("Sources:")
                    for c in entry["citations"]:
                        chat_lines.append(f"  - {c['label']}")
                chat_lines.append("")
            chat_export = "\n".join(chat_lines)

            st.download_button(
                t("download_chat_button"),
                data=chat_export,
                file_name="bhashadoc_chat_history.txt",
                mime="text/plain",
                use_container_width=True,
            )
        with col_b:
            if st.button(t("clear_chat_button"), use_container_width=True):
                st.session_state["chat_history"] = []
                st.rerun()


# ---------------------------------------------------------------------------
# Tab 2: Summary
# ---------------------------------------------------------------------------
with tab_summary:
    st.subheader(t("summary_header"))

    if st.button(t("summary_button"), type="primary"):
        if require_index() and require_api_key():
            with st.spinner(t("summary_generating")):
                try:
                    llm = get_llm()
                    summary = run_summary_chain(
                        llm, st.session_state["chunks"], st.session_state["language_code"]
                    )
                    st.session_state["summary_text"] = summary
                except MissingAPIKeyError:
                    st.error(t("error_no_api_key"))
                except Exception:
                    st.error(t("error_generic"))
                    with st.expander("Error details"):
                        st.code(traceback.format_exc())

    if st.session_state["summary_text"]:
        st.markdown(st.session_state["summary_text"])
    else:
        st.caption(t("summary_placeholder"))


# ---------------------------------------------------------------------------
# Tab 3: Key Insights
# ---------------------------------------------------------------------------
with tab_insights:
    st.subheader(t("insights_header"))

    if st.button(t("insights_button"), type="primary"):
        if require_index() and require_api_key():
            with st.spinner(t("insights_generating")):
                try:
                    llm = get_llm()
                    insights = run_insights_chain(
                        llm, st.session_state["chunks"], st.session_state["language_code"]
                    )
                    st.session_state["insights_text"] = insights
                except MissingAPIKeyError:
                    st.error(t("error_no_api_key"))
                except Exception:
                    st.error(t("error_generic"))
                    with st.expander("Error details"):
                        st.code(traceback.format_exc())

    if st.session_state["insights_text"]:
        st.markdown(st.session_state["insights_text"])
    else:
        st.caption(t("insights_placeholder"))


# ---------------------------------------------------------------------------
# Tab 4: Clause Finder
# ---------------------------------------------------------------------------
with tab_clauses:
    st.subheader(t("clause_header"))

    with st.form(key="clause_form", clear_on_submit=False):
        clause_query = st.text_input(
            t("clause_input_placeholder"),
            label_visibility="collapsed",
            placeholder=t("clause_input_placeholder"),
        )
        clause_submitted = st.form_submit_button(t("clause_button"), type="primary")

    if clause_submitted and clause_query.strip():
        if require_index() and require_api_key():
            with st.spinner(t("clause_searching")):
                try:
                    retriever = st.session_state["retriever"]
                    context_chunks = retriever.retrieve(clause_query, final_k=6)
                    llm = get_llm()
                    result = run_clause_chain(
                        llm, context_chunks, clause_query, st.session_state["language_code"]
                    )
                    st.session_state["clause_results"] = result
                    st.session_state["last_context"] = context_chunks
                except MissingAPIKeyError:
                    st.error(t("error_no_api_key"))
                except Exception:
                    st.error(t("error_generic"))
                    with st.expander("Error details"):
                        st.code(traceback.format_exc())

    if st.session_state["clause_results"]:
        st.markdown(st.session_state["clause_results"])
    else:
        st.caption(t("clause_placeholder"))


# ---------------------------------------------------------------------------
# Tab 5: Retrieved Context Viewer
# ---------------------------------------------------------------------------
with tab_context:
    st.subheader(t("context_header"))

    if not st.session_state["last_context"]:
        st.caption(t("no_context_available"))
    else:
        with st.expander(t("context_expander_label"), expanded=True):
            citations = format_citations_list(
                st.session_state["last_context"], labels_dict()
            )
            for c in citations:
                st.markdown(f"**{c['label']}**")
                st.text(c["content"])
                st.divider()
