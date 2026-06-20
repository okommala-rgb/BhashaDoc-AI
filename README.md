# BhashaDoc AI

A multilingual Retrieval-Augmented Generation (RAG) application that lets you upload and chat with large document collections (1000+ pages) in **English, Hindi, and Telugu**.

---

## Project Overview

BhashaDoc AI ("Bhasha" = language in Sanskrit/Hindi) is built for scenarios where users need to interrogate large sets of PDFs, Word documents, and text files — policies, contracts, regulations, reports — and get accurate, citation-backed answers in their preferred Indian language, regardless of what language the source documents are written in.

It combines semantic search (FAISS), keyword search (BM25), and cross-encoder reranking into a hybrid retrieval pipeline, then uses **Google Gemini 2.5 Flash** to generate grounded answers with page-level citations.

---

## Features

- 📂 **Multi-file upload** — PDF, DOCX, TXT, including collections exceeding 1000 pages
- 💬 **Chat with documents** — ask natural-language questions, get grounded answers
- 📌 **Source citations** — every answer references filename, page number, and chunk ID
- 🕘 **Conversation history** — stored per session, viewable and exportable
- 📥 **Download chat** — export the full Q&A history as a `.txt` file
- 📝 **Document summary** — map-reduce summarization scales to huge collections
- 💡 **Key insights** — automatic extraction of important points, risks, and obligations
- 🔍 **Clause finder** — locate policy clauses, legal sections, and regulations
- 📚 **Retrieved context viewer** — see exactly which chunks were used for the last answer
- 🌐 **Full i18n** — every UI label is translated; switch language instantly from the sidebar
- 🌓 **Dark-mode compatible** — clean, professional Streamlit theme

---

## Architecture

```
                 ┌─────────────┐
                 │  Documents  │  (PDF / DOCX / TXT)
                 └──────┬──────┘
                        ▼
                 ┌─────────────┐
                 │   Loader    │  per-file, per-page extraction + metadata
                 └──────┬──────┘
                        ▼
                 ┌─────────────┐
                 │  Cleaning   │  whitespace / control-char normalization
                 └──────┬──────┘
                        ▼
                 ┌─────────────┐
                 │  Chunking   │  RecursiveCharacterTextSplitter
                 │             │  size=1000, overlap=200
                 └──────┬──────┘
                        ▼
                 ┌─────────────┐
                 │ Embeddings  │  sentence-transformers/all-MiniLM-L6-v2
                 └──────┬──────┘
                        ▼
            ┌───────────┴───────────┐
            ▼                       ▼
     ┌─────────────┐         ┌─────────────┐
     │ FAISS Index │         │  BM25 Index │
     │ (semantic)  │         │  (keyword)  │
     └──────┬──────┘         └──────┬──────┘
            └───────────┬───────────┘
                         ▼
                ┌──────────────────┐
                │ Hybrid Retrieval │  Reciprocal Rank Fusion
                └────────┬─────────┘
                         ▼
                ┌──────────────────┐
                │    Reranking      │  cross-encoder/ms-marco-MiniLM-L-6-v2
                └────────┬─────────┘
                         ▼
                ┌──────────────────┐
                │  Gemini 2.5 Flash │
                └────────┬─────────┘
                         ▼
                ┌──────────────────┐
                │ Answer + Citations│
                └──────────────────┘
```

---

## Project Structure

```
BhashaDoc-AI/
│
├── app.py                     # Streamlit entry point
│
├── rag/
│   ├── __init__.py
│   ├── loader.py               # PDF / DOCX / TXT loading
│   ├── chunker.py              # RecursiveCharacterTextSplitter chunking
│   ├── embeddings.py           # all-MiniLM-L6-v2 embedding model
│   ├── vectorstore.py          # FAISS create / save / load
│   ├── retriever.py            # FAISS + BM25 hybrid retrieval + reranking
│   ├── llm.py                  # Gemini 2.5 Flash wrapper
│   └── chains.py                # QA / Summary / Insights / Clause chains
│
├── utils/
│   ├── __init__.py
│   ├── translator.py           # i18n loader (load_language / translate)
│   └── citations.py            # citation formatting
│
├── locales/
│   ├── en.json
│   ├── hi.json
│   └── te.json
│
├── data/                       # local scratch space (gitignored)
│
├── .streamlit/
│   └── config.toml             # dark-mode-friendly theme
│
├── .env.example
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Installation

```bash
git clone https://github.com/<your-username>/BhashaDoc-AI.git
cd BhashaDoc-AI

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

> The first run will download the `all-MiniLM-L6-v2` embedding model and the `ms-marco-MiniLM-L-6-v2` cross-encoder (a few hundred MB total) — this happens once and is cached locally.

---

## Environment Variables

Copy `.env.example` to `.env` and add your Gemini API key:

```bash
cp .env.example .env
```

```
GOOGLE_API_KEY=your_api_key_here
```

Get a free Gemini API key at: https://aistudio.google.com/apikey

---

## Running Locally

```bash
streamlit run app.py
```

Then open the URL shown in the terminal (usually `http://localhost:8501`).

**Usage:**
1. Pick your language in the sidebar (English / Hindi / Telugu).
2. Upload one or more PDF / DOCX / TXT files.
3. Click **Index Documents** and wait for indexing to finish.
4. Use the **Chat**, **Summary**, **Key Insights**, and **Clause Finder** tabs.
5. Check the **Retrieved Context** tab to see exactly what the model read.
6. Download your chat history any time from the Chat tab.

---

## GitHub Setup

```bash
git init
git add .
git commit -m "Initial commit: BhashaDoc AI"
git branch -M main
git remote add origin https://github.com/<your-username>/BhashaDoc-AI.git
git push -u origin main
```

`.env` is gitignored by default — never commit your real API key.

---

## Streamlit Cloud Deployment

1. Push this repository to GitHub (see above).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in.
3. Click **New app**, select your repo, branch `main`, and main file `app.py`.
4. Under **Advanced settings → Secrets**, add:
   ```
   GOOGLE_API_KEY = "your_api_key_here"
   ```
5. Click **Deploy**. The app will build and go live at `https://<app-name>.streamlit.app`.

---

## Future Improvements

- Persist FAISS/BM25 indexes to disk across sessions (multi-user, multi-collection support)
- Add more Indian languages (Tamil, Kannada, Marathi, Bengali)
- Streaming token-by-token answers
- Per-document access control and collection management
- OCR support for scanned PDFs
- Async/batched embedding generation for faster indexing of very large collections

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| Backend | Python |
| LLM | Google Gemini 2.5 Flash |
| RAG Orchestration | LangChain |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector DB | FAISS |
| Keyword Search | BM25 (rank-bm25) |
| Reranking | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Config | python-dotenv |
