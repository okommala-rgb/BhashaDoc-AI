# BhashaDoc AI — Project Constitution

## Purpose

BhashaDoc AI is a multilingual Retrieval-Augmented Generation (RAG) application
that allows users to upload and converse with large document collections
(1000+ pages) in English, Hindi, and Telugu.

## Core Principles

 **Accuracy over fluency** — Answers must be grounded in retrieved document
   context. The system must never fabricate citations or page numbers.
 **Multilingual parity** — Every user-facing feature must be available and
   fully translated in all supported languages (English, Hindi, Telugu).
 **Transparency** — Every answer must show its source citations (filename,
   page number, chunk id) and allow the user to inspect retrieved context.
 **Scalability** — The retrieval pipeline must remain performant on
   document collections exceeding 1000 pages, using chunked indexing and
   hybrid (semantic + keyword) retrieval rather than naive full-context
   stuffing.
 **Privacy** — Uploaded documents and API keys must never be committed to
   version control or logged in plaintext.

## Governance

- Changes to retrieval, chunking, or prompting logic require a corresponding
  update to `specs/document-qa/spec.md`.
- Breaking changes to the public interface (Streamlit UI, chain signatures)
  must be reflected in `CHANGELOG.md`.
- All code must pass lint (ruff, flake8, pylint), type-check (mypy), and
  security scan (bandit, semgrep) before merge.

## Amendment Process

This constitution may be amended via pull request with a clear rationale in
the commit message. Amendments should preserve backward compatibility with
existing indexed document stores where possible.
