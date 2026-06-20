"""
llm.py - Gemini 2.5 Flash LLM wrapper for BhashaDoc AI

Reads the API key from the GOOGLE_API_KEY environment variable
(populated via python-dotenv from a .env file) and exposes a
cached LangChain ChatGoogleGenerativeAI instance.
"""

import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

GEMINI_MODEL_NAME = "gemini-2.5-flash"


class MissingAPIKeyError(RuntimeError):
    """Raised when GOOGLE_API_KEY is not configured."""


def get_api_key() -> str:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key.strip() == "" or api_key == "your_api_key_here":
        raise MissingAPIKeyError(
            "GOOGLE_API_KEY not found. Please set it in your .env file."
        )
    return api_key


@lru_cache(maxsize=1)
def get_llm(temperature: float = 0.2) -> ChatGoogleGenerativeAI:
    """Return a cached Gemini 2.5 Flash chat model instance."""
    api_key = get_api_key()
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL_NAME,
        google_api_key=api_key,
        temperature=temperature,
        max_output_tokens=2048,
    )
