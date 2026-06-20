"""
translator.py - i18n/l10n module for BhashaDoc AI

Loads translation dictionaries from locales/*.json and exposes a
simple Translator class with load_language() and translate() methods,
used to drive every visible UI string in app.py.
"""

import json
import os
from functools import lru_cache
from typing import Dict

LOCALES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "locales")

SUPPORTED_LANGUAGES = {
    "en": "English",
    "hi": "हिन्दी (Hindi)",
    "te": "తెలుగు (Telugu)",
}

DEFAULT_LANGUAGE = "en"


@lru_cache(maxsize=8)
def _load_locale_file(language_code: str) -> Dict[str, str]:
    path = os.path.join(LOCALES_DIR, f"{language_code}.json")
    if not os.path.isfile(path):
        path = os.path.join(LOCALES_DIR, f"{DEFAULT_LANGUAGE}.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class Translator:
    """Loads a language's translation map and provides key -> string lookup."""

    def __init__(self, language_code: str = DEFAULT_LANGUAGE):
        self.language_code = language_code
        self.translations: Dict[str, str] = {}
        self.load_language(language_code)

    def load_language(self, language_code: str) -> None:
        """Load (or switch to) a given language code's translation file."""
        if language_code not in SUPPORTED_LANGUAGES:
            language_code = DEFAULT_LANGUAGE
        self.language_code = language_code
        self.translations = _load_locale_file(language_code)

    def translate(self, key: str) -> str:
        """Return the translated string for a key, falling back to the
        English value, and finally to the key itself, if missing."""
        if key in self.translations:
            return self.translations[key]
        fallback = _load_locale_file(DEFAULT_LANGUAGE)
        return fallback.get(key, key)

    def t(self, key: str) -> str:
        """Shorthand alias for translate()."""
        return self.translate(key)
