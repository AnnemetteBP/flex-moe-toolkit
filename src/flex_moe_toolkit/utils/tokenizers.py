from __future__ import annotations

from transformers import AutoTokenizer


def load_tokenizer_with_known_fixes(path_or_name: str, **kwargs):
    """Load a tokenizer while applying safe compatibility fixes when supported."""

    try:
        return AutoTokenizer.from_pretrained(path_or_name, fix_mistral_regex=True, **kwargs)
    except TypeError:
        return AutoTokenizer.from_pretrained(path_or_name, **kwargs)
