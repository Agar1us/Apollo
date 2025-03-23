# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Utilities for working with tokens."""

import logging

from graphrag.language_model.tokenizer import SingletonTokenizer

import graphrag.config.defaults as defs

DEFAULT_ENCODING_NAME = defs.ENCODING_MODEL

log = logging.getLogger(__name__)


def num_tokens_from_string(
    string: str, model: str | None = None, encoding_name: str | None = None
) -> int:
    """Return the number of tokens in a text string."""
    encoding = SingletonTokenizer(model or encoding_name)
    return len(encoding.encode(string))


def string_from_tokens(
    tokens: list[int], model: str | None = None, encoding_name: str | None = None
) -> str:
    """Return a text string from a list of tokens."""
    encoding = SingletonTokenizer(model or encoding_name)
    return encoding.decode(tokens)
