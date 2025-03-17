# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Utilities for working with tokens."""

import logging

import tiktoken
from transformers import AutoTokenizer

import graphrag.config.defaults as defs

DEFAULT_ENCODING_NAME = defs.ENCODING_MODEL

log = logging.getLogger(__name__)


def num_tokens_from_string(
    string: str, model: str | None = None, encoding_name: str | None = None
) -> int:
    """Return the number of tokens in a text string."""
    if model is not None:
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            # msg = f"Failed to get encoding for {model} when getting num_tokens_from_string. Fall back to default encoding {DEFAULT_ENCODING_NAME}"
            # log.warning(msg)
            # encoding = tiktoken.get_encoding(DEFAULT_ENCODING_NAME)
            encoding = AutoTokenizer.from_pretrained(model)
    else:
        encoding = tiktoken.get_encoding(encoding_name or DEFAULT_ENCODING_NAME) 
    if isinstance(encoding, tiktoken.Encoding):
        num_tokens = len(encoding.encode(string))
    else:
        num_tokens = len(encoding.encode(string, add_special_tokens=False))
    return num_tokens


def string_from_tokens(
    tokens: list[int], model: str | None = None, encoding_name: str | None = None
) -> str:
    """Return a text string from a list of tokens."""
    if model is not None:
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = AutoTokenizer.from_pretrained(model)
    elif encoding_name is not None:
        encoding = tiktoken.get_encoding(encoding_name)
    else:
        msg = "Either model or encoding_name must be specified."
        raise ValueError(msg)
    return encoding.decode(tokens)
