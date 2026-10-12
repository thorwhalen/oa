"""Pytest configuration: skip tests that call the OpenAI API when no key is set.

``import oa`` needs no key (it is resolved on first use), so there is no dummy key:
a dummy would make "is a key available?" checks pass and the API calls fail with 401.
Tests carry ``@pytest.mark.needs_api``; doctests that call the API are listed in
``API_DOCTESTS`` (a doctest can't carry a marker).
"""

import os

import pytest
from _pytest.doctest import DoctestItem

API_DOCTESTS = frozenset({"oa.base.embeddings", "oa.tools.constrained_answer"})


def _has_api_key() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY", "").strip())


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "needs_api: calls the OpenAI API; skipped when OPENAI_API_KEY is unset",
    )


def pytest_collection_modifyitems(config, items):
    if _has_api_key():
        return
    skip = pytest.mark.skip(reason="needs OPENAI_API_KEY (calls the OpenAI API)")
    for item in items:
        if "needs_api" in item.keywords or (
            isinstance(item, DoctestItem) and item.name in API_DOCTESTS
        ):
            item.add_marker(skip)
