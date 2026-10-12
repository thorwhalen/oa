"""Pytest configuration: skip tests that call the OpenAI API when no key is available.

``import oa`` needs no key (it is resolved on first use), so there is no dummy key:
a dummy would make "is a key available?" checks pass and the API calls fail with 401.
Tests carry ``@pytest.mark.needs_api``; doctests that call the API are listed in
``API_DOCTESTS`` (a doctest can't carry a marker).
"""

import pytest
from _pytest.doctest import DoctestItem

API_DOCTESTS = frozenset({"oa.base.embeddings", "oa.tools.constrained_answer"})


def _has_api_key() -> bool:
    """Whether oa can find a key (its config store or the environment) without asking."""
    from oa.util import _lookup_api_key_without_asking

    return bool(_lookup_api_key_without_asking())


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "needs_api: calls the OpenAI API; skipped when no OpenAI key is found",
    )


def pytest_collection_modifyitems(config, items):
    if _has_api_key():
        return
    skip = pytest.mark.skip(reason="needs OPENAI_API_KEY (calls the OpenAI API)")
    for item in items:
        if item.get_closest_marker("needs_api") is not None or (
            isinstance(item, DoctestItem) and item.name in API_DOCTESTS
        ):
            item.add_marker(skip)
