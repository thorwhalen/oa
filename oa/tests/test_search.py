"""Test search functionality"""

from collections.abc import Callable
from functools import wraps

import pytest

# ------------------------------------------------------------------------------
# Search functionality Testing Utils

from oa.vector_stores import Query, MaxNumResults, ResultT, SearchResults


def top_results_contain(results: SearchResults, expected: SearchResults) -> bool:
    """
    Check that the top results contain the expected elements.
    That is, the first len(expected) elements of results match the expected set,
    and if there are less results than expected, the only elements in results are
    contained in expected.
    """
    if len(results) < len(expected):
        return set(results) <= set(expected)
    return set(results[: len(expected)]) == set(expected)


def general_test_for_search_function(
    query,
    top_results_expected_to_contain: SearchResults,
    *,
    search_func: Callable[[Query], SearchResults],
    n_top_results=None,
):
    """
    General test function for search functionality.

    Args:
        query: Query string
        top_results_expected_to_contain: Set of expected document keys
        search_func: Search function to test (keyword-only)
        n_top_results: Number of top results to check. If None, defaults to min(len(results), len(top_results_expected_to_contain)) (keyword-only)

    Example use:

    >>> def search_docs_containing(query):
    ...     docs = {'doc1': 'apple pie recipe', 'doc2': 'car maintenance guide', 'doc3': 'apple varieties'}
    ...     return (key for key, text in docs.items() if query in text)
    >>> general_test_for_search_function(
    ...     query='apple',
    ...     top_results_expected_to_contain={'doc1', 'doc3'},
    ...     search_func=search_docs_containing
    ... )
    """
    # Execute search and collect results
    # TODO: Protect from cases where search_func(query) could be a long generator? Example, a max_results limit?
    results = list(search_func(query))

    # Determine the actual number of top results to check
    if n_top_results is None:
        effective_n_top_results = min(
            len(results), len(top_results_expected_to_contain)
        )
    else:
        effective_n_top_results = n_top_results

    # Get the slice of results to check
    top_results_to_check = results[:effective_n_top_results]

    # Generate helpful error message
    error_context = []
    error_context.append(f"Query: '{query}'")
    error_context.append(f"Expected docs: {top_results_expected_to_contain}")
    error_context.append(f"Actual results: {results}")
    error_context.append(
        f"Checking top {effective_n_top_results} results: {top_results_to_check}"
    )

    error_message = "\n".join(error_context)

    # Perform the assertion
    assert top_results_contain(top_results_to_check, top_results_expected_to_contain), (
        error_message
    )


# ─── Test Documents ────────────────────────────────────────────────────────────
docs = {
    "python": "Python is a high‑level programming language emphasizing readability and rapid development.",
    "java": "Java is a class‑based, object‑oriented language designed for portability across platforms.",
    "numpy": "NumPy provides support for large, multi‑dimensional arrays and matrices, along with a collection of mathematical functions.",
    "pandas": "Pandas is a Python library offering data structures and operations for manipulating numerical tables and time series.",
    "apple": "Apple is a fruit that grows on trees and comes in varieties such as Granny Smith, Fuji, and Gala.",
    "banana": "Banana is a tropical fruit with a soft, sweet interior and a peel that changes from green to yellow when ripe.",
    "microsoft": "Microsoft develops software products including the Windows operating system, Office suite, and cloud services.",
}

# ─── Semantic Search Examples ─────────────────────────────────────────────────


def check_search_func(
    search_func: Callable[[Query], SearchResults],
):
    """
    Test the search function with multiple queries using the general test framework.
    """
    # Test case 1: programming language search
    general_test_for_search_function(
        query="object‑oriented programming",
        top_results_expected_to_contain={"java", "python", "numpy"},
        search_func=search_func,
    )

    # Test case 2: fruit category search
    general_test_for_search_function(
        query="tropical fruit",
        top_results_expected_to_contain={"banana", "apple"},
        search_func=search_func,
    )


# ─── Retrieval‑Augmented Generation Example ────────────────────────────────────


def check_find_docs_to_answer_question(
    find_docs_to_answer_question: Callable[[Query], SearchResults],
):
    """
    Test the function that finds documents relevant to a question.
    """
    general_test_for_search_function(
        query="Which documents describe a fruit that is sweet and easy to eat?",
        top_results_expected_to_contain={"apple", "banana"},
        search_func=find_docs_to_answer_question,
    )


# ─── test these test functions with a docs_to_search_func factory function ──────


def check_search_func_factory(
    search_func_factory: Callable[[dict], Callable[[Query], SearchResults]],
):
    """
    Test the search function factory with a set of documents.
    """
    search_func = search_func_factory(docs)

    # Run the search function tests
    check_search_func(search_func)
    check_find_docs_to_answer_question(search_func)


# ------------------------------------------------------------------------------
# Tests

import oa.vector_stores as vector_stores_module
from oa.vector_stores import (
    OaStores,
    OaVectorStoreFiles,
    docs_to_vector_store,
    mk_search_func_for_oa_vector_store,
    docs_to_search_func_factory_via_vector_store,
)


def _delete_vector_store(vs_id, file_ids, *, client=None):
    """Best-effort removal of a test vector store and the files uploaded for it."""
    client = OaStores(client).client if client is None else client  # (or a fake)
    try:
        client.vector_stores.delete(vs_id)
    finally:
        for file_id in file_ids:
            try:
                client.files.delete(file_id)
            except Exception:  # already gone: nothing left to clean up
                pass


@pytest.mark.needs_api
def test_vector_store_search_functionality():
    """Create a vector store from ``docs``, search it, and always delete it."""
    vs_id, file_mapping = docs_to_vector_store(docs, "test_search_vs")
    try:
        assert vs_id and len(file_mapping) == len(docs)
        search_func = mk_search_func_for_oa_vector_store(vs_id, file_mapping)
        assert callable(search_func)
        assert isinstance(search_func("programming"), list)
    finally:
        _delete_vector_store(vs_id, file_mapping)


def _recording(real_docs_to_vector_store, created: list):
    """Wrap ``docs_to_vector_store`` so that every store it makes is noted in ``created``
    as ``(vector_store_id, {file_id: doc_key})``."""

    @wraps(real_docs_to_vector_store)
    def docs_to_vector_store_that_records(*args, **kwargs):
        vs_id, file_mapping = real_docs_to_vector_store(*args, **kwargs)
        created.append((vs_id, file_mapping))
        return vs_id, file_mapping

    return docs_to_vector_store_that_records


def _delete_all(created: list, *, client=None):
    """Delete every recorded store (and its files), then raise if any deletion failed,
    so that a leak is a visible failure rather than a silent one."""
    failures = []
    for vs_id, file_mapping in created:
        try:
            _delete_vector_store(vs_id, file_mapping, client=client)
        except Exception as error:
            failures.append(f"{vs_id}: {error!r}")
    if failures:
        raise AssertionError(f"could not delete test vector stores: {failures}")


def _record_created_vector_stores(monkeypatch) -> list:
    """Make ``oa.vector_stores.docs_to_vector_store`` (which the search factory calls)
    note every store it creates in the returned list."""
    created = []
    monkeypatch.setattr(
        vector_stores_module,
        "docs_to_vector_store",
        _recording(vector_stores_module.docs_to_vector_store, created),
    )
    return created


@pytest.fixture
def created_vector_stores(monkeypatch):
    """The vector stores that ``docs_to_vector_store`` makes during the test, deleted
    (with their files) afterwards, whether the test passed or not."""
    created = _record_created_vector_stores(monkeypatch)
    yield created
    _delete_all(created)


@pytest.mark.needs_api
def test_vector_store_search_with_api(created_vector_stores):
    """Run the full vector store search test with actual API calls."""
    check_search_func_factory(docs_to_search_func_factory_via_vector_store)
    assert created_vector_stores  # (they are deleted by the fixture)


# ------------------------------------------------------------------------------
# The clean-up machinery itself, with a fake client: no API call, no key needed


class _FakeClient:
    def __init__(self, *, failing_vector_stores=()):
        self.deleted_vector_stores, self.deleted_files = [], []
        self._failing = set(failing_vector_stores)
        client = self

        class _VectorStores:
            def delete(self, vs_id):
                if vs_id in client._failing:
                    raise RuntimeError("boom")
                client.deleted_vector_stores.append(vs_id)

        class _Files:
            def delete(self, file_id):
                client.deleted_files.append(file_id)

        self.vector_stores, self.files = _VectorStores(), _Files()


def test_recording_notes_what_the_wrapped_function_makes():
    created = []
    wrapped = _recording(lambda docs, name=None: ("vs_1", {"f1": "a"}), created)
    assert wrapped({"a": "x"}, "n") == ("vs_1", {"f1": "a"})
    assert created == [("vs_1", {"f1": "a"})]


def test_the_search_factory_goes_through_the_recorded_function(monkeypatch):
    """What the fixture relies on: the factory looks ``docs_to_vector_store`` up in the
    module, so the recorder sees the store it creates (fakes: no API call)."""
    monkeypatch.setattr(
        vector_stores_module, "docs_to_vector_store", lambda docs: ("vs_9", {"f9": "k"})
    )
    monkeypatch.setattr(
        vector_stores_module,
        "mk_search_func_for_oa_vector_store",
        lambda vs_id, file_mapping: lambda query: [],
    )
    created = _record_created_vector_stores(monkeypatch)
    search = docs_to_search_func_factory_via_vector_store({"k": "text"})
    assert search("q") == [] and created == [("vs_9", {"f9": "k"})]


def test_delete_all_removes_every_store_and_its_files():
    client = _FakeClient()
    _delete_all(
        [("vs_1", {"f1": "a", "f2": "b"}), ("vs_2", {"f3": "c"})], client=client
    )
    assert client.deleted_vector_stores == ["vs_1", "vs_2"]
    assert client.deleted_files == ["f1", "f2", "f3"]


def test_delete_all_still_deletes_the_rest_and_the_files_when_one_fails():
    client = _FakeClient(failing_vector_stores={"vs_1"})
    with pytest.raises(AssertionError, match="vs_1"):
        _delete_all([("vs_1", {"f1": "a"}), ("vs_2", {"f2": "b"})], client=client)
    assert client.deleted_vector_stores == ["vs_2"]
    assert client.deleted_files == ["f1", "f2"]


# ------------------------------------------------------------------------------
# Test runner

if __name__ == "__main__":
    print("=== Running Vector Store Search Tests ===")
    test_vector_store_search_functionality()

    print("\n=== Additional Test Functions Available ===")
    print("To run full API tests, call:")
    print("  test_vector_store_search_with_api()")
    print("  check_search_func_factory(docs_to_search_func_factory_via_vector_store)")
