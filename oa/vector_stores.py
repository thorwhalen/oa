"""Vector stores and search"""

import tempfile
from functools import partial
import os
from typing import Callable, Mapping, Iterable, Any
from oa.stores import OaStores, OaVectorStoreFiles, OaFiles

Query = str
MaxNumResults = int
ResultT = Any
SearchResults = Iterable[ResultT]


def docs_to_vector_store(
    docs: Mapping[str, str], vs_name: str = None, *, client=None
) -> tuple[str, dict[str, str]]:
    """
    Create an OpenAI vector store from a mapping of documents.

    Args:
        docs: Mapping of document keys to text content
        vs_name: Optional name for the vector store. If None, generates a unique name.

    Returns:
        tuple: (vector_store_id, file_id_to_doc_key_mapping)
    """
    if vs_name is None:
        import uuid

        vs_name = f"test_vs_{uuid.uuid4().hex[:8]}"

    # Initialize OA stores
    oa_stores = OaStores(client)

    # Create vector store
    vector_store = oa_stores.vector_stores_base.create(vs_name)

    # Create temporary files for each document and upload them
    vs_files = OaVectorStoreFiles(vector_store.id, oa_stores.client)
    file_id_to_doc_key = {}

    for doc_key, doc_text in docs.items():
        # Create a temporary file with the document content
        # Use doc_key as filename for easier identification
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", prefix=f"{doc_key}_", delete=False
        ) as tmp_file:
            tmp_file.write(doc_text)
            tmp_file.flush()

            # Upload the file to OpenAI with proper purpose for assistants/vector stores
            with open(tmp_file.name, "rb") as file_content:
                # Create a file store with the correct purpose for assistants/vector stores
                assistants_files = OaFiles(client, purpose="assistants")
                file_obj = assistants_files.append(file_content)

            # Add the file to the vector store
            vs_files.add_file(file_obj.id)

            # Store the mapping
            file_id_to_doc_key[file_obj.id] = doc_key

            # Clean up temporary file
            os.unlink(tmp_file.name)

    return vector_store.id, file_id_to_doc_key


from functools import partial
from inspect import Parameter
from i2 import Sig
from i2.wrapper import Ingress, wrap


def bind_and_modify(func, *bound_args, _param_changes: dict = (), **bound_kwargs):
    """
    Convenience function that both binds arguments and modifies signature.

    This is perfect for your vector store search use case.

    :param func: The function to wrap
    :param bound_args: Positional arguments to bind
    :param bound_kwargs: Dict of argument names to values to bind
    :param _param_changes: Parameter modifications
    :return: Wrapped function with bound arguments and modified signature

    Example for your vector store case:

    >>> def search(vector_store_id, *, query, filters=None, max_results=10):
    ...     return f"Searching {vector_store_id} for '{query}'"
    >>>
    >>> # Bind vector_store_id and make query positional
    >>> bound_search = bind_and_modify(
    ...     search,
    ...     vector_store_id='my_store',
    ...     _param_changes=dict(query={'kind': Parameter.POSITIONAL_OR_KEYWORD}),
    ... )
    >>>
    >>> bound_search('my query')
    "Searching my_store for 'my query'"
    """
    from i2.wrapper import Ingress, wrap

    # Get original signature and determine what we're binding
    original_sig = Sig(func)
    bound_kwargs = dict(bound_kwargs)

    # Map the bound arguments to parameter names
    bound_params = original_sig.map_arguments(
        bound_args,
        bound_kwargs,
        allow_partial=True,
    )

    # Remove bound parameters from signature
    remaining_sig = original_sig - list(bound_params.keys())

    # Apply parameter modifications to the remaining signature
    if _param_changes:
        remaining_sig = remaining_sig.modified(**_param_changes)

    # Create an ingress that transforms outer args/kwargs to inner args/kwargs
    def kwargs_trans(outer_kwargs):
        # Start with the bound parameters
        inner_kwargs = dict(bound_params)
        # Add the outer kwargs
        inner_kwargs.update(outer_kwargs)
        return inner_kwargs

    # Create ingress with the modified signature as outer and original as inner
    ingress = Ingress(
        outer_sig=remaining_sig, kwargs_trans=kwargs_trans, inner_sig=original_sig
    )

    # Wrap the function
    return wrap(func, ingress=ingress)


def mk_search_func_for_oa_vector_store(
    vector_store_id: str, doc_id_mapping: Mapping[str, str] = None, *, client=None
) -> Callable[[Query], SearchResults]:
    """
    Create a search function for an OpenAI vector store.

    Args:
        vector_store_id: The ID of the vector store to search
        doc_id_mapping: Optional mapping from file IDs to document keys for result translation

    Returns:
        A function that takes a query and returns search results
    """
    oa_stores = OaStores(client)

    # Create the basic search function with bound vector_store_id
    basic_search = bind_and_modify(
        oa_stores.client.vector_stores.search,
        vector_store_id,
        _param_changes=dict(query={"kind": Parameter.POSITIONAL_OR_KEYWORD}),
    )

    # If no doc_id_mapping, just return the basic search
    if not doc_id_mapping:
        return basic_search

    # Create a wrapper that maps file IDs back to document keys
    def search_with_doc_mapping(query, **kwargs):
        """Search function that maps results back to original document keys"""
        results = basic_search(query, **kwargs)

        # Map file IDs back to document keys
        mapped_results = []
        for result in results:
            # Assuming result has a file_id attribute or similar
            # This might need adjustment based on actual OpenAI response structure
            if hasattr(result, "file_id") and result.file_id in doc_id_mapping:
                # Replace or add the document key
                mapped_result = result
                mapped_result.doc_key = doc_id_mapping[result.file_id]
                mapped_results.append(mapped_result.doc_key)  # Return just the doc key
            elif hasattr(result, "id") and result.id in doc_id_mapping:
                mapped_results.append(doc_id_mapping[result.id])
            else:
                # If we can't find a mapping, include the original result
                mapped_results.append(result)

        return mapped_results

    return search_with_doc_mapping


def docs_to_search_func_factory_via_vector_store(
    docs: Mapping[str, str],
) -> Callable[[Query], SearchResults]:
    """
    Factory function that creates a search function via vector store.
    This can be used with check_search_func_factory.
    """
    # Create vector store from docs
    vs_id, file_id_mapping = docs_to_vector_store(docs)

    # Create and return search function with proper mapping
    return mk_search_func_for_oa_vector_store(vs_id, file_id_mapping)
