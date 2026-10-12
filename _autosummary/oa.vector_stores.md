# oa.vector_stores

Vector stores and search

### Functions

| [`bind_and_modify`](#oa.vector_stores.bind_and_modify)(func, \*bound_args[, ...])         | Convenience function that both binds arguments and modifies signature.       |
|-----------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------|
| [`docs_to_search_func_factory_via_vector_store`](#oa.vector_stores.docs_to_search_func_factory_via_vector_store)(docs) | Factory function that creates a search function via vector store.            |
| [`docs_to_vector_store`](#oa.vector_stores.docs_to_vector_store)(docs[, vs_name, client])      | Create an OpenAI vector store from a mapping of documents.                   |
| [`mk_search_func_for_oa_vector_store`](#oa.vector_stores.mk_search_func_for_oa_vector_store)(...[, ...])     | Create a search function for an OpenAI vector store using the Responses API. |

### oa.vector_stores.bind_and_modify(func, \*bound_args, \_param_changes=(), \*\*bound_kwargs)

Convenience function that both binds arguments and modifies signature.

This is perfect for your vector store search use case.

* **Parameters:**
  * **func** – The function to wrap
  * **bound_args** – Positional arguments to bind
  * **bound_kwargs** – Dict of argument names to values to bind
  * **\_param_changes** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)) – Parameter modifications
* **Returns:**
  Wrapped function with bound arguments and modified signature

Example for your vector store case:

```pycon
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
```

### oa.vector_stores.docs_to_search_func_factory_via_vector_store(docs)

Factory function that creates a search function via vector store.
This can be used with check_search_func_factory.

* **Return type:**
  [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]]

### oa.vector_stores.docs_to_vector_store(docs, vs_name=None, , client=None)

Create an OpenAI vector store from a mapping of documents.

* **Parameters:**
  * **docs** ([`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Mapping of document keys to text content
  * **vs_name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Optional name for the vector store. If None, generates a unique name.
* **Returns:**
  (vector_store_id, file_id_to_doc_key_mapping)
* **Return type:**
  [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]

### oa.vector_stores.mk_search_func_for_oa_vector_store(vector_store_id, doc_id_mapping=None, , client=None)

Create a search function for an OpenAI vector store using the Responses API.

* **Parameters:**
  * **vector_store_id** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – The ID of the vector store to search
  * **doc_id_mapping** ([`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Optional mapping from file IDs to document keys for result translation
* **Return type:**
  [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]]
* **Returns:**
  A function that takes a query and returns search results
