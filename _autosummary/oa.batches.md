# oa.batches

Batch functionality

Useful links:

- Web tool: [https://platform.openai.com/batches/](https://platform.openai.com/batches/)
- API docs: [https://platform.openai.com/docs/api-reference/batch](https://platform.openai.com/docs/api-reference/batch)

### Functions

| `batch_info_to_segments_and_embeddings`(...)                                                       |                                                                         |
|----------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------|
| `batches_within_range`(batches_base, min_date)                                                     |                                                                         |
| `get_batch_id_and_obj`(oa_stores, batch)                                                           |                                                                         |
| `get_batch_obj`(oa_stores, batch)                                                                  |                                                                         |
| [`get_output_file_data`](#oa.batches.get_output_file_data)(batch, \*, oa_stores[, ...]) | Get the output file data for a batch, if it has completed successfully. |
| `get_segments_and_embeddings`(batch_store, ...)                                                    |                                                                         |
| [`mk_batch_file_embeddings_task`](#oa.batches.mk_batch_file_embeddings_task)(texts, \*[, ...])   | Create a batch task (json-)dicts for generating embeddings.             |
| [`on_complete`](#oa.batches.on_complete)(oa_stores, batch_obj)                 | Return the output file if the batch completed successfully              |
| `raise_error`(error_obj)                                                                           |                                                                         |
| [`random_custom_id`](#oa.batches.random_custom_id)([prefix, suffix])                | Make a random custom_id by using the current time in nanoseconds        |
| `request_counts`(batch_list)                                                                       |                                                                         |

### Exceptions

| [`BatchCancelledError`](#oa.batches.BatchCancelledError)   |    |
|------------------------------------------------------------------------|----|
| [`BatchError`](#oa.batches.BatchError)            |    |
| [`BatchExpiredError`](#oa.batches.BatchExpiredError)     |    |
| [`BatchFailedError`](#oa.batches.BatchFailedError)      |    |
| [`BatchFinalizingError`](#oa.batches.BatchFinalizingError)  |    |
| [`BatchInProgressError`](#oa.batches.BatchInProgressError)  |    |

### *exception* oa.batches.BatchCancelledError

Bases: [`BatchError`](#oa.batches.BatchError)

### *exception* oa.batches.BatchError

Bases: [`ValueError`](https://docs.python.org/3/builtins/exceptions.html#ValueError)

### *exception* oa.batches.BatchExpiredError

Bases: [`BatchError`](#oa.batches.BatchError)

### *exception* oa.batches.BatchFailedError

Bases: [`BatchError`](#oa.batches.BatchError)

### *exception* oa.batches.BatchFinalizingError

Bases: [`BatchError`](#oa.batches.BatchError)

### *exception* oa.batches.BatchInProgressError

Bases: [`BatchError`](#oa.batches.BatchError)

### oa.batches.get_output_file_data(batch, \*, oa_stores, get_batch_obj=<function get_batch_obj>, on_complete=<function on_complete>, on_error=<function raise_error>)

Get the output file data for a batch, if it has completed successfully.

### oa.batches.mk_batch_file_embeddings_task(texts, \*, custom_id=None, validate=True, valid_text_getter=<function \_raise_if_any_invalid>, model='text-embedding-3-small', dimensions=NOT_GIVEN, custom_id_per_text=None, \*\*extra_embeddings_params)

Create a batch task (json-)dicts for generating embeddings.

These dictionaries (or list thereof) are destined to be used as input for the
OpenAI API batch endpoint. The endpoint will then generate embeddings for the
provided texts.

* **Parameters:**
  * **texts** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`KT`), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]) – The text or list of texts to generate embeddings for.
  * **custom_id** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – A custom identifier for the batch task. Defaults to None.
  * **validate** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool) | [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Whether to validate the texts. Defaults to True.
  * **valid_text_getter** (*Optional* *[**Callable* *]* *,* *optional*) – Function to retrieve valid texts. Defaults to \_raise_if_any_invalid.
  * **model** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str) *,* *optional*) – The embeddings model to use. Defaults to DFLT_EMBEDDINGS_MODEL.
  * **dimensions** ([`int`](https://docs.python.org/3/builtins/functions.html#int) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – The dimensions for the embeddings. Defaults to NOT_GIVEN.
  * **custom_id_per_text** (*Optional* *[*[*bool*](https://docs.python.org/3/builtins/functions.html#bool) *]* *,* *optional*) – Whether to include a custom ID per text item. Defaults to None.
  * **\*\*extra_embeddings_params** – Additional parameters for embeddings.
* **Returns:**
  A single task dictionary or a list of task dictionaries.
* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict) | [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]

### Examples

With a single text:

```pycon
>>> mk_batch_file_embeddings_task("Example text")
{'custom_id': '...',
 'method': 'POST',
 'url': '/v1/embeddings',
 'body': {'input': ['Example text'], 'model': 'text-embedding-3-small'}}
```

With a list of texts:

```pycon
>>> mk_batch_file_embeddings_task(["Text1", "Text2"])
{'custom_id': '...',
 'method': 'POST',
 'url': '/v1/embeddings',
 'body': {'input': ['Text1', 'Text2'], 'model': 'text-embedding-3-small'}}
```

With a dictionary of texts:

```pycon
>>> mk_batch_file_embeddings_task({"key1": "Text1", "key2": "Text2"})
[{'custom_id': 'key1',
  'method': 'POST',
  'url': '/v1/embeddings',
  'body': {'input': 'Text1', 'model': 'text-embedding-3-small'}},
 {'custom_id': 'key2',
  'method': 'POST',
  'url': '/v1/embeddings',
  'body': {'input': 'Text2', 'model': 'text-embedding-3-small'}}]
```

As you see, when a list is given, the batch task is created with a single custom_id,
with a list of texts as input.
When a dictionary is given, the batch task is created with
a custom_id per text item, and the input is a dictionary of key-value pairs.

The usefulness of giving a dictionary is that the tasks will contain the same keys
as your dictionary, which can be useful when having to link the results back to
the original data. But it comes with a disadvantage: it’s much faster to process
a list of texts than multiple tasks with a single text each.

When efficiency is a concern, you can keep a texts dictionary on the client side,
ask to embed list(texts.values()), and then use the keys to link the results back
to the original data.
You can force the function to ignore the type of the texts input and do it the way
you want it, by setting custom_id_per_text to True or False.

```pycon
>>> mk_batch_file_embeddings_task(
...     {"key1": "Text1", "key2": "Text2"}, custom_id_per_text=False
... )
{'custom_id': '...',
'method': 'POST',
'url': '/v1/embeddings',
'body': {'input': ['Text1', 'Text2'],
'model': 'text-embedding-3-small'}}
```

```pycon
>>> mk_batch_file_embeddings_task(
...     ["Text1", "Text2"], custom_id_per_text=True
... )
[{'custom_id': 0,
  'method': 'POST',
  'url': '/v1/embeddings',
  'body': {'input': 'Text1', 'model': 'text-embedding-3-small'}},
 {'custom_id': 1,
  'method': 'POST',
  'url': '/v1/embeddings',
  'body': {'input': 'Text2', 'model': 'text-embedding-3-small'}}]
```

### oa.batches.on_complete(oa_stores, batch_obj)

Return the output file if the batch completed successfully

### oa.batches.random_custom_id(prefix='custom_id-', suffix='')

Make a random custom_id by using the current time in nanoseconds
