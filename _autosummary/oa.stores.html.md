# oa.stores

Data object layers for openai resources

### Functions

| [`date_filter`](#oa.stores.date_filter)(objs[, min_date, max_date, ...])   | Filter objects by date.                                                                                                                                           |
|-------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `ensure_utc_date`(x, \*[, val_if_none])                                                         |                                                                                                                                                                   |
| [`extract_id`](#oa.stores.extract_id)([method, is_id, has_id, get_id])    | Decorator that will extract the id from the first non-instance argument of a method.                                                                              |
| [`get_json_data_from_response`](#oa.stores.get_json_data_from_response)(response)          | Extract the embeddings from a HttpxBinaryResponseContent object                                                                                                   |
| [`get_json_or_jsonl_data`](#oa.stores.get_json_or_jsonl_data)(response)               | Extract the json data from a response                                                                                                                             |
| [`is_task_dict`](#oa.stores.is_task_dict)(x)                                | Is a dict (or Mapping) the schema of an openai API 'task'                                                                                                         |
| [`is_task_dict_list`](#oa.stores.is_task_dict_list)(x)                           | Is a list (or iterable) of dicts (or Mappings) the schema of an openai API 'task'                                                                                 |
| [`object_of_data`](#oa.stores.object_of_data)(data)                           | This function extracts the object (value) from a {object: V, index: i, V} dict which is the format you'll find in the ['response']['body']['data'] of a response. |
| `print_some_jsonl_line_fields`(line)                                                            |                                                                                                                                                                   |
| `response_body_data`(response_dict)                                                             |                                                                                                                                                                   |
| `response_body_data_objects`(response_dict)                                                     |                                                                                                                                                                   |

### Classes

| [`DataObject`](#oa.stores.DataObject)                                    | dicts that have two required fields: 'object' and 'index'                              |
|------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------|
| [`EmbeddingsDataObject`](#oa.stores.EmbeddingsDataObject)                          | A DataObject with object='embeddings' and an 'embeddings' key that is a list of floats |
| [`MappingHooks`](#oa.stores.MappingHooks)()                                |                                                                                        |
| [`MutuableMappingHooks`](#oa.stores.MutuableMappingHooks)()                        |                                                                                        |
| [`OaBatches`](#oa.stores.OaBatches)([client])                           | A key-value store for OpenAI batches metadata.                                         |
| [`OaBatchesBase`](#oa.stores.OaBatchesBase)([client])                       |                                                                                        |
| `OaDacc`([client])                                                                             |                                                                                        |
| [`OaFiles`](#oa.stores.OaFiles)([client, purpose, ...])               | A key-value store for OpenAI files content data.                                       |
| [`OaFilesBase`](#oa.stores.OaFilesBase)([client, purpose, ...])           |                                                                                        |
| [`OaFilesJsonData`](#oa.stores.OaFilesJsonData)([client, purpose, ...])       | A key-value store for OpenAI files content data.                                       |
| [`OaFilesMetadata`](#oa.stores.OaFilesMetadata)([client, purpose, ...])       | A key-value store for OpenAI files metadata.                                           |
| [`OaJsonFiles`](#oa.stores.OaJsonFiles)([client, purpose, ...])           | A key-value store for OpenAI files content data.                                       |
| [`OaJsonLFiles`](#oa.stores.OaJsonLFiles)([client, purpose, ...])          | A key-value store for OpenAI files content data.                                       |
| [`OaMapping`](#oa.stores.OaMapping)()                                   |                                                                                        |
| `OaStores`([client])                                                                           |                                                                                        |
| [`OaVectorStoreFiles`](#oa.stores.OaVectorStoreFiles)(vector_store_id[, client]) | Mapping interface for files within a vector store.                                     |
| [`OaVectorStores`](#oa.stores.OaVectorStores)([client])                      | A key-value store for OpenAI vector stores.                                            |
| [`OaVectorStoresBase`](#oa.stores.OaVectorStoresBase)([client])                  | Base class for OpenAI vector stores mapping interface.                                 |
| [`ResponseDict`](#oa.stores.ResponseDict)                                  |                                                                                        |

### *class* oa.stores.DataObject

Bases: [`TypedDict`](https://docs.python.org/3/library/typing.html#typing.TypedDict)

dicts that have two required fields: ‘object’ and ‘index’

### *class* oa.stores.EmbeddingsDataObject

Bases: [`TypedDict`](https://docs.python.org/3/library/typing.html#typing.TypedDict)

A DataObject with object=’embeddings’ and an ‘embeddings’ key that is a list of
floats

### *class* oa.stores.MappingHooks

Bases: [`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)

### *class* oa.stores.MutuableMappingHooks

Bases: [`MappingHooks`](#oa.stores.MappingHooks)

### *class* oa.stores.OaBatches(client=None, \*\*extra_kwargs)

Bases: `Store`

A key-value store for OpenAI batches metadata.
Keys are the batch IDs.

#### append(input_file_id, , endpoint='/v1/embeddings', completion_window='24h', metadata=None)

Create and submit a new batch via submitting the input file (obj or id).

#### metadata(batch_id)

Retrieve metadata for a batch.

* **Return type:**
  `Batch`

### *class* oa.stores.OaBatchesBase(client=None, \*\*extra_kwargs)

Bases: [`OaMapping`](#oa.stores.OaMapping)

#### append(input_file_id, , endpoint='/v1/embeddings', completion_window='24h', metadata=None)

Create and submit a new batch via submitting the input file (obj or id).

#### metadata(batch_id)

Retrieve metadata for a batch.

* **Return type:**
  `Batch`

### *class* oa.stores.OaFiles(client=None, purpose='batch', iter_filter_purpose=False, encoding='utf-8', \*, file, expires_after=<openai.Omit object>, extra_headers=None, extra_query=None, extra_body=None, timeout=NOT_GIVEN)

Bases: `Store`

A key-value store for OpenAI files content data.
Keys are the file IDs.

### *class* oa.stores.OaFilesBase(client=None, purpose='batch', iter_filter_purpose=False, encoding='utf-8', \*, file, expires_after=<openai.Omit object>, extra_headers=None, extra_query=None, extra_body=None, timeout=NOT_GIVEN)

Bases: [`OaMapping`](#oa.stores.OaMapping)

### *class* oa.stores.OaFilesJsonData(client=None, purpose='batch', iter_filter_purpose=False, encoding='utf-8', \*, file, expires_after=<openai.Omit object>, extra_headers=None, extra_query=None, extra_body=None, timeout=NOT_GIVEN)

Bases: `Store`

A key-value store for OpenAI files content data.
Keys are the file IDs.
Values are the data extracted from the HttpxBinaryResponseContent object.

### *class* oa.stores.OaFilesMetadata(client=None, purpose='batch', iter_filter_purpose=False, encoding='utf-8', \*, file, expires_after=<openai.Omit object>, extra_headers=None, extra_query=None, extra_body=None, timeout=NOT_GIVEN)

Bases: [`OaFilesBase`](#oa.stores.OaFilesBase)

A key-value store for OpenAI files metadata.

### *class* oa.stores.OaJsonFiles(client=None, purpose='batch', iter_filter_purpose=False, encoding='utf-8', \*, file, expires_after=<openai.Omit object>, extra_headers=None, extra_query=None, extra_body=None, timeout=NOT_GIVEN)

Bases: `Store`

A key-value store for OpenAI files content data.
Keys are the file IDs.

### *class* oa.stores.OaJsonLFiles(client=None, purpose='batch', iter_filter_purpose=False, encoding='utf-8', \*, file, expires_after=<openai.Omit object>, extra_headers=None, extra_query=None, extra_body=None, timeout=NOT_GIVEN)

Bases: `Store`

A key-value store for OpenAI files content data.
Keys are the file IDs.

### *class* oa.stores.OaMapping

Bases: [`MappingHooks`](#oa.stores.MappingHooks), `KvReader`

### *class* oa.stores.OaVectorStoreFiles(vector_store_id, client=None, \*\*extra_kwargs)

Bases: [`OaMapping`](#oa.stores.OaMapping)

Mapping interface for files within a vector store.

#### add_file(file_id)

Add file to vector store.

* **Return type:**
  [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)

#### metadata(file_id)

Get file info from vector store.

* **Return type:**
  [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)

### *class* oa.stores.OaVectorStores(client=None, \*\*extra_kwargs)

Bases: `Store`

A key-value store for OpenAI vector stores.
Keys are the vector store IDs.

#### create(name, \*\*config)

Create a new vector store with given name.

* **Return type:**
  [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)

#### get_by_name(name)

Get vector store by name (since OpenAI API doesn’t support this directly).

#### metadata(vs_id)

Get vector store metadata by ID.

* **Return type:**
  [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)

### *class* oa.stores.OaVectorStoresBase(client=None, \*\*extra_kwargs)

Bases: [`OaMapping`](#oa.stores.OaMapping)

Base class for OpenAI vector stores mapping interface.

#### create(name, \*\*config)

Create a new vector store with given name.

* **Return type:**
  [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)

#### get_by_name(name)

Get vector store by name (since OpenAI API doesn’t support this directly).

#### metadata(vs_id)

Get vector store metadata by ID.

* **Return type:**
  [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)

### *class* oa.stores.ResponseDict

Bases: [`TypedDict`](https://docs.python.org/3/library/typing.html#typing.TypedDict)

### oa.stores.date_filter(objs, min_date=None, max_date=None, , date_extractor='created_at', stop_on_first_out_of_range=False)

Filter objects by date.

* **Parameters:**
  * **objs** – The objects to filter (must have a)
  * **min_date** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`int`](https://docs.python.org/3/builtins/functions.html#int), [`None`](https://docs.python.org/3/builtins/constants.html#None)]) – The minimum date (inclusive)
  * **max_date** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`int`](https://docs.python.org/3/builtins/functions.html#int), [`None`](https://docs.python.org/3/builtins/constants.html#None)]) – The maximum date (inclusive)
  * **date_attr** – The attribute to get the date from
  * **stop_on_first_out_of_range** – If True, stop iteration on first out-of-range date

### oa.stores.extract_id(method=None, \*, is_id=<function \_is_string>, has_id=<function \_has_id_attr>, get_id=operator.attrgetter('id'))

Decorator that will extract the id from the first non-instance argument of a method.

```pycon
>>> @extract_id
... def veni(self, vidi, vici):
...     return f"{vidi=}, {vici=}"
>>> assert (
...     veni(None, 'hi', vici=3)  # calling the function
...     == veni.__wrapped__(None, 'hi', vici=3)  # outputs the same as calling the original
...     == "vidi='hi', vici=3"
... )
```

Except it now has extra powers; if your id is contained in a attribute,
it’ll be extracted.

```pycon
>>> from types import SimpleNamespace
>>> obj = SimpleNamespace(id='hi')
>>> obj.id
'hi'
>>> assert veni(None, obj, vici=3) == "vidi='hi', vici=3"
```

You can also customize the filters and the id getter.
The following will resolve a string representation of an integer as an id.

```pycon
>>> @extract_id(is_id=lambda x: isinstance(x, int), has_id=str.isnumeric, get_id=int)
... def add_one(self, x):
...     return x
...
>>> assert add_one(None, 3) == 3
>>> assert add_one(None, "42") == 42
```

### oa.stores.get_json_data_from_response(response)

Extract the embeddings from a HttpxBinaryResponseContent object

* **Return type:**
  [*dict*](https://docs.python.org/3/builtins/stdtypes.html#dict)

### oa.stores.get_json_or_jsonl_data(response)

Extract the json data from a response

### oa.stores.is_task_dict(x)

Is a dict (or Mapping) the schema of an openai API ‘task’

### oa.stores.is_task_dict_list(x)

Is a list (or iterable) of dicts (or Mappings) the schema of an openai API ‘task’

### oa.stores.object_of_data(data)

This function extracts the object (value) from a {object: V, index: i, V} dict
which is the format you’ll find in the [‘response’][‘body’][‘data’] of a response.

* **Return type:**
  [`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`DataObjectValue`)

```pycon
>>> object_of_data({'object': 'embeddings', 'index': 3, 'embeddings': [1, 2, 3]})
[1, 2, 3]
```
