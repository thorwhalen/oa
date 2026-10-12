# oa.util

oa utils

### Functions

| [`chunk_iterable`](#oa.util.chunk_iterable)(iterable, chk_size, \*[, ...])   | Divide an iterable into chunks/batches of a specific size.                                                                                                   |
|--------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`concat_lists`](#oa.util.concat_lists)(lists)                             | Concatenate a list of lists into a single list.                                                                                                              |
| [`ensure_oa_client`](#oa.util.ensure_oa_client)(oa_client)                     | Ensure that an OpenAI client is available, either by using the provided one or creating a new one.                                                           |
| [`ensure_openai_api_key`](#oa.util.ensure_openai_api_key)()                         | Make sure the module-level `openai` client has an API key, resolving it lazily.                                                                              |
| [`get_api_key_from_config`](#oa.util.get_api_key_from_config)()                       | The OpenAI API key: oa's config store, then the environment, then ask the user.                                                                              |
| [`get_package_name`](#oa.util.get_package_name)()                              | Return current package name                                                                                                                                  |
| [`iso_date_to_utc_int`](#oa.util.iso_date_to_utc_int)(iso_date)                   | Convert iso date string to utc integer timestamp.                                                                                                            |
| [`jsonl_dumps`](#oa.util.jsonl_dumps)(x[, encoding])                      | Serialize an iterable as JSONL bytes                                                                                                                         |
| [`jsonl_loads_iter`](#oa.util.jsonl_loads_iter)(src, \*[, get_lines, ...])     | Deserialize JSONL bytes into a python iterable (dict or list of dicts)                                                                                       |
| `kv_strip_value`(k, v)                                                                           |                                                                                                                                                              |
| `merge_multiple_signatures`(iterable_of_sigs, ...)                                               |                                                                                                                                                              |
| `mk_client`([api_key])                                                                           |                                                                                                                                                              |
| [`mk_local_files_saves_callback`](#oa.util.mk_local_files_saves_callback)([rootdir, ...])   | Returns a function that takes two inputs (i: int, obj: Any) and saves the serializer(obj) bytes in a file named index_to_filename(i) in the rootdir.         |
| `mk_template_store`(template_store)                                                              |                                                                                                                                                              |
| [`num_tokens`](#oa.util.num_tokens)([text, model])                       | Return the number of tokens in a string, under given model.                                                                                                  |
| [`oa_extractors_obj`](#oa.util.oa_extractors_obj)(\*\*named_paths)              | Return a SimpleNamespace of extractors for the named paths                                                                                                   |
| [`pricing_info`](#oa.util.pricing_info)([category, print_data_date])       | Return the pricing info for the OpenAI API.                                                                                                                  |
| [`save_in_temp_dir`](#oa.util.save_in_temp_dir)(obj[, serializer])             | Saves obj in a temp file, using serializer to serialize it, and returns its path.                                                                            |
| [`source_parameter_props_from`](#oa.util.source_parameter_props_from)(parameters)         | A decorator that will change the annotation and default of the parameters of the decorated function, sourcing them from `parameters`, matching them by name. |
| `transpose_and_concatenate`(iterable_of_tuples)                                                  |                                                                                                                                                              |
| `transpose_iterable`(iterable_of_tuples)                                                         |                                                                                                                                                              |
| [`utc_int_to_iso_date`](#oa.util.utc_int_to_iso_date)(utc_time)                   | Convert utc integer timestamp to more human readable iso format.                                                                                             |

### Classes

| [`ProcessingManager`](#oa.util.ProcessingManager)(pending_items, ...[, ...])   | A class to manage and monitor the processing of a collection of items, allowing customizable processing functions, status handling, and timing control, using keys to track items.   |
|-------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

### Exceptions

| [`OpenAIKeyNotFoundError`](#oa.util.OpenAIKeyNotFoundError)   | No OpenAI API key is configured, and there is no one to ask for it.   |
|---------------------------------------------------------------------------|-----------------------------------------------------------------------|

### *exception* oa.util.OpenAIKeyNotFoundError

Bases: `ConfigNotFound`

No OpenAI API key is configured, and there is no one to ask for it.

### *class* oa.util.ProcessingManager(pending_items, processing_function, handle_status_function, wait_time_function, status_check_interval=5.0, max_cycles=None, completed_items=<factory>, cycles=0)

Bases: [`Generic`](https://docs.python.org/3/library/typing.html#typing.Generic)[`KT`, `VT`, `Result`]

A class to manage and monitor the processing of a collection of items, allowing customizable
processing functions, status handling, and timing control, using keys to track items.

**Use Case**:

- Ideal for scenarios where you have a set of items identified by keys that require periodic status
  checks until they reach a completed state.
- Useful for managing batch jobs, asynchronous tasks, or any operations where items may not complete
  processing immediately.

**Attributes**:

- **pending_items** (`MutableMapping[KT, VT]`): The mapping of keys to pending items.
  : - If an iterable is provided, it is converted to a dict using `dict(enumerate(iterable))`.
- **processing_function** (`Callable[[VT], Tuple[Status, Result]]`): A function that takes a value (item)
  and returns a tuple of `(status, result)`.
  > - `status` (`Status`): Indicates the current state of the item (e.g., `'completed'`, `'in_progress'`, `'failed'`).
  > - `result` (`Result`): Additional data or context about the item’s processing result.
- **handle_status_function** (`Callable[[VT, Status, Result], bool]`): A function that decides whether to remove
  an item from `pending_items` based on its `status` and `result`.
  > - Returns `True` if the item should be removed (e.g., processing is complete or failed irrecoverably).
- **wait_time_function** (`Callable[[float, Dict], float]`): A function that determines how long to wait
  before the next processing cycle.
  > - Takes `cycle_duration` (time taken for the last cycle) and `locals()` dictionary as inputs.
  > - Returns the sleep time in seconds.
- **status_check_interval** (`float`): Desired minimum time (in seconds) between status checks. Defaults to `5.0`.
- **max_cycles** (`Optional[int]`): Maximum number of processing cycles to perform. If `None`, there is no limit.
- **completed_items** (`MutableMapping[KT, Result]`): Mapping of keys to results for items that have been processed.
- **cycles** (`int`): Number of processing cycles that have been performed.

#### process_items()

Runs the processing loop until all pending items are processed or max_cycles is reached.

In each cycle:

- Calls `process_pending_items()` to process the current pending items.
- Increments the cycle count.
- Calculates the duration of the cycle and determines how long to sleep before the next cycle
  using `wait_time_function`.
- Sleeps for the calculated duration if there are still pending items.

The loop continues until:

- `pending_items` is empty (all items have been processed), or
- The number of cycles reaches `max_cycles`, if specified.

#### process_pending_items()

Processes the pending items once.

This method iterates over each key-value pair in `pending_items`, applies the `processing_function`
to determine its status, and then uses `handle_status_function` to decide whether to
remove the item from `pending_items`. Items removed are added to `completed_items` with their results.

#### *property* status *: [bool](https://docs.python.org/3/builtins/functions.html#bool)*

Indicates whether all pending items have been processed.

* **Returns:**
  `True` if there are no more pending items, `False` otherwise.
* **Return type:**
  [*bool*](https://docs.python.org/3/builtins/functions.html#bool)

### oa.util.chunk_iterable(iterable, chk_size, , chunk_type=None)

Divide an iterable into chunks/batches of a specific size.

Handles both mappings (e.g. dicts) and non-mappings (lists, tuples, sets…)
as you probably expect it to (if you give a dict input, it will chunk on the
(key, value) items and return dicts of these).
Thought note that you always can control the type of the chunks with the
`chunk_type` argument.

* **Parameters:**
  * **iterable** ([`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`T`)] | [`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`KT`), [`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`VT`)]) – The iterable or mapping to divide.
  * **chk_size** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – The size of each chunk.
  * **chunk_type** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[`...`](https://docs.python.org/3/builtins/constants.html#Ellipsis), [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`T`)] | [`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`KT`), [`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`VT`)]] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – The type of the chunks (list, tuple, set, dict…).
* **Return type:**
  [`Iterator`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterator)[[`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`T`)] | [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`T`), [`...`](https://docs.python.org/3/builtins/constants.html#Ellipsis)] | [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`KT`), [`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`VT`)]]
* **Returns:**
  An iterator of dicts if the input is a Mapping, otherwise an iterator
  of collections (list, tuple, set…).

### Examples

```pycon
>>> list(chunk_iterable([1, 2, 3, 4, 5], 2))
[[1, 2], [3, 4], [5]]
```

```pycon
>>> list(chunk_iterable((1, 2, 3, 4, 5), 3, chunk_type=tuple))
[(1, 2, 3), (4, 5)]
```

```pycon
>>> list(chunk_iterable({"a": 1, "b": 2, "c": 3}, 2))
[{'a': 1, 'b': 2}, {'c': 3}]
```

```pycon
>>> list(chunk_iterable({"x": 1, "y": 2, "z": 3}, 1, chunk_type=dict))
[{'x': 1}, {'y': 2}, {'z': 3}]
```

### oa.util.concat_lists(lists)

Concatenate a list of lists into a single list.

```pycon
>>> concat_lists([[1, 2], [3, 4], [5, 6]])
[1, 2, 3, 4, 5, 6]
```

### oa.util.ensure_oa_client(oa_client)

Ensure that an OpenAI client is available, either by using the provided one or creating a new one.

* **Return type:**
  `OpenAI`

### oa.util.ensure_openai_api_key()

Make sure the module-level `openai` client has an API key, resolving it lazily.

The key is looked up (config store, environment, then asking the user) the
first time something needs it, never at `import oa` time, so importing
`oa` works – and dependents’ test suites can run – when no key is set.
Resolution follows oa’s own order (store before environment), as when the key
was set at import, and happens once: a key set on `openai.api_key`
afterwards is respected. If no key is stored or in the environment, a key
already set on `openai.api_key` is kept rather than asking the user.
Thread-safe.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### oa.util.get_api_key_from_config()

The OpenAI API key: oa’s config store, then the environment, then ask the user.

Asking only happens when the user can be asked (see `_can_prompt_user`);
otherwise an informative `OpenAIKeyNotFoundError` is raised. Thread-safe: when
several threads need the key at once, only one asks, and the others reuse the answer.

### oa.util.get_package_name()

Return current package name

### oa.util.iso_date_to_utc_int(iso_date)

Convert iso date string to utc integer timestamp.
Inverse of utc_int_to_iso_date.

* **Return type:**
  [`int`](https://docs.python.org/3/builtins/functions.html#int)

```pycon
>>> iso_date_to_utc_int('2024-08-12T14:01:57+00:00')
1723471317
```

### oa.util.jsonl_dumps(x, encoding='utf-8')

Serialize an iterable as JSONL bytes

* **Return type:**
  [`bytes`](https://docs.python.org/3/builtins/stdtypes.html#bytes)

```pycon
>>> jsonl_dumps([{'a': 1}, {'b': 2}])
b'{"a": 1}\n{"b": 2}'
```

### oa.util.jsonl_loads_iter(src, \*, get_lines=<method 'splitlines' of 'bytes' objects>, line_egress=operator.methodcaller('strip'))

Deserialize JSONL bytes into a python iterable (dict or list of dicts)

* **Return type:**
  [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]

```pycon
>>> list(jsonl_loads(b'\n{"a": 1}\n\n{"b": 2}'))
[{'a': 1}, {'b': 2}]
```

### oa.util.mk_local_files_saves_callback(rootdir=None, \*, serializer=<built-in function dumps>, index_to_filename=<built-in method format of str object>, print_dir_path=True)

Returns a function that takes two inputs (i: int, obj: Any) and saves the
serializer(obj) bytes in a file named index_to_filename(i) in the rootdir.
If rootdir, a temp dir is used.

### oa.util.num_tokens(text=None, model='gpt-3.5-turbo')

Return the number of tokens in a string, under given model.

keywords: token count, number of tokens

* **Return type:**
  [`int`](https://docs.python.org/3/builtins/functions.html#int)

### oa.util.oa_extractors_obj(\*\*named_paths)

Return a SimpleNamespace of extractors for the named paths

### oa.util.pricing_info(category=None, , print_data_date=False)

Return the pricing info for the OpenAI API.

#### NOTE
These are not live prices. Live prices can be found here:

The information pricing_info returns is taken from the file `openai_api_pricing_info.json`
in the `data` directory of the package.
To print a message with the data date, do `pricing_info(print_data_date=True)`.

### oa.util.save_in_temp_dir(obj, serializer=<built-in function dumps>)

Saves obj in a temp file, using serializer to serialize it, and returns its path.

### oa.util.source_parameter_props_from(parameters)

A decorator that will change the annotation and default of the parameters of the
decorated function, sourcing them from `parameters`, matching them by name.

### oa.util.utc_int_to_iso_date(utc_time)

Convert utc integer timestamp to more human readable iso format.
Inverse of iso_date_to_utc_int.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

```pycon
>>> utc_int_to_iso_date(1723471317)
'2024-08-12T14:01:57+00:00'
```
