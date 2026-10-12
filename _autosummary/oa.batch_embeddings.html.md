# oa.batch_embeddings

Simplified interface for computing embeddings in bulk using OpenAI’s batch API.

This module provides a clean, reusable interface for generating embeddings from
text segments using OpenAI’s batch API, handling the async nature of the API
and providing status monitoring, error handling, and result aggregation.

### Functions

| [`compute_embeddings`](#oa.batch_embeddings.compute_embeddings)(segments[, model, ...])   | Compute embeddings for text segments using OpenAI's batch API.   |
|-----------------------------------------------------------------------------------------------|------------------------------------------------------------------|
| [`compute_embeddings_df`](#oa.batch_embeddings.compute_embeddings_df)(segments[, model])     | Compute embeddings and return results as a pandas DataFrame.     |

### Classes

| [`BatchRequestCounts`](#oa.batch_embeddings.BatchRequestCounts)([completed, failed, total])   | Counts of batch requests by status                       |
|---------------------------------------------------------------------------------------------------|----------------------------------------------------------|
| [`BatchStatus`](#oa.batch_embeddings.BatchStatus)()                                    | Status constants for batch processing                    |
| [`EmbeddingsBatchProcess`](#oa.batch_embeddings.EmbeddingsBatchProcess)([segments, ...])          | Manages the lifecycle of batch embedding requests.       |
| [`ProcessingMall`](#oa.batch_embeddings.ProcessingMall)(\*[, current, segments, ...])     | Container for all stores needed during batch processing. |

### Exceptions

| [`BatchError`](#oa.batch_embeddings.BatchError)   | Base exception for batch processing errors   |
|---------------------------------------------------------------|----------------------------------------------|

### *exception* oa.batch_embeddings.BatchError

Bases: [`Exception`](https://docs.python.org/3/builtins/exceptions.html#Exception)

Base exception for batch processing errors

### *class* oa.batch_embeddings.BatchRequestCounts(completed=0, failed=0, total=0)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Counts of batch requests by status

### *class* oa.batch_embeddings.BatchStatus

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Status constants for batch processing

#### *classmethod* is_success(status)

Check if a status represents successful completion

* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

#### *classmethod* is_terminal(status)

Check if a status represents a terminal state

* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

### *class* oa.batch_embeddings.EmbeddingsBatchProcess(segments=None, processing_mall=None, , model='text-embedding-3-small', batch_size=1000, poll_interval=5.0, max_polls=None, verbosity=1, keep_processing_info=False, dacc=None, logger=None, \*\*embeddings_kwargs)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Manages the lifecycle of batch embedding requests.

This class handles submitting batch requests to the OpenAI API,
monitoring their status, retrieving results, and aggregating them.
It can be used as a context manager for automatic cleanup.

#### aggregate_results()

Aggregate all segments and embeddings from completed batches.

* **Return type:**
  [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`float`](https://docs.python.org/3/builtins/functions.html#float)]]]
* **Returns:**
  Tuple of (all_segments, all_embeddings)

#### get_status_summary()

Get a summary of batch statuses.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`int`](https://docs.python.org/3/builtins/functions.html#int)]
* **Returns:**
  Dictionary with counts of batches in each status

#### *property* is_complete *: [bool](https://docs.python.org/3/builtins/functions.html#bool)*

Check if batch processing is complete

#### *property* is_running *: [bool](https://docs.python.org/3/builtins/functions.html#bool)*

Check if batch processing is currently running

#### monitor_batches()

Monitor the status of all submitted batches until completion.

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

#### *property* result *: [tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[list](https://docs.python.org/3/builtins/stdtypes.html#list)[[str](https://docs.python.org/3/builtins/stdtypes.html#str)], [list](https://docs.python.org/3/builtins/stdtypes.html#list)[[list](https://docs.python.org/3/builtins/stdtypes.html#list)[[float](https://docs.python.org/3/builtins/functions.html#float)]]] | [None](https://docs.python.org/3/builtins/constants.html#None)*

Get the aggregated results if available

#### run()

Execute the complete batch embedding workflow and return results.

This method submits batches, monitors their status until completion,
and returns the aggregated results.

* **Return type:**
  [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`float`](https://docs.python.org/3/builtins/functions.html#float)]]]
* **Returns:**
  Tuple of (all_segments, all_embeddings)

#### submit_batches()

Submit all segment batches to the OpenAI API.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), `Batch`]
* **Returns:**
  Dictionary mapping batch IDs to batch objects

### *class* oa.batch_embeddings.ProcessingMall(, current=None, segments=None, finished=None, erred=None, embeddings=None)

Bases: [`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)

Container for all stores needed during batch processing.

The ProcessingMall contains five stores, all keyed by batch_id:

- current: Batches that are currently being processed
- segments: The text segments corresponding to each batch
- finished: Completed batches
- erred: Batches that encountered errors
- embeddings: The computed embeddings for each batch

#### clear()

Clear all stores

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

#### is_complete()

Check if processing is complete (no batches in current)

* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

#### *classmethod* with_folder(rootdir, store_cls=<class 'dol.filesys.JsonFiles'>)

Create a ProcessingMall with stores that persist to local files.

* **Parameters:**
  **rootdir** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Directory where the stores will be saved

### oa.batch_embeddings.compute_embeddings(segments, model='text-embedding-3-small', , batch_size=1000, poll_interval=5.0, max_polls=None, verbosity=1, processing_mall=None, keep_processing_info=False, dacc=None, return_process=False, logger=None, \*\*embeddings_kwargs)

Compute embeddings for text segments using OpenAI’s batch API.

This function manages the complete lifecycle of batch embedding requests,
from submitting batches to the API, monitoring their status, and
aggregating the results.

* **Parameters:**
  * **segments** (`Union`[[`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]) – Text segments to embed, either as a list or a dictionary
  * **model** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – OpenAI embedding model to use
  * **batch_size** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Maximum number of segments per batch
  * **poll_interval** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Seconds between status checks
  * **max_polls** ([`int`](https://docs.python.org/3/builtins/functions.html#int) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Maximum number of status checks before timing out
  * **verbosity** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Level of logging detail (0-2)
  * **processing_mall** ([`ProcessingMall`](#oa.batch_embeddings.ProcessingMall) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Optional custom ProcessingMall
  * **keep_processing_info** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Whether to keep mall data after completion
  * **dacc** (`OaDacc` | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Optional custom OaDacc instance
  * **return_process** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – If True, return the EmbeddingsBatchProcess object instead of results
  * **logger** ([`Logger`](https://docs.python.org/3/library/logging.html#logging.Logger) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Optional custom logger
  * **\*\*embeddings_kwargs** – Additional parameters for embedding generation
* **Returns:**
  Tuple of (segments, embeddings)
  If return_process is True:
  > EmbeddingsBatchProcess object for further interaction
* **Return type:**
  [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`float`](https://docs.python.org/3/builtins/functions.html#float)]]] | [`EmbeddingsBatchProcess`](#oa.batch_embeddings.EmbeddingsBatchProcess)

### oa.batch_embeddings.compute_embeddings_df(segments, model='text-embedding-3-small', \*\*kwargs)

Compute embeddings and return results as a pandas DataFrame.

* **Parameters:**
  * **segments** (`Union`[[`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]) – Text segments to embed
  * **model** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – OpenAI embedding model to use
  * **\*\*kwargs** – Additional arguments passed to compute_embeddings
* **Return type:**
  pandas.DataFrame
* **Returns:**
  DataFrame with ‘segment’ and ‘embedding’ columns
