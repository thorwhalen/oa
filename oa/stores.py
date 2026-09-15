"""Data object layers for openai resources"""

from collections.abc import Mapping
from operator import attrgetter, methodcaller
from typing import Optional, Union, Any, List, Literal, T
from collections.abc import Iterable, Callable
import json
from functools import wraps, partial, cached_property

import openai  # TODO: Import from oa.util instead?

# TODO: Move openai imports to go through oa.util
from openai.resources.files import Files as OpenaiFiles, FileTypes, FileObject
from openai.types import FileObject

from oa.base import TextOrTexts
from oa.batches import (
    mk_batch_file_embeddings_task,
    BatchSpec,
    BatchObj,
    BatchId,
)
from oa.util import (
    OaClientSpec,
    mk_client,
    ensure_oa_client,
    merge_multiple_signatures,
    source_parameter_props_from,
    utc_int_to_iso_date,
    iso_date_to_utc_int,
    Purpose,
    BatchesEndpoint,
    DFLT_ENCODING,
    jsonl_loads_iter,
    jsonl_dumps,
    jsonl_loads,
)
from i2 import Sig
from i2.signatures import SignatureAble, ParamsAble
from i2 import postprocess
from dol import wrap_kvs, KvReader, Pipe

FilterFunc = Callable[[Any], bool]


DFLT_PURPOSE = "batch"
DFLT_BATCHES_ENDPOINT = "/v1/embeddings"

openai_files_cumul_sig = merge_multiple_signatures(
    [
        OpenaiFiles.create,
        OpenaiFiles.retrieve,
        OpenaiFiles.delete,
        Sig(OpenaiFiles.list),
    ],
    default_conflict_method="take_first",
)

params_from_openai_files_cls = source_parameter_props_from(openai_files_cumul_sig)

files_create_sig = (
    Sig(OpenaiFiles.create)
    .ch_defaults(purpose=DFLT_PURPOSE)
    .ch_kinds(purpose=Sig.POSITIONAL_OR_KEYWORD)
)


def _is_string(x):
    return isinstance(x, str)


def _has_id_attr(x):
    return hasattr(x, "id")


def _is_instance_and_has_id(x=None, *, type_: type):
    if x is None:
        return partial(_is_instance_and_has_id, type_=type_)
    else:
        return isinstance(x, type_) and hasattr(x, "id")


def extract_id(
    method: Callable | None = None,
    *,
    is_id: FilterFunc = _is_string,
    has_id: FilterFunc = _has_id_attr,
    get_id: Callable = attrgetter("id"),
):
    """
    Decorator that will extract the id from the first non-instance argument of a method.

    >>> @extract_id
    ... def veni(self, vidi, vici):
    ...     return f"{vidi=}, {vici=}"
    >>> assert (
    ...     veni(None, 'hi', vici=3)  # calling the function
    ...     == veni.__wrapped__(None, 'hi', vici=3)  # outputs the same as calling the original
    ...     == "vidi='hi', vici=3"
    ... )

    Except it now has extra powers; if your id is contained in a attribute,
    it'll be extracted.

    >>> from types import SimpleNamespace
    >>> obj = SimpleNamespace(id='hi')
    >>> obj.id
    'hi'
    >>> assert veni(None, obj, vici=3) == "vidi='hi', vici=3"

    You can also customize the filters and the id getter.
    The following will resolve a string representation of an integer as an id.

    >>> @extract_id(is_id=lambda x: isinstance(x, int), has_id=str.isnumeric, get_id=int)
    ... def add_one(self, x):
    ...     return x
    ...
    >>> assert add_one(None, 3) == 3
    >>> assert add_one(None, "42") == 42

    """
    if method is None:
        return partial(extract_id, is_id=is_id, has_id=has_id, get_id=get_id)
    else:

        @wraps(method)
        def _wrapped_method(self, x, *args, **kwargs):
            if not is_id(x):
                if has_id(x):
                    x = get_id(x)
                else:
                    raise ValueError(f"Can't resolve id from {type(x)}: {x}")
            return method(self, x, *args, **kwargs)

        return _wrapped_method


class MappingHooks(Mapping):
    """Implement `Mapping`'s dunder methods by delegating to `_iter`/`_len`/`_contains`/`_getitem` hooks.

    A subclass defines those hooks (rather than the dunders directly) to get a
    read-only mapping interface.
    """

    def __iter__(self):
        # TODO: I'd like to just return the (iterable) object, but that doesn't work (why?)
        yield from self._iter()

    def __len__(self):
        return self._len()

    def __contains__(self, key):
        return self._contains(key)

    def __getitem__(self, key):
        return self._getitem(key)


class MutuableMappingHooks(MappingHooks):
    """`MappingHooks` plus mutation, delegating to `_delitem`/`_setitem` hooks."""

    def __delitem__(self, key):
        return self._delitem(key)

    def __setitem__(self, key, value):
        return self._setitem(key, value)


class OaMapping(MappingHooks, KvReader):
    """Base class for the OpenAI-resource key-value stores below: a `client`-backed `MappingHooks`."""

    client: openai.Client
    _list_kwargs = {}  # default, overriden in __init__

    def _len(self) -> int:
        """Return the number of batches."""
        # TODO: Does the API have a direct way to get the number of items?
        c = 0
        for _ in self:
            c += 1
        return c

    def _contains(self, k) -> bool:
        """Check if an item is contained in the mapping."""
        try:
            self[k]
            return True
        except openai.NotFoundError:
            return False

    def __delitem__(self, key):
        return self._delitem(key)


def is_task_dict(x):
    """Is a dict (or Mapping) the schema of an openai API 'task'"""
    # TODO: Get this schema dynamically from the API's swagger spec (or similar)
    task_keys = {"custom_id", "method", "url", "body"}
    return isinstance(x, Mapping) and all(k in x for k in task_keys)


def is_task_dict_list(x):
    """
    Is a list (or iterable) of dicts (or Mappings) the schema of an openai API 'task'
    """
    # TODO: Get this schema dynamically from the API's swagger spec (or similar)
    return all(is_task_dict(item) for item in x)


class OaFilesBase(OaMapping):
    """A key-value store of OpenAI files: keys are file ids, values are their content."""

    # @params_from_openai_files_cls
    @Sig.replace_kwargs_using(files_create_sig - "purpose")
    def __init__(
        self,
        client: openai.Client | None = None,
        purpose: Purpose | None = DFLT_PURPOSE,  # type: ignore
        iter_filter_purpose: bool = False,  # type: ignore
        encoding: str = DFLT_ENCODING,
        **extra_kwargs,
    ):
        if client is None:
            client = mk_client()
        self.client = client
        self.purpose = purpose
        self.iter_filter_purpose = iter_filter_purpose
        if self.iter_filter_purpose:
            self._list_kwargs = {"purpose": self.purpose}
        self.encoding = encoding
        self.extra_kwargs = extra_kwargs

    def _iter(self):
        return self.client.files.list(**self._list_kwargs)

    @extract_id
    def metadata(self, file_id) -> FileObject:
        """Get the `FileObject` metadata for `file_id`."""
        return self.client.files.retrieve(file_id, **self.extra_kwargs)

    @extract_id
    def content(self, file_id):
        """Get the raw content of the file `file_id`."""
        return self.client.files.content(file_id, **self.extra_kwargs)

    _getitem = content

    @extract_id
    def _delitem(self, file_id):
        # Delete the file using the API (you might need to implement this)
        return self.client.files.delete(file_id)  # Assuming there's a delete method

    @params_from_openai_files_cls
    def append(self, file: FileTypes | dict) -> FileObject:
        """Upload `file` (or a task dict/list, jsonl-encoded first) to OpenAI and return its `FileObject`."""
        # Note: self.client.create can be found in openai.resources.files.Files.create
        if is_task_dict(file) or is_task_dict_list(file):
            file = jsonl_dumps(file, self.encoding)
        return self.client.files.create(
            file=file, purpose=self.purpose, **self.extra_kwargs
        )

    @Sig.replace_kwargs_using(mk_batch_file_embeddings_task)
    def create_embedding_task(self, texts: TextOrTexts, **extra_kwargs):
        """Build an embeddings batch task for `texts` (via `mk_batch_file_embeddings_task`) and upload it."""
        # Note: self.client.create can be found in openai.resources.files.Files.create
        task = mk_batch_file_embeddings_task(texts, **extra_kwargs)
        return self.append(task)


class OaFilesMetadata(OaFilesBase):
    """
    A key-value store for OpenAI files metadata.
    """

    _getitem = OaFilesBase.metadata


@wrap_kvs(key_decoder=attrgetter("id"), value_decoder=attrgetter("content"))
class OaFiles(OaFilesBase):
    """
    A key-value store for OpenAI files content data.
    Keys are the file IDs.
    """


@wrap_kvs(key_decoder=attrgetter("id"), value_decoder=jsonl_loads)
class OaJsonLFiles(OaFilesBase):
    """
    A key-value store for OpenAI files content data.
    Keys are the file IDs.
    """


def get_json_or_jsonl_data(response):
    """Extract the json data from a response"""
    try:
        return response.json()
    except json.JSONDecodeError:
        return jsonl_loads(response.content)


@wrap_kvs(key_decoder=attrgetter("id"), value_decoder=get_json_or_jsonl_data)
class OaJsonFiles(OaFilesBase):
    """
    A key-value store for OpenAI files content data.
    Keys are the file IDs.
    """


# TODO: Find a non-underscored place to import HttpxBinaryResponseContent from
# Note: This is just used for annotation purposes
from openai._legacy_response import HttpxBinaryResponseContent

from typing import TypedDict, TypeVar


# TODO: Find some where to import this definition from
class ResponseDict(TypedDict):
    """One line of a batch output file: `id`, `custom_id`, `response` body, and `error`."""

    id: str  # Example type, you can replace it with whatever type is appropriate
    custom_id: str
    response: dict
    error: Any  # TODO: Look up what type this can be


class DataObject(TypedDict, total=False):
    """dicts that have two required fields: 'object' and 'index'"""

    object: str  # Required field
    index: int  # Required field


class EmbeddingsDataObject(TypedDict):
    """
    A DataObject with object='embeddings' and an 'embeddings' key that is a list of
    floats
    """

    object: Literal["embeddings"]  # This enforces the value to be 'embeddings'
    index: int  # Required field
    embeddings: list[float]  # The third known field, 'embeddings'


DataObjectValue = TypeVar("DataObjectValue")
DataObjectValue.__doc__ = (
    "The value that a DataObject holds in the field indicated by the object field"
)

jsonl_loads_response_lines = partial(
    jsonl_loads_iter, get_lines=methodcaller("iter_lines")
)

jsonl_loads_list = Pipe(jsonl_loads_response_lines, list)


def response_body_data(response_dict: ResponseDict) -> DataObject:
    """Get `response_dict['response']['body']['data']`."""
    return response_dict["response"]["body"]["data"]


def object_of_data(data: DataObject) -> DataObjectValue:
    """
    This function extracts the object (value) from a {object: V, index: i, V} dict
    which is the format you'll find in the ['response']['body']['data'] of a response.

    >>> object_of_data({'object': 'embeddings', 'index': 3, 'embeddings': [1, 2, 3]})
    [1, 2, 3]

    """
    return data[data["object"]]


def response_body_data_objects(
    response_dict: ResponseDict,
) -> Iterable[DataObjectValue]:
    """Yield each data object's value (via `object_of_data`) from `response_dict`'s `response.body.data`."""
    data = response_body_data(response_dict)
    for d in data:
        yield object_of_data(d)


# Note: This is to be used as a content decoder
# TODO: The looping logic is messy, consider refactoring
@postprocess(dict)
def get_json_data_from_response(response: HttpxBinaryResponseContent):
    """Extract the embeddings from a HttpxBinaryResponseContent object"""
    for d in jsonl_loads_response_lines(response):
        custom_id = d["custom_id"]
        dd = response_body_data(d)
        if isinstance(dd, list):
            for ddd in dd:
                yield custom_id, object_of_data(ddd)
        else:
            yield custom_id, object_of_data(dd)


@wrap_kvs(key_decoder=attrgetter("id"), value_decoder=get_json_data_from_response)
class OaFilesJsonData(OaFilesBase):
    """
    A key-value store for OpenAI files content data.
    Keys are the file IDs.
    Values are the data extracted from the HttpxBinaryResponseContent object.
    """


import openai
from collections.abc import Mapping
from typing import Optional, Dict
from openai.resources.batches import Batches as OaBatches
from typing import Literal


# TODO: Why does go to definition here (for self.client.batches,
#    but also for self.client.files), but not in OpenAIFilesBase?
class OaBatchesBase(OaMapping):
    """A key-value store of OpenAI batches: keys are batch ids, values their `BatchObj` metadata."""

    def __init__(self, client: openai.Client | None = None, **extra_kwargs):
        if client is None:
            client = mk_client()
        self.client = client
        self.extra_kwargs = extra_kwargs

    def _iter(self) -> Iterable[BatchObj]:
        """Return an iterator over batch IDs"""
        return self.client.batches.list(**self._list_kwargs)

    @extract_id
    def metadata(self, batch_id: str) -> BatchObj:
        """Retrieve metadata for a batch."""
        return self.client.batches.retrieve(batch_id, **self.extra_kwargs)

    _getitem = metadata

    @extract_id
    def _delitem(self, batch_id: str):
        """Cancel the batch via the API."""
        self.client.batches.cancel(batch_id)

    @extract_id
    def append(
        self,
        input_file_id: str,
        *,
        endpoint: BatchesEndpoint = DFLT_BATCHES_ENDPOINT,  # type: ignore
        completion_window: Literal["24h"] = "24h",
        metadata: dict[str, str] | None = None,
    ):
        """Create and submit a new batch via submitting the input file (obj or id)."""
        return self.client.batches.create(
            input_file_id=input_file_id,
            endpoint=endpoint,
            completion_window=completion_window,
            metadata=metadata,
            **self.extra_kwargs,
        )


@wrap_kvs(key_decoder=attrgetter("id"))
class OaBatches(OaBatchesBase):
    """
    A key-value store for OpenAI batches metadata.
    Keys are the batch IDs.
    """


# --------------------------------------------------------------------------------------
# Batches/Files API utilities

origin_date = 0
end_of_times_date = iso_date_to_utc_int("9999-12-31T23:59:59Z")
DateSpec = Optional[Union[str, int]]

NotGiven = type("NotGiven", (), {})()


def ensure_utc_date(x, *, val_if_none=NotGiven):
    """Coerce `x` (an ISO date string, a UTC int, or `None`) to a UTC integer timestamp.

    Args:
        x: An ISO date string, an already-UTC int, or `None`.
        val_if_none: Value returned when `x` is `None`; if left as the
            sentinel `NotGiven`, `x=None` raises instead.

    Raises:
        TypeError: If `x` is not a str, int, or (when allowed) `None`.
    """
    if isinstance(x, str):
        return iso_date_to_utc_int(x)
    elif isinstance(x, int):
        return x
    elif val_if_none is not NotGiven and x is None:
        return val_if_none
    else:
        raise TypeError(f"Expected a date string or int, got {x}")


def date_filter(
    objs,  # Iterable of file/batch API objects
    min_date: DateSpec = None,
    max_date: DateSpec = None,
    *,
    date_extractor: str | Callable = "created_at",
    stop_on_first_out_of_range=False,
):
    """
    Filter objects by date.

    Args:
        objs: The objects to filter (must have a)
        min_date: The minimum date (inclusive)
        max_date: The maximum date (inclusive)
        date_attr: The attribute to get the date from
        stop_on_first_out_of_range: If True, stop iteration on first out-of-range date
    """
    if isinstance(date_extractor, str):
        date_attr = date_extractor
        date_extractor = attrgetter(date_attr)

    min_date = ensure_utc_date(min_date, val_if_none=origin_date)
    max_date = ensure_utc_date(max_date, val_if_none=end_of_times_date)

    for obj in objs:
        date = date_extractor(obj)
        if min_date <= date <= max_date:
            yield obj
        elif stop_on_first_out_of_range:
            break


# --------------------------------------------------------------------------------------
# General classes that tie everything together


class OaStores:
    """Facade bundling every OpenAI-resource key-value store (files, batches, vector stores) behind one client.

    Each attribute is a `functools.cached_property` returning a `MappingHooks`
    store built from the same `client`, so accessing `.files`, `.batches`, etc.
    only pays for the stores you actually use.

    Attributes:
        client: The `openai.Client` (or client spec) shared by every store.
    """

    def __init__(self, client: OaClientSpec = None) -> None:
        client = ensure_oa_client(client)
        self.client = client

    @cached_property
    def data_files(self):
        """`OaFilesJsonData`: files store decoding content as embeddings/task json(l) data."""
        return OaFilesJsonData(self.client)

    @cached_property
    def json_files(self):
        """`OaJsonFiles`: files store decoding content as parsed json/jsonl."""
        return OaJsonFiles(self.client)

    @cached_property
    def jsonl_files(self):
        """`OaJsonFiles`: alias of `json_files`."""
        return OaJsonFiles(self.client)

    @cached_property
    def files(self):
        """`OaFiles`: files store, values as raw bytes content."""
        return OaFiles(self.client)

    @cached_property
    def batches(self):
        """`OaBatches`: batches store, keyed by batch id."""
        return OaBatches(self.client)

    @cached_property
    def files_base(self):
        """`OaFilesBase`: the undecorated files store `data_files`/`json_files`/`files` build on."""
        return OaFilesBase(self.client)

    @cached_property
    def batches_base(self):
        """`OaBatchesBase`: the undecorated batches store `batches` builds on."""
        return OaBatchesBase(self.client)

    @cached_property
    def files_metadata(self):
        """`OaFilesMetadata`: files store, values as file metadata (`FileObject`)."""
        return OaFilesMetadata(self.client)

    @cached_property
    def vector_stores(self):
        """`OaVectorStores`: vector stores store, keyed by vector store id."""
        return OaVectorStores(self.client)

    @cached_property
    def vector_stores_base(self):
        """`OaVectorStoresBase`: the undecorated vector stores store `vector_stores` builds on."""
        return OaVectorStoresBase(self.client)


# TODO: There's a lot of functions in stores and batch that take an oa_stores argument.
#    Perhaps it's better for them to just be here in the OaDacc class instead?
from typing import Tuple
from oa.batches import get_output_file_data, get_batch_id_and_obj, get_batch_obj
from oa.util import concat_lists, extractors

Segment = str  # TODO: Replace by more specific, global type
Vector = list[float]  # TODO: Replace by more specific, global type


class OaDacc:
    """Data-access facade over `OaStores`, adding batch/embeddings convenience methods.

    Wraps an `OaStores` instance (`self.s`) built from `client`, and exposes
    higher-level operations (resolving a batch, fetching its output file,
    segments and embeddings) on top of the raw stores.
    """

    def __init__(self, client: openai.Client | None = None) -> None:
        if client is None:
            client = mk_client()
        self.client = client
        self.s = OaStores(self.client)

    extractors = extractors

    @property
    def files(self):
        """Shortcut for `self.s.files`."""
        return self.s.files

    @property
    def json_files(self):
        """Shortcut for `self.s.json_files`."""
        return self.s.json_files

    @property
    def batches(self):
        """Shortcut for `self.s.batches`."""
        return self.s.batches

    @property
    def vector_stores(self):
        """Shortcut for `self.s.vector_stores`."""
        return self.s.vector_stores

    def ensure_batch_obj(self, batch) -> BatchObj:
        """Resolve `batch` (a `BatchObj` or id) to its `BatchObj`, via `self.s`."""
        return get_batch_obj(self.s, batch)

    def batch_id_and_obj(self, batch) -> tuple[BatchId, BatchObj]:
        """Resolve `batch` (a `BatchObj` or id) to `(batch_id, batch_obj)`, via `self.s`."""
        batch_id, batch_obj = get_batch_id_and_obj(self.s, batch)
        return batch_id, batch_obj

    def get_output_file_data(self, batch):
        """Get the output file data of `batch`, if it completed successfully."""
        return get_output_file_data(batch, oa_stores=self.s)

    def check_status(self, batch: BatchSpec) -> str:
        """Check the status of a batch process."""
        batch_obj = self.ensure_batch_obj(batch)
        return batch_obj.status

    def retrieve_embeddings(self, batch: BatchSpec) -> list[Vector]:
        """Retrieve output embeddings for a completed batch."""
        output_data_obj = self.get_output_file_data(batch)

        # batch = self.s.batches[batch_id]

        return concat_lists(
            map(
                extractors.embeddings_from_output_data,
                jsonl_loads_iter(output_data_obj.content),
            )
        )

    def segments_from_file(self, file) -> list[Segment]:
        """
        Retrieve output embeddings for a completed batch, from the file it's stored in.
        """
        input_data = self.json_files[file]
        return extractors.inputs_from_file_obj(input_data)

    def segments_from_batch(self, batch: BatchSpec) -> list[Segment]:
        """
        Retrieve output embeddings for a completed batch, given the batch object or id.
        """
        batch_obj = self.ensure_batch_obj(batch)
        input_data_file_id = batch_obj.input_file_id
        return self.segments_from_file(input_data_file_id)

    def embeddings_from_file(self, file) -> list[Vector]:
        """
        Retrieve output embeddings for a completed batch, from the file it's stored in.
        """
        output_data_obj = self.s.files_base[file]
        return concat_lists(
            map(
                extractors.embeddings_from_output_data,
                jsonl_loads_iter(output_data_obj.content),
            )
        )

    def embeddings_from_batch(self, batch: BatchSpec) -> list[Vector]:
        """
        Retrieve output embeddings for a completed batch, given the batch object or id.
        """
        batch_obj = self.ensure_batch_obj(batch)
        return self.embeddings_from_file(batch_obj.output_file_id)

    def segments_and_embeddings(
        self, batch: BatchSpec
    ) -> tuple[list[Segment], list[Vector]]:
        """Retrieve segments nad embeddings for a completed batch."""
        batch_obj = self.ensure_batch_obj(batch)
        return (
            self.segments_from_batch(batch_obj),
            self.embeddings_from_batch(batch_obj),
        )

    def launch_embedding_task(
        self, segments: Iterable[Segment], **imbed_task_dict_kwargs
    ):
        """Upload `segments` as an embeddings task file and submit it as a batch."""
        # Upload files and get input file IDs
        input_file_id = self.files.create_embedding_task(
            segments, **imbed_task_dict_kwargs
        )
        batch = self.batches.append(input_file_id, endpoint="/v1/embeddings")
        return batch

    date_filter = staticmethod(date_filter)


# --------------------------------------------------------------------------------------
# debugging and diagnosis tools


def print_some_jsonl_line_fields(line):
    """Print the shape of a batch-output jsonl `line`'s fields, for interactive debugging."""
    print(f"{list(line)=}")
    print(f"{list(line['response'])=}")
    print(f"{list(line['response']['body'])=}")
    print(f"{line['response']['body']['object']=}")
    print(f"{len(line['response']['body']['data'])=}")
    print(f"{list(line['response']['body']['data'][0])=}")


# --------------------------------------------------------------------------------------
# Vector Stores


class OaVectorStoresBase(OaMapping):
    """Base class for OpenAI vector stores mapping interface."""

    def __init__(self, client: openai.Client | None = None, **extra_kwargs):
        if client is None:
            client = mk_client()
        self.client = client
        self.extra_kwargs = extra_kwargs

    def _iter(self):
        """Iterate over vector store objects."""
        return self.client.vector_stores.list(**self._list_kwargs)

    @extract_id
    def metadata(self, vs_id) -> Any:  # TODO: Add proper vector store type
        """Get vector store metadata by ID."""
        return self.client.vector_stores.retrieve(vs_id, **self.extra_kwargs)

    _getitem = metadata

    @extract_id
    def _delitem(self, vs_id):
        """Delete a vector store."""
        return self.client.vector_stores.delete(vs_id)

    def create(self, name: str, **config) -> Any:
        """Create a new vector store with given name."""
        return self.client.vector_stores.create(
            name=name, **config, **self.extra_kwargs
        )

    def get_by_name(self, name: str):
        """Get vector store by name (since OpenAI API doesn't support this directly)."""
        for vs in self.client.vector_stores.list():
            if vs.name == name:
                return vs
        raise KeyError(f"Vector store with name '{name}' not found")


@wrap_kvs(key_decoder=attrgetter("id"))
class OaVectorStores(OaVectorStoresBase):
    """
    A key-value store for OpenAI vector stores.
    Keys are the vector store IDs.
    """


class OaVectorStoreFiles(OaMapping):
    """Mapping interface for files within a vector store."""

    def __init__(
        self,
        vector_store_id: str,
        client: openai.Client | None = None,
        **extra_kwargs,
    ):
        if client is None:
            client = mk_client()
        self.client = client
        self.vector_store_id = vector_store_id
        self.extra_kwargs = extra_kwargs

    def _iter(self):
        """Iterate over file objects in the vector store."""
        return self.client.vector_stores.files.list(
            vector_store_id=self.vector_store_id, **self._list_kwargs
        )

    @extract_id
    def metadata(self, file_id) -> Any:
        """Get file info from vector store."""
        return self.client.vector_stores.files.retrieve(
            vector_store_id=self.vector_store_id, file_id=file_id, **self.extra_kwargs
        )

    _getitem = metadata

    @extract_id
    def _delitem(self, file_id):
        """Remove file from vector store."""
        return self.client.vector_stores.files.delete(
            vector_store_id=self.vector_store_id, file_id=file_id
        )

    def add_file(self, file_id: str) -> Any:
        """Add file to vector store."""
        return self.client.vector_stores.files.create(
            vector_store_id=self.vector_store_id, file_id=file_id, **self.extra_kwargs
        )

    def __setitem__(self, file_id, _):
        """Add file to vector store (for MutableMapping interface)."""
        self.add_file(file_id)
