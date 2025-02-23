"""Data object layers for openai resources"""

from collections.abc import Mapping
from operator import attrgetter, methodcaller
from typing import Optional, Union, Iterable, Callable, Any, List, Literal, T
import json
from functools import wraps, partial, cached_property

import openai  # TODO: Import from oa.util instead?

# TODO: Move openai imports to go through oa.util
from openai.resources.files import Files as OpenaiFiles, FileTypes, FileObject
from openai.types import FileObject

from oa.base import mk_client, TextOrTexts
from oa.batches import (
    mk_batch_file_embeddings_task,
    BatchSpec,
    BatchObj,
    BatchId,
)
from oa.util import (
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
    method: Optional[Callable] = None,
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
    def __delitem__(self, key):
        return self._delitem(key)

    def __setitem__(self, key, value):
        return self._setitem(key, value)


class OaMapping(MappingHooks, KvReader):
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
    # @params_from_openai_files_cls
    @Sig.replace_kwargs_using(files_create_sig - "purpose")
    def __init__(
        self,
        client: Optional[openai.Client] = None,
        purpose: Optional[Purpose] = DFLT_PURPOSE,  # type: ignore
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
        return self.client.files.retrieve(file_id, **self.extra_kwargs)

    @extract_id
    def content(self, file_id):
        return self.client.files.content(file_id, **self.extra_kwargs)

    _getitem = content

    @extract_id
    def _delitem(self, file_id):
        # Delete the file using the API (you might need to implement this)
        return self.client.files.delete(file_id)  # Assuming there's a delete method

    @params_from_openai_files_cls
    def append(self, file: Union[FileTypes, dict]) -> FileObject:
        # Note: self.client.create can be found in openai.resources.files.Files.create
        if is_task_dict(file) or is_task_dict_list(file):
            file = jsonl_dumps(file, self.encoding)
        return self.client.files.create(
            file=file, purpose=self.purpose, **self.extra_kwargs
        )

    @Sig.replace_kwargs_using(mk_batch_file_embeddings_task)
    def create_embedding_task(self, texts: TextOrTexts, **extra_kwargs):
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
    embeddings: List[float]  # The third known field, 'embeddings'


DataObjectValue = TypeVar("DataObjectValue")
DataObjectValue.__doc__ = (
    "The value that a DataObject holds in the field indicated by the object field"
)

jsonl_loads_response_lines = partial(
    jsonl_loads_iter, get_lines=methodcaller("iter_lines")
)

jsonl_loads_list = Pipe(jsonl_loads_response_lines, list)


def response_body_data(response_dict: ResponseDict) -> DataObject:
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
    def __init__(self, client: Optional[openai.Client] = None, **extra_kwargs):
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
        metadata: Optional[Dict[str, str]] = None,
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
    date_extractor: Union[str, Callable] = "created_at",
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
    def __init__(self, client: Optional[openai.Client] = None) -> None:
        if client is None:
            client = mk_client()
        self.client = client

    @cached_property
    def data_files(self):
        return OaFilesJsonData(self.client)

    @cached_property
    def json_files(self):
        return OaJsonFiles(self.client)

    @cached_property
    def jsonl_files(self):
        return OaJsonFiles(self.client)

    @cached_property
    def files(self):
        return OaFiles(self.client)

    @cached_property
    def batches(self):
        return OaBatches(self.client)

    @cached_property
    def files_base(self):
        return OaFilesBase(self.client)

    @cached_property
    def batches_base(self):
        return OaBatchesBase(self.client)

    @cached_property
    def files_metadata(self):
        return OaFilesMetadata(self.client)


# TODO: There's a lot of functions in stores and batch that take an oa_stores argument.
#    Perhaps it's better for them to just be here in the OaDacc class instead?
from typing import Tuple
from oa.batches import get_output_file_data, get_batch_id_and_obj, get_batch_obj
from oa.util import concat_lists, extractors

Segment = str  # TODO: Replace by more specific, global type
Vector = List[float]  # TODO: Replace by more specific, global type


class OaDacc:
    def __init__(self, client: Optional[openai.Client] = None) -> None:
        if client is None:
            client = mk_client()
        self.client = client
        self.s = OaStores(self.client)

    extractors = extractors

    @property
    def files(self):
        return self.s.files

    @property
    def json_files(self):
        return self.s.json_files

    @property
    def batches(self):
        return self.s.batches

    def ensure_batch_obj(self, batch) -> BatchObj:
        return get_batch_obj(self.s, batch)

    def batch_id_and_obj(self, batch) -> Tuple[BatchId, BatchObj]:
        batch_id, batch_obj = get_batch_id_and_obj(self.s, batch)
        return batch_id, batch_obj

    def get_output_file_data(self, batch):
        return get_output_file_data(batch, oa_stores=self.s)

    def check_status(self, batch: BatchSpec) -> str:
        """Check the status of a batch process."""
        batch_obj = self.ensure_batch_obj(batch)
        return batch_obj.status

    def retrieve_embeddings(self, batch: BatchSpec) -> List[Vector]:
        """Retrieve output embeddings for a completed batch."""
        output_data_obj = self.get_output_file_data(batch)

        # batch = self.s.batches[batch_id]

        return concat_lists(
            map(
                extractors.embeddings_from_output_data,
                jsonl_loads_iter(output_data_obj.content),
            )
        )

    def segments_from_file(self, file) -> List[Segment]:
        """
        Retrieve output embeddings for a completed batch, from the file it's stored in.
        """
        input_data = self.json_files[file]
        return extractors.inputs_from_file_obj(input_data)

    def segments_from_batch(self, batch: BatchSpec) -> List[Segment]:
        """
        Retrieve output embeddings for a completed batch, given the batch object or id.
        """
        batch_obj = self.ensure_batch_obj(batch)
        input_data_file_id = batch_obj.input_file_id
        return self.segments_from_file(input_data_file_id)

    def embeddings_from_file(self, file) -> List[Vector]:
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

    def embeddings_from_batch(self, batch: BatchSpec) -> List[Vector]:
        """
        Retrieve output embeddings for a completed batch, given the batch object or id.
        """
        batch_obj = self.ensure_batch_obj(batch)
        return self.embeddings_from_file(batch_obj.output_file_id)

    def segments_and_embeddings(
        self, batch: BatchSpec
    ) -> Tuple[List[Segment], List[Vector]]:
        """Retrieve segments nad embeddings for a completed batch."""
        batch_obj = self.ensure_batch_obj(batch)
        return (
            self.segments_from_batch(batch_obj),
            self.embeddings_from_batch(batch_obj),
        )

    def launch_embedding_task(
        self, segments: Iterable[Segment], **imbed_task_dict_kwargs
    ):
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
    print(f"{list(line)=}")
    print(f"{list(line['response'])=}")
    print(f"{list(line['response']['body'])=}")
    print(f"{line['response']['body']['object']=}")
    print(f"{len(line['response']['body']['data'])=}")
    print(f"{list(line['response']['body']['data'][0])=}")
