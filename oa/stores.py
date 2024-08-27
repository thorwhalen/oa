"""Data object layers for openai resources"""

from collections.abc import Mapping
from operator import attrgetter, methodcaller
from typing import Optional, Union, Iterable, Callable, Any, List, Literal
import json
from functools import wraps, partial, cached_property

import openai  # TODO: Import from oa.util instead?
from openai.resources.files import Files as OpenaiFiles, FileTypes, FileObject
from openai.types import FileObject, Batch
from oa.base import mk_client, TextOrTexts
from oa.batches import mk_batch_file_embeddings_task
from oa.util import (
    merge_multiple_signatures,
    source_parameter_props_from,
    utc_int_to_iso_date,
    Purpose,
    BatchesEndpoint,
)
from i2 import Sig
from i2.signatures import SignatureAble, ParamsAble
from i2 import postprocess
from dol import wrap_kvs, KvReader

FilterFunc = Callable[[Any], bool]


DFLT_PURPOSE = 'batch'
DFLT_BATCHES_ENDPOINT = '/v1/embeddings'

openai_files_cumul_sig = merge_multiple_signatures(
    [
        OpenaiFiles.create,
        OpenaiFiles.retrieve,
        OpenaiFiles.delete,
        Sig(OpenaiFiles.list),
    ],
    default_conflict_method='take_first',
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
    return hasattr(x, 'id')


def _is_instance_and_has_id(x=None, *, type_: type):
    if x is None:
        return partial(_is_instance_and_has_id, type_=type_)
    else:
        return isinstance(x, type_) and hasattr(x, 'id')


def extract_id(
    method: Optional[Callable] = None,
    *,
    is_id: FilterFunc = _is_string,
    has_id: FilterFunc = _has_id_attr,
    get_id: Callable = attrgetter('id'),
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


DFLT_ENCODING = 'utf-8'


def jsonl_dumps(x: Iterable, encoding: str = DFLT_ENCODING) -> bytes:
    r"""
    Serialize an iterable as JSONL bytes

    >>> jsonl_dumps([{'a': 1}, {'b': 2}])
    b'{"a": 1}\n{"b": 2}'

    """
    if isinstance(x, Mapping):
        return json.dumps(x).encode(encoding)
    else:
        return b'\n'.join(json.dumps(task).encode(encoding) for task in x)


def jsonl_loads(bytes_: bytes, encoding: str = DFLT_ENCODING) -> List[dict]:
    r"""
    Deserialize JSONL bytes into a python iterable (dict or list of dicts)

    >>> jsonl_loads(b'\n{"a": 1}\n\n{"b": 2}')
    [{'a': 1}, {'b': 2}]

    """

    def gen():
        for line in filter(None, map(methodcaller('strip'), bytes_.split(b'\n'))):
            yield json.loads(line.decode(encoding))

    return list(gen())


class OaFilesBase(OaMapping):
    # @params_from_openai_files_cls
    @Sig.replace_kwargs_using(files_create_sig - 'purpose')
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
            self._list_kwargs = {'purpose': self.purpose}
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
            file = json1_dumps(file, self.encoding)
        return self.client.files.create(
            file=file, purpose=self.purpose, **self.extra_kwargs
        )

    # TODO: Inject signature from mk_batch_file_embeddings_task
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


@wrap_kvs(key_decoder=attrgetter('id'), value_decoder=attrgetter('content'))
class OaFiles(OaFilesBase):
    """
    A key-value store for OpenAI files content data.
    Keys are the file IDs.
    """


# TODO: Find a non-underscored place to import HttpxBinaryResponseContent from
# Note: This is just used for annotation purposes
from openai._legacy_response import HttpxBinaryResponseContent

from typing import TypedDict


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

    object: Literal['embeddings']  # This enforces the value to be 'embeddings'
    index: int  # Required field
    embeddings: List[float]  # The third known field, 'embeddings'


def jsonl_loads_iter(response: HttpxBinaryResponseContent) -> Iterable[ResponseDict]:
    return map(json.loads, response.iter_lines())


def response_body_data(response_dict: ResponseDict) -> DataObject:
    return response_dict['response']['body']['data']


def object_of_data(data: DataObject) -> Any:
    """
    This function extracts the object (value) from a {object: V, index: i, V} dict
    which is the format you'll find in the ['response']['body']['data'] of a response.

    >>> object_of_data({'object': 'embeddings', 'index': 3, 'embeddings': [1, 2, 3]})
    [1, 2, 3]

    """
    return data[data['object']]


# Note: This is to be used as a content decoder
# TODO: The looping logic is messy, consider refactoring
@postprocess(dict)
def get_json_data_from_response(response: HttpxBinaryResponseContent):
    """Extract the embeddings from a HttpxBinaryResponseContent object"""
    for d in jsonl_loads_iter(response):
        custom_id = d['custom_id']
        dd = response_body_data(d)
        if isinstance(dd, list):
            for ddd in dd:
                yield custom_id, object_of_data(ddd)
        else:
            yield custom_id, object_of_data(dd)


@wrap_kvs(key_decoder=attrgetter('id'), value_decoder=get_json_data_from_response)
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

    def _iter(self) -> Iterable[Batch]:
        """Return an iterator over batch IDs"""
        return self.client.batches.list(**self._list_kwargs)

    @extract_id
    def metadata(self, batch_id: str) -> Batch:
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


@wrap_kvs(key_decoder=attrgetter('id'))
class OaBatches(OaBatchesBase):
    """
    A key-value store for OpenAI batches metadata.
    Keys are the batch IDs.
    """


class OaStores:
    def __init__(self, client: Optional[openai.Client] = None) -> None:
        if client is None:
            client = mk_client()
        self.client = client

    @cached_property
    def data_files(self):
        return OaFilesJsonData(self.client)

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


from oa.batches import get_output_file_data


class OaDacc:
    def __init__(self, client: Optional[openai.Client] = None) -> None:
        if client is None:
            client = mk_client()
        self.client = client
        self.s = OaStores(self.client)

    def get_output_file_data(self, batch):
        return get_output_file_data(batch, oa_stores=self.s)
