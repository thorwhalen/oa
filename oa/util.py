"""oa utils"""

from importlib.resources import files
import os
from functools import partial, lru_cache
from typing import Mapping, Union, get_args, Literal
from types import SimpleNamespace

from i2 import Sig, get_app_data_folder
import dol
import graze
from config2py import (
    get_config,
    ask_user_for_input,
    get_configs_local_store,
    simple_config_getter,
    user_gettable,
)

import openai  # pip install openai (see https://pypi.org/project/openai/)
from openai.resources.files import FileObject
from openai.resources.batches import Batches as OpenaiBatches, Batch as BatchObj

BatchObj  # to avoid unused import warning (the import here is for other modules)


def get_package_name():
    """Return current package name"""
    # return __name__.split('.')[0]
    # TODO: See if this works in all cases where module is in highest level of package
    #  but meanwhile, hardcode it:
    return "oa"


# get app data dir path and ensure it exists
pkg_name = get_package_name()
data_files = files(pkg_name) / "data"
templates_files = data_files / "templates"
_root_app_data_dir = get_app_data_folder()
app_data_dir = os.environ.get(
    f"{pkg_name.upper()}_APP_DATA_DIR",
    os.path.join(_root_app_data_dir, pkg_name),
)
app_data_dir = dol.ensure_dir(app_data_dir)
djoin = partial(os.path.join, app_data_dir)

# _open_api_key_env_name = 'OPENAI_API_KEY'
# _api_key = os.environ.get(_open_api_key_env_name, None)
# if _api_key is None:
#     _api_key = getpass.getpass(
#         f"Please set your OpenAI API key and press enter to continue. "
#         f"I will put it in the environment variable {_open_api_key_env_name} "
#     )
# openai.api_key = _api_key

configs_local_store = get_configs_local_store(pkg_name)

_DFLT_CONFIGS = {
    "OPENAI_API_KEY_ENV_NAME": "OPENAI_API_KEY",
    "OA_DFLT_TEMPLATES_SOURCE_ENV_NAME": "OA_DFLT_TEMPLATES_SOURCE",
    "OA_DFLT_ENGINE": "gpt-3.5-turbo-instruct",
    "OA_DFLT_MODEL": "gpt-3.5-turbo",
}

# write the defaults to the local store, if key missing there
for k, v in _DFLT_CONFIGS.items():
    if k not in configs_local_store:
        configs_local_store[k] = v


config_sources = [
    configs_local_store,  # look in the local store
    os.environ,  # look in the environment variables
    user_gettable(
        configs_local_store
    ),  # ask the user (and save response in local store)
]


def kv_strip_value(k, v):
    return v.strip()


# The main config getter for this package
config_getter = get_config(sources=config_sources, egress=kv_strip_value)


# Get the OPENAI_API_KEY_ENV_NAME and DFLT_TEMPLATES_SOURCE_ENV_NAME
OPENAI_API_KEY_ENV_NAME = config_getter("OPENAI_API_KEY_ENV_NAME")
DFLT_TEMPLATES_SOURCE_ENV_NAME = config_getter("OA_DFLT_TEMPLATES_SOURCE_ENV_NAME")

# TODO: Understand the model/engine thing better and merge defaults if possible
DFLT_ENGINE = config_getter("OA_DFLT_ENGINE")
DFLT_MODEL = config_getter("OA_DFLT_MODEL")

# TODO: Add the following to config_getter mechanism
DFLT_EMBEDDINGS_MODEL = "text-embedding-3-small"


Purpose = FileObject.model_fields["purpose"].annotation
DFLT_PURPOSE = "batch"
BatchesEndpoint = eval(Sig(OpenaiBatches.create).annotations["endpoint"])
batch_endpoints_values = get_args(BatchesEndpoint)
batch_endpoints_keys = [
    k.replace("/v1/", "").replace("/", "_") for k in batch_endpoints_values
]
batch_endpoints = SimpleNamespace(
    **dict(zip(batch_endpoints_keys, batch_endpoints_values))
)


_pricing_category_aliases = {
    "text": "Latest models - Text tokens",
    "audio": "Latest models - Audio tokens",
    "finetune": "Fine tuning",
    "tools": "Built-in tools",
    "search": "Web search",
    "speech": "Transcription and speech generation",
    "images": "Image generation",
    "embeddings": "Embeddings",
    "moderation": "Moderation",
    "other": "Other models",
}

PricingCategory = Literal[tuple(_pricing_category_aliases.keys())]


def pricing_info(category: PricingCategory = None, *, print_data_date=False):
    """
    Return the pricing info for the OpenAI API.

    Note: These are not live prices. Live prices can be found here:

    The information pricing_info returns is taken from the file `openai_api_pricing_info.json`
    in the `data` directory of the package.
    To print a message with the data date, do `pricing_info(print_data_date=True)`.
    """
    info_filepath = data_files / "openai_api_pricing_info.json"
    if print_data_date:
        print(f"Data date: {info_filepath.stat().st_mtime}")
    info = json.loads(info_filepath.read_text())

    if category is None:

        def _pricing_info():
            for category in _pricing_category_aliases:
                for d in pricing_info(category):
                    yield dict(category=category, **d)

        return list(_pricing_info())
    else:
        return info[_pricing_category_aliases.get(category, category)]["pricing_table"]


pricing_info.category_aliases = _pricing_category_aliases

# TODO: Write tools to update mteb_eval
# Note: OpenAI API live prices: https://platform.openai.com/docs/pricing
embeddings_models = {
    "text-embedding-3-small": {
        "price_per_million_tokens": 0.02,  # in dollars
        "pages_per_dollar": 62500,  # to do:
        "performance_on_mteb_eval": 62.3,
        "max_input": 8191,
    },
    "text-embedding-3-large": {
        "price_per_million_tokens": 0.13,  # in dollars
        "pages_per_dollar": 9615,
        "performance_on_mteb_eval": 64.6,
        "max_input": 8191,
    },
    "text-embedding-ada-002": {
        "price_per_million_tokens": 0.10,  # in dollars
        "pages_per_dollar": 12500,
        "performance_on_mteb_eval": 61.0,
        "max_input": 8191,
    },
}


# add batch-api models
def _generate_batch_api_models_info(models_info_dict, batch_api_discount=0.5):
    for model_name, model_info in models_info_dict.items():
        m = model_info.copy()
        m["price_per_million_tokens"] = round(
            m["price_per_million_tokens"] * batch_api_discount, 4
        )
        m["pages_per_dollar"] = int(1 / m["price_per_million_tokens"])
        # the rest remains the same
        yield f"batch__{model_name}", m


embeddings_models = dict(
    embeddings_models,
    **dict(_generate_batch_api_models_info(embeddings_models, batch_api_discount=0.5)),
)

# Note: OpenAI API live prices: https://platform.openai.com/docs/pricing
chat_models = {
    "gpt-4": {
        "price_per_million_tokens": 30.00,  # in dollars
        "price_per_million_tokens_output": 60.00,  # in dollars
        "pages_per_dollar": 134,  # approximately
        "performance_on_eval": "Advanced reasoning for complex tasks",
        "max_input": 8192,  # tokens
    },
    "gpt-4-32k": {
        "price_per_million_tokens": 60.00,  # in dollars
        "price_per_million_tokens_output": 120.00,  # in dollars
        "pages_per_dollar": 67,
        "performance_on_eval": "Extended context window for long documents",
        "max_input": 32768,  # tokens
    },
    "gpt-4-turbo": {
        "price_per_million_tokens": 10.00,  # in dollars
        "price_per_million_tokens_output": 30.00,  # in dollars
        "pages_per_dollar": 402,
        "performance_on_eval": "Cost-effective version of GPT-4",
        "max_input": 8192,  # tokens
    },
    "o1": {
        "price_per_million_tokens": 15.00,  # in dollars
        "price_per_million_tokens_output": 60.00,  # in dollars
        "pages_per_dollar": 268,
        "performance_on_eval": "Optimized for complex reasoning in STEM fields",
        "max_input": 8192,  # tokens
    },
    "o1-mini": {
        "price_per_million_tokens": 1.10,  # in dollars
        "price_per_million_tokens_output": 4.40,  # in dollars
        "pages_per_dollar": 1341,
        "performance_on_eval": "Cost-effective reasoning for simpler tasks",
        "max_input": 8192,  # tokens
    },
    "gpt-4o": {
        "price_per_million_tokens": 2.50,  # in dollars
        "price_per_million_tokens_output": 10.0,  # in dollars
        "pages_per_dollar": 804,  # approximately
        "performance_on_eval": "Efficiency-optimized version of GPT-4 for better performance on reasoning tasks",
        "max_input": 8192,  # tokens
    },
    "gpt-4o-mini": {
        "price_per_million_tokens": 0.15,  # in dollars,
        "price_per_million_tokens_output": 0.60,  # in dollars
        "pages_per_dollar": 13410,
        "performance_on_eval": "Highly cost-effective, optimized for simple tasks with faster response times",
        "max_input": 8192,  # tokens
    },
}

model_information_dict = dict(
    **embeddings_models,
    **chat_models,
    # TODO: Add more model information dicts here
)


# Have a particular way to get this api key
@lru_cache
def get_api_key_from_config():
    return get_config(
        OPENAI_API_KEY_ENV_NAME,
        sources=[
            # Try to find it in oa config
            configs_local_store,
            # Try to find it in os.environ (environmental variables)
            os.environ,
            # If not, ask the user to input it
            lambda k: ask_user_for_input(
                f"Please set your OpenAI API key and press enter to continue. "
                "If you don't have one, you can get one at "
                "https://platform.openai.com/account/api-keys. ",
                mask_input=True,
                masking_toggle_str="",
                egress=lambda v: configs_local_store.__setitem__(k, v),
            ),
        ],
        egress=kv_strip_value,
    )


openai.api_key = get_api_key_from_config()


@lru_cache
def mk_client(api_key=None, **client_kwargs) -> openai.Client:
    api_key = api_key or get_api_key_from_config()
    return openai.OpenAI(api_key=api_key, **client_kwargs)


# TODO: Pros and cons of using a default client
#   Reason was that I was fed up of having to pass the client to every function
try:
    dflt_client = mk_client()
except Exception as e:
    dflt_client = None

_grazed_dir = dol.ensure_dir(os.path.join(app_data_dir, "grazed"))
grazed = graze.Graze(rootdir=_grazed_dir)


chatgpt_templates_dir = os.path.join(templates_files, "chatgpt")

DFLT_TEMPLATES_SOURCE = get_config(
    DFLT_TEMPLATES_SOURCE_ENV_NAME,
    sources=[os.environ],
    default=f"{chatgpt_templates_dir}",
)


# TODO: This is general: Bring this in dol or dolx
def _extract_folder_and_suffixes(
    string: str, default_suffixes=(), *, default_folder="", root_sep=":", suffix_sep=","
):
    root_folder, *suffixes = string.split(root_sep)
    if root_folder == "":
        root_folder = default_folder
    if len(suffixes) == 0:
        suffixes = default_suffixes
    elif len(suffixes) == 1:
        suffixes = suffixes[0].split(suffix_sep)
    else:
        raise ValueError(
            f"template_store must be a path to a directory of templates, "
            f"optionally followed by a colon and a list of file suffixes to use"
        )
    return root_folder, suffixes


def mk_template_store(template_store: Union[Mapping, str]):
    if isinstance(template_store, Mapping):
        return template_store
    elif isinstance(template_store, str):
        root_folder, suffixes = _extract_folder_and_suffixes(template_store)
        suffix_filter = dol.filt_iter.suffixes(suffixes)
        return suffix_filter(dol.TextFiles(root_folder))
    else:
        raise TypeError(
            f"template_store must be a Mapping or a path to a directory of templates"
        )


import tiktoken


def num_tokens(text: str = None, model: str = DFLT_MODEL) -> int:
    """Return the number of tokens in a string, under given model.

    keywords: token count, number of tokens
    """
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text, disallowed_special=()))


# --------------------------------------------------------------------------------------
# Extraction

from oa.oa_types import BatchRequest, EmbeddingResponse, InputDataJsonL
from ju import ModelExtractor
from types import SimpleNamespace
from operator import itemgetter
import pickle
import tempfile

from dol import add_ipython_key_completions, Pipe

models = [BatchRequest, EmbeddingResponse, InputDataJsonL]

oa_extractor = Pipe(ModelExtractor(models), add_ipython_key_completions)


def oa_extractors_obj(**named_paths):
    """
    Return a SimpleNamespace of extractors for the named paths
    """
    return SimpleNamespace(
        **{
            name: Pipe(oa_extractor, itemgetter(path))
            for name, path in named_paths.items()
        }
    )


extractors = oa_extractors_obj(
    embeddings_from_output_data="response.body.data.*.embedding",
    inputs_from_file_obj="body.input",
)


# --------------------------------------------------------------------------------------
# misc utils
from typing import Iterable
from dateutil.parser import parse as parse_date
from datetime import datetime, timezone
from itertools import chain, islice
from typing import (
    Iterable,
    Union,
    Dict,
    List,
    Tuple,
    Mapping,
    TypeVar,
    Iterator,
    Callable,
    Optional,
    T,
)

KT = TypeVar("KT")  # there's a typing.KT, but pylance won't allow me to use it!
VT = TypeVar("VT")  # there's a typing.VT, but pylance won't allow me to use it!


def chunk_iterable(
    iterable: Union[Iterable[T], Mapping[KT, VT]],
    chk_size: int,
    *,
    chunk_type: Optional[Callable[..., Union[Iterable[T], Mapping[KT, VT]]]] = None,
) -> Iterator[Union[List[T], Tuple[T, ...], Dict[KT, VT]]]:
    """
    Divide an iterable into chunks/batches of a specific size.

    Handles both mappings (e.g. dicts) and non-mappings (lists, tuples, sets...)
    as you probably expect it to (if you give a dict input, it will chunk on the
    (key, value) items and return dicts of these).
    Thought note that you always can control the type of the chunks with the
    `chunk_type` argument.

    Args:
        iterable: The iterable or mapping to divide.
        chk_size: The size of each chunk.
        chunk_type: The type of the chunks (list, tuple, set, dict...).

    Returns:
        An iterator of dicts if the input is a Mapping, otherwise an iterator
        of collections (list, tuple, set...).

    Examples:
        >>> list(chunk_iterable([1, 2, 3, 4, 5], 2))
        [[1, 2], [3, 4], [5]]

        >>> list(chunk_iterable((1, 2, 3, 4, 5), 3, chunk_type=tuple))
        [(1, 2, 3), (4, 5)]

        >>> list(chunk_iterable({"a": 1, "b": 2, "c": 3}, 2))
        [{'a': 1, 'b': 2}, {'c': 3}]

        >>> list(chunk_iterable({"x": 1, "y": 2, "z": 3}, 1, chunk_type=dict))
        [{'x': 1}, {'y': 2}, {'z': 3}]
    """
    if isinstance(iterable, Mapping):
        if chunk_type is None:
            chunk_type = dict
        it = iter(iterable.items())
        for first in it:
            yield {
                key: value for key, value in chain([first], islice(it, chk_size - 1))
            }
    else:
        if chunk_type is None:
            if isinstance(iterable, (list, tuple, set)):
                chunk_type = type(iterable)
            else:
                chunk_type = list
        it = iter(iterable)
        for first in it:
            yield chunk_type(chain([first], islice(it, chk_size - 1)))


def concat_lists(lists: Iterable[Iterable]):
    """Concatenate a list of lists into a single list.

    >>> concat_lists([[1, 2], [3, 4], [5, 6]])
    [1, 2, 3, 4, 5, 6]
    """
    return list(chain.from_iterable(lists))


# a function to translate utc time in 1723557717 format into a human readable format
def utc_int_to_iso_date(utc_time: int) -> str:
    """
    Convert utc integer timestamp to more human readable iso format.
    Inverse of iso_date_to_utc_int.

    >>> utc_int_to_iso_date(1723471317)
    '2024-08-12T14:01:57+00:00'
    """
    return datetime.utcfromtimestamp(utc_time).replace(tzinfo=timezone.utc).isoformat()


def iso_date_to_utc_int(iso_date: str) -> int:
    """
    Convert iso date string to utc integer timestamp.
    Inverse of utc_int_to_iso_date.

    >>> iso_date_to_utc_int('2024-08-12T14:01:57+00:00')
    1723471317
    """
    return int(parse_date(iso_date).timestamp())


# just to have the inverse of a function close to the function itself:
utc_int_to_iso_date.inverse = iso_date_to_utc_int
iso_date_to_utc_int.inverse = utc_int_to_iso_date


def transpose_iterable(iterable_of_tuples):
    return zip(*iterable_of_tuples)


def transpose_and_concatenate(iterable_of_tuples):
    return map(list, map(chain.from_iterable, transpose_iterable(iterable_of_tuples)))


def save_in_temp_dir(obj, serializer=pickle.dumps):
    """
    Saves obj in a temp file, using serializer to serialize it, and returns its path.
    """
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(serializer(obj))
    return f.name


from typing import Any, Optional, Callable


def mk_local_files_saves_callback(
    rootdir: Optional[str] = None,
    *,
    serializer: Callable[[Any], bytes] = pickle.dumps,
    index_to_filename: Callable[[int], str] = "{:05.0f}".format,
    print_dir_path: bool = True,
):
    """
    Returns a function that takes two inputs (i: int, obj: Any) and saves the
    serializer(obj) bytes in a file named index_to_filename(i) in the rootdir.
    If rootdir, a temp dir is used.
    """
    if rootdir is None:
        rootdir = tempfile.mkdtemp()
    assert os.path.isdir(rootdir), f"rootdir {rootdir} is not a directory"
    if print_dir_path:
        print(f"Files will be saved in {rootdir}")

    def save_to_file(i, obj):
        with open(os.path.join(rootdir, index_to_filename(i)), "wb") as f:
            f.write(serializer(obj))

    return save_to_file


import json
from operator import methodcaller
from typing import Iterable, Callable, T
from dol import Pipe

DFLT_ENCODING = "utf-8"


def jsonl_dumps(x: Iterable, encoding: str = DFLT_ENCODING) -> bytes:
    r"""
    Serialize an iterable as JSONL bytes

    >>> jsonl_dumps([{'a': 1}, {'b': 2}])
    b'{"a": 1}\n{"b": 2}'

    """
    if isinstance(x, Mapping):
        return json.dumps(x).encode(encoding)
    else:
        return b"\n".join(json.dumps(line).encode(encoding) for line in x)


def jsonl_loads_iter(
    src: T,
    *,
    get_lines: Callable[[T], Iterable[bytes]] = bytes.splitlines,
    line_egress: Callable = methodcaller("strip"),
) -> Iterable[dict]:
    r"""
    Deserialize JSONL bytes into a python iterable (dict or list of dicts)

    >>> list(jsonl_loads(b'\n{"a": 1}\n\n{"b": 2}'))
    [{'a': 1}, {'b': 2}]

    """

    for line in filter(None, map(line_egress, get_lines(src))):
        yield json.loads(line)


jsonl_loads = Pipe(jsonl_loads_iter, list)
jsonl_loads.__doc__ = jsonl_loads_iter.__doc__


from typing import Iterable
import openai
from i2.signatures import SignatureAble
from inspect import Parameter


@Sig.replace_kwargs_using(Sig.merge_with_sig)
def merge_multiple_signatures(
    iterable_of_sigs: Iterable[SignatureAble], **merge_with_sig_options
):
    sig = Sig()
    for input_sig in map(Sig, iterable_of_sigs):
        sig = sig.merge_with_sig(input_sig, **merge_with_sig_options)
    return sig


# TODO: Control whether to only overwrite if defaults and/or annotations don't already exist
# TODO: Control if matching by name or annotation
def source_parameter_props_from(parameters: Mapping[str, Parameter]):
    """
    A decorator that will change the annotation and default of the parameters of the
    decorated function, sourcing them from `parameters`, matching them by name.
    """

    def decorator(func):
        sig = Sig(func)
        common_names = set(sig.names) & set(parameters.keys())
        sig = sig.ch_defaults(
            **{name: parameters[name].default for name in common_names}
        )
        sig = sig.ch_annotations(
            **{name: parameters[name].annotation for name in common_names}
        )
        return sig(func)

    return decorator


# --------------------------------------------------------------------------------------
# ProcessingManager
# Monitoring the processing of a collection of items

from dataclasses import dataclass, field
import time
from typing import (
    Callable,
    Any,
    Optional,
    Tuple,
    Dict,
    MutableMapping,
    Iterable,
    Union,
    TypeVar,
    Generic,
)

# Define type variables and aliases
KT = TypeVar("KT")  # Key type
VT = TypeVar("VT")  # Value type (pending item type)
Result = TypeVar("Result")  # Result type returned by processing_function
Status = str  # Could be an Enum in future


@dataclass
class ProcessingManager(Generic[KT, VT, Result]):
    """
    A class to manage and monitor the processing of a collection of items, allowing customizable
    processing functions, status handling, and timing control, using keys to track items.

    **Use Case**:
    - Ideal for scenarios where you have a set of items identified by keys that require periodic status
      checks until they reach a completed state.
    - Useful for managing batch jobs, asynchronous tasks, or any operations where items may not complete
      processing immediately.

    **Attributes**:
    - **pending_items** (`MutableMapping[KT, VT]`): The mapping of keys to pending items.
        - If an iterable is provided, it is converted to a dict using `dict(enumerate(iterable))`.
    - **processing_function** (`Callable[[VT], Tuple[Status, Result]]`): A function that takes a value (item)
      and returns a tuple of `(status, result)`.
        - `status` (`Status`): Indicates the current state of the item (e.g., `'completed'`, `'in_progress'`, `'failed'`).
        - `result` (`Result`): Additional data or context about the item's processing result.
    - **handle_status_function** (`Callable[[VT, Status, Result], bool]`): A function that decides whether to remove
      an item from `pending_items` based on its `status` and `result`.
        - Returns `True` if the item should be removed (e.g., processing is complete or failed irrecoverably).
    - **wait_time_function** (`Callable[[float, Dict], float]`): A function that determines how long to wait
      before the next processing cycle.
        - Takes `cycle_duration` (time taken for the last cycle) and `locals()` dictionary as inputs.
        - Returns the sleep time in seconds.
    - **status_check_interval** (`float`): Desired minimum time (in seconds) between status checks. Defaults to `5.0`.
    - **max_cycles** (`Optional[int]`): Maximum number of processing cycles to perform. If `None`, there is no limit.
    - **completed_items** (`MutableMapping[KT, Result]`): Mapping of keys to results for items that have been processed.
    - **cycles** (`int`): Number of processing cycles that have been performed.
    """

    pending_items: Union[MutableMapping[KT, VT], Iterable[VT]]
    processing_function: Callable[[VT], Tuple[Status, Result]]
    handle_status_function: Callable[[VT, Status, Result], bool]
    wait_time_function: Callable[[float, Dict], float]
    status_check_interval: float = 5.0
    max_cycles: Optional[int] = None
    completed_items: MutableMapping[KT, Result] = field(default_factory=dict)
    cycles: int = 0  # Tracks the number of cycles performed

    def __post_init__(self):
        # Convert pending_items to a MutableMapping if it's not one already
        if not isinstance(self.pending_items, MutableMapping):
            self.pending_items = dict(enumerate(self.pending_items))
        # Ensure completed_items is a MutableMapping
        if not isinstance(self.completed_items, MutableMapping):
            self.completed_items = {}

    @property
    def status(self) -> bool:
        """
        Indicates whether all pending items have been processed.

        Returns:
            bool: `True` if there are no more pending items, `False` otherwise.
        """
        return not self.pending_items

    def process_pending_items(self):
        """
        Processes the pending items once.

        This method iterates over each key-value pair in `pending_items`, applies the `processing_function`
        to determine its status, and then uses `handle_status_function` to decide whether to
        remove the item from `pending_items`. Items removed are added to `completed_items` with their results.
        """
        keys_to_remove = set()

        for k, v in list(self.pending_items.items()):
            # Apply the processing_function to get the item's status and result
            status, result = self.processing_function(v)

            # Decide whether to remove the item based on its status and result
            should_remove = self.handle_status_function(v, status, result)

            if should_remove:
                keys_to_remove.add(k)
                # Store the result associated with the item's key
                self.completed_items[k] = result

        # Remove items that are done processing from pending_items
        for k in keys_to_remove:
            del self.pending_items[k]

    def process_items(self):
        """
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
        """
        # Continue looping while there are pending items and the cycle limit hasn't been reached
        while not self.status and self.cycles < (self.max_cycles or float("inf")):
            # Record the start time of the cycle
            cycle_start_time = time.time()

            # Process the pending items once
            self.process_pending_items()

            # Increment the cycle counter
            self.cycles += 1

            # Calculate how long the processing took
            cycle_duration = time.time() - cycle_start_time

            # Determine how long to wait before the next cycle
            sleep_duration = self.wait_time_function(cycle_duration, locals())

            # Sleep if there are still pending items and the sleep duration is positive
            if not self.status and sleep_duration > 0:
                time.sleep(sleep_duration)

        return self.completed_items
