r"""
Tools to work with shared chats (conversations).

For instance: Extract information from them.

The main object here is `ChatDacc` (Chat Data Accessor), which is a class that allows
you to access the data in a shared chat in a structured way.

## TODO: Temporarily commenting out tests, since html structure of chats has changed.
# >>> from oa.chats import ChatDacc
# >>>
# >>> url = 'https://chatgpt.com/share/6788d539-0f2c-8013-9535-889bf344d7d5'
# >>> dacc = ChatDacc(url)
# >>>
# >>> basic_turns_data = dacc.basic_turns_data
# >>> len(basic_turns_data)
# 4
# >>> first_turn = basic_turns_data[0]
# >>> isinstance(first_turn, dict)
# True
# >>> list(first_turn)
# ['id', 'role', 'content', 'time']
# >>> print(first_turn['content'])  # doctest: +NORMALIZE_WHITESPACE
# This conversation is meant to be used as an example, for testing, and/or for figuring out how to parse the html and json of a conversation.
# <BLANKLINE>
# As such, we'd like to keep it short.
# <BLANKLINE>
# Just say "Hello World!" back to me for now, and then in a second line write 10 random words.
# >>>
# >>> from pprint import pprint
# >>> first_response = basic_turns_data[1]
# >>> pprint(first_response)  # doctest: +NORMALIZE_WHITESPACE
# {'content': 'Hello World!  \n'
#             'apple, river, galaxy, chair, melody, shadow, cloud, puzzle, '
#             'flame, whisper.',
#     'id': '38ee4a3f-8487-4b35-9f92-ddee57c25d0a',
#     'role': 'assistant',
#     'time': 1737020652.436654}


In the example above, we just extracted the basic turn data.
But there's a lot more in the conversation data that you can access.

See this notebook for plenty more:
https://github.com/thorwhalen/oa/blob/main/misc/oa%20-%20chats%20acquisition%20and%20parsing.ipynb

And for you contributors out there, there are also tools to help you maintain the parser.

>>> url = 'https://chatgpt.com/share/6788d539-0f2c-8013-9535-889bf344d7d5'
>>> string_contained_in_conversation = 'apple, river, galaxy, chair, melody, shadow, cloud, puzzle, flame, whisper'
>>> chat_html = url_to_html(url)  # doctest: +SKIP
>>> chat_json_dict = parse_shared_chatGPT_chat(chat_html, string_contained_in_conversation)  # doctest: +SKIP

"""

import re
import json
from typing import Callable, Optional, Union, Iterable
from functools import partial, cached_property

import requests
from dol import path_filter


def url_to_html(url: str) -> str:
    """Get the html for a url"""
    response = requests.get(url)
    if response.status_code != 200:
        raise RuntimeError(f"Failed to fetch {url}: {response.status_code}")
    return response.text


def raise_run_time_error(msg: str):
    raise RuntimeError(msg)


def extract_json_dict(
    string: str,
    object_filter: Callable,
    *,
    decoder: json.JSONDecoder = json.JSONDecoder(),
    not_found_callback: Optional[
        Callable
    ] = lambda string, object_filter: raise_run_time_error(
        "Object not found in string"
    ),
) -> dict:
    """
    Searches `string` from the beginning, attempting to decode consecutive
    JSON objects using `decoder.raw_decode`. When an object satisfies
    `object_filter`, returns it as a Python dictionary.

    :param string: The string to search.
    :param object_filter: A function that takes a Python object and returns a boolean.
        The first object that satisfies this condition will be returned.
    :param decoder: A JSONDecoder instance to use for decoding.
    :param not_found_callback: A function to call if no object is found.

    """
    pos = 0
    while pos < len(string):
        try:
            obj, pos = decoder.raw_decode(string, pos)
            if object_filter(obj):
                return obj
        except json.JSONDecodeError:
            pos += 1
    return not_found_callback(string, object_filter)


# NOTE: This is one of the fragile parts of the parsing. Who knows if this is consistent now, or will be in the future!
DFLT_VARIABLE_NAME = "window.__remixContext"

DFLT_JSON_PATTERN = r"(\{.*?\});"

DFLT_TARGET_DATA_PATTERN_STR = r"data.*mapping.*message"


def ensure_filter_func(target: Union[str, Callable]) -> Callable:
    if isinstance(target, str):
        target_pattern = re.compile(target, re.DOTALL)
        return lambda x: isinstance(x, str) and target_pattern.search(x)
    assert callable(target), f"target must be a string or a callable: {target=}"
    return target


def parse_shared_chatGPT_chat(
    html: str,
    is_target_string: Union[str, Callable] = DFLT_TARGET_DATA_PATTERN_STR,
    *,
    variable_name: str = DFLT_VARIABLE_NAME,
    json_pattern: str = DFLT_JSON_PATTERN,
) -> dict:
    """
    Locates the big JSON structure assigned to window.__remixContext,
    checks that the conversation portion includes `string_contained_in_conversation`,
    and returns the JSON as a dict.

    :param html: The HTML content of the chat page.
    :param string_contained_in_conversation: A string that should be present in the conversation data.
    :param context_pattern: A regex pattern to match the window.__remixContext assignment.

    """
    is_target_string = ensure_filter_func(is_target_string)
    pattern = re.escape(variable_name) + r"\s*=\s*" + f"({json_pattern})"

    # Regex pattern to capture the object assigned to window.__remixContext = {...};
    assignment_match = re.search(pattern, html, flags=re.DOTALL)
    if not assignment_match:
        raise RuntimeError(
            f"Could not locate the JSON assigned, given the pattern\n{pattern}\n"
            "Consider changing the variable_name and/or json_pattern."
        )

    # Extract the raw JSON text (removing the trailing semicolon if needed)
    raw_json_text = assignment_match.group(1).strip()

    # We'll define a filter that checks if the conversation data includes the target string.
    # Because the JSON is quite large, we can simply check if the substring is present in the raw text:
    # but if we want to be more precise, we can parse first and only return if there's conversation data.
    def conversation_filter(obj):
        return is_target_string(json.dumps(obj, ensure_ascii=False))

    # Now parse the JSON, returning only if the filter passes
    extracted_dict = extract_json_dict(
        raw_json_text,
        object_filter=conversation_filter,
    )

    return extracted_dict


def find_all_matching_paths_in_list_values(
    nested_dict, target_value: Union[Callable, str]
):
    """
    Find all paths in a nested dictionary where target_value evaluates to True
    for elements in a list contained within the value.

    :param nested_dict: The dictionary to search.
    :param target_value: A function that takes a value and returns a boolean.
        If a string, will be converted to a regex search.
    :return: A generator yielding paths that match the condition.

    >>> nested_dict = {
    ...     "a": {"b": {"c": [1, 2, 3], "d": [4, 5]}},
    ...     "e": {"f": [6, 7], "g": {"h": [8, 9]}},
    ... }
    >>> target_value_fn = lambda x: x % 2 == 0  # Find even numbers
    >>> list(find_all_matching_paths_in_list_values(nested_dict, target_value_fn))
    [('a', 'b', 'c'), ('a', 'b', 'd'), ('e', 'f'), ('e', 'g', 'h')]

    """
    # if isinstance(target_value, str):
    #     target_value_string = target_value
    #     target_value = lambda x: isinstance(x, str) and re.compile(
    #         target_value_string
    #     ).search(x)

    target_value = ensure_filter_func(target_value)

    return path_filter(
        pkv_filt=lambda p, k, v: (
            isinstance(v, list) and any(target_value(item) for item in v)
        ),
        d=nested_dict,
    )


from typing import Literal
from dol import path_get, paths_getter


paths_get_or_none = partial(
    paths_getter,
    get_value=dict.get,
    on_error=path_get.return_none_on_error,
)


def turn_is_not_visually_hidden(turn):
    visually_hidden = paths_get_or_none(
        {"visually_hidden": "message.metadata.is_visually_hidden_from_conversation"},
        turn,
    )
    return (
        visually_hidden is not None
        and visually_hidden.get("visually_hidden") is not True
    )


def remove_utm_source(text):
    """
    Removes the "?utm_source=chatgpt.com" suffix from all URLs in the given text.

    Args:
        text (str): The input text containing URLs.

    Returns:
        str: The text with the specified query parameter removed from all URLs.

    Example:

        >>> input_text = (
        ...     "not_a_url "  # not even a url (won't be touched)
        ...     "http://abc?utm_source=chatgpt.com_not_at_the_end "  # target not at the end
        ...     "https://abc?utm_source=chatgpt.com "  # with ?
        ...     "https://abc&utm_source=chatgpt.com "  # with &
        ...     "http://abc?utm_source=chatgpt.com"  # with http instead of https
        ... )
        >>> remove_utm_source(input_text)
        'not_a_url http://abc_not_at_the_end https://abc https://abc http://abc'

    """
    pattern = r"(https?://[^\s]+)[&\?]utm_source=chatgpt\.com"
    cleaned_text = re.sub(pattern, r"\1", text)
    return cleaned_text


# --------------------------------------------------------------------------------------
# A manager class that operates on a shared chat


class ChatDacc:
    metadata_path = (
        "state",
        "loaderData",
        "routes/share.$shareId.($action)",
        "serverResponse",
        "data",
    )
    turns_field = "mapping"
    turns_path = metadata_path + ("mapping",)
    turns_df_paths = {
        "id": "id",
        "role": "message.author.role",
        "content": "message.content.parts",
        "time": "message.create_time",
    }
    _matadata_exclude_fields = {"mapping", "linear_conversation"}

    def __init__(
        self,
        src,
    ):
        self.src = src
        if isinstance(src, str):
            if src.startswith("http"):
                url = src
                html = url_to_html(url)
                self.full_json_dict = parse_shared_chatGPT_chat(html)
            else:
                raise ValueError(f"Invalid (string) src. Must be a url: {src}")
        elif isinstance(src, dict):
            self.full_json_dict = src
        else:
            raise ValueError(f"Invalid src: {src}")

    @cached_property
    def metadata(self):
        d = path_get(self.full_json_dict, self.metadata_path)
        return {k: v for k, v in d.items() if k not in self._matadata_exclude_fields}

    @cached_property
    def turns_data(self):
        return path_get(self.full_json_dict, self.turns_path)

    @cached_property
    def turns_data_keys(self):
        return list(self.turns_data.keys())

    def extract_turns(
        self,
        paths: Optional[Iterable[str]] = None,
        *,
        filter_turns: Callable = turn_is_not_visually_hidden,
        egress: Callable = lambda turns: turns,
    ):
        def gen():
            for id_, turn in self.turns_data.items():
                if filter_turns(turn):
                    if paths:
                        yield paths_get_or_none(paths, turn)
                    else:
                        yield turn

        return egress(gen())

    @cached_property
    def basic_turns_data(self):
        # same as turns_df, but list of dicts
        def gen():
            for turn in self.extract_turns(self.turns_df_paths):
                if turn.get("role", None) is None or turn.get("content", None) is None:
                    continue
                if "content" in turn:
                    turn["content"] = "\n".join(map(str, turn["content"]))
                yield turn

        return list(gen())

    @property
    def basic_turns_df(self):
        import pandas as pd  # pip install pandas

        t = pd.DataFrame(self.basic_turns_data)
        if "time" in t:
            t["time"] = pd.to_datetime(t["time"], unit="s").dt.floor("s")
            # # take the time up to seconds only
            # t['time'] = t['time'].dt.floor('s')
        return t

    def copy_turns_json(
        self,
        paths: Optional[Iterable[str]] = None,
        *,
        filter_turns: Callable = turn_is_not_visually_hidden,
    ):
        from pyperclip import copy  # pip install pyperclip

        t = list(self.extract_turns(paths, filter_turns=filter_turns))
        return copy(json.dumps(t, indent=4))

    @cached_property
    def url_paths(self):
        return list(find_url_keys(self.turns_data))

    @cached_property
    def paths_containing_urls(self):
        replace_array_index_with_star = partial(re.sub, "\[\d+\]", "[*]")
        ignore_first_part_of_path = lambda x: ".".join(x.split(".")[1:])
        transform = Pipe(replace_array_index_with_star, ignore_first_part_of_path)
        return sorted(set(map(transform, self.url_paths)))

    def url_data(
        self, prior_levels_to_include=0, *, only_values=True, remove_chatgpt_utm=True
    ):
        """
        Get the urls, and optionally, the data associated with them.

        Note: More levels you ask for, more you might get several urls under a same path
        so you'll get less output items (since this method works on paths).
        For example, if both 'a.b' and 'a.c' are urls, then asking for prior_levels_to_include=1
        will give the data under 'a' only (with two urls).

        Args:
            prior_levels_to_include (int): If 0, returns the urls. If >0, returns the data associated with the urls, but only up to the specified level.
            only_values (bool): If True, only returns the values associated with the urls. If False, returns the full data associated with the urls.
        """
        if prior_levels_to_include == 0:
            _url_paths = self.url_paths
        else:
            _url_paths = map(lambda x: x.split("."), self.url_paths)
            _url_paths = list(
                map(lambda x: ".".join(x[:-prior_levels_to_include]), _url_paths)
            )

        d = paths_getter(
            paths=_url_paths,
            obj=self.turns_data,
            sep=path_get.keys_and_indices_path,
        )

        if remove_chatgpt_utm:
            for k, v in d.items():
                if isinstance(v, str):
                    d[k] = remove_utm_source(v)
        if only_values:
            d = list(d.values())
        return d


# --------------------------------------------------------------------------------------
# SSOT and other data
from oa._params import turns_data_ssot, metadata_ssot

ChatDacc.turns_data_ssot = turns_data_ssot
ChatDacc.metadata_ssot = metadata_ssot

# --------------------------------------------------------------------------------------
# Tools for parser maintenance
# See https://github.com/thorwhalen/oa/blob/main/misc/oa%20-%20chats%20acquisition%20and%20parsing.ipynb
# for more info (namely how to use mk_json_field_documentation to make descriptions)
"""
Notes:

metadata['linear_conversation'] == list(metadata['mapping'].values())

"""
from oa.tools import prompt_function

mk_json_field_documentation = prompt_function(
    """
You are a technical writer specialized in documenting JSON fields. 
Below is a JSON object. I'd like you to document each field in a markdown table.
The table should contain the name, description, and example value of each field.
                                              
The context is:
{context: just a general context}
                                              
Here's an example json object:

{example_json}
""",
    egress=lambda x: x["result"],
)


def mk_merged_turn_data():
    """
    Make a merged turn data dict from a selection of shared chats,
    to feed to mk_json_field_documentation
    """
    from lkj import merge_dicts

    # A selection of shared chats with a variety of features
    urls = dict(
        simple_url="https://chatgpt.com/share/6788d539-0f2c-8013-9535-889bf344d7d5",
        url_with_searches="https://chatgpt.com/share/67877101-6708-8013-ba9e-2e770186db58",
        url_with_image_gen="https://chatgpt.com/share/678a1339-d14c-8013-bfcb-288d367a9079",
    )

    # Note, the turns_data is a dict whose keys are "turn ids" and values are the metadata for that turn, so we want to merge those
    turn_datas = {
        k: merge_dicts(*ChatDacc(url).turns_data.values()) for k, url in urls.items()
    }
    merge_turn_data = merge_dicts(*turn_datas.values())
    return truncate_dict_values(merge_turn_data)


# In order to write a "link extractor" I want to see where links (in searches,
# in citations, etc.) are found in the JSON
# The following code helps
# Note: Could have used dol.paths.path_filter here
# TODO: Refactor this to use dol.paths.path_filter
def find_url_keys(data, current_path=""):
    """
    Recursively finds all paths in a JSON-like object (nested dicts/lists) where URL strings are present.

    This is a generator function that yields each path as a dot-separated string. Supports paths through
    both dictionaries and lists.

    Args:
        data (dict or list): The JSON-like object to search.
        current_path (str): The current path being traversed, used internally during recursion.

    Yields:
        str: Dot-separated path to a value containing a URL.

    Example:
        >>> example_data = {
        ...     "key1": {
        ...         "nested": [
        ...             {"url": "http://example.com"},
        ...             {"url": "https://example.org"}
        ...         ]
        ...     },
        ...     "key2": "http://another-example.com"
        ... }
        >>> list(find_url_keys(example_data))
        ['key1.nested[0].url', 'key1.nested[1].url', 'key2']

        One thing you'll probably want to do sometimes is transform, filter, and
        aggregate these paths. Here's an example of how you might get a list of
        unique paths, with all array indices replaced with a wildcard, so thay
        don't appear as separate paths:

        >>> from functools import partial
        >>> import re
        >>> paths = find_url_keys(example_data)
        >>> transform = partial(re.sub, '\[\d+\]', '[*]')
        >>> unique_paths = set(map(transform, paths))
        >>> sorted(unique_paths)
        ['key1.nested[*].url', 'key2']
    """

    def _is_url(x):
        return isinstance(x, str) and ("http://" in x or "https://" in x)

    if isinstance(data, dict):
        for key, value in data.items():
            new_path = f"{current_path}.{key}" if current_path else key
            if isinstance(value, (dict, list)):
                yield from find_url_keys(value, new_path)
            elif _is_url(value):
                yield new_path
    elif isinstance(data, list):
        for index, item in enumerate(data):
            new_path = f"{current_path}[{index}]"
            yield from find_url_keys(item, new_path)
