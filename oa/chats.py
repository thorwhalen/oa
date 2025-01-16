"""
Tools to work with shared chats (conversations).

For instance: Extract information from them.


>>> url = 'https://chatgpt.com/share/6788d539-0f2c-8013-9535-889bf344d7d5'
>>> string_contained_in_conversation = 'apple, river, galaxy, chair, melody, shadow, cloud, puzzle, flame, whisper'
>>> chat_html = url_to_html(url)
>>> chat_json_dict = parse_shared_chatGPT_chat(chat_html, string_contained_in_conversation)

"""

import re
import json
from typing import Callable, Optional, Union
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
DFLT_VARIABLE_NAME = 'window.__remixContext'

DFLT_JSON_PATTERN = r"(\{.*?\});"

DFLT_TARGET_DATA_PATTERN_STR = r'data.*mapping.*message'


def ensure_filter_func(target: Union[str, Callable]) -> Callable:
    if isinstance(target, str):
        target_pattern = re.compile(target, re.DOTALL)
        return lambda x: isinstance(x, str) and target_pattern.search(x)
    assert callable(target), "target must be a string or a callable"


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


class ChatDacc:
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
    def all_matching_paths(self):
        return find_all_matching_paths_in_list_values(
            self.full_json_dict, DFLT_TARGET_DATA_PATTERN_STR
        )

# --------------------------------------------------------------------------------------
# Tools for parser maintenance

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
"""
)
