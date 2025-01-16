"""
Tools to work with shared chats (conversations).

For instance: Extract information from them.

The main object here is `ChatDacc` (Chat Data Accessor), which is a class that allows
you to access the data in a shared chat in a structured way.

>>> from oa.chats import ChatDacc
>>>
>>> url = 'https://chatgpt.com/share/6788d539-0f2c-8013-9535-889bf344d7d5'
>>> dacc = ChatDacc(url)
>>>
>>> basic_turns_data = dacc.basic_turns_data
>>> len(basic_turns_data)
4
>>> first_turn = basic_turns_data[0]
>>> isinstance(first_turn, dict)
True
>>> list(first_turn)
['id', 'role', 'content', 'time']
>>> print(first_turn['content'])
This conversation is meant to be used as an example, for testing, and/or for figuring out how to parse the html and json of a conversation.
<BLANKLINE>
As such, we'd like to keep it short.
<BLANKLINE>
Just say "Hello World!" back to me for now, and then in a second line write 10 random words.
>>>
>>> from pprint import pprint
>>> first_response = basic_turns_data[1]
>>> pprint(first_response)
{'content': 'Hello World!  \n'
            'apple, river, galaxy, chair, melody, shadow, cloud, puzzle, '
            'flame, whisper.',
 'id': '38ee4a3f-8487-4b35-9f92-ddee57c25d0a',
 'role': 'assistant',
 'time': 1737020652.436654}


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


from typing import Literal
from dol import path_get, paths_getter


paths_get_or_none = partial(
    paths_getter,
    get_value=dict.get,
    on_error=path_get.return_none_on_error,
)


def turn_is_not_visually_hidden(turn):
    visually_hidden = paths_get_or_none(
        {'visually_hidden': 'message.metadata.is_visually_hidden_from_conversation'},
        turn,
    )
    return (
        visually_hidden is not None
        and visually_hidden.get('visually_hidden') is not True
    )


class ChatDacc:
    metadata_path = (
        'state',
        'loaderData',
        'routes/share.$shareId.($action)',
        'serverResponse',
        'data',
    )
    turns_field = 'mapping'
    turns_path = metadata_path + ('mapping',)
    turns_df_paths = {
        'id': 'id',
        'role': 'message.author.role',
        'content': 'message.content.parts',
        'time': 'message.create_time',
    }
    _matadata_exclude_fields = {'mapping', 'linear_conversation'}

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
        d = path_get(self.full_json_dict, self.chat_data_path)
        return {k: v for k, v in d.items() if k not in self._matadata_exclude_fields}

    @cached_property
    def turns_data(self):
        return path_get(self.full_json_dict, self.turns_path)

    @cached_property
    def turns_data_keys(self):
        return list(self.turns_data.keys())

    def extract_turns(
        self, paths, *, filter_turns: Callable = turn_is_not_visually_hidden
    ):
        for id_, turn in self.turns_data.items():
            if filter_turns(turn):
                yield paths_get_or_none(paths, turn)

    @cached_property
    def basic_turns_data(self):
        # same as turns_df, but list of dicts
        def gen():
            for turn in self.extract_turns(self.turns_df_paths):
                if turn.get('role', None) is None or turn.get('content', None) is None:
                    continue
                if 'content' in turn:
                    turn['content'] = '\n'.join(turn['content'])
                yield turn

        return list(gen())

    @property
    def basic_turns_df(self):
        import pandas as pd  # pip install pandas

        t = pd.DataFrame(self.basic_turns_data)
        if 'time' in t:
            t['time'] = pd.to_datetime(t['time'], unit='s').dt.floor('s')
            # # take the time up to seconds only
            # t['time'] = t['time'].dt.floor('s')
        return t

    def extract_metadata(self, paths):
        extractor = self.make_extractor(paths, source='metadata')
        return extractor()


# --------------------------------------------------------------------------------------
# SSOT and other data

turns_data_ssot = {
    'id': {
        'description': 'The unique identifier for a message or turn in the conversation.',
        'example': 'adff303b-75cc-493c-b757-605adadb8e56',
    },
    'children': {
        'description': 'An array of identifiers for the child messages that are part of this turn.',
        'example': ['1473f2d9-ba09-4cd7-90c4-1452898676de'],
    },
    'message': {
        'description': 'An object containing details about the message sent during this turn.',
        'example': {
            'id': '1473f2d9-ba09-4cd7-90c4-1452898676de',
            'author': {'role': 'system', 'metadata': {}},
            'content': {'content_type': 'text', 'parts': ['']},
            'status': 'finished_successfully',
            'end_turn': True,
            'weight': 0,
            'metadata': {
                'is_visually_hidden_from_conversation': True,
                'shared_conversation_id': '6788d539-0f2c-8013-9535-889bf344d7d5',
            },
            'recipient': 'all',
        },
    },
    'parent': {
        'description': 'The identifier of the parent message in the conversation tree.',
        'example': 'adff303b-75cc-493c-b757-605adadb8e56',
    },
    'create_time': {
        'description': 'Timestamp indicating when the message was created, typically represented as a Unix timestamp.',
        'example': 1737020650.866734,
    },
    'author': {
        'description': 'An object containing details about the author of the message, including role and any additional metadata.',
        'example': {'role': 'system', 'metadata': {}},
    },
    'content_type': {
        'description': 'The type of content included in the message (e.g., text, model_editable_context).',
        'example': 'text',
    },
    'parts': {
        'description': 'An array containing the actual message content as individual parts.',
        'example': ['Original custom instructions no longer available'],
    },
    'status': {
        'description': 'The status of the message processing (e.g., finished_successfully, error).',
        'example': 'finished_successfully',
    },
    'end_turn': {
        'description': 'Indicates whether this message concludes the current turn in the conversation.',
        'example': True,
    },
    'weight': {
        'description': 'A numeric value representing the weight or importance of the message.',
        'example': 0,
    },
    'metadata': {
        'description': 'An object containing additional metadata about the message, such as visibility and contextual data.',
        'example': {
            'is_visually_hidden_from_conversation': True,
            'shared_conversation_id': '6788d539-0f2c-8013-9535-889bf344d7d5',
        },
    },
    'recipient': {
        'description': 'Indicates who the recipient of the message is (e.g., all, specific user).',
        'example': 'all',
    },
    'finish_details': {
        'description': 'An object detailing how the message processing finished, including any stop tokens that could affect subsequent content.',
        'example': {'type': 'stop', 'stop_tokens': [200002]},
    },
    'is_complete': {
        'description': 'Indicates whether the message processing is complete.',
        'example': True,
    },
    'citations': {
        'description': 'An array of citations (if any) included with the message.',
        'example': [],
    },
    'content_references': {
        'description': 'An array of content references associated with the message, can be empty.',
        'example': [],
    },
    'model_slug': {
        'description': 'Indicates the model used to generate the response, if applicable.',
        'example': 'gpt-4o',
    },
    'default_model_slug': {
        'description': 'The default model slug that was utilized for generating responses.',
        'example': 'gpt-4o',
    },
    'parent_id': {
        'description': 'Identifier of the parent message for hierarchical relationships in the conversation.',
        'example': '3b469b70-b069-4640-98af-5417491bb626',
    },
    'request_id': {
        'description': 'Unique identifier for the request made to the model during the interaction.',
        'example': '902d2b91cd44e15c-MRS',
    },
    'timestamp_': {
        'description': 'Timestamp type indicating whether it is relative or absolute, often used for synchronization purposes.',
        'example': 'absolute',
    },
    'shared_conversation_id': {
        'description': 'Identifier for the shared conversation session to track context.',
        'example': '6788d539-0f2c-8013-9535-889bf344d7d5',
    },
    'is_redacted': {
        'description': 'Indicates whether the message content has been redacted for privacy or security reasons.',
        'example': True,
    },
}

metadata_ssot = {
    'title': {
        'description': 'The title of the conversation.',
        'example': 'Test Chat 1',
    },
    'create_time': {
        'description': 'The timestamp (in seconds since epoch) when the conversation was created.',
        'example': 1737020729.060687,
    },
    'update_time': {
        'description': 'The timestamp (in seconds since epoch) when the conversation was last updated.',
        'example': 1737020733.031014,
    },
    'moderation_results': {
        'description': 'An array that holds results from any moderation applied to the conversation. It is empty if no moderation has been done.',
        'example': [],
    },
    'current_node': {
        'description': 'The unique identifier of the current node in the conversation flow.',
        'example': 'be4486db-894f-4e6f-bd0a-22d9d2facf69',
    },
    'conversation_id': {
        'description': 'A unique identifier for the entire conversation.',
        'example': '6788d539-0f2c-8013-9535-889bf344d7d5',
    },
    'is_archived': {
        'description': 'A boolean indicating whether the conversation is archived.',
        'example': False,
    },
    'safe_urls': {
        'description': 'An array of URLs deemed safe within the context of the conversation. It is empty if there are no safe URLs.',
        'example': [],
    },
    'default_model_slug': {
        'description': 'The identifier for the default model used in the conversation.',
        'example': 'gpt-4o',
    },
    'disabled_tool_ids': {
        'description': 'An array of tool identifiers that have been disabled for this conversation. It is empty if no tools are disabled.',
        'example': [],
    },
    'is_public': {
        'description': 'A boolean indicating whether the conversation is accessible to the public.',
        'example': True,
    },
    'linear_conversation': {
        'description': 'An array representing the sequential flow of the conversation, where each object contains an id and its children (sub-sequent nodes).',
        'example': [
            {
                'id': 'adff303b-75cc-493c-b757-605adadb8e56',
                'children': ['1473f2d9-ba09-4cd7-90c4-1452898676de'],
            }
        ],
    },
    'has_user_editable_context': {
        'description': 'A boolean indicating if the user can edit the context of the conversation.',
        'example': False,
    },
    'continue_conversation_url': {
        'description': 'A URL that allows users to continue the conversation from its current state.',
        'example': 'https://chatgpt.com/share/6788d539-0f2c-8013-9535-889bf344d7d5/continue',
    },
    'moderation_state': {
        'description': 'An object that encapsulates the moderation status of the conversation, detailing various moderation flags.',
        'example': {
            'has_been_moderated': False,
            'has_been_blocked': False,
            'has_been_accepted': False,
            'has_been_auto_blocked': False,
            'has_been_auto_moderated': False,
        },
    },
    'is_indexable': {
        'description': 'A boolean indicating if this conversation can be indexed by search engines or other indexing tools.',
        'example': False,
    },
    'is_better_metatags_enabled': {
        'description': 'A boolean indicating if better metatags are enabled for this conversation.',
        'example': True,
    },
}


ChatDacc.turns_data_ssot = turns_data_ssot
ChatDacc.metadata_ssot = metadata_ssot

# --------------------------------------------------------------------------------------
# Tools for parser maintenance

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
"""
)
