# oa.chats

Tools to work with shared chats (conversations).

For instance: Extract information from them.

The main object here is `ChatDacc` (Chat Data Accessor), which is a class that allows
you to access the data in a shared chat in a structured way.

# >>> from oa.chats import ChatDacc
# >>>
# >>> url = ‘[https://chatgpt.com/share/6788d539-0f2c-8013-9535-889bf344d7d5](https://chatgpt.com/share/6788d539-0f2c-8013-9535-889bf344d7d5)’
# >>> dacc = ChatDacc(url)
# >>>
# >>> basic_turns_data = dacc.basic_turns_data
# >>> len(basic_turns_data)
# 4
# >>> first_turn = basic_turns_data[0]
# >>> isinstance(first_turn, dict)
# True
# >>> list(first_turn)
# [‘id’, ‘role’, ‘content’, ‘message_id’]
# >>> print(first_turn[‘content’])  # doctest: +NORMALIZE_WHITESPACE
# This conversation is meant to be used as an example, for testing, and/or for figuring out how to parse the html and json of a conversation.
# <BLANKLINE>
# As such, we’d like to keep it short.
# <BLANKLINE>
# Just say “Hello World!” back to me for now, and then in a second line write 10 random words.

### Functions

| [`find_all_matching_paths_in_list_values`](#oa.chats.find_all_matching_paths_in_list_values)(...)       | Find all paths in a nested dictionary where target_value evaluates to True for elements in a list contained within the value.                                                |
|----------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`find_url_keys`](#oa.chats.find_url_keys)(data[, current_path])               | Recursively finds all paths in a JSON-like object (nested dicts/lists) where URL strings are present.                                                                        |
| [`get_rendered_html`](#oa.chats.get_rendered_html)(url, \*[, headless, ...])       | Return the fully rendered HTML for `url` using Playwright (sync or async).                                                                                                   |
| [`get_rendered_html_async`](#oa.chats.get_rendered_html_async)(url, \*[, headless, ...]) | Asynchronous version: return fully rendered HTML using Playwright async API.                                                                                                 |
| [`parse_chat_html`](#oa.chats.parse_chat_html)(html)                             | Parses conversation data from a shared chat HTML file into a structured list of messages.                                                                                    |
| [`reduce_chat_html`](#oa.chats.reduce_chat_html)(html)                            | Strips away all irrelevant metadata, headers, footers, and scripts from the raw HTML, keeping only the essential conversation content enclosed in the <div id="thread"> tag. |
| [`remove_utm_source`](#oa.chats.remove_utm_source)(text)                           | Removes the "?utm_source=chatgpt.com" suffix from all URLs in the given text.                                                                                                |
| [`truncate_dict_values`](#oa.chats.truncate_dict_values)(d[, max_length])             | Truncate string values in a dict to a maximum length for display purposes.                                                                                                   |

### Classes

| [`ChatDacc`](#oa.chats.ChatDacc)(src)   | Chat Data Accessor - manages and extracts data from shared ChatGPT conversations.   |
|------------------------------------------------------------------|-------------------------------------------------------------------------------------|

### *class* oa.chats.ChatDacc(src)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Chat Data Accessor - manages and extracts data from shared ChatGPT conversations.

#### *property* basic_turns_data

Return the parsed conversation as-is (list of dicts with id, role, content, message_id).

#### *property* basic_turns_df

Convert the conversation to a pandas DataFrame.

#### copy_turns_json()

Copy the conversation data to clipboard as JSON.

#### *property* paths_containing_urls

Get unique paths containing URLs with array indices replaced by wildcards.

#### url_data(, remove_chatgpt_utm=True)

Extract all URLs from the conversation.

* **Parameters:**
  **remove_chatgpt_utm** – If True, remove utm_source=chatgpt.com from URLs
* **Returns:**
  List of URLs found in the conversation

#### *property* url_paths

Find all paths containing URLs in the conversation data.

### oa.chats.find_all_matching_paths_in_list_values(nested_dict, target_value)

Find all paths in a nested dictionary where target_value evaluates to True
for elements in a list contained within the value.

* **Parameters:**
  * **nested_dict** – The dictionary to search.
  * **target_value** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable) | [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – A function that takes a value and returns a boolean.
    If a string, will be converted to a regex search.
* **Returns:**
  A generator yielding paths that match the condition.

```pycon
>>> nested_dict = {
...     "a": {"b": {"c": [1, 2, 3], "d": [4, 5]}},
...     "e": {"f": [6, 7], "g": {"h": [8, 9]}},
... }
>>> target_value_fn = lambda x: x % 2 == 0  # Find even numbers
>>> list(find_all_matching_paths_in_list_values(nested_dict, target_value_fn))
[('a', 'b', 'c'), ('a', 'b', 'd'), ('e', 'f'), ('e', 'g', 'h')]
```

### oa.chats.find_url_keys(data, current_path='')

Recursively finds all paths in a JSON-like object (nested dicts/lists) where URL strings are present.

This is a generator function that yields each path as a dot-separated string. Supports paths through
both dictionaries and lists.

* **Parameters:**
  * **data** ([*dict*](https://docs.python.org/3/builtins/stdtypes.html#dict) *or* [*list*](https://docs.python.org/3/builtins/stdtypes.html#list)) – The JSON-like object to search.
  * **current_path** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)) – The current path being traversed, used internally during recursion.
* **Yields:**
  *str* – Dot-separated path to a value containing a URL.

### Example

```pycon
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
```

One thing you’ll probably want to do sometimes is transform, filter, and
aggregate these paths. Here’s an example of how you might get a list of
unique paths, with all array indices replaced with a wildcard, so thay
don’t appear as separate paths:

```pycon
>>> from functools import partial
>>> import re
>>> paths = find_url_keys(example_data)
>>> transform = partial(re.sub, '\[\d+\]', '[*]')
>>> unique_paths = set(map(transform, paths))
>>> sorted(unique_paths)
['key1.nested[*].url', 'key2']
```

### oa.chats.get_rendered_html(url, , headless=True, timeout=30000, use_async=False)

Return the fully rendered HTML for `url` using Playwright (sync or async).

By default this uses the synchronous Playwright API. If `use_async=True` the
async variant will be executed via asyncio.run.

#### NOTE
This function only imports Playwright when called. Install with:

pip install playwright
playwright install chromium

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### *async* oa.chats.get_rendered_html_async(url, , headless=True, timeout=30000)

Asynchronous version: return fully rendered HTML using Playwright async API.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### oa.chats.parse_chat_html(html)

Parses conversation data from a shared chat HTML file into a structured list of messages.

It extracts the primary conversation data along with essential metadata:
‘id’ (from the turn ID), ‘role’, ‘content’, and ‘message_id’. The ‘time’
field is excluded as it cannot be reliably extracted from the visible HTML.

* **Parameters:**
  **html** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – The full HTML content of the conversation file.
* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]]
* **Returns:**
  A list of dictionaries, where each dictionary represents a message and
  contains ‘id’, ‘role’, ‘content’, and ‘message_id’ keys.

### oa.chats.reduce_chat_html(html)

Strips away all irrelevant metadata, headers, footers, and scripts from
the raw HTML, keeping only the essential conversation content enclosed in
the <div id=”thread”> tag.

This dramatically reduces the size of the HTML fed into the main parser
while preserving all necessary information for `parse_chat_html`.

* **Parameters:**
  **html** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – The full HTML content of the conversation file.
* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)
* **Returns:**
  A stripped-down HTML string containing only the conversation thread.

### oa.chats.remove_utm_source(text)

Removes the “?utm_source=chatgpt.com” suffix from all URLs in the given text.

* **Parameters:**
  **text** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)) – The input text containing URLs.
* **Returns:**
  The text with the specified query parameter removed from all URLs.
* **Return type:**
  [*str*](https://docs.python.org/3/builtins/stdtypes.html#str)

### Example

```pycon
>>> input_text = (
...     "not_a_url "  # not even a url (won't be touched)
...     "http://abc?utm_source=chatgpt.com_not_at_the_end "  # target not at the end
...     "https://abc?utm_source=chatgpt.com "  # with ?
...     "https://abc&utm_source=chatgpt.com "  # with &
...     "http://abc?utm_source=chatgpt.com"  # with http instead of https
... )
>>> remove_utm_source(input_text)
'not_a_url http://abc_not_at_the_end https://abc https://abc http://abc'
```

### oa.chats.truncate_dict_values(d, max_length=100)

Truncate string values in a dict to a maximum length for display purposes.
