r"""
Tools to work with shared chats (conversations).

For instance: Extract information from them.

The main object here is `ChatDacc` (Chat Data Accessor), which is a class that allows
you to access the data in a shared chat in a structured way.

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
# ['id', 'role', 'content', 'message_id']
# >>> print(first_turn['content'])  # doctest: +NORMALIZE_WHITESPACE
# This conversation is meant to be used as an example, for testing, and/or for figuring out how to parse the html and json of a conversation.
# <BLANKLINE>
# As such, we'd like to keep it short.
# <BLANKLINE>
# Just say "Hello World!" back to me for now, and then in a second line write 10 random words.
"""

import re
import json
import asyncio
from typing import Callable, Optional, Union, Iterable, List, Dict, Any
from functools import partial, cached_property
from bs4 import BeautifulSoup
from dol import path_filter, Pipe, path_get, paths_getter


# --------------------------------------------------------------------------------------
# HTML rendering and parsing functions from chats3.py


def get_rendered_html(
    url: str, *, headless: bool = True, timeout: int = 30000, use_async: bool = False
) -> str:
    """Return the fully rendered HTML for `url` using Playwright (sync or async).

    By default this uses the synchronous Playwright API. If `use_async=True` the
    async variant will be executed via asyncio.run.

    Note: This function only imports Playwright when called. Install with:
        pip install playwright
        playwright install chromium

    """
    # If user explicitly requested async, or if there's a running asyncio loop
    # (e.g. in Jupyter), prefer the async implementation.
    try:
        import asyncio

        loop_running = asyncio.get_running_loop().is_running()
    except RuntimeError:
        loop_running = False
    except Exception:
        loop_running = False

    if use_async or loop_running:
        # In notebook-style environments there's already an event loop; use
        # the async Playwright API. If there's a running loop, we need to use
        # nest_asyncio or run in a new task — here we'll try nest_asyncio first
        # and then fall back to running the async helper in a new thread.
        try:
            import nest_asyncio

            nest_asyncio.apply()
            return asyncio.run(
                get_rendered_html_async(url, headless=headless, timeout=timeout)
            )
        except Exception:
            # If nest_asyncio isn't available or something else fails, run the
            # async helper in a separate thread to avoid running inside the
            # already-running loop.
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    get_rendered_html_async(url, headless=headless, timeout=timeout),
                )
                return future.result()

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        raise ImportError(
            "playwright is required for rendered HTML extraction. "
            "Install with: pip install playwright && playwright install chromium"
        )

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        page = browser.new_page()
        try:
            page.goto(url, wait_until="networkidle", timeout=timeout)
            # return fully rendered HTML
            return page.content()
        finally:
            browser.close()


async def get_rendered_html_async(
    url: str, *, headless: bool = True, timeout: int = 30000
) -> str:
    """Asynchronous version: return fully rendered HTML using Playwright async API."""
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        raise ImportError(
            "playwright is required for rendered HTML extraction. "
            "Install with: pip install playwright && playwright install chromium"
        )

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless)
        page = await browser.new_page()
        try:
            await page.goto(url, wait_until="networkidle", timeout=timeout)
            return await page.content()
        finally:
            await browser.close()


def reduce_chat_html(html: str) -> str:
    """
    Strips away all irrelevant metadata, headers, footers, and scripts from
    the raw HTML, keeping only the essential conversation content enclosed in
    the <div id="thread"> tag.

    This dramatically reduces the size of the HTML fed into the main parser
    while preserving all necessary information for `parse_chat_html`.

    Args:
        html: The full HTML content of the conversation file.

    Returns:
        A stripped-down HTML string containing only the conversation thread.
    """
    # 1. Use BeautifulSoup to quickly find the root of the conversation thread
    soup = BeautifulSoup(html, "html.parser")

    # 2. The entire conversation content lives inside the element with id="thread".
    thread_div = soup.find("div", id="thread")

    if thread_div:
        # Get the inner HTML content of the thread div
        thread_content = str(thread_div)

        # 3. Wrap it in a minimal valid HTML structure for the parser
        # The surrounding tags are needed for BeautifulSoup to treat it as a proper document
        reduced_html = f"<html><body>{thread_content}</body></html>"
        return reduced_html
    else:
        # If the thread element is not found, return the original HTML
        return html


def parse_chat_html(html: str) -> List[Dict[str, Any]]:
    """
    Parses conversation data from a shared chat HTML file into a structured list of messages.

    It extracts the primary conversation data along with essential metadata:
    'id' (from the turn ID), 'role', 'content', and 'message_id'. The 'time'
    field is excluded as it cannot be reliably extracted from the visible HTML.

    Args:
        html: The full HTML content of the conversation file.

    Returns:
        A list of dictionaries, where each dictionary represents a message and
        contains 'id', 'role', 'content', and 'message_id' keys.
    """
    soup = BeautifulSoup(html, "html.parser")
    conversation = []

    # Find all conversation turns (user or assistant messages)
    turns = soup.find_all("article", {"data-turn": ["user", "assistant"]})

    for turn in turns:
        role = turn["data-turn"]
        content_parts = []

        # Extract turn metadata, using 'id' as requested for compatibility
        turn_id = turn.get("data-turn-id")

        # Find the inner div that contains the message ID
        message_div = turn.find("div", {"data-message-author-role": role})
        message_id = message_div.get("data-message-id") if message_div else None

        # --- 1. Handle User Messages ---
        if role == "user":
            text_container = turn.find("div", class_="whitespace-pre-wrap")
            if text_container:
                content = text_container.get_text("\n").strip()
                if content:
                    content_parts.append({"type": "text", "text": content})

        # --- 2. Handle Assistant Messages ---
        elif role == "assistant":
            markdown_div = turn.find(
                "div", class_=lambda c: c and "markdown" in c and "prose" in c
            )

            if markdown_div:
                for element in markdown_div.children:
                    # Handle Paragraphs/Text (<p> tags)
                    if element.name == "p":
                        text_content = element.get_text("\n").strip()
                        if text_content:
                            cleaned_text = text_content.replace("\n\n", "\n").strip()
                            if cleaned_text:
                                content_parts.append(
                                    {"type": "text", "text": cleaned_text}
                                )

                    # Handle Code Blocks (<pre> tags for fenced code blocks)
                    elif element.name == "pre":
                        lang_div = element.find(
                            "div",
                            class_=lambda c: c
                            and "justify-between" in c
                            and "h-9" in c,
                        )
                        language = "plaintext"
                        if lang_div:
                            language_text = lang_div.get_text(strip=True).lower()
                            if language_text and language_text != "copy code":
                                language = language_text

                        code_tag = element.find("code")
                        code_content = (
                            code_tag.get_text(strip=False).strip("\n")
                            if code_tag
                            else ""
                        )

                        if code_content:
                            content_parts.append(
                                {
                                    "type": "code",
                                    "language": language,
                                    "code": code_content,
                                }
                            )

        # --- Combine and Format Output ---
        if content_parts:
            final_content = ""
            for part in content_parts:
                if part["type"] == "text":
                    final_content += part["text"] + "\n\n"
                elif part["type"] == "code":
                    final_content += f"```{part['language']}\n{part['code']}\n```\n\n"

            final_content = final_content.strip()

            if final_content:
                message = {
                    "id": turn_id,
                    "role": role,
                    "content": final_content,
                    "message_id": message_id,
                }
                conversation.append(message)

    return conversation


# --------------------------------------------------------------------------------------
# Side extras - utility functions kept from original


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
    target_value = _ensure_filter_func(target_value)

    return path_filter(
        pkv_filt=lambda p, k, v: (
            isinstance(v, list) and any(target_value(item) for item in v)
        ),
        d=nested_dict,
    )


def _ensure_filter_func(target: Union[str, Callable]) -> Callable:
    """Convert a string to a regex filter function, or pass through callable."""
    if isinstance(target, str):
        target_pattern = re.compile(target, re.DOTALL)
        return lambda x: isinstance(x, str) and target_pattern.search(x)
    assert callable(target), f"target must be a string or a callable: {target=}"
    return target


paths_get_or_none = partial(
    paths_getter,
    get_value=dict.get,
    on_error=path_get.return_none_on_error,
)


# --------------------------------------------------------------------------------------
# A manager class that operates on a shared chat


class ChatDacc:
    """Chat Data Accessor - manages and extracts data from shared ChatGPT conversations."""

    def __init__(self, src):
        """
        Initialize ChatDacc with a source (URL or parsed conversation data).

        Args:
            src: Either a URL string or a list of conversation dictionaries
        """
        self.src = src
        if isinstance(src, str):
            if src.startswith("http"):
                self.url = src
                html = get_rendered_html(src)
                self.parsed_conversation = parse_chat_html(html)
            else:
                raise ValueError(f"Invalid (string) src. Must be a url: {src}")
        elif isinstance(src, list):
            # Assume it's already parsed conversation data
            self.parsed_conversation = src
        else:
            raise ValueError(f"Invalid src: {src}")

    @cached_property
    def basic_turns_data(self):
        """Return the parsed conversation as-is (list of dicts with id, role, content, message_id)."""
        return self.parsed_conversation

    @property
    def basic_turns_df(self):
        """Convert the conversation to a pandas DataFrame."""
        import pandas as pd  # pip install pandas

        return pd.DataFrame(self.basic_turns_data)

    def copy_turns_json(self):
        """Copy the conversation data to clipboard as JSON."""
        from pyperclip import copy  # pip install pyperclip

        return copy(json.dumps(self.parsed_conversation, indent=4))

    @cached_property
    def url_paths(self):
        """Find all paths containing URLs in the conversation data."""
        # Convert list to dict for compatibility with find_url_keys
        conversation_dict = {
            str(i): msg for i, msg in enumerate(self.parsed_conversation)
        }
        return list(find_url_keys(conversation_dict))

    @cached_property
    def paths_containing_urls(self):
        """Get unique paths containing URLs with array indices replaced by wildcards."""
        replace_array_index_with_star = partial(re.sub, r"\[\d+\]", "[*]")
        ignore_first_part_of_path = lambda x: (
            ".".join(x.split(".")[1:]) if "." in x else x
        )
        transform = Pipe(replace_array_index_with_star, ignore_first_part_of_path)
        return sorted(set(map(transform, self.url_paths)))

    def url_data(self, *, remove_chatgpt_utm=True):
        """
        Extract all URLs from the conversation.

        Args:
            remove_chatgpt_utm: If True, remove utm_source=chatgpt.com from URLs

        Returns:
            List of URLs found in the conversation
        """
        urls = []
        for msg in self.parsed_conversation:
            content = msg.get("content", "")
            # Find URLs in content
            url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
            found_urls = re.findall(url_pattern, content)
            urls.extend(found_urls)

        if remove_chatgpt_utm:
            urls = [remove_utm_source(url) for url in urls]

        return urls


# --------------------------------------------------------------------------------------
# SSOT and other data - placeholder for compatibility
# These would need to be imported from oa._params if they exist
try:
    from oa._params import turns_data_ssot, metadata_ssot

    ChatDacc.turns_data_ssot = turns_data_ssot
    ChatDacc.metadata_ssot = metadata_ssot
except ImportError:
    pass  # These might not exist in the new structure


# --------------------------------------------------------------------------------------
# Tools for parser maintenance
# See https://github.com/thorwhalen/oa/blob/main/misc/oa%20-%20chats%20acquisition%20and%20parsing.ipynb
# for more info (namely how to use mk_json_field_documentation to make descriptions)
"""
Notes:

metadata['linear_conversation'] == list(metadata['mapping'].values())

"""
try:
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
except ImportError:
    pass  # prompt_function might not be available


def truncate_dict_values(d, max_length=100):
    """Truncate string values in a dict to a maximum length for display purposes."""
    result = {}
    for k, v in d.items():
        if isinstance(v, str) and len(v) > max_length:
            result[k] = v[:max_length] + "..."
        elif isinstance(v, dict):
            result[k] = truncate_dict_values(v, max_length)
        elif isinstance(v, list) and len(v) > 3:
            result[k] = v[:3] + ["..."]
        else:
            result[k] = v
    return result
