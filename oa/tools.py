"""Interface tools"""

from functools import partial
from typing import Optional, Callable
import string

from i2 import Sig, Pipe
from lkj import add_attr
from oa.base import chat

# -----------------------------------------------------------------------------
# Helpers

import re
from collections import namedtuple
from typing import List, NamedTuple, Literal, Union, Optional

Pattern = Union[re.Pattern, str]
string_formatter = string.Formatter()

DFLT_IGNORE_PATTERN = re.compile(r"```.*?```", re.DOTALL)


def remove_pattern(string, pattern_to_remove: Optional[Pattern] = DFLT_IGNORE_PATTERN):
    """
    Returns a where a given regular expression pattern has been removed.

    >>> string = 'this ```is a``` string ```with several``` backticks'
    >>> remove_pattern(string)
    'this  string  backticks'

    """
    pattern_to_remove = re.compile(pattern_to_remove)
    return pattern_to_remove.sub("", string)


def extract_parts(string: str, pattern: Pattern) -> NamedTuple:
    PartResult = namedtuple("PartResult", ["matched", "unmatched"])
    matched: List[str] = []
    unmatched: List[str] = []
    last_end = 0

    for match in re.finditer(pattern, string):
        start, end = match.span()
        unmatched.append(string[last_end:start])
        matched.append(string[start:end])
        last_end = end

    unmatched.append(string[last_end:])

    return PartResult(matched=matched, unmatched=unmatched)


def pattern_based_map(
    func: Callable,
    string: str,
    pattern: Pattern,
    apply_to: Literal["matched", "unmatched"] = "unmatched",
):
    """
    Applies a function to parts of the string that are either matching or non-matching based on a regex pattern,
    depending on the value of apply_to.

    Example:
    >>> func = str.upper
    >>> string = "the good the ```bad``` and the ugly"
    >>> ignore_pattern = r'```.*?```'
    >>> pattern_based_map(func, string, ignore_pattern)
    'THE GOOD THE ```bad``` AND THE UGLY'
    >>> pattern_based_map(func, string, ignore_pattern, 'matched')
    'the good the ```BAD``` and the ugly'
    """
    parts = extract_parts(string, pattern)
    result = ""

    # Apply the function to the appropriate parts
    if apply_to == "matched":
        transformed_matched = [func(part) for part in parts.matched]
        # Interleave transformed matched parts with untouched unmatched parts
        result_parts = sum(zip(parts.unmatched, transformed_matched), ())
    else:
        transformed_unmatched = [func(part) for part in parts.unmatched]
        # Interleave untouched matched parts with transformed unmatched parts
        result_parts = sum(zip(transformed_unmatched, parts.matched), ())

    # Ensure all parts are added, including the last unmatched if unmatched is longer
    result = "".join(result_parts)
    if len(parts.unmatched) > len(parts.matched):
        result += (
            transformed_unmatched[-1]
            if apply_to == "unmatched"
            else parts.unmatched[-1]
        )

    return result


def _extract_names_from_format_string(
    template: str, *, ignore_pattern: Optional[Pattern] = DFLT_IGNORE_PATTERN
):
    """Extract names from a string format template

    >>> _extract_names_from_format_string("Hello {name}! I am {bot_name}.")
    ('name', 'bot_name')

    """
    if ignore_pattern is not None:
        template = remove_pattern(template, ignore_pattern)
    return tuple(
        name for _, name, _, _ in string_formatter.parse(template) if name is not None
    )


def _extract_defaults_from_format_string(
    template: str, *, ignore_pattern=DFLT_IGNORE_PATTERN
) -> dict:
    """Extract (name, specifier) from a string format template.

    >>> _extract_defaults_from_format_string(
    ...     "Hello {name}! I am {bot_name:chatGPT}."
    ... )
    {'bot_name': 'chatGPT'}

    """
    if ignore_pattern is not None:
        template = remove_pattern(template, ignore_pattern)
    return dict(
        (name, specifier)
        for _, name, specifier, _ in string_formatter.parse(template)
        if name is not None and specifier != ""
    )


def _template_without_specifiers(
    template: str, *, ignore_pattern: Optional[Pattern] = DFLT_IGNORE_PATTERN
) -> str:
    """Uses remove any extras from a template string, leaving only text and fields.

    >>> template = "A {normal}, {stra:nge}, an ```{igno:red}``` and an empty: {}."
    >>> _template_without_specifiers(template, ignore_pattern=None)
    'A {normal}, {stra}, an ```{igno}``` and an empty: {}.'
    >>> _template_without_specifiers(template, ignore_pattern=r'```.*?```')
    'A {normal}, {stra}, an ```{igno:red}``` and an empty: {}.'

    """

    def gen(template):
        for text, field_name, *_ in string.Formatter().parse(template):
            text_ = text or ""
            if field_name is None:
                yield text_
            else:
                yield text_ + "{" + field_name + "}"

    def rm_specifiers(template):
        return "".join(gen(template))

    if ignore_pattern is None:
        return rm_specifiers(template)
    else:
        return pattern_based_map(rm_specifiers, template, ignore_pattern)


def _template_with_double_braces_in_ignored_sections(
    template, *, ignore_pattern: Optional[Pattern] = DFLT_IGNORE_PATTERN
) -> str:
    """double the braces of the parts of the template that should be ignored"""
    double_braces = lambda string: string.replace("{", "{{").replace("}", "}}")
    return pattern_based_map(
        double_braces, template, ignore_pattern, apply_to="matched"
    )


def string_format_embodier(
    template, *, ignore_pattern: Optional[Pattern] = DFLT_IGNORE_PATTERN
):

    names = _extract_names_from_format_string(template, ignore_pattern=ignore_pattern)
    names = tuple(dict.fromkeys(names))  # get unique names, but conserving order
    sig = Sig(names).ch_kinds(**{name: Sig.KEYWORD_ONLY for name in names})

    @sig
    def templated_string_embodier(**kwargs):
        return template.format(**kwargs)

    return templated_string_embodier


add_name = add_attr("__name__")
add_doc = add_attr("__doc__")
add_module = add_attr("__module__")

# -----------------------------------------------------------------------------
# The meat


# TODO: template_to_names, template_to_defaults and embodier are implicitly bound by
#   their ignore_pattern argument (set to DFLT_IGNORE_PATTERN). Find a cleaner way.
def prompt_function(
    template,
    *,
    defaults: Optional[dict] = None,
    template_to_names: Callable = _extract_names_from_format_string,
    template_to_defaults: Callable = _extract_defaults_from_format_string,
    embodier: Callable = string_format_embodier,
    arg_kinds: Optional[dict] = None,
    name="prompt",
    prompt_func=chat,
    prompt_func_kwargs=None,
    ingress=None,
    egress=None,
    doc="The function composes a prompt and asks an LLM to respond to it.",
    module=__name__,
):
    r"""Convert a string template to a function that will produce a prompt string
    and ask an LLM (`prompt_func`) to respond to it.

    :param template: A string template with placeholders.
    :param defaults: A dictionary of default values for placeholders.
    :param template_to_names: A function that extracts names from a template.
    :param template_to_defaults: A function that extracts defaults from a template.
    :param embodier: A function that converts a template to a function that will
        produce a prompt string.
    :param arg_kinds: A dictionary of argument kinds for the function.
    :param name: The name of the function.
    :param prompt_func: The function that will be used to ask the LLM to respond to
        the prompt. If None, the output function will only produce the prompt string,
        not ask the LLM to respond to it.
    :param prompt_func_kwargs: Keyword arguments to pass to `prompt_func`.
    :param ingress: A function to apply to the input of `prompt_func`.
    :param egress: A function to apply to the output of `prompt_func`.
    :param doc: The docstring of the function.
    :param module: The module of the function.

    In the following example, we'll use the `prompt_func=None` argument to get a
    function that simply injects inputs in a prompt template, without actually calling
    an AI-enabled `prompt_func`.
    Note in this example, how a block of the prompt template string is ignored for
    injection purposes, via a triple-backtick marker.

    >>> prompt_template = '''
    ... ```
    ... In this block, all {placeholders} are {igno:red} so that they can appear in prompt.
    ... ```
    ... But outside {inputs} are {injected:normally}
    ... '''
    >>> f = prompt_function(prompt_template, prompt_func=None)
    >>> from inspect import signature
    >>> assert str(signature(f)) == "(inputs, *, injected='normally')"
    >>> print(f('INPUTS', injected="INJECTED"))  # doctest: +NORMALIZE_WHITESPACE
    ```
    In this block, all {placeholders} are {igno:red} so that they can appear in prompt.
    ```
    But outside INPUTS are INJECTED

    """

    template_original = template
    defaults = dict(template_to_defaults(template), **(defaults or {}))
    template = _template_without_specifiers(template)
    template = _template_with_double_braces_in_ignored_sections(template)
    template_embodier = embodier(template)
    prompt_func_kwargs = prompt_func_kwargs or {}
    egress = egress or (lambda x: x)
    ingress = ingress or (lambda x: x)

    # TODO: Same logic replicated in string_format_embodier (what can we do?)
    names = template_to_names(template)
    arg_kinds = dict({name: Sig.KEYWORD_ONLY for name in names}, **(arg_kinds or {}))
    names = tuple(dict.fromkeys(names))  # get unique names, but conserving order
    sig = Sig(names)

    # Inject defaults
    sig = sig.ch_defaults(
        _allow_reordering=True, **{name: default for name, default in defaults.items()}
    )
    # Handle kinds (make all but first keyword only)) and inject defaults
    sig = sig.ch_kinds(**arg_kinds)
    if sig.names:
        # Change the first argument to position or keyword kind
        first_arg_name = sig.names[0]
        sig = sig.ch_kinds(**{first_arg_name: Sig.POSITIONAL_OR_KEYWORD})

    sig = sig.sort_params()
    func_wrap = Pipe(sig, add_name(name), add_doc(doc), add_module(module))

    @func_wrap
    def embody_prompt(*ask_oa_args, **ask_oa_kwargs):
        _kwargs = sig.map_arguments(ask_oa_args, ask_oa_kwargs, apply_defaults=True)
        _kwargs = ingress(_kwargs)
        __args, __kwargs = Sig(template_embodier).mk_args_and_kwargs(_kwargs)
        embodied_template = template_embodier(*__args, **__kwargs)
        return embodied_template

    @func_wrap
    def ask_oa(*ask_oa_args, **ask_oa_kwargs):
        embodied_template = embody_prompt(*ask_oa_args, **ask_oa_kwargs)
        return egress(prompt_func(embodied_template, **prompt_func_kwargs))

    if prompt_func is not None:
        f = ask_oa
    else:
        f = embody_prompt
    f.template = template
    f.template_original = template_original

    return f


import json
from i2 import Sig, Pipe
from typing import Mapping


def identity(x):
    return x


json_types = {
    "object": dict,
    "array": list,
    "string": str,
    "number": float,
    "integer": int,
    "boolean": bool,
    "null": type(None),
}

py_to_json_types = {v: k for k, v in json_types.items()}

json_type_specs = json_types.keys() | py_to_json_types.keys()

# TODO: Define from json_type_specs when possible (e.g. in python 3.11 it will be)
# JsonTypes = Literal[
#     'object', 'array', 'string', 'number', 'integer', 'boolean', 'null',
#     dict, list, str, float, int, bool, type(None),
# ]

from enum import Enum


class JsonTypes(Enum):
    string = "string"
    number = "number"
    object = "object"
    array = "array"
    boolean = "boolean"
    null = "null"
    dict = dict
    list = list
    float = float
    int = int
    bool = bool
    none = type(None)


def ensure_json_type(json_type: JsonTypes) -> str:
    """
    Ensure that the json type is a string that is a valid json type

    >>> ensure_json_type('string')
    'string'
    >>> ensure_json_type(str)
    'string'
    >>> ensure_json_type(object)  # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    ValueError: Cannot convert ...

    """
    # Get the string representation of the json type, given as a python type
    if isinstance(json_type, type):
        if json_type in py_to_json_types:
            return py_to_json_types[json_type]
        else:
            raise ValueError(
                f"Cannot convert {json_type} to a json type. "
                f"Should be one of {json_type_specs}"
            )
    # ensure that the json type is a string that is listed in json_types
    if not isinstance(json_type, str) or json_type not in json_types:
        raise ValueError(
            f"json_type should be a string or a type, not {type(json_type)}"
        )
    return json_type


def make_generic_json_schema(json_type: JsonTypes) -> dict:
    """
    Make a generic json schema for a given json type.

    >>> make_generic_json_schema('string')
    {'name': 'generic_string_schema', 'schema': {'properties': {'result': {'type': 'string'}}, 'required': ['result']}}
    """

    json_type = ensure_json_type(json_type)
    return {
        "name": f"generic_{json_type}_schema",
        "schema": {
            "properties": {"result": {"type": json_type}},
            "required": ["result"],
        },
    }


# Note: Deprecated, but Keeping around for reference
_generic_json_schema = {
    "name": "generic_json_schema",
    "schema": {
        "properties": {"result": {"type": "string"}},
        "required": ["result"],
        #  "additionalProperties": True,
    },
    #     "strict": False,
}


def _might_be_a_json_string(string):
    """
    Returns True if the string might have a chance of being decoded by `json.loads`.

    More precisely, will check if the first non-whitespace character is a '{' or a '['.

    >>> _might_be_a_json_string('   {"a": 1}  ')
    True
    >>> _might_be_a_json_string('    ["lists", "of", "stuff"]  ')
    True
    >>> _might_be_a_json_string('    not a json string  ')
    False
    """
    return re.compile(r"^\s*[\[\{]").match(string) is not None


def _ensure_json_schema(json_schema: Union[str, bytes, Mapping]) -> dict:
    """
    A few things to make it more probable that the input is a oa valid json schema
    """
    if isinstance(json_schema, type):
        json_schema = make_generic_json_schema(json_schema)
    elif isinstance(json_schema, str):
        if json_schema in json_types:  # make a generic json schema for that type
            json_schema = make_generic_json_schema(json_schema)
        elif _might_be_a_json_string(json_schema):  # assume it is a json string
            json_schema = json.loads(json_schema)
        else:  # assume it's free text, from which AI will try to infer a schema
            verbal_description = json_schema
            _json_schema = infer_schema_from_verbal_description(verbal_description)
            json_schema = _json_schema  # ["json_schema"]

    if "name" not in json_schema:  # OpenAI forces you to put a name
        json_schema["name"] = "json_schema"

    if "schema" not in json_schema:  # the schema actually has to be under a schema key
        json_schema = {"schema": json_schema, "name": "json_schema"}

    if "type" not in json_schema["schema"]:  # OpenAI forces you to put a type
        json_schema["schema"]["type"] = "object"

    return json_schema


# TODO: model could be present in prompt_func_kwargs or in partial of prompt_func
#   --> need to ensure that all work well together (no obfuscated conflicts)
def prompt_json_function(
    template,
    json_schema: Union[str, bytes, Mapping] = "string",
    *,
    defaults: Optional[dict] = None,
    embodier: Callable = string_format_embodier,
    arg_kinds: Optional[dict] = None,
    name="prompt",
    prompt_func=chat,
    prompt_func_kwargs=None,
    model="gpt-4o-mini",
    ingress=None,
    egress=None,
    doc="The function composes a prompt and asks an LLM to respond to it with json.",
    module=__name__,
) -> dict:
    """
    Make prompt functions that return jsons (dicts) with a given schema.
    """

    json_schema = _ensure_json_schema(json_schema)

    assert isinstance(json_schema, Mapping)

    prompt_func_kwargs = dict(
        dict(
            model=model,  # TODO: Change to just ensure model is compatible
            response_format={
                "type": "json_schema",
                "json_schema": json_schema,
            },
        ),
        **(prompt_func_kwargs or {}),
    )

    egress = Pipe(json.loads, egress or identity)

    return prompt_function(
        template,
        defaults=defaults,
        embodier=embodier,
        arg_kinds=arg_kinds,
        name=name,
        prompt_func=prompt_func,
        prompt_func_kwargs=prompt_func_kwargs,
        ingress=ingress,
        egress=egress,
        doc=doc,
        module=module,
    )


def infer_schema_from_verbal_description(verbal_description: str):
    template = """
    Generate a valid JSON Schema based on the the verbal description of the desired 
    JSON output below. 
    The schema must be properly formatted for use with OpenAI’s Chat API 
    in "JSON mode" and should accurately define the structure, 
    data types, required fields, and any constraints specified by the user. 
    Ensure correctness and completeness.

    Note that you need to provide not only a valid schema but also a valid name for it.

    Here is the verbal description of the desired JSON output:
    {verbal_description}
    """
    output_schema = {
        "name": "infered_json_schema",
        "schema": {
            "properties": {
                "name": {"type": "string"},
                "properties": {"type": "object"},
                "type": {"type": "string"},
            },
            "required": ["name", "properties"],
        },
    }
    f = prompt_json_function(template, output_schema)
    return f(verbal_description=verbal_description)


from typing import Mapping, Optional, KT, Union
from dol import filt_iter
from oa.util import mk_template_store, DFLT_TEMPLATES_SOURCE
import os

# chatgpt_templates_dir = os.path.join(templates_files, "chatgpt")
# _ends_with_txt = filt_iter.suffixes(".txt")
# chatgpt_templates = filt_iter(TextFiles(chatgpt_templates_dir), filt=_ends_with_txt)
dflt_function_key = lambda f: os.path.splitext(os.path.basename(f))[0]
dflt_factory_key = lambda f: os.path.splitext(os.path.basename(f))[-1]
_dflt_factories = {
    ".txt": prompt_function,
    "": prompt_function,
}
dflt_factories = tuple(_dflt_factories.items())

_suffixes_csv = ",".join(_dflt_factories.keys())
DFLT_TEMPLATE_SOURCE_WITH_SUFFIXES = f"{DFLT_TEMPLATES_SOURCE}:{_suffixes_csv}"

StoreKey = str
FuncName = str
FactoryKey = KT


class PromptFuncs:
    """Make AI enabled functions"""

    def __init__(
        self,
        template_store: Union[Mapping, str] = DFLT_TEMPLATE_SOURCE_WITH_SUFFIXES,
        *,
        function_key: Callable[[StoreKey], FuncName] = dflt_function_key,
        factory_key: Callable[[StoreKey], FactoryKey] = dflt_factory_key,
        factories: Callable[[FactoryKey], Callable] = dflt_factories,
        extra_template_kwargs: Optional[Mapping[StoreKey, Mapping]] = None,
    ):
        self._template_store = mk_template_store(template_store)
        self._function_key = function_key
        self._factory_key = factory_key
        self._factories = dict(factories)
        self._extra_template_kwargs = dict(extra_template_kwargs or {})
        self._functions = dict(self._mk_functions())
        self._inject_functions()

    def _mk_functions(self):
        for store_key, template in self._template_store.items():
            factory = self._factories[self._factory_key(store_key)]
            func_key = self._function_key(store_key)
            yield func_key, factory(
                template,
                name=func_key,
                **self._extra_template_kwargs.get(store_key, {}),
            )

    def _inject_functions(self):
        self.__dict__.update(self._functions)

    def __iter__(self):
        return iter(self._functions)

    def __getitem__(self, name):
        return self._functions[name]

    def __len__(self):
        return len(self._functions)

    def reload(self):
        """Reload all functions"""
        self._functions = dict(self._mk_functions())
        self._inject_functions()
        return self

    def funcs_and_sigs(self):
        """Return a mapping of function names to signatures"""
        return {name: Sig(func) for name, func in self._functions.items()}

    def print_signatures(self):
        """Print signatures of all functions"""
        print("\n".join(f"{k}{v}" for k, v in self.funcs_and_sigs().items()))
