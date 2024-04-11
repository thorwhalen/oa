"""Interface tools"""

from functools import partial
from typing import Optional, Callable
import string

from i2 import Sig, Pipe
from lkj import add_attr
from oa.base import chat

# -----------------------------------------------------------------------------
# Helpers

string_formatter = string.Formatter()

import re

DFLT_IGNORE_PATTERN = re.compile(r'```.*?```', re.DOTALL)


def remove_pattern(string, pattern_to_remove=DFLT_IGNORE_PATTERN):
    """
    Returns a where a given regular expression pattern has been removed.

    >>> string = 'this ```is a``` string ```with several``` backticks'
    >>> remove_pattern(string)
    'this  string  backticks'

    """
    pattern_to_remove = re.compile(pattern_to_remove)
    return pattern_to_remove.sub('', string)


def process_string(func, string, ignore_pattern=DFLT_IGNORE_PATTERN):
    """
    Applies a function to segments of a string that do not match a given regular expression pattern,
    while leaving segments that do match the pattern unchanged.

    Parameters:
    - func: A function to apply to non-matching segments of the string.
    - string: The input string to process.
    - ignore_pattern: A regular expression pattern. Segments of `string` that match this pattern will not be altered.

    Returns:
    - A new string with `func` applied to non-matching segments.

    Example:
    >>> func = str.upper
    >>> string = "the good the ```bad``` and the ugly"
    >>> ignore_pattern = r'```.*?```'
    >>> process_string(func, string, ignore_pattern)
    'THE GOOD THE ```bad``` AND THE UGLY'
    """
    # Initialize an empty result string
    result = ''
    # The end position of the last match
    last_end = 0

    # Find all matches of ignore_pattern in the string
    for match in re.finditer(ignore_pattern, string):
        # Get the start and end of the current match
        start, end = match.span()
        # Apply func to the substring from the end of the last match to the start of the current match
        result += func(string[last_end:start])
        # Add the current match unchanged
        result += string[start:end]
        # Update the end position of the last match
        last_end = end

    # Apply func to the remainder of the string after the last match
    result += func(string[last_end:])
    return result


def _extract_names_from_format_string(
    template: str, *, ignore_pattern=DFLT_IGNORE_PATTERN
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
    template: str, *, ignore_pattern=DFLT_IGNORE_PATTERN
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
        return process_string(rm_specifiers, template, ignore_pattern)


def string_format_embodier(template):
    names = _extract_names_from_format_string(template)
    sig = Sig(names).ch_kinds(**{name: Sig.KEYWORD_ONLY for name in names})

    @sig
    def templated_string_embodier(**kwargs):
        return template.format(**kwargs)

    return templated_string_embodier


add_name = add_attr('__name__')
add_doc = add_attr('__doc__')
add_module = add_attr('__module__')

# -----------------------------------------------------------------------------
# The meat


def prompt_function(
    template,
    *,
    defaults: Optional[dict] = None,
    template_to_names: Callable = _extract_names_from_format_string,
    template_to_defaults: Callable = _extract_defaults_from_format_string,
    embodier: Callable = string_format_embodier,
    arg_kinds: Optional[dict] = None,
    name='prompt',
    prompt_func=chat,
    prompt_func_kwargs=None,
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
    :param egress: A function to apply to the output of `prompt_func`.
    :param doc: The docstring of the function.
    :param module: The module of the function.

    >>> prompt_template = '''
    ... ```
    ... In this block, all {placeholders} are {igno:red} so that they can appear in prompt.
    ... ```
    ... But outside {inputs} are {injected:normally}
    ... '''
    >>> f = prompt_function(prompt_template, prompt_func=None)
    >>> from inspect import signature
    >>> assert str(signature(f)) == "(inputs, *, injected='normally')"

    """
    template_original = template
    defaults = dict(template_to_defaults(template), **(defaults or {}))
    template = _template_without_specifiers(template)
    template_embodier = embodier(template)
    prompt_func_kwargs = prompt_func_kwargs or {}
    egress = egress or (lambda x: x)

    # TODO: Same logic replicated in string_format_embodier (what can we do?)
    names = template_to_names(template)
    arg_kinds = dict({name: Sig.KEYWORD_ONLY for name in names}, **(arg_kinds or {}))
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
        _kwargs = sig.kwargs_from_args_and_kwargs(
            ask_oa_args, ask_oa_kwargs, apply_defaults=True
        )
        __args, __kwargs = Sig(template_embodier).args_and_kwargs_from_kwargs(_kwargs)
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

_suffixes_csv = ','.join(_dflt_factories.keys())
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
        print('\n'.join(f"{k}{v}" for k, v in self.funcs_and_sigs().items()))
