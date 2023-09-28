"""Interface tools"""

from functools import partial
from typing import Optional, Callable
import string

from i2 import Sig
from oa.base import chat

# -----------------------------------------------------------------------------
# Helpers

string_formatter = string.Formatter()


def _extract_names_from_format_string(template: str):
    """Extract names from a string format template

    >>> _extract_names_from_format_string("Hello {name}! I am {bot_name}.")
    ('name', 'bot_name')

    """
    return tuple(
        name for _, name, _, _ in string_formatter.parse(template) if name is not None
    )


def _extract_defaults_from_format_string(template: str) -> dict:
    """Extract (name, specifier) from a string format template.

    >>> _extract_defaults_from_format_string(
    ...     "Hello {name}! I am {bot_name:chatGPT}."
    ... )
    {'bot_name': 'chatGPT'}

    """
    return dict(
        (name, specifier)
        for _, name, specifier, _ in string_formatter.parse(template)
        if name is not None and specifier != ""
    )


def _template_without_specifiers(template):
    """Uses remove any extras from a template string, leaving only text and fields.

    >>> _template_without_specifiers("A {normal}, {stra:nge} and an empty: {}.")
    'A {normal}, {stra} and an empty: {}.'

    """
    import string

    def gen():
        for text, field_name, *_ in string.Formatter().parse(template):
            text_ = text or ""
            if field_name is None:
                yield text_
            else:
                yield text_ + "{" + field_name + "}"

    return "".join(gen())


def string_format_embodier(template):
    names = _extract_names_from_format_string(template)
    sig = Sig(names).ch_kinds(**{name: Sig.KEYWORD_ONLY for name in names})

    @sig
    def templated_string_embodier(**kwargs):
        return template.format(**kwargs)

    return templated_string_embodier


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
    name=None,
    prompt_func=chat,
    prompt_func_kwargs=None,
    egress=None,
):
    """Convert a string template to a function that will call"""

    defaults = dict(template_to_defaults(template), **(defaults or {}))
    template = _template_without_specifiers(template)
    template_embodier = embodier(template)
    prompt_func_kwargs = prompt_func_kwargs or {}
    egress = egress or (lambda x: x)

    # TODO: Same logic replicated in string_format_embodier (what can we do?)
    names = template_to_names(template)
    # TODO: Too restrictive to require all names to be keyword only?
    arg_kinds = dict({name: Sig.KEYWORD_ONLY for name in names}, **(arg_kinds or {}))
    sig = Sig(names).ch_kinds(**arg_kinds)
    sig = sig.ch_defaults(
        _allow_reordering=True, **{name: default for name, default in defaults.items()}
    )

    @sig
    def ask_oa(*ask_oa_args, **ask_oa_kwargs):
        _kwargs = sig.kwargs_from_args_and_kwargs(
            ask_oa_args, ask_oa_kwargs, apply_defaults=True
        )
        # print(f"kwargs: {_kwargs}")
        __args, __kwargs = Sig(template_embodier).args_and_kwargs_from_kwargs(_kwargs)
        embodied_template = template_embodier(*__args, **__kwargs)
        # embodied_template = _call_forgivingly(
        #     template_embodier, args, kwargs #, enforce_sig=sig
        # )
        # embodied_template = call_forgivingly(template_embodier, *args, **kwargs)

        return egress(prompt_func(embodied_template, **prompt_func_kwargs))

    if name is not None:
        ask_oa.__name__ = name

    return ask_oa


from typing import Mapping, Optional, KT
from dol import TextFiles, filt_iter
from oa.util import templates_files
import os

chatgpt_templates_dir = os.path.join(templates_files, "chatgpt")
_ends_with_txt = lambda f: f.endswith(".txt")
chatgpt_templates = filt_iter(TextFiles(chatgpt_templates_dir), filt=_ends_with_txt)
dflt_function_key = lambda f: os.path.splitext(os.path.basename(f))[0]
dflt_factory_key = lambda f: os.path.splitext(os.path.basename(f))[-1]
_dflt_factories = {
    ".txt": prompt_function,
    "": prompt_function,
}
dflt_factories = tuple(_dflt_factories.items())

StoreKey = str
FuncName = str
FactoryKey = KT


class PromptFuncs:
    """Make AI enabled functions"""

    def __init__(
        self,
        template_store: Mapping = chatgpt_templates,
        *,
        function_key: Callable[[StoreKey], FuncName] = dflt_function_key,
        factory_key: Callable[[StoreKey], FactoryKey] = dflt_factory_key,
        factories: Callable[[FactoryKey], Callable] = dflt_factories,
        extra_template_kwargs: Optional[Mapping[StoreKey, Mapping]] = None,
    ):
        self._template_store = template_store
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
