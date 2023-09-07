"""Interface tools"""

from functools import partial
from typing import Optional, Iterable
import string

from i2 import Sig, call_forgivingly
from oa.base import chat

string_formatter = string.Formatter()


def _extract_names_from_format_string(template: str):
    """Extract names from a string format template

    >>> _extract_names_from_format_string("Hello {name}! I am {bot_name}.}")
    ('name', 'bot_name')

    """
    return tuple(
        name for _, name, _, _ in string_formatter.parse(template) if name is not None
    )


def string_format_embodier(template):
    names = _extract_names_from_format_string(template)
    sig = Sig(names).ch_kinds(**{name: Sig.KEYWORD_ONLY for name in names})

    @sig
    def templated_string_embodier(**kwargs):
        return template.format(**kwargs)

    return templated_string_embodier


def prompt_function(
    template,
    *,
    defaults: Optional[dict] = None,
    template_to_names=_extract_names_from_format_string,
    embodier=string_format_embodier,
    name=None,
    prompt_func=chat,
    prompt_func_kwargs=None,
    egress=None
):
    """Convert a string template to a function that will call"""

    template_embodier = embodier(template)
    defaults = defaults or {}
    prompt_func_kwargs = prompt_func_kwargs or {}
    egress = egress or (lambda x: x)

    # TODO: Same logic replicated in string_format_embodier (what can we do?)
    names = template_to_names(template)
    # TODO: Too restrictive to require all names to be keyword only?
    sig = Sig(names).ch_kinds(**{name: Sig.KEYWORD_ONLY for name in names})
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
