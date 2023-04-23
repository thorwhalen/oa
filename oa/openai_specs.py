"""Tools to extract specs from the openai interface

See raw schemas by doing

>>> from oa.openai_specs import schemas
>>> sorted(schemas)  # doctest: +ELLIPSIS
['ChatCompletionRequestMessage', 'ChatCompletionResponseMessage', ...
>>> schema = schemas['CreateCompletionRequest']
>>> schema['properties']  # doctest: +SKIP
 {'model': {'description': 'ID of the model to use...

Get resulting signatures by doing.

>>> from oa.openai_specs import sig
>>> sig.CreateCompletionRequest  # doctest: +ELLIPSIS
<Sig (model: str, *, prompt='<|endoftext|>', ...

Or see argument descriptions in rst format by doing:

>>> from oa.openai_specs import schema_to_rst_argument_descriptions, schemas
>>> schema_to_rst_argument_descriptions(schemas['CreateCompletionRequest'])  # doctest: +SKIP

```

"""
import os
import re
from functools import lru_cache
from typing import Dict, List, TypedDict, Optional, Literal
import yaml

import dol
from i2 import Sig, Param, empty_param_attr as empty

from oa.util import grazed


@lru_cache
def get_openapi_spec_dict(
    openapi_spec_url: str = (
        'https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml'
    ),
    *,
    refresh: bool = False,
    expand_refs: bool = True,
):
    """Get the dict of the openapi spec"""
    if refresh:
        # TODO: Graze should have a refresh method to make sure to not delete before
        #  re-grazing
        del grazed[openapi_spec_url]

    d = yaml.safe_load(grazed[openapi_spec_url])
    if expand_refs:
        import jsonref  # pip install jsonref

        d = jsonref.JsonRef.replace_refs(d)
    return d


specs = get_openapi_spec_dict()
schemas = specs['components']['schemas']

# A mapping between the OpenAI API's schema names and the corresponding Python types.
pytype_of_jstype = {
    'string': str,
    'boolean': bool,
    'integer': int,
    'number': float,
    'array': list,
    'object': dict,
}


# TODO: ?Extract this directly from the ChatCompletionRequestMessage schema
#  Could use TypedDict with annotations:
#  {p.name: p.annotation for p in sig.ChatCompletionRequestMessage.params}
class Message(TypedDict):
    role: str
    content: str
    name: Optional[str]


Messages = List[Message]
Model = str  # TODO: should be Literal parsed from the models list

pytype_of_name = {
    'messages': Messages,
    'model': Model,
}

# Some arguments don't have defaults in the schema, but are secondary, so shouldn't be
# required. The lazy way to handle this case is to give defaults to these arguments.
# The better way would be to give these arguments a sentinel default (say, None) and
# wrap our python functions so they ignore it (don't include it in the request.
# Further -- some schema defaults have values (like max_tokens='inf' that don't match
# their type, AND make the request fail. So we a mechanism to overwride the schema
# default too.
pre_defaults_of_name = {  # defaults that should override the schema
    'max_tokens': None,
}
post_defaults_of_name = {  # defaults used if schema doesn't have a default
    'user': None,
}


def properties_to_annotation(name: str, props: dict):
    """Get annotations from a schema"""
    annotation = pytype_of_name.get(name, None)  # get type from name if exists
    if annotation is None:  # if not, fallback on schemas enum or type
        if 'enum' in props:
            annotation = Literal[tuple(props['enum'])]
        else:
            annotation = pytype_of_jstype.get(props.get('type', None), empty)
    return annotation


def properties_to_param_dict(name: str, props: dict):
    """Get all but kind fields of a Parameter object"""
    annotation = pytype_of_name.get(name, None)  # get type from name if exists
    if annotation is None:  # if not, fallback on schemas type
        annotation = pytype_of_jstype.get(props.get('type', None), empty)
    if name in pre_defaults_of_name:
        default = pre_defaults_of_name[name]
    else:
        default = props.get('default', post_defaults_of_name.get(name, empty))
    return dict(name=name, default=default, annotation=annotation)


def schema_to_signature(schema):
    """Get a signature from a schema

    >>> from oa.openai_specs import specs
    >>> schema = specs['components']['schemas']['CreateChatCompletionRequest']
    >>> print(str(schema_to_signature(schema)))  # doctest: +ELLIPSIS
    (model: str, messages: List[openai_specs.Message], *, temperature: float = 1, ...

    """
    def gen():
        required = schema.get('required', [])
        for name, props in schema['properties'].items():
            if name in required:
                yield Param(
                    **properties_to_param_dict(name, props),
                    kind=Param.POSITIONAL_OR_KEYWORD,
                )
            else:
                yield Param(
                    **properties_to_param_dict(name, props), kind=Param.KEYWORD_ONLY
                )

    return Sig(sorted(gen()))


def _clean_up_whitespace(s):
    """Clean up whitespace in a string.
    Replace all whitespaces with single space, and strip.
    """
    return re.sub(r'\s+', ' ', s).strip()


def schema_to_rst_argument_descriptions(
        schema, process_description=_clean_up_whitespace
):
    """Yield rst-formatted argument descriptions for a schema

    >>> from oa.openai_specs import specs
    >>> schema = specs['components']['schemas']['CreateChatCompletionRequest']
    >>> print(*schema_to_rst_argument_descriptions(schema), '\\n')  # doctest: +SKIP
    :param model: ID of the model to use. Currently, only ...

    `process_description` is a function that will be applied to each schema description
    (so you can clean it up, or add links, etc.). By default, it will replace all
    whitespaces with single space, and strip.

    """
    process_description = process_description or (lambda x: x)
    for name, props in schema['properties'].items():
        description = process_description(props.get('description', ''))
        yield f':param {name}: {description}'


def schema_to_docs(name, schema: dict,  prefix='', line_prefix: str = '\t',):
    """Get the docs for a schema

    >>> from oa.openai_specs import specs
    >>> schema = specs['components']['schemas']['CreateChatCompletionRequest']
    >>> print(schema_to_docs('chatcompletion', schema))  # doctest: +SKIP
    chatcompletion(...

    :param model: ID of the model to use. Currently, only ...

    """
    s = schema_to_signature(schema)
    doc = prefix
    doc += f"{name}(\n\t" + '\n\t'.join(str(s)[1:-1].split(', ')) + '\n)'
    doc += '\n\n'
    doc += '\n\n'.join(schema_to_rst_argument_descriptions(schema))
    return _prefix_all_lines(doc, line_prefix)


def _prefix_all_lines(string: str, prefix: str = '\t'):
    return '\n'.join(map(lambda line: prefix + line, string.splitlines()))


from dol import path_filter, path_get
from dol.sources import AttrContainer, Attrs
from types import SimpleNamespace

sig = AttrContainer(
    **{name: schema_to_signature(schema) for name, schema in schemas.items()}
)
_docs = AttrContainer(
    **{name: schema_to_docs(name.lower(), schema) for name, schema in schemas.items()}
)

from i2 import wrap

from functools import partial, cached_property


# TODO: Write a types.SimpleNamespace version of this.
class SpecNames:
    specs = specs

    @cached_property
    def op_paths(self):
        return list(
            path_filter(lambda p, k, v: k == 'operationId' or v == 'operationId', specs)
        )

    @cached_property
    def route_and_op(self):
        return [(p[1], path_get(specs, p, get_value=dict.get)) for p in self.op_paths]

    @cached_property
    def create_route_and_op(self):
        return list(filter(lambda x: x[1].startswith('create'), self.route_and_op))

    @cached_property
    def sigs_ending_with_req(self):
        return [x for x in vars(sig) if x.endswith('Request')]

    @cached_property
    def creation_actions(self):
        return set(
            map(
                lambda x: x[len('create') : -len('request')].lower(),
                self.sigs_ending_with_req,
            )
        )

    @cached_property
    def attrs(self):
        import oa

        return Attrs(oa.openai)

    @cached_property
    def matched_names(self):
        return [
            x
            for x in list(self.attrs)
            if 'create' in self.attrs[x] and x.lower() in self.creation_actions
        ]

    @cached_property
    def doc_for_name(self):
        import oa
        return {
            op_name: (getattr(oa.openai, op_name).create.__doc__ or '').strip()
            for op_name in self.matched_names
        }

    @cached_property
    def schema_for_name(self):
        return {
            name: schemas[f"Create{name[0].upper()}{name[1:]}Request"]
            for name in self.matched_names
        }

    def _assert_that_everything_matches(self):
        assert set(
            map(lambda x: x[: -len('request')].lower(), self.sigs_ending_with_req)
        ) == set(map(str.lower, [x[1] for x in self.create_route_and_op]))


spec_names = SpecNames()


## rm_when_none-parametrizale version of current version below
# def _kwargs_cast_ingress(func_sig, rm_when_none=(), /, *args, **kwargs):
#     kwargs = func_sig.kwargs_from_args_and_kwargs(args, kwargs)
#     if rm_when_none:
#         for k in rm_when_none:
#             if kwargs[k] is None:
#                 del kwargs[k]
#     return (), kwargs


def _kwargs_cast_ingress(func_sig, /, *args, **kwargs):
    kwargs = func_sig.kwargs_from_args_and_kwargs(args, kwargs)
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    return (), kwargs


def _mk_raw():
    import oa

    def gen():
        for name in spec_names.matched_names:
            lo_name = name.lower()
            func_sig = getattr(sig, f'Create{name}Request')

            func = wrap(
                getattr(oa.openai, name).create,
                ingress=func_sig(partial(_kwargs_cast_ingress, func_sig)),
                name=lo_name,
            )
            func.__doc__ = schema_to_docs(
                lo_name,
                spec_names.schema_for_name[name],
                prefix=spec_names.doc_for_name[name] + '\n\n',
            )
            yield lo_name, func

    return AttrContainer(**dict(gen()))


raw = _mk_raw()


def normalized_file_name(prompt: str) -> str:
    """Convert prompt to a normalized valid file/folder name

    >>> normalized_file_name("This is a prompt")
    'this is a prompt'
    >>> normalized_file_name("This is: a PROMPT!  (with punctuation)")
    'this is a prompt with punctuation'
    """
    return re.sub(r'\W+', ' ', prompt).lower().strip()


def prompt_path(prompt, prefix=''):
    filepath = os.path.join(prefix, normalized_file_name(prompt))
    return dol.ensure_dir(filepath)


def merge_keys_to_values(d: dict, key_name='key'):
    """Merge the keys of a dict into the values of the dict.

    >>> d = {'a': {'b': 1}, 'c': {'d': 2, 'e': 3}}
    >>> dict(merge_keys_to_values(d))
    {'a': {'key': 'a', 'b': 1}, 'c': {'key': 'c', 'd': 2, 'e': 3}}

    Useful when, for example, you want to make a table containing both keys and values.
    """
    for k, v in d.items():
        assert key_name not in v, (
            f'The key_name {key_name} was found in the value dict. Choose a different '
            f'key_name.'
        )
        yield k, dict({key_name: k}, **v)


def schemas_df(schema: dict):
    import pandas as pd

    print(pd.Series(schema['properties']))
    print(schema['required'])
    pd.DataFrame(schema['properties']).T.fillna('')
