"""Base oa functionality"""

from itertools import chain
from functools import partial
from typing import Union, Iterable, Optional, Mapping, KT, Callable

from types import SimpleNamespace
from i2 import Sig, Param

from oa.util import (
    openai,
    djoin,
    mk_client,
    num_tokens,
    embeddings_models,
    DFLT_ENGINE,
    DFLT_MODEL,
    DFLT_EMBEDDINGS_MODEL,
)
from oa.openai_specs import prompt_path, sig

Text = str
TextStrings = Iterable[Text]
TextStore = Mapping[KT, Text]
Texts = Union[TextStrings, TextStore]
TextOrTexts = Union[Text, Texts]

api = None
# TODO: Expand this to include all the other endpoints (automatically?)
# api = SimpleNamespace(
#     chat=sig.CreateChatCompletionRequest(openai.chat.completions.create),
#     complete=sig.CreateCompletionRequest(openai.completions.create),
#     dalle=sig.CreateImageRequest(openai.images.generate),
# )

_model_id_aliases = {
    'davinci': 'text-davinci-003',
    'ada v2': 'text-embedding-ada-002',
}

_model_information_aliases = {
    'max_tokens': 'max_input',
    'price': 'price_per_million_tokens',
}


# TODO: Literal type of models and information?
def model_information(model, information):
    """Get information about a model"""
    if model in _model_id_aliases:
        model = _model_id_aliases[model]
    if information in _model_information_aliases:
        information = _model_information_aliases[information]

    return embeddings_models[model][information]


# TODO: Parse more info and complete this function
#     (see https://github.com/thorwhalen/oa/discussions/8#discussioncomment-9138661)
def compute_price(
    model: str, num_input_tokens: int = None, num_output_tokens: Optional[int] = None
):
    """Compute the price of a model given the number of input and output tokens"""
    assert num_output_tokens is None, 'num_output_tokens not yet implemented'
    if num_input_tokens is None:
        return partial(compute_price, model)
    price_per_million_tokens = model_information(model, 'price_per_million_tokens')
    return price_per_million_tokens * (tokens / 1_000_000)


prompt_dalle_path = partial(prompt_path, prefix=djoin('dalle'))
prompt_davinci_path = partial(prompt_path, prefix=djoin('davinci'))

# TODO: Use oa.openai_specs sig to provide good signatures


def complete(prompt, model=None, **complete_params):
    if 'engine' in complete_params:
        model = complete_params.pop('engine')
    model = model or getattr(complete, 'engine', DFLT_ENGINE)
    text_resp = openai.completions.create(model=model, prompt=prompt, **complete_params)
    return text_resp.choices[0].text


complete.engine = DFLT_ENGINE


def _raw_chat(prompt=None, model=DFLT_MODEL, *, messages=None, **chat_params):
    if not ((prompt is None) ^ (messages is None)):
        raise ValueError('Either prompt or messages must be specified, but not both.')
    if prompt is not None:
        messages = [{'role': 'user', 'content': prompt}]
    return openai.chat.completions.create(messages=messages, model=model, **chat_params)


# chat_sig = sig.CreateChatCompletionRequest
# chat_sig = chat_sig.ch_defaults(model=DFLT_MODEL, messages=None)
# chat_sig = Sig([Param(name='prompt', default=None, annotation=str), *chat_sig.params])


def chat(prompt=None, model=DFLT_MODEL, *, messages=None, **chat_params):
    resp = _raw_chat(prompt=prompt, model=model, messages=messages, **chat_params)
    # TODO: Make attr and item getters more robust (use glom?)
    return resp.choices[0].message.content


chat.raw = _raw_chat


def _raw_dalle(prompt, n=1, size='512x512', **image_create_params):
    return openai.images.generate(prompt=prompt, n=n, size=size, **image_create_params)


def dalle(prompt, n=1, size='512x512', **image_create_params):
    r = _raw_dalle(prompt=prompt, n=n, size=size, **image_create_params)
    return r.data[0].url


def list_engine_ids():
    models_list = mk_client().models.list()
    return [x.id for x in models_list.data]


def _raise_if_any_invalid(
    validation_vector: Iterable[bool],
    texts: Iterable[Text] = None,
    print_invalid_texts=True,
):
    if any(validation_vector):
        if print_invalid_texts:
            print(
                'Invalid text(s):\n',
                '\n'.join(
                    item
                    for is_valid, item in zip(validation_vector, texts)
                    if not is_valid
                ),
            )
        raise ValueError('Some of the texts are invalid')
    return texts

# TODO: Make a few useful validation_callback functions 
#    (e.g. return list or dict where invalid texts are replaced with None)
#    (e.g. return dict containing only valid texts (if input was list, uses indices as keys)
def embeddings(
    texts: TextOrTexts,
    *,
    validate: Optional[Union[bool, Callable]] = True,
    validation_callback=_raise_if_any_invalid,
    model=DFLT_EMBEDDINGS_MODEL,
    client=None,
):
    if validate:
        if validate is True:
            validate = partial(text_is_valid, model=model)
        validation_vector = validate(texts)
        texts = validation_callback(validation_vector, texts=texts)

    if client is None:
        client = mk_client()
    if isinstance(texts, str):
        texts = [texts]
        return client.embeddings.create(input=texts, model=model).data[0].embedding
    elif isinstance(texts, Mapping):
        vectors = embeddings(texts.values(), model=model, client=client)
        return {k: v for k, v in zip(texts.keys(), vectors)}
    else:
        return [
            x.embedding for x in client.embeddings.create(input=texts, model=model).data
        ]


def text_is_valid(
    texts: TextOrTexts,
    token_count=True,
    *,
    model: str = DFLT_EMBEDDINGS_MODEL,
    max_tokens=None,
):
    """Check if text (a string or iterable of strings) is/are valid for a given model.

    Text is valid if
    - it is not empty
    - the number of tokens in the text is less than or equal to the max_tokens

    :param texts: a string or an iterable of strings
    :param token_count: Specification of the token count of the text or texts
    :param model: The model to use for token count
    :param max_tokens: The maximum number of tokens allowed by the model

    If token_count is an integer, it will check if it is less than or equal to the
    `max_tokens`.

    If token_count is True, it will compute the number of tokens in the text using the
    model specified by `model` and check if it is less than or equal to `max_tokens`.

    If token_count is False, it will not check the token count.

    If token_count is an iterable, it will apply the same mechanism as above, to each
    text in `texts` and the corresponding token count in `token_count`.
    This means both texts and token_count(s) must be of the same length.

    Examples:

    >>> text_is_valid('Hello, world!')
    True
    >>> text_is_valid('')
    False
    >>> text_is_valid('Alice ' * 9000)
    False
    >>> text_is_valid('Alice ' * 9000, token_count=False)
    True
    >>> list(text_is_valid(['Bob', '', 'Alice ' * 9000]))
    [True, False, False]
    >>> list(text_is_valid(['Bob', '', 'Alice ' * 9000], token_count=False))
    [True, False, True]

    """
    if isinstance(texts, str):
        text = texts  # renaming to be clear it's a single text
        if not text:
            return False
        if token_count:
            max_tokens = max_tokens or embeddings_models[model]['max_input']
            if token_count is True:
                token_count = num_tokens(text, model=model)
            return token_count <= max_tokens
        else:
            return True
    elif isinstance(texts, Iterable):
        _text_is_valid = partial(text_is_valid, model=model, max_tokens=max_tokens)
        if isinstance(token_count, Iterable):
            return map(_text_is_valid, texts, token_count)
        else:
            __text_is_valid = partial(_text_is_valid, token_count=token_count)
            return map(__text_is_valid, texts)
    else:
        raise TypeError('texts must be a str or an iterable of str')


# df['ada_embedding'] = df.combined.apply(lambda x: get_embedding(x, model='text-embedding-3-small'))
# df.to_csv('output/embedded_1k_reviews.csv', index=False)
