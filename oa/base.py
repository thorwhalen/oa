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
    model_information_dict,
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

    if model not in model_information_dict:
        raise ValueError(f'Unknown model: {model}')
    if information not in model_information_dict[model]:
        raise ValueError(f'Unknown information: {information}')

    return model_information_dict[model][information]


model_information.model_information_dict = model_information_dict


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
    return price_per_million_tokens * (num_input_tokens / 1_000_000)


compute_price.model_information_dict = model_information_dict

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
    if isinstance(validation_vector, bool):
        # if it's a single validation boolean, make it a list of one boolean
        validation_vector = [validation_vector]
    else:
        validation_vector = list(validation_vector)
    if not all(validation_vector):
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


from openai import NOT_GIVEN
from typing import Union

# from collections.abc import Mapping, Iterable


# TODO: Make a few useful validation_callback functions
#    (e.g. return list or dict where invalid texts are replaced with None)
#    (e.g. return dict containing only valid texts (if input was list, uses indices as keys)
def embeddings(
    texts: TextOrTexts,
    *,
    validate: Optional[Union[bool, Callable]] = True,
    valid_text_getter=_raise_if_any_invalid,
    model=DFLT_EMBEDDINGS_MODEL,
    client=None,
    dimensions: Optional[int] = NOT_GIVEN,
    **extra_embeddings_params,
):
    """
    Get embeddings for a text or texts.

    :param texts: A string, an iterable of strings, or a dictionary of strings
    :param validate: If True, validate the text(s) before getting embeddings
    :param valid_text_getter: A function that gets valid texts from the input texts
    :param model: The model to use for embeddings
    :param client: The OpenAI client to use
    :param dimensions: If given will reduce the dimensions of the full size embedding
        vectors to that size
    :param extra_embeddings_params: Extra parameters to pass to the embeddings API


    >>> from functools import partial
    >>> dimensions = 3
    >>> embeddings_ = partial(embeddings, dimensions=dimensions, validate=True)

    # Test with a single word
    >>> text = "vector"
    >>> result = embeddings_(text)
    >>> result  # doctest: +SKIP
    [-0.4096039831638336, 0.3794299364089966, -0.8296127915382385]
    >>> isinstance(result, list)
    True
    >>> len(result) == dimensions == 3
    True

    # Test with a list of words
    >>> texts = ["semantic", "vector"]
    >>> result = embeddings_(texts)
    >>> isinstance(result, list)
    True
    >>> len(result)
    2

    Two vectors; one for each word. Note that the second vector is the vector of
    "vector", which we've seen before.
    >>> result[1]  # doctest: +SKIP
    [-0.4096039831638336, 0.3794299364089966, -0.8296127915382385]

    >>> len(result[1]) == dimensions == 3
    True


    # Test with a dictionary of words
    >>> texts = {"adj": "semantic", "noun": "vector"}
    >>> result = embeddings_(texts)
    >>> isinstance(result, dict)
    True
    >>> len(result)
    2
    >>> result["noun"]  # doctest: +SKIP
    [-0.4096039831638336, 0.3794299364089966, -0.8296127915382385]
    >>> len(result["adj"]) == len(result["noun"]) == dimensions == 3
    True

    """
    texts, texts_type, keys = _prepare_embeddings_args(
        validate, texts, valid_text_getter, model
    )

    if client is None:
        client = mk_client()

    # Note: validate set to False, as we've already validated
    vectors = [
        x.embedding
        for x in client.embeddings.create(
            input=texts,
            model=model,
            dimensions=dimensions,
            **extra_embeddings_params,
        ).data
    ]

    if texts_type is Mapping:
        return {k: v for k, v in zip(keys, vectors)}
    elif texts_type is str:
        return vectors[0]
    else:
        return vectors


import time
from dataclasses import dataclass


def random_custom_id(prefix='custom_id-', suffix=''):
    """Make a random custom_id by using the current time in nanoseconds"""
    return f"{prefix}{int(time.time() * 1e9)}{suffix}"


# @dataclass
# class EmbeddingsMaker:
#     texts: TextOrTexts,

#     custom_id: str = None,
#     validate: Optional[Union[bool, Callable]] = True,
#     valid_text_getter=_raise_if_any_invalid,
#     model=DFLT_EMBEDDINGS_MODEL,
#     client=None,
#     dimensions: Optional[int] = NOT_GIVEN,
#     **extra_embeddings_params,


def _rm_not_given_values(d):
    return {k: v for k, v in d.items() if v is not NOT_GIVEN}


def mk_batch_file_embeddings_task(
    texts: TextOrTexts,
    *,
    custom_id: Optional[str] = None,
    validate: Optional[Union[bool, Callable]] = True,
    valid_text_getter=_raise_if_any_invalid,
    # client=None,
    model=DFLT_EMBEDDINGS_MODEL,
    dimensions: Optional[int] = NOT_GIVEN,
    **extra_embeddings_params,
):
    # Make a random custom_id if not provided
    if custom_id is None:
        custom_id = random_custom_id('embeddings_batch_id-')

    texts, texts_type, keys = _prepare_embeddings_args(
        validate, texts, valid_text_getter, model
    )
    body = _rm_not_given_values(
        dict(
            input=texts,
            model=model,
            dimensions=dimensions,
        )
    )
    task = {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/embeddings",
        "body": body,
    }
    return task


import tempfile
from pathlib import Path
import json


def mk_embeddings_batch_file(
    texts, *, purpose='batch', client=None, embeddings_params: Union[dict, tuple] = ()
):
    client = client or mk_client()

    embeddings_params = dict(embeddings_params)
    task_dict = mk_batch_file_embeddings_task(texts, **embeddings_params)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jsonl', mode='w')
    Path(temp_file.name).write_text(json.dumps(task_dict))
    batch_input_file = client.files.create(file=open(temp_file.name, 'rb'), purpose=purpose)
    batch_input_file._local_filepath = temp_file.name
    batch_input_file._task_dict = task_dict
    return batch_input_file


def _prepare_embeddings_args(validate, texts, valid_text_getter, model):
    if validate:
        texts = validate_texts_for_embeddings(texts, valid_text_getter, model)

    texts, texts_type, keys = normalize_text_input(texts)

    return texts, texts_type, keys


def validate_texts_for_embeddings(
    texts, valid_text_getter, model=DFLT_EMBEDDINGS_MODEL
):
    validate = partial(text_is_valid, model=model)
    validation_vector = validate(texts)
    texts = valid_text_getter(validation_vector, texts=texts)
    return texts


def normalize_text_input(texts: TextOrTexts) -> TextStrings:
    """Ensures the type of texts is an iterable of strings"""
    if isinstance(texts, str):
        return [texts], str, None  # Single string case
    elif isinstance(texts, Mapping):
        return texts.values(), Mapping, list(texts.keys())
    elif isinstance(texts, Iterable):
        return texts, Iterable, None  # Iterable case
    else:
        raise ValueError("Input type not supported")


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
    # Normalize the input
    texts, texts_type, keys = normalize_text_input(texts)

    # Set the maximum tokens allowed if not provided
    max_tokens = max_tokens or model_information_dict[model]['max_input']

    # Define the validation logic for a single text
    def is_text_valid(text, token_count):
        if not text:
            return False
        if token_count:
            if token_count is True:
                token_count = num_tokens(text, model=model)
            return token_count <= max_tokens
        return True

    # Handle the validation for different input types
    if isinstance(token_count, Iterable):
        results = map(is_text_valid, texts, token_count)
    else:
        results = map(partial(is_text_valid, token_count=token_count), texts)

    if texts_type is Mapping:
        return {
            k: v for k, v in zip(keys, results)
        }  # Return a mapping if input was a mapping
    elif texts_type is str:
        return next(
            results
        )  # Return the boolean directly if the input was a single string
    else:
        return list(results)  # Return the list of booleans for an iterable of strings


# df['ada_embedding'] = df.combined.apply(lambda x: get_embedding(x, model='text-embedding-3-small'))
# df.to_csv('output/embedded_1k_reviews.csv', index=False)
