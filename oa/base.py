"""Base oa functionality"""

import re
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
from oa.openai_specs import prompt_path

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
    "davinci": "text-davinci-003",
    "ada v2": "text-embedding-ada-002",
}

_model_information_aliases = {
    "max_tokens": "max_input",
    "price": "price_per_million_tokens",
}


# TODO: Literal type of models and information?
def model_information(model, information):
    """Get information about a model"""
    if model in _model_id_aliases:
        model = _model_id_aliases[model]
    if information in _model_information_aliases:
        information = _model_information_aliases[information]

    if model not in model_information_dict:
        raise ValueError(f"Unknown model: {model}")
    if information not in model_information_dict[model]:
        raise ValueError(f"Unknown information: {information}")

    return model_information_dict[model][information]


model_information.model_information_dict = model_information_dict


# TODO: Parse more info and complete this function
#     (see https://github.com/thorwhalen/oa/discussions/8#discussioncomment-9138661)
def compute_price(
    model: str, num_input_tokens: int = None, num_output_tokens: Optional[int] = None
):
    """Compute the price of a model given the number of input and output tokens"""
    assert num_output_tokens is None, "num_output_tokens not yet implemented"
    if num_input_tokens is None:
        return partial(compute_price, model)
    price_per_million_tokens = model_information(model, "price_per_million_tokens")
    return price_per_million_tokens * (num_input_tokens / 1_000_000)


compute_price.model_information_dict = model_information_dict

prompt_dalle_path = partial(prompt_path, prefix=djoin("dalle"))
prompt_davinci_path = partial(prompt_path, prefix=djoin("davinci"))

# TODO: Use oa.openai_specs sig to provide good signatures


@Sig.replace_kwargs_using(openai.completions.create)
def complete(prompt, model=None, **complete_params):
    if "engine" in complete_params:
        model = complete_params.pop("engine")
    model = model or getattr(complete, "engine", DFLT_ENGINE)
    text_resp = openai.completions.create(model=model, prompt=prompt, **complete_params)
    return text_resp.choices[0].text


complete.engine = DFLT_ENGINE


@Sig.replace_kwargs_using(openai.chat.completions.create)
def _raw_chat(prompt=None, model=DFLT_MODEL, *, messages=None, **chat_params):
    if not ((prompt is None) ^ (messages is None)):
        raise ValueError("Either prompt or messages must be specified, but not both.")
    if prompt is not None:
        messages = [{"role": "user", "content": prompt}]
    return openai.chat.completions.create(messages=messages, model=model, **chat_params)


# chat_sig = sig.CreateChatCompletionRequest
# chat_sig = chat_sig.ch_defaults(model=DFLT_MODEL, messages=None)
# chat_sig = Sig([Param(name='prompt', default=None, annotation=str), *chat_sig.params])


@Sig.replace_kwargs_using(_raw_chat)
def chat(prompt=None, *, model=DFLT_MODEL, messages=None, **chat_params):
    resp = _raw_chat(prompt=prompt, model=model, messages=messages, **chat_params)
    # TODO: Make attr and item getters more robust (use glom?)
    return resp.choices[0].message.content


chat.raw = _raw_chat


@Sig.replace_kwargs_using(openai.images.generate)
def _raw_dalle(prompt, n=1, size="512x512", **image_create_params):
    return openai.images.generate(prompt=prompt, n=n, size=size, **image_create_params)


@Sig.replace_kwargs_using(_raw_dalle)
def dalle(prompt, n=1, size="512x512", **image_create_params):
    r = _raw_dalle(prompt=prompt, n=n, size=size, **image_create_params)
    return r.data[0].url


def list_engine_ids(pattern: Optional[str] = None):
    """List the available engine IDs. Optionally filter by a regex pattern."""
    models_list = mk_client().models.list()
    model_ids = [x.id for x in models_list.data]
    if pattern:
        # filter model_ids by pattern, taken to be a regex pattern
        pattern = re.compile(pattern)
        model_ids = list(filter(pattern.search, model_ids))
    return model_ids


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
                "Invalid text(s):\n",
                "\n".join(
                    item
                    for is_valid, item in zip(validation_vector, texts)
                    if not is_valid
                ),
            )
        raise ValueError("Some of the texts are invalid")
    return texts


# --------------------------------------------------------------------------------------
# Embeddings
"""
There are (at least) three ways to compute embeddings, which are more or less ideal 
depending on the situation.

* One by one, locally and serially (that is, we wait for the response of the request 
    before sending another). This is **VERY** slow, and you don't want to do this with 
    a lot of data. But it has the advantage of being simple and straightforward, and, 
    if one of your segments has a problem, you'll know easily exactly which one does.
* In batches, locally and serially. 
* In batches, remotely, in parallel, asynchronously. Advantages here are that it's 
    remote, so you're not hogging down the resources of your computer, and the remote 
    server will manage the persistence, status, etc. It's also cheaper (with OpenAI, 
    at the time of writing this, half the price). But it's more complex, and though 
    often faster to get your response every time I've ever tried, you are "only" 
    guaranteed getting your batch jobs within 24h of launching them.
"""

from openai import NOT_GIVEN
from typing import Union, List, Any
from oa.util import chunk_iterable, mk_local_files_saves_callback

# from collections.abc import Mapping, Iterable

extra_embeddings_params = Sig(openai.embeddings.create) - {"input", "model"}


# TODO: Added a lot of options, but not really clean. Should be cleaned up.
# TODO: The dict-versus-list types should be handled more cleanly!
# TODO: Integrate the batch API way of doing embeddings
# TODO: Batches should be able to be done in paralel, with async/await
# TODO: Make a few useful validation_callback functions
#    (e.g. return list or dict where invalid texts are replaced with None)
#    (e.g. return dict containing only valid texts (if input was list, uses indices as keys)
@Sig.replace_kwargs_using(extra_embeddings_params)
def embeddings(
    texts: TextOrTexts = None,
    *,
    batch_size: Optional[int] = 2048,  # found on unofficial OpenAI API docs
    egress: Optional[str] = None,
    batch_callback: Optional[Callable[[int, List[list]], Any]] = None,
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
    :param egress: A function that takes the embeddings and returns the desired output.
        If None, the output will be a list of embeddings. If False, no output will be
        returned, so the batch_callback had better be set to accumulate the results.
    :param batch_callback: A function that is called after each batch of embeddings is
        computed. This can be used for logging, saving, etc.
        One common use case is to save the intermediate results, in files, database,
        or in a list. This can be useful if you're worried about the process failing
        and want to be able to resume from where you left off instead of having
        to start over (wasting time, and money).
        To accumulate the results in a list, you can set `results = []` and then
        use a lambda function like this:
        `batch_callback = lambda i, batch: results.extend(batch)`.
    :param validate: If True, validate the text(s) before getting embeddings
    :param valid_text_getter: A function that gets valid texts from the input texts
    :param model: The model to use for embeddings
    :param client: The OpenAI client to use
    :param dimensions: If given will reduce the dimensions of the full size embedding
        vectors to that size
    :param batch_size: The maximum number of texts to send in a single request
    :param extra_embeddings_params: Extra parameters to pass to the embeddings API

    >>> from functools import partial
    >>> dimensions = 3
    >>> embeddings_ = partial(embeddings, dimensions=dimensions, validate=True)

    Test with a single word:

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

    If you don't specify `texts`, you will get a "partial" function that you can
    use later to compute embeddings for texts. This is useful if you want to
    set some parameters (like `dimensions`, `validate`, etc.) and then use the
    resulting function to compute embeddings for different texts later on.
    >>> embeddings_with_10_dimensions = embeddings_(texts=None, dimensions=10)
    >>> isinstance(embeddings_with_10_dimensions, Callable)
    True

    """
    if texts is None:
        _kwargs = locals()
        _ = _kwargs.pop("texts")
        extra_embeddings_params = _kwargs.pop("extra_embeddings_params", {})
        return partial(embeddings, **_kwargs, **extra_embeddings_params)

    if egress is False:
        assert batch_callback, (
            "batch_callback must be provided if egress is False: "
            "It will be the batch_callback's responsibility to accumulate the batches "
            "of embeddings!"
        )

    if batch_callback == "temp_files":  # an extra not annotated or documented
        # convenience to get intermediary results saved to file
        batch_callback = mk_local_files_saves_callback()
    batch_callback = batch_callback or (lambda i, batch: None)
    assert callable(batch_callback) & (
        len(Sig(batch_callback)) >= 2
    ), "batch_callback must be callable with at least two arguments (i, batch)"
    texts, texts_type, keys = _prepare_embeddings_args(
        validate, texts, valid_text_getter, model
    )

    if texts_type is str and egress is not None:
        raise ValueError("egress should be None if texts is a single string")

    if client is None:
        client = mk_client()

    def _embeddings_batches():
        for i, batch in enumerate(chunk_iterable(texts, batch_size)):
            batch_result = _embeddings_batch(
                batch,
                model=model,
                client=client,
                dimensions=dimensions,
                **extra_embeddings_params,
            )
            batch_callback(i, batch_result)
            yield from batch_result

    # vectors = chain.from_iterable(_embeddings_batches())
    vectors = _embeddings_batches()

    if egress is False:
        # the batch
        for _ in vectors:
            pass
    else:
        if egress is None:
            if issubclass(texts_type, Mapping):
                egress = lambda vectors: {k: v for k, v in zip(keys, vectors)}
            else:
                egress = list

        if texts_type is str:
            return next(iter(vectors))  # there's one and only one (note: no egress)
        else:
            return egress(vectors)


def _embeddings_batch(
    texts: TextOrTexts,
    model=DFLT_EMBEDDINGS_MODEL,
    client=None,
    dimensions: Optional[int] = NOT_GIVEN,
    **extra_embeddings_params,
):

    # Note: validate set to False, as we've already validated
    return [
        x.embedding
        for x in client.embeddings.create(
            input=texts,
            model=model,
            dimensions=dimensions,
            **extra_embeddings_params,
        ).data
    ]


# --------------------------------------------------------------------------------------
# embeddings utils


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

    # TODO: Too many places where we have to check if it's a dict or a list. Need to clean up.
    if isinstance(validation_vector, Mapping):
        keys = list(validation_vector.keys())
        validation_vector = list(validation_vector.values())
    else:
        keys = None

    texts = valid_text_getter(validation_vector, texts=texts)

    # TODO: Too many places where we have to check if it's a dict or a list. Need to clean up.
    if isinstance(validation_vector, Mapping):
        return {k: v for k, v in zip(keys, texts)}
    else:
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
    max_tokens = max_tokens or model_information_dict[model]["max_input"]

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
