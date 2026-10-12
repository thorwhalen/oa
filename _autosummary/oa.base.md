# oa.base

Base oa functionality

### Functions

| `chat`([prompt, model, messages, audio, ...])                                                    |                                                                                 |
|--------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------|
| `complete`(prompt[, model, best_of, echo, ...])                                                  |                                                                                 |
| [`compute_price`](#oa.base.compute_price)(model[, num_input_tokens, ...])   | Compute the price of a model given the number of input and output tokens        |
| `dalle`(prompt[, n, size, background, model, ...])                                               |                                                                                 |
| [`embeddings`](#oa.base.embeddings)([texts, batch_size, egress, ...])    | Get embeddings for a text or texts.                                             |
| [`list_engine_ids`](#oa.base.list_engine_ids)([pattern])                      | List the available engine IDs.                                                  |
| [`model_information`](#oa.base.model_information)(model, information)           | Get information about a model                                                   |
| [`normalize_text_input`](#oa.base.normalize_text_input)(texts)                     | Ensures the type of texts is an iterable of strings                             |
| [`text_is_valid`](#oa.base.text_is_valid)(texts[, token_count, model, ...]) | Check if text (a string or iterable of strings) is/are valid for a given model. |
| `validate_texts_for_embeddings`(texts, ...[, ...])                                               |                                                                                 |

### oa.base.compute_price(model, num_input_tokens=None, num_output_tokens=None)

Compute the price of a model given the number of input and output tokens

### oa.base.embeddings(texts=None, \*, batch_size=2048, egress=None, batch_callback=None, validate=True, valid_text_getter=<function \_raise_if_any_invalid>, model='text-embedding-3-small', client=None, dimensions=NOT_GIVEN, encoding_format=<openai.Omit object>, user=<openai.Omit object>, extra_headers=None, extra_query=None, extra_body=None, timeout=NOT_GIVEN)

Get embeddings for a text or texts.

* **Parameters:**
  * **texts** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`KT`), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]) – A string, an iterable of strings, or a dictionary of strings
  * **egress** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – A function that takes the embeddings and returns the desired output.
    If None, the output will be a list of embeddings. If False, no output will be
    returned, so the batch_callback had better be set to accumulate the results.
  * **batch_callback** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`int`](https://docs.python.org/3/builtins/functions.html#int), [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`list`](https://docs.python.org/3/builtins/stdtypes.html#list)]], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – A function that is called after each batch of embeddings is
    computed. This can be used for logging, saving, etc.
    One common use case is to save the intermediate results, in files, database,
    or in a list. This can be useful if you’re worried about the process failing
    and want to be able to resume from where you left off instead of having
    to start over (wasting time, and money).
    To accumulate the results in a list, you can set `results = []` and then
    use a lambda function like this:
    `batch_callback = lambda i, batch: results.extend(batch)`.
  * **validate** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool) | [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – If True, validate the text(s) before getting embeddings
  * **valid_text_getter** – A function that gets valid texts from the input texts
  * **model** – The model to use for embeddings
  * **client** – The OpenAI client to use
  * **dimensions** ([`int`](https://docs.python.org/3/builtins/functions.html#int) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – If given will reduce the dimensions of the full size embedding
    vectors to that size
  * **batch_size** ([`int`](https://docs.python.org/3/builtins/functions.html#int) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – The maximum number of texts to send in a single request
  * **extra_embeddings_params** – Extra parameters to pass to the embeddings API

```pycon
>>> from functools import partial
>>> dimensions = 3
>>> embeddings_ = partial(embeddings, dimensions=dimensions, validate=True)
```

Test with a single word:

```pycon
>>> text = "vector"
>>> result = embeddings_(text)
>>> result
[-0.4096039831638336, 0.3794299364089966, -0.8296127915382385]
>>> isinstance(result, list)
True
>>> len(result) == dimensions == 3
True
```

### Test with a list of words

```pycon
>>> texts = ["semantic", "vector"]
>>> result = embeddings_(texts)
>>> isinstance(result, list)
True
>>> len(result)
2
```

Two vectors; one for each word. Note that the second vector is the vector of
“vector”, which we’ve seen before.

```pycon
>>> result[1]
[-0.4096039831638336, 0.3794299364089966, -0.8296127915382385]
```

```pycon
>>> len(result[1]) == dimensions == 3
True
```

### Test with a dictionary of words

```pycon
>>> texts = {"adj": "semantic", "noun": "vector"}
>>> result = embeddings_(texts)
>>> isinstance(result, dict)
True
>>> len(result)
2
>>> result["noun"]
[-0.4096039831638336, 0.3794299364089966, -0.8296127915382385]
>>> len(result["adj"]) == len(result["noun"]) == dimensions == 3
True
```

If you don’t specify `texts`, you will get a “partial” function that you can
use later to compute embeddings for texts. This is useful if you want to
set some parameters (like `dimensions`, `validate`, etc.) and then use the
resulting function to compute embeddings for different texts later on.

```pycon
>>> embeddings_with_10_dimensions = embeddings_(texts=None, dimensions=10)
>>> isinstance(embeddings_with_10_dimensions, Callable)
True
```

### oa.base.list_engine_ids(pattern=None)

List the available engine IDs. Optionally filter by a regex pattern.

### oa.base.model_information(model, information)

Get information about a model

### oa.base.normalize_text_input(texts)

Ensures the type of texts is an iterable of strings

* **Return type:**
  [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

### oa.base.text_is_valid(texts, token_count=True, , model='text-embedding-3-small', max_tokens=None)

Check if text (a string or iterable of strings) is/are valid for a given model.

Text is valid if

- it is not empty
- the number of tokens in the text is less than or equal to the max_tokens

* **Parameters:**
  * **texts** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`KT`), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]) – a string or an iterable of strings
  * **token_count** – Specification of the token count of the text or texts
  * **model** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – The model to use for token count
  * **max_tokens** – The maximum number of tokens allowed by the model

If token_count is an integer, it will check if it is less than or equal to the
`max_tokens`.

If token_count is True, it will compute the number of tokens in the text using the
model specified by `model` and check if it is less than or equal to `max_tokens`.

If token_count is False, it will not check the token count.

If token_count is an iterable, it will apply the same mechanism as above, to each
text in `texts` and the corresponding token count in `token_count`.
This means both texts and token_count(s) must be of the same length.

### Examples

```pycon
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
```
