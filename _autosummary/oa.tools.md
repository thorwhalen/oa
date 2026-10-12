# oa.tools

Interface tools

### Functions

| [`constrained_answer`](#oa.tools.constrained_answer)(prompt, valid_answers, \*)    | Get an answer from the LLM constrained to a set of valid answers or types.                                                                       |
|---------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------|
| `dflt_factory_key`(f)                                                                             |                                                                                                                                                  |
| `dflt_function_key`(f)                                                                            |                                                                                                                                                  |
| [`ensure_json_type`](#oa.tools.ensure_json_type)(json_type)                      | Ensure that the json type is a string that is a valid json type                                                                                  |
| `extract_parts`(string, pattern)                                                                  |                                                                                                                                                  |
| `identity`(x)                                                                                     |                                                                                                                                                  |
| [`infer_schema_from_verbal_description`](#oa.tools.infer_schema_from_verbal_description)(...[, ...]) | Infer a JSON Schema (`{name, properties, type}`) from a verbal description.                                                                      |
| [`make_generic_json_schema`](#oa.tools.make_generic_json_schema)(json_type)              | Make a generic json schema for a given json type.                                                                                                |
| [`pattern_based_map`](#oa.tools.pattern_based_map)(func, string, pattern[, ...])  | Applies a function to parts of the string that are either matching or non-matching based on a regex pattern, depending on the value of apply_to. |
| [`prompt_function`](#oa.tools.prompt_function)(template, \*[, defaults, ...])   | Convert a string template to a function that will produce a prompt string and ask an LLM (`prompt_func`) to respond to it.                       |
| [`prompt_json_function`](#oa.tools.prompt_json_function)(template[, ...])            | Make prompt functions that return jsons (dicts) with a given schema.                                                                             |
| [`remove_pattern`](#oa.tools.remove_pattern)(string[, pattern_to_remove])      | Returns a where a given regular expression pattern has been removed.                                                                             |
| `string_format_embodier`(template, \*[, ...])                                                     |                                                                                                                                                  |

### Classes

| [`JsonTypes`](#oa.tools.JsonTypes)(\*values)                              |                           |
|---------------------------------------------------------------------------------------------------|---------------------------|
| [`PromptFuncs`](#oa.tools.PromptFuncs)([template_store, function_key, ...]) | Make AI enabled functions |

### *class* oa.tools.JsonTypes(\*values)

Bases: [`Enum`](https://docs.python.org/3/library/enum.html#enum.Enum)

### *class* oa.tools.PromptFuncs(template_store='/home/runner/work/oa/oa/oa/data/templates/chatgpt:.txt, ', \*, function_key=<function <lambda>>, factory_key=<function <lambda>>, factories=(('.txt', <function prompt_function>), ('', <function prompt_function>)), extra_template_kwargs=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Make AI enabled functions

#### funcs_and_sigs()

Return a mapping of function names to signatures

#### print_signatures()

Print signatures of all functions

#### reload()

Reload all functions

### oa.tools.constrained_answer(prompt, valid_answers, , model='gpt-4o-mini', n=1)

Get an answer from the LLM constrained to a set of valid answers or types.

Uses `prompt_json_function` to ensure the LLM returns a valid response based on
constraints.

This can be seen as a facade for some common structured output use cases, as well
as a convenient tool to do response statistics and validation (via n>1).

* **Parameters:**
  * **prompt** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – The question or prompt to ask the LLM
  * **valid_answers** (`Union`[[`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`int`](https://docs.python.org/3/builtins/functions.html#int)], [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`float`](https://docs.python.org/3/builtins/functions.html#float)], [`type`](https://docs.python.org/3/builtins/functions.html#type), [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`float`](https://docs.python.org/3/builtins/functions.html#float), [`float`](https://docs.python.org/3/builtins/functions.html#float)]]) – 

    Can be:
    - list[str]: List of valid string options
    - list[int]: List of valid integer options
    - list[float]: List of valid float options
    - bool: Constrains answer to True or False
    - int: Any integer
    - float: Any number
    - tuple[float, float]: Numerical range (min, max) inclusive
  * **model** – The model to use for the LLM (default: “gpt-4o-mini”)
  * **n** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Number of times to call the LLM (default: 1)
* **Returns:**
  One of the valid answers, respecting the type constraint

### Examples

```pycon
>>> # String options
>>> answer = constrained_answer(
...     "Is Python compiled or interpreted?",
...     ["compiled", "interpreted", "both"]
... )
>>> answer in ["compiled", "interpreted", "both"]
True
```

```pycon
>>> # Boolean
>>> answer = constrained_answer(
...     "Is Python dynamically typed?",
...     bool
... )
>>> isinstance(answer, bool)
True
```

```pycon
>>> # Integer options
>>> answer = constrained_answer(
...     "How many wheels does a car have?",
...     [2, 3, 4, 6, 8]
... )
>>> answer in [2, 3, 4, 6, 8]
True
```

```pycon
>>> # Numerical range
>>> answer = constrained_answer(
...     "What is a reasonable hourly rate for a senior Python developer? (USD)",
...     (50.0, 300.0)
... )
>>> 50.0 <= answer <= 300.0
True
```

### oa.tools.ensure_json_type(json_type)

Ensure that the json type is a string that is a valid json type

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

```pycon
>>> ensure_json_type('string')
'string'
>>> ensure_json_type(str)
'string'
>>> ensure_json_type(object)
Traceback (most recent call last):
...
ValueError: Cannot convert ...
```

### oa.tools.infer_schema_from_verbal_description(verbal_description, \*, prompt_func=<function chat>, prompt_func_kwargs=None, model='gpt-4o-mini')

Infer a JSON Schema (`{name, properties, type}`) from a verbal description.

The LLM backend is **injectable** so this is not hardwired to OpenAI: pass any
`prompt_func` with the `(prompt, **kwargs) -> str` shape (e.g. a
provider-agnostic `aix.chat`) to route the inference through a different
provider, optionally pinning `model` / extra `prompt_func_kwargs`. The
defaults preserve the original behavior (oa’s `chat` on `gpt-4o-mini`).

* **Parameters:**
  * **verbal_description** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Natural-language description of the desired JSON output.
  * **prompt_func** – The LLM callable to use (default: oa’s `chat`). Forwarded to
    [`prompt_json_function()`](#oa.tools.prompt_json_function).
  * **prompt_func_kwargs** – Extra keyword arguments merged into the `prompt_func`
    call (e.g. provider options).
  * **model** – Model id passed to `prompt_func` (pass `None` to let an
    `aix`-style backend resolve its own configured default).
* **Returns:**
  A `dict` with at least `name` and `properties` keys (the inferred
  JSON Schema, formatted for OpenAI “JSON mode”).

### oa.tools.make_generic_json_schema(json_type)

Make a generic json schema for a given json type.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

```pycon
>>> make_generic_json_schema('string')
{'name': 'generic_string_schema', 'schema': {'properties': {'result': {'type': 'string'}}, 'required': ['result']}}
```

### oa.tools.pattern_based_map(func, string, pattern, apply_to='unmatched')

Applies a function to parts of the string that are either matching or non-matching based on a regex pattern,
depending on the value of apply_to.

### Example

```pycon
>>> func = str.upper
>>> string = "the good the ```bad``` and the ugly"
>>> ignore_pattern = r'```.*?```'
>>> pattern_based_map(func, string, ignore_pattern)
'THE GOOD THE ```bad``` AND THE UGLY'
>>> pattern_based_map(func, string, ignore_pattern, 'matched')
'the good the ```BAD``` and the ugly'
```

### oa.tools.prompt_function(template, \*, defaults=None, template_to_names=<function \_extract_names_from_format_string>, template_to_defaults=<function \_extract_defaults_from_format_string>, embodier=<function string_format_embodier>, arg_kinds=None, name='prompt', prompt_func=<function chat>, prompt_func_kwargs=None, ingress=None, egress=None, doc='The function composes a prompt and asks an LLM to respond to it.', module='oa.tools')

Convert a string template to a function that will produce a prompt string
and ask an LLM (`prompt_func`) to respond to it.

* **Parameters:**
  * **template** – A string template with placeholders.
  * **defaults** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – A dictionary of default values for placeholders.
  * **template_to_names** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)) – A function that extracts names from a template.
  * **template_to_defaults** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)) – A function that extracts defaults from a template.
  * **embodier** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)) – A function that converts a template to a function that will
    produce a prompt string.
  * **arg_kinds** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – A dictionary of argument kinds for the function.
  * **name** – The name of the function.
  * **prompt_func** – The function that will be used to ask the LLM to respond to
    the prompt. If None, the output function will only produce the prompt string,
    not ask the LLM to respond to it.
  * **prompt_func_kwargs** – Keyword arguments to pass to `prompt_func`.
  * **ingress** – A function to apply to the input of `prompt_func`.
  * **egress** – A function to apply to the output of `prompt_func`.
  * **doc** – The docstring of the function.
  * **module** – The module of the function.

In the following example, we’ll use the `prompt_func=None` argument to get a
function that simply injects inputs in a prompt template, without actually calling
an AI-enabled `prompt_func`.
Note in this example, how a block of the prompt template string is ignored for
injection purposes, via a triple-backtick marker.

```pycon
>>> prompt_template = '''
... ```
... In this block, all {placeholders} are {igno:red} so that they can appear in prompt.
... ```
... But outside {inputs} are {injected:normally}
... '''
>>> f = prompt_function(prompt_template, prompt_func=None)
>>> from inspect import signature
>>> assert str(signature(f)) == "(inputs, *, injected='normally')"
>>> print(f('INPUTS', injected="INJECTED"))
```

```text
In this block, all {placeholders} are {igno:red} so that they can appear in prompt.
```

But outside INPUTS are INJECTED

### oa.tools.prompt_json_function(template, json_schema='string', \*, defaults=None, embodier=<function string_format_embodier>, arg_kinds=None, name='prompt', prompt_func=<function chat>, prompt_func_kwargs=None, model='gpt-4o-mini', ingress=None, egress=None, doc='The function composes a prompt and asks an LLM to respond to it with json.', module='oa.tools')

Make prompt functions that return jsons (dicts) with a given schema.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### oa.tools.remove_pattern(string, pattern_to_remove=re.compile('\`\`\`.\*?\`\`\`', re.DOTALL))

Returns a where a given regular expression pattern has been removed.

```pycon
>>> string = 'this ```is a``` string ```with several``` backticks'
>>> remove_pattern(string)
'this  string  backticks'
```
