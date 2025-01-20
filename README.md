# oa
Python interface to OpenAi

To install:	```pip install oa```

- [oa](#oa)
- [Usage](#usage)
  - [A collection of prompt-enabled functions](#a-collection-of-prompt-enabled-functions)
    - [PromptFuncs](#promptfuncs)
  - [Functionalizing prompts](#functionalizing-prompts)
  - [Enforcing json formatted outputs](#enforcing-json-formatted-outputs)
  - [Just-do-it: A minimal-boilerplate facade to OpenAI stuff](#just-do-it-a-minimal-boilerplate-facade-to-openai-stuff)
  - [Raw form - When you need to be closer to the metal](#raw-form---when-you-need-to-be-closer-to-the-metal)


# Usage

Sure, you can do many things in English now with our new AI superpowers, but still, to be able to really reuse and compose your best prompts, you had better parametrize them -- that is, distill them down to the minimal necessary interface. The function.

What `oa` does for you is enable you to easily -- really easily -- harness the newly available super-powers of AI from python. 

Below, you'll see how 

See notebooks:
* [oa - An OpenAI facade.ipynb](https://github.com/thorwhalen/oa/blob/main/misc/oa%20-%20An%20OpenAI%20facade.ipynb)
* [oa - Making an Aesop fables children's book oa.ipynb](https://github.com/thorwhalen/oa/blob/main/misc/oa%20-%20Making%20an%20Aesop%20fables%20children's%20book%20oa.ipynb)

Below are a few snippets from there. 


## A collection of prompt-enabled functions

One main functionality that `oa` offers is an easy way to define python functions based 
on AI prompts. 
In order to demo these, we've made a few ready-to-use ones, which you can access via
`oa.ask.ai`:

```python
from oa.ask import ai

list(ai)
```

    ['define_jargon', 'suggest_names', ..., 'make_synopsis']

These are the names of functions automatically generated from a (for now small) folder of prompt templates. 

These functions all have propert signatures:

```python
import inspect
print(inspect.signature(ai.suggest_names))
```

(*, thing, n='30', min_length='1', max_length='15')


```python
answer = ai.suggest_names(
    thing="""
    A python package that provides python functions to do things,
    enabled by prompts sent to an OpenAI engine.
    """
)
print(answer)
```

    GyroPy
    PromptCore
    FlexiFunc
    ProperPy
    PyCogito
    ...
    PyPrompter
    FuncDomino
    SmartPy
    PyVirtuoso



### PromptFuncs

Above, all we did was scan some local text files that specify prompt templates and make an object that contained the functions they define. We used `oa.PromptFuncs` for that. You can do the same. What `PromptFuncs` uses itself, is a convenient `oa.prompt_function` function that transforms a template into a function. See more details in the next "Functionalizing prompts" section.

Let's just scratch the surface of what `PromptFuncs` can do. For more, you can look at the documentation, including the docs for `ai.prompt_function`.


```python
from oa import PromptFuncs

funcs = PromptFuncs(
    template_store = {
        "haiku": "Write haiku about {subject}. Only output the haiku.",
        "stylize": """
            Reword what I_SAY, using the style: {style:funny}.
            Only output the reworded text.
            I_SAY:
            {something}
        """,
    }
)

list(funcs)
```

    ['haiku', 'stylize']



```python
import inspect
for name in funcs:
    print(f"{name}: {inspect.signature(funcs[name])}")

```

    haiku: (*, subject)
    stylize: (*, something, style='funny')



```python
print(funcs.haiku(subject="The potential elegance of code"))
```

    Code speaks a language,
    Elegant syntax dances,
    Beauty in function.



```python
print(funcs.stylize(something="The mess that is spagetti code!"))
```

    Spaghetti code, the tangled web of chaos!



```python
print(funcs.stylize(something="The mess that is spagetti code!", style="poetic"))
```

    The tangled strands of code, a chaotic tapestry!


We used a `dict` to express our `func_name:template` specification, but note that it can be any `Mapping`. Therefore, you can source `PromptFuncs` with local files (example, using `dol.TextFiles`, like we did), a DB, or anything you can map to a key-value `Mapping` interface.

(We suggest you use the [dol](https://pypi.org/project/dol/) package, and ecosystem, to help out with that.)


## Functionalizing prompts

The `oa.prompt_function` is an easy to use, yet extremely configurable, tool to do that.


```python
from oa import prompt_function

template = """
I'd like you to give me help me understand domain-specific jargon. 
I will give you a CONTEXT and some WORDS. 
You will then provide me with a tab separated table (with columns name and definition)
that gives me a short definition of each word in the context of the context.
Only output the table, with no words before or after it, since I will be parsing the output
automatically.

CONTEXT:
{context}

WORDS:
{words}
"""

define_jargon = prompt_function(template, defaults=dict(context='machine learning'))
```


```python
# Let's look at the signature
import inspect
print(inspect.signature(define_jargon))
```

    (*, words, context='machine learning')



```python
response = define_jargon(words='supervised learning\tunsupervised learning\treinforcement learning')
print(response)
```

    name	definition
    supervised learning	A type of machine learning where an algorithm learns from labeled training data to make predictions or take actions. The algorithm is provided with input-output pairs and uses them to learn patterns and make accurate predictions on new, unseen data.
    unsupervised learning	A type of machine learning where an algorithm learns patterns and structures in input data without any labeled output. The algorithm identifies hidden patterns and relationships in the data to gain insights and make predictions or classifications based on the discovered patterns.
    reinforcement learning	A type of machine learning where an algorithm learns to make a sequence of decisions in an environment to maximize a cumulative reward. The algorithm interacts with the environment, receives feedback in the form of rewards or punishments, and adjusts its actions to achieve the highest possible reward over time.



```python
def table_str_to_dict(table_str, *, newline='\n', sep='   '):
    return dict([x.split('   ') for x in table_str.split('\n')[1:]])

table_str_to_dict(define_jargon(
    words='\n'.join(['allomorph', 'phonology', 'phonotactic constraints']),
    context='linguistics'
))

```


    {'allomorph': 'A variant form of a morpheme that is used in a specific linguistic context, often resulting in different phonetic realizations.',
     'phonology': 'The study of speech sounds and their patterns, including the way sounds are organized and used in a particular language or languages.',
     'phonotactic constraints': 'The rules or restrictions that govern the possible combinations of sounds within a language, specifying what sound sequences are allowed and which ones are not.'}



Check out the many ways you can configure your function with `prompt_function`:


```python
str(inspect.signature(prompt_function)).split(', ')
```




    ['(template',
     '*',
     'defaults: Optional[dict] = None',
     'template_to_names=<function _extract_names_from_format_string at 0x106d20940>',
     'embodier=<function string_format_embodier at 0x106d204c0>',
     'name=None',
     'prompt_func=<function chat at 0x128420af0>',
     'prompt_func_kwargs=None',
     'egress=None)']


## Enforcing json formatted outputs

With some newer models (example, "gpt4o-mini") you can request that only valid 
json be given as a response, or even more: A json obeying a specific schema. 
You control this via the `response_format` argument. 

Let's first use AI to get a json schema for characteristics of a programming language.
That's a json, so why not use the `response_format` with `{"type": "json_object"}` to 
get that schema!


```python
from oa import chat
from oa.util import data_files

# To make sure we get a json schema that is openAI compliant, we'll use an example of 
# one in our prompt to AI to give us one...
example_of_a_openai_json_schema = example_of_a_openai_json_schema = (
    data_files.joinpath('json_schema_example.json').read_text()
)

json_schema_str = chat(
    "Give me the json of a json_schema I can use different characteristics of "
    "programming languages. This schema should be a valid schema to use as a "
    "response_format in the OpenAI API. "
    f"Here's an example:\n{example_of_a_openai_json_schema}", 
    model='gpt-4o-mini',
    response_format={'type': 'json_object'}
)
print(json_schema_str[:500] + '...')
```

```
{
  "name": "programming_language_characteristics",
  "strict": false,
  "schema": {
    "type": "object",
    "properties": {
      "name": {
        "type": "string",
        "description": "The name of the programming language."
      },
      "paradigm": {
        "type": "array",
        "items": {
          "type": "string",
          "description": "Programming paradigms the language follows, e.g., 'object-oriented', 'functional', etc."
        }
      },
      "designed_by": {
        "t...
```

Now we can use this schema to make an AI-enabled python function that will give 
us characteristics of a language, but always using that fixed format.
This also means we'll be able to stick an `egress` to our prompt function, so 
that we always get our output in the form of an already decoded json (a `dict`).

```python
from oa import prompt_function
import json

properties_of_language = prompt_function(
    "Give me a json that describes characteristics of the programming language: {language}.",
    prompt_func=chat, 
    prompt_func_kwargs=dict(
        model='gpt-4o-mini', 
        response_format={
            'type': 'json_schema',
            'json_schema': json.loads(json_schema_str)
        }
    ),
    egress=json.loads
)

info = properties_of_language('Python')
print(f"{type(info)=}\n")

from pprint import pprint
pprint(info)
```

```
type(info)=<class 'dict'>

{'designed_by': ['Guido van Rossum'],
 'first_appeared': 1991,
 'influenced_by': ['ABC', 'C', 'C++', 'Java', 'Modula-3', 'Lisp'],
 'influences': ['Ruby', 'Swift', 'Matlab', 'Go'],
 'latest_release': '3.11.5',
 'name': 'Python',
 'paradigm': ['object-oriented', 'imperative', 'functional', 'procedural'],
 'typing_discipline': 'dynamic',
 'website': 'https://www.python.org'}
```




## Just-do-it: A minimal-boilerplate facade to OpenAI stuff

For the typical tasks you might want to use OpenAI for.

Note there's no "enter API KEY here" code. That's because if you don't have it in the place(s) it'll look for it, it will simply ask you for it, and, with your permission, put it in a hidden file for you, so you don't have to do this every time.


```python
import oa
```


```python
print(oa.complete('chatGPT is a'))
```

     chatbot based on OpenAI's GPT-2, a natural language processing



```python
print(oa.chat('Act as a chatGPT expert. List 5 useful prompt templates'))
```

    Sure, here are 5 useful prompt templates that can be used in a chatGPT session:
    
    1. Can you provide some more details about [topic]?
    - Examples: Can you provide some more details about the symptoms you're experiencing? Or Can you provide some more details about the issue you're facing with the website?
    
    2. How long have you been experiencing [issue]?
    - Examples: How long have you been experiencing the trouble with your internet connection? Or How long have you been experiencing the pain in your back?
    
    3. Have you tried any solutions to resolve [issue]?
    - Examples: Have you tried any solutions to resolve the error message you're seeing? Or Have you tried any solutions to resolve the trouble you're having with the application?
    
    4. What is the specific error message you are receiving?
    - Examples: What is the specific error message you are receiving when you try to log in? Or What is the specific error message you are receiving when you try to submit the form?
    
    5. Is there anything else you would like to add that might be helpful for me to know?
    - Examples: Is there anything else you would like to add that might be helpful for me to know about your situation? Or Is there anything else you would like to add that might be helpful for me to know about the product you are using?



```python
url = oa.dalle('An image of Davinci, pop art style')
print(url)
```

    https://oaidalleapiprodscus.blob.core.windows.net/private/org-AY3lr3H3xB9yPQ0HGR498f9M/user-7ZNCDYLWzP0GT48V6DCiTFWt/img-pNE6fCWGN3eJGj7ycFwZREhi.png?st=2023-04-22T22%3A17%3A03Z&se=2023-04-23T00%3A17%3A03Z&sp=r&sv=2021-08-06&sr=b&rscd=inline&rsct=image/png&skoid=6aaadede-4fb3-4698-a8f6-684d7786b067&sktid=a48cca56-e6da-484e-a814-9c849652bcb3&skt=2023-04-22T21%3A08%3A14Z&ske=2023-04-23T21%3A08%3A14Z&sks=b&skv=2021-08-06&sig=5j6LPVO992R95dllAAjbmOXzS0MORD06Fo8unwtGNl0%3D



```python
from IPython.display import Image

Image(url=url)
```

<img width="608" alt="image" src="https://github.com/thorwhalen/oa/assets/1906276/6e7b2ac4-648c-4ec0-81bf-078208f4ac39">


## Raw form - When you need to be closer to the metal

The `raw` object is a thin layer on top of the `openai` package, which is itself a thin layer over the web requests. 

What was unsatisfactory with the `openai` package is (1) finding the right function, (2) the signature of the function once you found it, and (3) the documentation of the function. 
What raw contains is pointers to the main functionalities (not all available -- yet), with nice signatures and documentation, extracted from the web service openAPI specs themselves. 

For example, to ask chatGPT something, the openai function is `openai.ChatCompletion.create`, or to get simple completions, the function is `openai.Completion.create` whose help is:

```
Help on method create in module openai.api_resources.completion:

create(*args, **kwargs) method of builtins.type instance
    Creates a new completion for the provided prompt and parameters.
    
    See https://platform.openai.com/docs/api-reference/completions/create for a list
    of valid parameters.
```

Not super helpful. It basically tells you to got read the docs elsewhere. 

The corresponding `raw` function is `raw.completion`, and it's help is a bit more like what you'd expect in a python function.



```python
help(oa.raw.chatcompletion)
```

    Help on Wrap in module openai.api_resources.chat_completion:
    
    chatcompletion
        Creates a new chat completion for the provided messages and parameters.
        
                See https://platform.openai.com/docs/api-reference/chat-completions/create
                for a list of valid parameters.
        
        chatcompletion(
                model: str
                messages: List[oa.openai_specs.Message]
                *
                temperature: float = 1
                top_p: float = 1
                n: int = 1
                stream: bool = False
                stop=None
                max_tokens: int = None
                presence_penalty: float = 0
                frequency_penalty: float = 0
                logit_bias: dict = None
                user: str = None
        )
        
        :param model: ID of the model to use. Currently, only `gpt-3.5-turbo` and `gpt-3.5-turbo-0301` are supported.
        
        :param messages: The messages to generate chat completions for, in the [chat format](/docs/guides/chat/introduction).
        
        :param temperature: What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic. We generally recommend altering this or `top_p` but not both.
        
        :param top_p: An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are considered. We generally recommend altering this or `temperature` but not both.
        
        :param n: How many chat completion choices to generate for each input message.
        
        :param stream: If set, partial message deltas will be sent, like in ChatGPT. Tokens will be sent as data-only [server-sent events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#Event_stream_format) as they become available, with the stream terminated by a `data: [DONE]` message.
        
        :param stop: Up to 4 sequences where the API will stop generating further tokens.
        
        :param max_tokens: The maximum number of tokens allowed for the generated answer. By default, the number of tokens the model can return will be (4096 - prompt tokens).
        
        :param presence_penalty: Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics. [See more information about frequency and presence penalties.](/docs/api-reference/parameter-details)
        
        :param frequency_penalty: Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim. [See more information about frequency and presence penalties.](/docs/api-reference/parameter-details)
        
        :param logit_bias: Modify the likelihood of specified tokens appearing in the completion. Accepts a json object that maps tokens (specified by their token ID in the tokenizer) to an associated bias value from -100 to 100. Mathematically, the bias is added to the logits generated by the model prior to sampling. The exact effect will vary per model, but values between -1 and 1 should decrease or increase likelihood of selection; values like -100 or 100 should result in a ban or exclusive selection of the relevant token.
        
        :param user: A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse. [Learn more](/docs/guides/safety-best-practices/end-user-ids).
    



```python
prompt = 'List 5 top prompt engineering tricks to write good prompts for chatGPT'

resp = oa.raw.chatcompletion(
    messages=[
        {"role": "system", "content": "You are an expert at chatGPT"},
        {"role": "user", "content": prompt},
    ],
    model='gpt-3.5-turbo-0301',
    temperature=0.5,
    max_tokens=300
)
resp
```




    <OpenAIObject chat.completion id=chatcmpl-78HMPgn3oy2fuvm6sLCgOsQvnTVYr at 0x11fd467a0> JSON: {
      "choices": [
        {
          "finish_reason": "stop",
          "index": 0,
          "message": {
            "content": "Sure, here are 5 top prompt engineering tricks to write good prompts for chatGPT:\n\n1. Be Specific: Ensure that your prompts are specific and clear. The more specific your prompt, the better the response from chatGPT. Avoid using vague or ambiguous language.\n\n2. Use Open-Ended Questions: Open-ended questions encourage chatGPT to provide more detailed and personalized responses. Avoid using closed-ended questions that can be answered with a simple yes or no.\n\n3. Include Context: Providing context to your prompts helps chatGPT to better understand the topic and provide more relevant responses. Include any necessary background information or details to help guide chatGPT's response.\n\n4. Use Emotion: Including emotion in your prompts can help chatGPT generate more engaging and relatable responses. Consider using prompts that evoke emotions such as happiness, sadness, or excitement.\n\n5. Test and Refine: Experiment with different prompts and evaluate the responses from chatGPT. Refine your prompts based on the quality of the responses and continue to test and improve over time.",
            "role": "assistant"
          }
        }
      ],
      "created": 1682207713,
      "id": "chatcmpl-78HMPgn3oy2fuvm6sLCgOsQvnTVYr",
      "model": "gpt-3.5-turbo-0301",
      "object": "chat.completion",
      "usage": {
        "completion_tokens": 214,
        "prompt_tokens": 36,
        "total_tokens": 250
      }
    }


```python
print(resp['choices'][0]['message']['content'])
```

    Sure, here are 5 top prompt engineering tricks to write good prompts for chatGPT:
    
    1. Be Specific: Ensure that your prompts are specific and clear. The more specific your prompt, the better the response from chatGPT. Avoid using vague or ambiguous language.
    
    2. Use Open-Ended Questions: Open-ended questions encourage chatGPT to provide more detailed and personalized responses. Avoid using closed-ended questions that can be answered with a simple yes or no.
    
    3. Include Context: Providing context to your prompts helps chatGPT to better understand the topic and provide more relevant responses. Include any necessary background information or details to help guide chatGPT's response.
    
    4. Use Emotion: Including emotion in your prompts can help chatGPT generate more engaging and relatable responses. Consider using prompts that evoke emotions such as happiness, sadness, or excitement.
    
    5. Test and Refine: Experiment with different prompts and evaluate the responses from chatGPT. Refine your prompts based on the quality of the responses and continue to test and improve over time.



# ChatDacc: Shared chats parser

The `ChatDacc` class gives you access of the main functionalities of the `chats.py` module.
It provides tools for analyzing and extracting data from shared ChatGPT conversations. 
Here’s an overview of its main features, for more details, see 
[this demo notebook](https://github.com/thorwhalen/oa/blob/main/misc/oa%20-%20chats%20acquisition%20and%20parsing.ipynb)


## Initialize with a URL

Begin by creating a ChatDacc object with a conversation’s shared URL:

```python
from oa.chats import ChatDacc

url = 'https://chatgpt.com/share/6788d539-0f2c-8013-9535-889bf344d7d5'
dacc = ChatDacc(url)
```

## Access Basic Conversation Data

Retrieve minimal metadata (e.g., roles, timestamps, and content):

```python
dacc.basic_turns_data
```

Or directly get it as a Pandas DataFrame:

```python
dacc.basic_turns_df
```

## Explore Full Turn Data

Access all available fields for each message in the conversation:

```python
dacc.turns_data
```

Indexed access simplifies specific turn retrieval:

```python
dacc.turns_data_keys
turn_data = dacc.turns_data[dacc.turns_data_keys[3]]
```

## Extract Metadata

Metadata summarizing the conversation is available through:

```python
dacc.metadata
```

## Extract and Analyze URLs

Identify all URLs referenced within the conversation, including quoted and embedded sources:

```python
urls = dacc.url_data()
```

For richer context, you can include prior levels or retain tracking parameters:

urls_in_context = dacc.url_data(prior_levels_to_include=1, remove_chatgpt_utm=False)

## Get Full JSON

The raw JSON for the entire conversation can be accessed for in-depth analysis:

```python
dacc.full_json_dict
```

This tool simplifies data extraction and analysis from ChatGPT shared conversations, making it ideal for developers, researchers, and data analysts.