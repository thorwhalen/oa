"""Base oa functionality"""

from itertools import chain
from functools import partial

from types import SimpleNamespace
from i2 import Sig, Param

from oa.util import openai, djoin, mk_client
from oa.openai_specs import prompt_path, sig

api = None
# TODO: Expand this to include all the other endpoints (automatically?)
# api = SimpleNamespace(
#     chat=sig.CreateChatCompletionRequest(openai.chat.completions.create),
#     complete=sig.CreateCompletionRequest(openai.completions.create),
#     dalle=sig.CreateImageRequest(openai.images.generate),
# )

prompt_dalle_path = partial(prompt_path, prefix=djoin('dalle'))
prompt_davinci_path = partial(prompt_path, prefix=djoin('davinci'))

# TODO: Understand the model/engine thing better and merge defaults if possible
DFLT_ENGINE = 'gpt-3.5-turbo-instruct'
DFLT_MODEL = 'gpt-3.5-turbo'

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


def embeddings(texts, model="text-embedding-3-small", client=None):
    if client is None:
        client = mk_client()
    if isinstance(texts, str):
        texts = [texts]
        return client.embeddings.create(input=texts, model=model).data[0].embedding
    else:
        return [
            x.embedding for x in client.embeddings.create(input=texts, model=model).data
        ]


# df['ada_embedding'] = df.combined.apply(lambda x: get_embedding(x, model='text-embedding-3-small'))
# df.to_csv('output/embedded_1k_reviews.csv', index=False)
