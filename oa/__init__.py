"""
Python facade to OpenAi functionality.
"""

from oa.util import (
    openai,
    grazed,
    djoin,
    app_data_dir,
    num_tokens,
    model_information_dict,
    utc_int_to_iso_date,
    DFLT_ENGINE,
    DFLT_MODEL,
    DFLT_EMBEDDINGS_MODEL,
)

from oa.base import (
    chat,
    complete,
    dalle,
    api,
    embeddings,
    model_information,
    compute_price,
    text_is_valid,
)
from oa.openai_specs import raw
from oa.tools import (
    prompt_function,  # Make a python function from a prompt template
    PromptFuncs,  # make a collection of AI-enabled functions
    prompt_json_function,  # Make a python function (returning a valid json) from a prompt template
    infer_schema_from_verbal_description,  # Get a schema from a verbal description
)
from oa import ask
from oa.stores import OaStores
from oa.chats import ChatDacc
