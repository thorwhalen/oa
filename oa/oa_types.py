"""Types for oa"""

from typing import Any, List, Optional, TypeVar, Generic

from pydantic import BaseModel, RootModel, Field

from ju.pydantic_util import is_pydantic_model, is_type_hint

import openai.types as oat

pydantic_models = {k: v for k, v in vars(oat).items() if is_pydantic_model(v)}
type_hints = {k: v for k, v in vars(oat).items() if is_type_hint(v)}


T = TypeVar("T", bound=BaseModel)

# --------------------------------------------------------------------------------------
# JsonL (lists of dicts)


class JsonL(RootModel[List[T]], Generic[T]):
    """
    A generic class for JSONL (JSON Lines) files, which are lists of dictionaries.
    """


class InputText(BaseModel):
    """Used to specify the input data in some OpenAI API endpounts."""

    input: str


InputDataJsonL = JsonL[InputText]


# --------------------------------------------------------------------------------------
# BatchRequest (e.g. embeddings)


class BatchRequestBody(BaseModel):
    input: List[str]
    model: str


# Note: leaf model
class BatchRequest(BaseModel):
    custom_id: str
    method: str
    url: str
    body: BatchRequestBody


# --------------------------------------------------------------------------------------
# OpenAI Responses
from pydantic import BaseModel, Field
from typing import List, TypeVar, Generic, Any

# Define a generic type for Datum
DatumT = TypeVar("DatumT")


# Usage model remains the same
class Usage(BaseModel):
    prompt_tokens: int
    total_tokens: int


# ResponseBody is now parameterized by DatumT
class ResponseBody(BaseModel, Generic[DatumT]):
    object: str
    data: List[DatumT]  # Generic list of DatumT
    model: str
    usage: Usage


class RequestResponse(BaseModel, Generic[DatumT]):
    status_code: int = Field(..., ge=100, le=599)
    request_id: str
    body: ResponseBody[DatumT]


class Response(BaseModel, Generic[DatumT]):
    id: str
    custom_id: str
    response: RequestResponse[DatumT]
    error: Any


from openai.types import Embedding as EmbeddingT

EmbeddingResponse = Response[EmbeddingT]
EmbeddingResponse.__name__ = "EmbeddingResponse"

# --------------------------------------------------------------------------------------
# Extras


def heatmap_of_models_and_their_fields():
    import pandas as pd  # pylint: disable=import-outside-toplevel
    from oplot.matrix import heatmap_sns  # pylint: disable=import-outside-toplevel

    models_and_their_fields = pd.DataFrame(
        [{k: 1 for k in model.model_fields} for model in pydantic_models.values()],
        index=pydantic_models.keys(),
    ).transpose()

    return heatmap_sns(models_and_their_fields, figsize=13)


# # --------------------------------------------------------------------------------------

# class Datum(BaseModel):
#     object: str
#     index: int
#     embedding: List[float]


# class Usage(BaseModel):
#     prompt_tokens: int
#     total_tokens: int


# class ResponseBody(BaseModel):
#     object: str
#     data: List[Datum]
#     model: str
#     usage: Usage


# class Response(BaseModel):
#     status_code: int = Field(..., ge=100, le=599)
#     request_id: str
#     body: ResponseBody


# # Note: leaf model
# class ResponseRoot(BaseModel):
#     id: str
#     custom_id: str
#     response: Response
#     error: Any
