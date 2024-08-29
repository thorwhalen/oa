"""Types for oa """

from typing import Any, List, Optional
from pydantic import BaseModel



# --------------------------------------------------------------------------------------
# BatchRequest (e.g. embeddings)


class BatchRequestBody(BaseModel):
    input: List[str]
    model: str


# main
class BatchRequest(BaseModel):
    custom_id: str
    method: str
    url: str
    body: BatchRequestBody


# --------------------------------------------------------------------------------------
# OpenAI Responses


class Datum(BaseModel):
    object: str
    index: int
    embedding: List[float]


class Usage(BaseModel):
    prompt_tokens: int
    total_tokens: int


class ResponseBody(BaseModel):
    object: str
    data: List[Datum]
    model: str
    usage: Usage


class Response(BaseModel):
    status_code: int
    request_id: str
    body: ResponseBody


# main
class ResponseRoot(BaseModel):
    id: str
    custom_id: str
    response: Response
    error: Any

