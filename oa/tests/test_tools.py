"""Tests for oa.tools — focused on the injectable LLM backend.

These exercise prompt-function composition with a *fake* ``prompt_func`` so they
run offline (no provider call), verifying that the backend is genuinely
injectable (not hardwired to OpenAI).
"""

import json

from oa.tools import (
    prompt_function,
    prompt_json_function,
    infer_schema_from_verbal_description,
)


def test_prompt_function_uses_injected_backend():
    calls = []

    def fake_llm(prompt, **kwargs):
        calls.append((prompt, kwargs))
        return "FAKE:" + prompt

    f = prompt_function("Say hi to {name}", prompt_func=fake_llm)
    out = f(name="Ada")
    assert out == "FAKE:Say hi to Ada"
    assert calls and "Ada" in calls[0][0]


def test_prompt_json_function_uses_injected_backend():
    captured = {}

    def fake_llm(prompt, **kwargs):
        captured["prompt"] = prompt
        captured["kwargs"] = kwargs
        return json.dumps({"answer": 42})

    f = prompt_json_function(
        "Answer about {topic}",
        {"name": "ans", "schema": {"properties": {"answer": {"type": "integer"}}}},
        prompt_func=fake_llm,
        model=None,  # let the backend resolve its own default
    )
    result = f(topic="life")
    assert result == {"answer": 42}
    # the schema is forwarded as response_format, and model is threaded through
    assert captured["kwargs"]["response_format"]["type"] == "json_schema"
    assert "model" in captured["kwargs"]


def test_infer_schema_backend_is_injectable():
    """The named NL→schema helper must route through an injected backend (not OpenAI)."""
    seen = {}

    def fake_llm(prompt, **kwargs):
        seen["model"] = kwargs.get("model")
        return json.dumps(
            {
                "name": "person",
                "properties": {"age": {"type": "integer"}},
                "type": "object",
            }
        )

    schema = infer_schema_from_verbal_description(
        "an object with an integer age",
        prompt_func=fake_llm,
        model=None,
    )
    assert schema["name"] == "person"
    assert schema["properties"]["age"]["type"] == "integer"
    # model=None was threaded through verbatim (backend resolves its own default)
    assert seen["model"] is None
