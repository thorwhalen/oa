from functools import partial
import pytest
from oa.base import embeddings


# Note: Moved to doctests
# def test_embeddings():
#     dimensions = 3
#     embeddings_ = partial(embeddings, dimensions=dimensions, validate=True)
#     # Test with a single word
#     text = "vector"
#     result = embeddings_(text)
#     assert isinstance(result, list)
#     assert len(result) == dimensions

#     # Test with a list of words
#     texts = ["semantic", "vector"]
#     result = embeddings_(texts)
#     assert isinstance(result, list)
#     assert len(result) == len(texts) == 2
#     assert isinstance(result[0], list)
#     assert len(result[0]) == dimensions
#     assert isinstance(result[1], list)
#     assert len(result[1]) == dimensions

#     # Test with a dictionary of words
#     texts = {"adj": "semantic", "noun": "vector"}
#     result = embeddings_(texts)
#     assert isinstance(result, dict)
#     assert len(result) == len(texts) == 2
#     assert isinstance(result["adj"], list)
#     assert len(result["adj"]) == dimensions
#     assert isinstance(result["noun"], list)
#     assert len(result["noun"]) == dimensions

