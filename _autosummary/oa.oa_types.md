# oa.oa_types

Types for oa

### Functions

| `heatmap_of_models_and_their_fields`()   |    |
|------------------------------------------|----|

### Classes

| [`BatchRequest`](#oa.oa_types.BatchRequest)(\*\*data)        |                                                                                |
|--------------------------------------------------------------------------------|--------------------------------------------------------------------------------|
| [`BatchRequestBody`](#oa.oa_types.BatchRequestBody)(\*\*data)    |                                                                                |
| [`EmbeddingResponse`](#oa.oa_types.EmbeddingResponse)(\*\*data)   |                                                                                |
| [`InputDataJsonL`](#oa.oa_types.InputDataJsonL)                |                                                                                |
| [`InputText`](#oa.oa_types.InputText)(\*\*data)           | Used to specify the input data in some OpenAI API endpounts.                   |
| [`JsonL`](#id0)([root])                 | A generic class for JSONL (JSON Lines) files, which are lists of dictionaries. |
| `JsonL[InputText]`([root])                                                     | A generic class for JSONL (JSON Lines) files, which are lists of dictionaries. |
| [`RequestResponse`](#oa.oa_types.RequestResponse)(\*\*data)     |                                                                                |
| [`Response`](#id2)(\*\*data)            |                                                                                |
| [`ResponseBody`](#oa.oa_types.ResponseBody)(\*\*data)        |                                                                                |
| [`Response[Embedding]`](#oa.oa_types.EmbeddingResponse)(\*\*data) |                                                                                |
| [`Usage`](#oa.oa_types.Usage)(\*\*data)               |                                                                                |

### *class* oa.oa_types.BatchRequest(\*\*data)

Bases: `BaseModel`

#### model_config *: [ClassVar](https://docs.python.org/3/library/typing.html#typing.ClassVar)[ConfigDict]* *= {}*

Configuration for the model, should be a dictionary conforming to [`ConfigDict`][pydantic.config.ConfigDict].

### *class* oa.oa_types.BatchRequestBody(\*\*data)

Bases: `BaseModel`

#### model_config *: [ClassVar](https://docs.python.org/3/library/typing.html#typing.ClassVar)[ConfigDict]* *= {}*

Configuration for the model, should be a dictionary conforming to [`ConfigDict`][pydantic.config.ConfigDict].

### *class* oa.oa_types.EmbeddingResponse(\*\*data)

Bases: [`Response`](#id2)

#### model_config *: [ClassVar](https://docs.python.org/3/library/typing.html#typing.ClassVar)[ConfigDict]* *= {}*

Configuration for the model, should be a dictionary conforming to [`ConfigDict`][pydantic.config.ConfigDict].

### oa.oa_types.InputDataJsonL

alias of `JsonL[InputText]`

### *class* oa.oa_types.InputText(\*\*data)

Bases: `BaseModel`

Used to specify the input data in some OpenAI API endpounts.

#### model_config *: [ClassVar](https://docs.python.org/3/library/typing.html#typing.ClassVar)[ConfigDict]* *= {}*

Configuration for the model, should be a dictionary conforming to [`ConfigDict`][pydantic.config.ConfigDict].

### *class* oa.oa_types.JsonL(root=PydanticUndefined, \*\*data)

Bases: `RootModel[list[TypeVar]]`, [`Generic`](https://docs.python.org/3/library/typing.html#typing.Generic)[`T`]

A generic class for JSONL (JSON Lines) files, which are lists of dictionaries.

#### model_config *: [ClassVar](https://docs.python.org/3/library/typing.html#typing.ClassVar)[ConfigDict]* *= {}*

Configuration for the model, should be a dictionary conforming to [`ConfigDict`][pydantic.config.ConfigDict].

### *class* oa.oa_types.JsonL(root=PydanticUndefined, \*\*data)

Bases: `RootModel[list[TypeVar]]`, [`Generic`](https://docs.python.org/3/library/typing.html#typing.Generic)[`T`]

A generic class for JSONL (JSON Lines) files, which are lists of dictionaries.

#### model_config *: [ClassVar](https://docs.python.org/3/library/typing.html#typing.ClassVar)[ConfigDict]* *= {}*

Configuration for the model, should be a dictionary conforming to [`ConfigDict`][pydantic.config.ConfigDict].

### *class* oa.oa_types.RequestResponse(\*\*data)

Bases: `BaseModel`, [`Generic`](https://docs.python.org/3/library/typing.html#typing.Generic)[`DatumT`]

#### model_config *: [ClassVar](https://docs.python.org/3/library/typing.html#typing.ClassVar)[ConfigDict]* *= {}*

Configuration for the model, should be a dictionary conforming to [`ConfigDict`][pydantic.config.ConfigDict].

### *class* oa.oa_types.Response(\*\*data)

Bases: `BaseModel`, [`Generic`](https://docs.python.org/3/library/typing.html#typing.Generic)[`DatumT`]

#### model_config *: [ClassVar](https://docs.python.org/3/library/typing.html#typing.ClassVar)[ConfigDict]* *= {}*

Configuration for the model, should be a dictionary conforming to [`ConfigDict`][pydantic.config.ConfigDict].

### *class* oa.oa_types.Response(\*\*data)

Bases: `BaseModel`, [`Generic`](https://docs.python.org/3/library/typing.html#typing.Generic)[`DatumT`]

#### model_config *: [ClassVar](https://docs.python.org/3/library/typing.html#typing.ClassVar)[ConfigDict]* *= {}*

Configuration for the model, should be a dictionary conforming to [`ConfigDict`][pydantic.config.ConfigDict].

### *class* oa.oa_types.ResponseBody(\*\*data)

Bases: `BaseModel`, [`Generic`](https://docs.python.org/3/library/typing.html#typing.Generic)[`DatumT`]

#### model_config *: [ClassVar](https://docs.python.org/3/library/typing.html#typing.ClassVar)[ConfigDict]* *= {}*

Configuration for the model, should be a dictionary conforming to [`ConfigDict`][pydantic.config.ConfigDict].

### *class* oa.oa_types.Usage(\*\*data)

Bases: `BaseModel`

#### model_config *: [ClassVar](https://docs.python.org/3/library/typing.html#typing.ClassVar)[ConfigDict]* *= {}*

Configuration for the model, should be a dictionary conforming to [`ConfigDict`][pydantic.config.ConfigDict].
