# oa.openai_specs

Tools to extract specs from the openai interface

See raw schemas by doing

```pycon
>>> from oa.openai_specs import schemas
>>> sorted(schemas)
['AssistantFileObject', 'AssistantObject', 'AssistantToolsCode', 'AssistantToolsFunction', 'AssistantToolsRetrieval', 'ChatCompletionFunctionCallOption',
'ChatCompletionFunctions', 'ChatCompletionMessageToolCall', 'ChatCompletionMessageToolCallChunk', 'ChatCompletionMessageToolCalls',
'ChatCompletionNamedToolChoice', 'ChatCompletionRequestAssistantMessage', 'ChatCompletionRequestFunctionMessage', 'ChatCompletionRequestMessage',
'ChatCompletionRequestMessageContentPart', 'ChatCompletionRequestMessageContentPartImage', 'ChatCompletionRequestMessageContentPartText',
'ChatCompletionRequestSystemMessage', 'ChatCompletionRequestToolMessage', 'ChatCompletionRequestUserMessage', 'ChatCompletionResponseMessage',
'ChatCompletionRole', 'ChatCompletionStreamResponseDelta', 'ChatCompletionTool', 'ChatCompletionToolChoiceOption', 'CompletionUsage',
'CreateAssistantFileRequest', 'CreateAssistantRequest', 'CreateChatCompletionFunctionResponse', 'CreateChatCompletionImageResponse',
'CreateChatCompletionRequest', 'CreateChatCompletionResponse', 'CreateChatCompletionStreamResponse', 'CreateCompletionRequest',
'CreateCompletionResponse', 'CreateEditRequest', 'CreateEditResponse', 'CreateEmbeddingRequest', 'CreateEmbeddingResponse', 'CreateFileRequest',
'CreateFineTuneRequest', 'CreateFineTuningJobRequest', 'CreateImageEditRequest', 'CreateImageRequest', 'CreateImageVariationRequest', 'CreateMessageRequest',
'CreateModerationRequest', 'CreateModerationResponse', 'CreateRunRequest', 'CreateSpeechRequest', 'CreateThreadAndRunRequest', 'CreateThreadRequest',
'CreateTranscriptionRequest', 'CreateTranscriptionResponse', 'CreateTranslationRequest', 'CreateTranslationResponse', 'DeleteAssistantFileResponse',
'DeleteAssistantResponse', 'DeleteFileResponse', 'DeleteMessageResponse', 'DeleteModelResponse', 'DeleteThreadResponse', 'Embedding', 'Error',
'ErrorResponse', 'FineTune', 'FineTuneEvent', 'FineTuningJob', 'FineTuningJobEvent', 'FunctionObject', 'FunctionParameters', 'Image',
'ImagesResponse', 'ListAssistantFilesResponse', 'ListAssistantsResponse', 'ListFilesResponse', 'ListFineTuneEventsResponse', 'ListFineTunesResponse',
'ListFineTuningJobEventsResponse', 'ListMessageFilesResponse', 'ListMessagesResponse', 'ListModelsResponse', 'ListPaginatedFineTuningJobsResponse',
'ListRunStepsResponse', 'ListRunsResponse', 'ListThreadsResponse', 'MessageContentImageFileObject', 'MessageContentTextAnnotationsFileCitationObject',
'MessageContentTextAnnotationsFilePathObject', 'MessageContentTextObject', 'MessageFileObject', 'MessageObject', 'Model', 'ModifyAssistantRequest',
'ModifyMessageRequest', 'ModifyRunRequest', 'ModifyThreadRequest', 'OpenAIFile', 'RunObject', 'RunStepDetailsMessageCreationObject', 'RunStepDetailsToolCallsCodeObject',
'RunStepDetailsToolCallsCodeOutputImageObject', 'RunStepDetailsToolCallsCodeOutputLogsObject', 'RunStepDetailsToolCallsFunctionObject', 'RunStepDetailsToolCallsObject',
'RunStepDetailsToolCallsRetrievalObject', 'RunStepObject', 'RunToolCallObject', 'SubmitToolOutputsRunRequest', 'ThreadObject']
>>> schema = schemas['CreateCompletionRequest']
>>> schema['properties']
 {'model': {'description': 'ID of the model to use...
```

Get resulting signatures by doing.

```pycon
>>> from oa.openai_specs import sig
>>> sig.CreateCompletionRequest)
<Sig (model: str, prompt='<|endoftext|>', *, seed: int, best_of: int = 1, echo: bool = False,
frequency_penalty: float = 0, logit_bias: dict = None, logprobs: int = None, max_tokens: int = None,
n: int = 1, presence_penalty: float = 0, stop=None, stream: bool = False, suffix: str = None,
temperature: float = 1, top_p: float = 1, user: str = None)>
```

Or see argument descriptions in rst format by doing:

```pycon
>>> from oa.openai_specs import schema_to_rst_argument_descriptions, schemas
>>> schema_to_rst_argument_descriptions(schemas['CreateCompletionRequest'])
```

```text
```

### Functions

| [`get_openapi_spec_dict`](#oa.openai_specs.get_openapi_spec_dict)([openapi_spec_url, ...])   | Get the dict of the openapi spec                       |
|---------------------------------------------------------------------------------------------------|--------------------------------------------------------|
| [`merge_keys_to_values`](#oa.openai_specs.merge_keys_to_values)(d[, key_name])              | Merge the keys of a dict into the values of the dict.  |
| [`normalized_file_name`](#oa.openai_specs.normalized_file_name)(prompt)                     | Convert prompt to a normalized valid file/folder name  |
| `prompt_path`(prompt[, prefix])                                                                   |                                                        |
| [`properties_to_annotation`](#oa.openai_specs.properties_to_annotation)(name, props)            | Get annotations from a schema                          |
| [`properties_to_param_dict`](#oa.openai_specs.properties_to_param_dict)(name, props)            | Get all but kind fields of a Parameter object          |
| [`schema_to_docs`](#oa.openai_specs.schema_to_docs)(name, schema[, prefix, ...])      | Get the docs for a schema                              |
| [`schema_to_rst_argument_descriptions`](#oa.openai_specs.schema_to_rst_argument_descriptions)(schema)      | Yield rst-formatted argument descriptions for a schema |
| [`schema_to_signature`](#oa.openai_specs.schema_to_signature)(schema)                      | Get a signature from a schema                          |
| `schemas_df`(schema)                                                                              |                                                        |

### Classes

| [`Message`](#oa.openai_specs.Message)   |    |
|------------------------------------------------------------|----|
| `SpecNames`()                                              |    |

### *class* oa.openai_specs.Message

Bases: [`TypedDict`](https://docs.python.org/3/library/typing.html#typing.TypedDict)

### oa.openai_specs.get_openapi_spec_dict(openapi_spec_url='https://raw.githubusercontent.com/thorwhalen/oa/main/misc/openapi.yaml', , refresh=False, expand_refs=True)

Get the dict of the openapi spec

### oa.openai_specs.merge_keys_to_values(d, key_name='key')

Merge the keys of a dict into the values of the dict.

```pycon
>>> d = {'a': {'b': 1}, 'c': {'d': 2, 'e': 3}}
>>> dict(merge_keys_to_values(d))
{'a': {'key': 'a', 'b': 1}, 'c': {'key': 'c', 'd': 2, 'e': 3}}
```

Useful when, for example, you want to make a table containing both keys and values.

### oa.openai_specs.normalized_file_name(prompt)

Convert prompt to a normalized valid file/folder name

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

```pycon
>>> normalized_file_name("This is a prompt")
'this is a prompt'
>>> normalized_file_name("This is: a PROMPT!  (with punctuation)")
'this is a prompt with punctuation'
```

### oa.openai_specs.properties_to_annotation(name, props)

Get annotations from a schema

### oa.openai_specs.properties_to_param_dict(name, props)

Get all but kind fields of a Parameter object

### oa.openai_specs.schema_to_docs(name, schema, prefix='', line_prefix='\\\\t')

Get the docs for a schema

```pycon
>>> from oa.openai_specs import specs
>>> schema = specs['components']['schemas']['CreateChatCompletionRequest']
>>> print(schema_to_docs('chatcompletion', schema))
chatcompletion(...
```

* **Parameters:**
  **model** – ID of the model to use. Currently, only …

### oa.openai_specs.schema_to_rst_argument_descriptions(schema, process_description=<function \_clean_up_whitespace>)

Yield rst-formatted argument descriptions for a schema

```pycon
>>> from oa.openai_specs import specs
>>> schema = specs['components']['schemas']['CreateChatCompletionRequest']
>>> print(*schema_to_rst_argument_descriptions(schema), '\n')
:param model: ID of the model to use. Currently, only ...
```

`process_description` is a function that will be applied to each schema description
(so you can clean it up, or add links, etc.). By default, it will replace all
whitespaces with single space, and strip.

### oa.openai_specs.schema_to_signature(schema)

Get a signature from a schema

```pycon
>>> from oa.openai_specs import specs
>>> schema = specs['components']['schemas']['CreateChatCompletionRequest']
>>> print(str(schema_to_signature(schema)))
(model: str, messages: List[openai_specs.Message], *, temperature: float = 1, ...
```
