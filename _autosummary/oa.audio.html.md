# oa.audio

OpenAI Audio tools

### Functions

| [`text_to_speech`](#oa.audio.text_to_speech)(text, \*, input[, model, ...])   | Convert text to speech using OpenAI's TTS API.                |
|--------------------------------------------------------------------------------------------------|---------------------------------------------------------------|
| [`transcribe`](#oa.audio.transcribe)(audio, \*, file[, model, ...])       | Transcribe an audio or video file using OpenAI's Whisper API. |
| [`transcription_to_srt`](#oa.audio.transcription_to_srt)(transcription, \*[, ...])  | Convert a transcription dict to SRT format string.            |
| [`tts`](#oa.audio.tts)(text, \*, input[, model, voice, ...])       | Convert text to speech using OpenAI's TTS API.                |

### oa.audio.text_to_speech(text, \*, input, model='tts-1', voice='alloy', response_format='mp3', client=None, instructions=<openai.Omit object>, speed=<openai.Omit object>, stream_format=<openai.Omit object>, extra_headers=None, extra_query=None, extra_body=None, timeout=NOT_GIVEN)

Convert text to speech using OpenAI’s TTS API.

* **Parameters:**
  * **text** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Text to convert to speech
  * **model** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – TTS model to use
  * **voice** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Voice to use (alloy, echo, fable, onyx, nova, shimmer)
  * **response_format** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Audio format (‘mp3’, ‘opus’, ‘aac’, ‘flac’, ‘wav’, ‘pcm’)
  * **client** (`Union`[`OpenAI`, [`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict), [`None`](https://docs.python.org/3/builtins/constants.html#None)]) – OpenAI client instance, API key string, config dict, or None
  * **kwargs** – Additional parameters for the API
* **Returns:**
  Audio content as bytes

#### NOTE
Actual API calls require a valid API key

### oa.audio.transcribe(audio, \*, file, model='whisper-1', response_format='json', language=None, client=None, chunking_strategy=<openai.Omit object>, include=<openai.Omit object>, keywords=<openai.Omit object>, known_speaker_names=<openai.Omit object>, known_speaker_references=<openai.Omit object>, languages=<openai.Omit object>, prompt=<openai.Omit object>, stream=<openai.Omit object>, temperature=<openai.Omit object>, timestamp_granularities=<openai.Omit object>, extra_headers=None, extra_query=None, extra_body=None, timeout=NOT_GIVEN)

Transcribe an audio or video file using OpenAI’s Whisper API.

* **Parameters:**
  * **audio_file_path** – Path to the audio/video file to transcribe
  * **model** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Transcription model to use
  * **response_format** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Format for the response (‘json’, ‘text’, ‘srt’, ‘verbose_json’)
  * **language** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Optional ISO-639-1 language code (e.g., ‘en’, ‘fr’)
  * **client** (`Union`[`OpenAI`, [`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict), [`None`](https://docs.python.org/3/builtins/constants.html#None)]) – OpenAI client instance, API key string, config dict, or None
  * **kwargs** – Additional parameters for the API
* **Returns:**
  Dict with ‘text’, ‘segments’, ‘language’, and ‘raw_response’

#### NOTE
Actual API calls require a valid audio file and API key

### oa.audio.transcription_to_srt(transcription, , segment_duration=None)

Convert a transcription dict to SRT format string.

* **Parameters:**
  * **transcription** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)) – Dict with ‘text’ and optionally ‘segments’
  * **segment_duration** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – If segments missing, duration per generated segment
* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)
* **Returns:**
  SRT formatted string

```pycon
>>> result = transcription_to_srt({'text': 'Hello world'})
>>> 'Hello world' in result
True
```

### oa.audio.tts(text, \*, input, model='tts-1', voice='alloy', response_format='mp3', client=None, instructions=<openai.Omit object>, speed=<openai.Omit object>, stream_format=<openai.Omit object>, extra_headers=None, extra_query=None, extra_body=None, timeout=NOT_GIVEN)

Convert text to speech using OpenAI’s TTS API.

* **Parameters:**
  * **text** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Text to convert to speech
  * **model** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – TTS model to use
  * **voice** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Voice to use (alloy, echo, fable, onyx, nova, shimmer)
  * **response_format** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Audio format (‘mp3’, ‘opus’, ‘aac’, ‘flac’, ‘wav’, ‘pcm’)
  * **client** (`Union`[`OpenAI`, [`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict), [`None`](https://docs.python.org/3/builtins/constants.html#None)]) – OpenAI client instance, API key string, config dict, or None
  * **kwargs** – Additional parameters for the API
* **Returns:**
  Audio content as bytes

#### NOTE
Actual API calls require a valid API key
