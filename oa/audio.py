"""OpenAI Audio tools"""

from functools import partial
from typing import Optional, Literal
from collections.abc import Callable, Iterable

from i2 import Sig

from oa.util import openai, mk_client, ensure_oa_client, OaClientSpec

# --------------------------------------------------------------------------------------
# Defaults

DFLT_TRANSCRIPTION_MODEL = "whisper-1"
DFLT_TTS_MODEL = "tts-1"
DFLT_TTS_VOICE = "alloy"
DFLT_RESPONSE_FORMAT = "json"
DFLT_AUDIO_FORMAT = "mp3"

# --------------------------------------------------------------------------------------
# Transcription


def _parse_transcription_response(resp, response_format: str) -> dict:
    """Parse the transcription API response into a standardized dict."""
    if response_format in ("json", "verbose_json"):
        full_text = resp.get("text", "")
        segments = resp.get("segments", None)
        language = resp.get("language", None)
    elif response_format == "srt":
        full_text = resp
        segments = None
        language = None
    else:
        full_text = resp if isinstance(resp, str) else str(resp)
        segments = None
        language = None

    return {
        "text": full_text,
        "segments": segments,
        "language": language,
        "raw_response": resp,
    }


@Sig.replace_kwargs_using(openai.audio.transcriptions.create)
def transcribe(
    audio: str,
    *,
    model: str = DFLT_TRANSCRIPTION_MODEL,
    response_format: str = DFLT_RESPONSE_FORMAT,
    language: Optional[str] = None,
    client: OaClientSpec = None,
    **kwargs,
) -> dict:
    """
    Transcribe an audio or video file using OpenAI's Whisper API.

    :param audio_file_path: Path to the audio/video file to transcribe
    :param model: Transcription model to use
    :param response_format: Format for the response ('json', 'text', 'srt', 'verbose_json')
    :param language: Optional ISO-639-1 language code (e.g., 'en', 'fr')
    :param client: OpenAI client instance, API key string, config dict, or None
    :param kwargs: Additional parameters for the API
    :return: Dict with 'text', 'segments', 'language', and 'raw_response'

    Note: Actual API calls require a valid audio file and API key
    """
    client = ensure_oa_client(client)

    with open(audio, "rb") as f:
        resp = client.audio.transcriptions.create(
            file=f,
            model=model,
            response_format=response_format,
            language=language,
            **kwargs,
        )

    return _parse_transcription_response(resp, response_format)


# --------------------------------------------------------------------------------------
# SRT (SubRip) utilities


def _format_srt_timestamp(seconds: float) -> str:
    """
    Format seconds as SRT timestamp (HH:MM:SS,mmm).

    >>> _format_srt_timestamp(0)
    '00:00:00,000'
    >>> _format_srt_timestamp(3661.5)
    '01:01:01,500'
    """
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hrs:02}:{mins:02}:{secs:02},{millis:03}"


def _segments_to_srt_entries(segments):
    """Generate SRT entries from segments with timestamps."""
    for idx, seg in enumerate(segments, start=1):
        start_str = _format_srt_timestamp(seg["start"])
        end_str = _format_srt_timestamp(seg["end"])
        text = seg["text"].strip()
        yield f"{idx}\n{start_str} --> {end_str}\n{text}\n"


def _text_to_equal_duration_segments(
    text: str, segment_duration: float, *, words_per_segment: int = None
):
    """
    Split text into segments of roughly equal duration.

    >>> segments = list(_text_to_equal_duration_segments("one two three four", 10.0))
    >>> segments[0]['text']
    'one two three four'
    >>> segments[0]['end']
    10.0
    """
    words = text.split()
    if not words:
        return

    if words_per_segment is None:
        # Simple heuristic: assume uniform word rate
        words_per_segment = max(1, len(words) // max(1, int(len(words) / 10)))

    idx = 0
    for i in range(0, len(words), words_per_segment):
        chunk = " ".join(words[i : i + words_per_segment])
        start_t = idx * segment_duration
        end_t = start_t + segment_duration
        yield {"start": start_t, "end": end_t, "text": chunk}
        idx += 1


def transcription_to_srt(transcription: dict, *, segment_duration: float = None) -> str:
    """
    Convert a transcription dict to SRT format string.

    :param transcription: Dict with 'text' and optionally 'segments'
    :param segment_duration: If segments missing, duration per generated segment
    :return: SRT formatted string

    >>> result = transcription_to_srt({'text': 'Hello world'})
    >>> 'Hello world' in result
    True
    """
    segments = transcription.get("segments")

    if segments:
        # Use existing timestamped segments
        entries = _segments_to_srt_entries(segments)
    elif segment_duration:
        # Generate equal-duration segments from text
        text = transcription["text"].strip()
        synthetic_segments = _text_to_equal_duration_segments(text, segment_duration)
        entries = _segments_to_srt_entries(synthetic_segments)
    else:
        # Single segment spanning the whole text
        text = transcription["text"].strip()
        entries = ["1\n00:00:00,000 --> 99:59:59,000\n" + text + "\n"]

    return "\n".join(entries)


# --------------------------------------------------------------------------------------
# Text-to-Speech


@Sig.replace_kwargs_using(openai.audio.speech.create)
def text_to_speech(
    text: str,
    *,
    model: str = DFLT_TTS_MODEL,
    voice: str = DFLT_TTS_VOICE,
    response_format: str = DFLT_AUDIO_FORMAT,
    client: OaClientSpec = None,
    **kwargs,
) -> bytes:
    """
    Convert text to speech using OpenAI's TTS API.

    :param text: Text to convert to speech
    :param model: TTS model to use
    :param voice: Voice to use (alloy, echo, fable, onyx, nova, shimmer)
    :param response_format: Audio format ('mp3', 'opus', 'aac', 'flac', 'wav', 'pcm')
    :param client: OpenAI client instance, API key string, config dict, or None
    :param kwargs: Additional parameters for the API
    :return: Audio content as bytes

    Note: Actual API calls require a valid API key
    """
    client = ensure_oa_client(client)

    response = client.audio.speech.create(
        input=text, model=model, voice=voice, response_format=response_format, **kwargs
    )

    # Handle different response types
    if hasattr(response, "content"):
        return response.content
    elif isinstance(response, bytes):
        return response
    else:
        return response.read()


# Alias for consistency with other modules
tts = text_to_speech
