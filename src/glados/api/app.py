from dataclasses import dataclass
import io
import os
from typing import Literal

import requests
from litestar import Litestar, post
from litestar.config.cors import CORSConfig
from litestar.response import Stream

from pathlib import Path
from ..core.engine import GladosConfig
from .log import structlog_plugin
from .tts import write_glados_audio_file

Voice = Literal["glados"]
ResponseFormat = Literal["mp3", "wav", "ogg"]


@dataclass
class RequestData:
    input: str
    model: str = "glados"  # Placeholder; LLM model is taken from config
    voice: Voice = "glados"
    response_format: ResponseFormat = "mp3"
    speed: float = 1.0


CONTENT_TYPES: dict[ResponseFormat, str] = {"mp3": "audio/mpeg", "wav": "audio/wav", "ogg": "audio/ogg"}


def _load_config() -> GladosConfig:
    """Load Glados configuration for LLM settings.

    Tries env var GLADOS_CONFIG_PATH first, then falls back to the repository configs/glados_config.yaml.
    """
    cfg_path = os.environ.get("GLADOS_CONFIG_PATH")
    if cfg_path:
        return GladosConfig.from_yaml(cfg_path)

    # Fallback to repo root relative to this file: src/glados/api/app.py -> project root
    project_root = Path(__file__).resolve().parents[3]
    return GladosConfig.from_yaml(str(project_root / "configs" / "glados_config.yaml"))


def _llm_complete(user_input: str, config: GladosConfig) -> str:
    """Send text to the configured LLM and return assistant reply text.

    Attempts to support both OpenAI-compatible and Ollama-compatible APIs.
    """
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"

    messages = config.to_chat_messages() + [{"role": "user", "content": user_input}]

    payload = {
        "model": config.llm_model,
        "messages": messages,
        "stream": False,
    }

    resp = requests.post(str(config.completion_url), headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    # OpenAI-style
    if isinstance(data, dict) and "choices" in data:
        choice = data.get("choices", [{}])[0] or {}
        msg = (choice.get("message") or {}).get("content")
        if msg:
            return str(msg)
    # Ollama-style
    if isinstance(data, dict):
        msg = (data.get("message") or {}).get("content")
        if msg:
            return str(msg)
        # Some variants may return a top-level 'response'
        if data.get("response"):
            return str(data["response"])

    raise ValueError("Unrecognized LLM response format")


@post("/v1/audio/speech")
async def create_speech(data: RequestData) -> Stream:
    """
    Generate speech audio from input text (direct TTS, no LLM).

    Parameters:
        data: The request data containing input text and speech parameters

    Returns:
        Stream: Stream of bytes data containing the generated speech
    """
    # TODO: Handle other voices and speed
    buffer = io.BytesIO()
    write_glados_audio_file(buffer, data.input, format=data.response_format)
    buffer.seek(0)
    return Stream(
        buffer,
        headers={
            "Access-Control-Allow-Origin": "http://localhost:3000",
            "content-type": CONTENT_TYPES[data.response_format],
            "content-disposition": f'attachment; filename="speech.{data.response_format}"',
        },
    )


@post("/v1/core/answer")
async def glados_answer(data: RequestData) -> Stream:
    """
    Send GLaDOS a text message: it will be passed to the LLM, then rendered via GLaDOS TTS.

    Parameters:
        data: The request data containing input text and speech parameters

    Returns:
        Stream: Stream of bytes data containing the generated speech (LLM → TTS)
        Additionally returns the assistant text via headers:
        - X-Glados-Text: UTF-8 text (sanitized and possibly truncated)
        - X-Glados-Text-Base64: Base64-encoded UTF-8 of the full text
    """
    buffer = io.BytesIO()

    try:
        config = _load_config()
        assistant_text = _llm_complete(data.input, config)
    except Exception:
        # Keep responses useful even if LLM fails
        assistant_text = (
            "An error occurred while contacting my language module."
            " Consider checking the LLM service configuration."
        )

    # Prepare headers with text response as well
    # Sanitize for header safety and truncate to a reasonable size
    text_header = assistant_text.replace("\r", " ").replace("\n", " ").strip()
    MAX_HEADER_LEN = 1024
    if len(text_header) > MAX_HEADER_LEN:
        text_header = text_header[: MAX_HEADER_LEN - 3] + "..."

    # TODO: Handle other voices and speed
    write_glados_audio_file(buffer, assistant_text, format=data.response_format)
    buffer.seek(0)
    return Stream(
        buffer,
        headers={
            "Access-Control-Allow-Origin": "http://localhost:3000",
            "Access-Control-Expose-Headers": "X-Glados-Text, X-Glados-Text-Base64",
            "X-Glados-Text": text_header,
            "content-type": CONTENT_TYPES[data.response_format],
            "content-disposition": f'attachment; filename="speech.{data.response_format}"',
        },
    )


# --- The CORS Configuration ---
# This is where the magic happens.
cors_config = CORSConfig(
    allow_origins=["https://glados-web-chat.web.app", "http://localhost:3000"],
    allow_methods=["GET", "POST"],  # Specify methods you want to allow
    allow_credentials=True,  # If you need to send cookies/auth headers
)

app = Litestar([create_speech, glados_answer], plugins=[structlog_plugin], cors_config=cors_config)
