"""Application service shared by the CLI and web interface."""

import asyncio
from pathlib import Path

from delaida_transcriber.models import FileTranscription
from delaida_transcriber.transcriber import WhisperTranscriber

SUPPORTED_LANGUAGE_HINTS = {"auto", "bs", "en"}
SUPPORTED_SUFFIXES = {".ogg", ".mp3", ".mp4", ".m4a"}


class TranscriptionService:
    def __init__(self, transcriber: WhisperTranscriber, max_upload_bytes: int) -> None:
        self.transcriber = transcriber
        self.max_upload_bytes = max_upload_bytes
        self._lock = asyncio.Lock()

    async def transcribe(self, path: Path, language: str = "auto") -> FileTranscription:
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            formats = ", ".join(sorted(SUPPORTED_SUFFIXES))
            raise ValueError(f"Only these media formats are supported: {formats}.")
        if not path.is_file():
            raise ValueError("The selected file does not exist.")
        if language not in SUPPORTED_LANGUAGE_HINTS:
            raise ValueError("Language must be auto, bs, or en.")
        if path.stat().st_size > self.max_upload_bytes:
            raise ValueError("The file is larger than the configured upload limit.")

        async with self._lock:
            return await self.transcriber.transcribe_file(
                path, language=None if language == "auto" else language
            )
