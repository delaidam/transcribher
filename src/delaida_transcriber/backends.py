"""Pluggable transcription backends.

The local faster-whisper backend is the only one implemented. The seam exists so
a hosted backend (ElevenLabs Scribe) can be added without touching the CLI, the
web app, or the dictation command -- they all go through ``create_backend``.
"""

from pathlib import Path
from typing import Protocol, runtime_checkable

from delaida_transcriber.config import Settings
from delaida_transcriber.models import FileTranscription
from delaida_transcriber.transcriber import WhisperTranscriber

SUPPORTED_BACKENDS = {"local", "elevenlabs"}


@runtime_checkable
class TranscriptionBackend(Protocol):
    """Anything that can turn a media file into a transcript."""

    async def transcribe_file(
        self, path: Path, language: str | None = None
    ) -> FileTranscription: ...


def create_backend(settings: Settings | None = None) -> TranscriptionBackend:
    settings = settings or Settings()
    if settings.backend == "local":
        return WhisperTranscriber(settings)
    if settings.backend == "elevenlabs":
        raise NotImplementedError(
            "The ElevenLabs Scribe backend is not built yet. Set STT_BACKEND=local, "
            "or ask for the hosted backend to be added -- it needs ELEVENLABS_API_KEY "
            "in .env and bills per hour of audio."
        )
    supported = ", ".join(sorted(SUPPORTED_BACKENDS))
    raise ValueError(f"Unknown STT_BACKEND {settings.backend!r}; expected one of: {supported}.")
