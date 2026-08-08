"""Local faster-whisper transcription."""

import asyncio
import logging
import warnings
from pathlib import Path
from typing import Any

from faster_whisper import WhisperModel

from delaida_transcriber.config import Settings
from delaida_transcriber.models import FileTranscription, TranscriptionSegment

logger = logging.getLogger(__name__)


class WhisperTranscriber:
    """Lazy-loading local Whisper model shared by all interfaces."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self._model: WhisperModel | None = None

    def _load(self) -> WhisperModel:
        if self._model is None:
            try:
                self._model = WhisperModel(
                    self.settings.model,
                    device=self.settings.device,
                    compute_type=self.settings.compute_type,
                )
            except Exception as error:
                if self.settings.device != "cuda":
                    raise
                message = (
                    f"CUDA model loading failed; falling back to CPU/int8 transcription: {error}"
                )
                logger.warning(message)
                warnings.warn(
                    message,
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._model = WhisperModel(
                    self.settings.model,
                    device="cpu",
                    compute_type="int8",
                )
        return self._model

    def _transcribe_source(self, source: Any, language: str | None) -> FileTranscription:
        segments, info = self._load().transcribe(source, language=language, vad_filter=True)
        collected = [
            TranscriptionSegment(
                start=float(segment.start),
                end=float(segment.end),
                text=segment.text.strip(),
            )
            for segment in segments
            if segment.text.strip()
        ]
        return FileTranscription(
            text=" ".join(segment.text for segment in collected),
            language=getattr(info, "language", None),
            language_probability=getattr(info, "language_probability", None),
            segments=collected,
        )

    async def transcribe_file(self, path: Path, language: str | None = None) -> FileTranscription:
        """Transcribe one audio file; ``None`` asks Whisper to detect language."""
        return await asyncio.to_thread(self._transcribe_source, str(path), language)
