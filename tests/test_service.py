from pathlib import Path

import pytest

from delaida_transcriber.models import FileTranscription
from delaida_transcriber.service import TranscriptionService


class FakeTranscriber:
    def __init__(self) -> None:
        self.calls: list[tuple[Path, str | None]] = []

    async def transcribe_file(self, path: Path, language: str | None = None) -> FileTranscription:
        self.calls.append((path, language))
        return FileTranscription("hello", "en", 0.9, [])


@pytest.mark.asyncio
async def test_service_accepts_ogg_and_converts_auto_to_detection(tmp_path: Path) -> None:
    path = tmp_path / "recording.ogg"
    path.write_bytes(b"ogg")
    transcriber = FakeTranscriber()
    service = TranscriptionService(transcriber, max_upload_bytes=100)

    result = await service.transcribe(path)

    assert result.text == "hello"
    assert transcriber.calls == [(path, None)]


@pytest.mark.asyncio
async def test_service_rejects_wrong_language_extension_and_size(tmp_path: Path) -> None:
    text = tmp_path / "notes.txt"
    text.write_text("x", encoding="utf-8")
    huge = tmp_path / "huge.ogg"
    huge.write_bytes(b"12345")
    service = TranscriptionService(FakeTranscriber(), max_upload_bytes=4)

    with pytest.raises(ValueError, match="Only .ogg"):
        await service.transcribe(text)
    with pytest.raises(ValueError, match="larger"):
        await service.transcribe(huge)
    with pytest.raises(ValueError, match="auto, bs, or en"):
        await service.transcribe(huge, "de")
