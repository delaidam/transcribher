from pathlib import Path

from fastapi.testclient import TestClient

from delaida_transcriber.config import Settings
from delaida_transcriber.models import FileTranscription
from delaida_transcriber.service import TranscriptionService
from delaida_transcriber.web import create_app


def test_phone_web_health_and_upload_validation() -> None:
    client = TestClient(create_app(Settings(model="base", device="cpu")))

    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/").status_code == 200

    response = client.post(
        "/transcribe",
        files={"file": ("notes.txt", b"not audio", "text/plain")},
        data={"language": "auto"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Only these media formats are supported: .m4a, .mp3, .mp4, .ogg, .wav, .webm."
    )


class FakeTranscriber:
    async def transcribe_file(self, path: object, language: str | None = None) -> FileTranscription:
        return FileTranscription("hello", "en", 0.9, [])


def test_phone_web_accepts_m4a_upload() -> None:
    service = TranscriptionService(FakeTranscriber())
    client = TestClient(create_app(Settings(model="base", device="cpu"), service=service))

    response = client.post(
        "/transcribe",
        files={"file": ("recording.m4a", b"audio", "audio/mp4")},
        data={"language": "en"},
    )

    assert response.status_code == 200
    assert response.json()["filename"] == "recording.m4a"


def test_phone_web_accepts_a_browser_recording() -> None:
    """What the record button sends: MediaRecorder's WebM/Opus, named for the
    container it actually is, since the suffix is all the endpoint can check."""
    service = TranscriptionService(FakeTranscriber())
    client = TestClient(create_app(Settings(model="base", device="cpu"), service=service))

    response = client.post(
        "/transcribe",
        files={"file": ("dictation.webm", b"audio", "audio/webm")},
        data={"language": "auto"},
    )

    assert response.status_code == 200
    assert response.json()["filename"] == "dictation.webm"


class ReadingTranscriber:
    """Opens the upload by path, the way faster-whisper does."""

    async def transcribe_file(self, path: Path, language: str | None = None) -> FileTranscription:
        return FileTranscription(f"read {len(path.read_bytes())} bytes", "en", 0.9, [])


def test_the_upload_can_be_opened_by_the_transcriber() -> None:
    """The uploaded bytes reach a file that something else can open.

    Held open by NamedTemporaryFile this failed on Windows for every upload,
    with "Permission denied" on the temporary file the server had just written,
    and no test caught it because the fakes never opened the path.
    """
    service = TranscriptionService(ReadingTranscriber())
    client = TestClient(create_app(Settings(model="base", device="cpu"), service=service))

    response = client.post(
        "/transcribe",
        files={"file": ("dictation.webm", b"audio bytes", "audio/webm")},
        data={"language": "auto"},
    )

    assert response.status_code == 200
    assert response.json()["text"] == "read 11 bytes"


def test_the_page_offers_recording_and_upload() -> None:
    client = TestClient(create_app(Settings(model="base", device="cpu")))

    page = client.get("/").text

    assert 'id="record"' in page
    assert "MediaRecorder" in page
    assert 'type="file"' in page


def test_phone_web_rejects_upload_over_the_limit() -> None:
    """The cap belongs to the upload path; CLI runs on local files are not capped."""
    service = TranscriptionService(FakeTranscriber())
    settings = Settings(model="base", device="cpu", max_upload_mb=1)
    client = TestClient(create_app(settings, service=service))

    response = client.post(
        "/transcribe",
        files={"file": ("recording.m4a", b"x" * (1024 * 1024 + 1), "audio/mp4")},
        data={"language": "en"},
    )

    assert response.status_code == 413
