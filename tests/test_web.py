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
        "Only these media formats are supported: .m4a, .mp3, .mp4, .ogg, .wav."
    )


class FakeTranscriber:
    async def transcribe_file(self, path: object, language: str | None = None) -> FileTranscription:
        return FileTranscription("hello", "en", 0.9, [])


def test_phone_web_accepts_m4a_upload() -> None:
    service = TranscriptionService(FakeTranscriber(), max_upload_bytes=100)
    client = TestClient(create_app(Settings(model="base", device="cpu"), service=service))

    response = client.post(
        "/transcribe",
        files={"file": ("recording.m4a", b"audio", "audio/mp4")},
        data={"language": "en"},
    )

    assert response.status_code == 200
    assert response.json()["filename"] == "recording.m4a"
