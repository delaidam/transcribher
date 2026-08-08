from fastapi.testclient import TestClient

from delaida_transcriber.config import Settings
from delaida_transcriber.web import create_app


def test_phone_web_health_and_upload_validation() -> None:
    client = TestClient(create_app(Settings(model="base", device="cpu", compute_type="int8")))

    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/").status_code == 200

    response = client.post(
        "/transcribe",
        files={"file": ("notes.txt", b"not audio", "text/plain")},
        data={"language": "auto"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Only .ogg, .mp3, .mp4, and .m4a files are supported."
