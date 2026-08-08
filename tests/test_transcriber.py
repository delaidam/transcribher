import pytest

from delaida_transcriber.config import Settings
from delaida_transcriber.transcriber import WhisperTranscriber


def test_cuda_load_failure_falls_back_to_cpu(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []

    class FakeModel:
        def __init__(self, model: str, device: str, compute_type: str) -> None:
            calls.append((device, compute_type))
            if device == "cuda":
                raise RuntimeError("CUDA unavailable")

    monkeypatch.setattr("delaida_transcriber.transcriber.WhisperModel", FakeModel)
    transcriber = WhisperTranscriber(Settings(model="base", device="cuda", compute_type="float16"))

    with pytest.warns(RuntimeWarning, match="falling back"):
        transcriber._load()

    assert calls == [("cuda", "float16"), ("cpu", "int8")]
    assert transcriber.settings.device == "cuda"
    assert transcriber.settings.compute_type == "float16"
