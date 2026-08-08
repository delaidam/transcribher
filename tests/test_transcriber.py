import asyncio
from types import SimpleNamespace

import pytest

from delaida_transcriber.config import Settings
from delaida_transcriber.transcriber import WhisperTranscriber


def test_priming_is_passed_through_to_the_model(monkeypatch, tmp_path) -> None:
    """The prompt and hotwords are worth ~2.6 points of accuracy, so a silent
    failure to forward them would be an invisible quality regression."""
    captured: dict[str, object] = {}

    class FakeModel:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def transcribe(self, source, **kwargs):
            captured.update(kwargs)
            return [], SimpleNamespace(language="bs", language_probability=0.9)

    monkeypatch.setattr("delaida_transcriber.transcriber.WhisperModel", FakeModel)
    transcriber = WhisperTranscriber(Settings(device="cpu", compute_type="int8"))

    asyncio.run(transcriber.transcribe_file(tmp_path / "recording.ogg", language="hr"))

    assert captured["language"] == "hr"
    assert captured["vad_filter"] is True
    assert "bosanskom" in captured["initial_prompt"]
    assert "GPT" in captured["hotwords"]


def test_empty_priming_becomes_none_not_empty_string(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}

    class FakeModel:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def transcribe(self, source, **kwargs):
            captured.update(kwargs)
            return [], SimpleNamespace(language="bs", language_probability=0.9)

    monkeypatch.setattr("delaida_transcriber.transcriber.WhisperModel", FakeModel)
    transcriber = WhisperTranscriber(
        Settings(device="cpu", compute_type="int8", initial_prompt="", hotwords="")
    )

    asyncio.run(transcriber.transcribe_file(tmp_path / "recording.ogg"))

    assert captured["initial_prompt"] is None
    assert captured["hotwords"] is None


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
