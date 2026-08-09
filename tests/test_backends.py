import pytest

from delaida_transcriber.backends import TranscriptionBackend, create_backend
from delaida_transcriber.config import Settings
from delaida_transcriber.transcriber import WhisperTranscriber


def test_local_backend_is_the_default() -> None:
    backend = create_backend(Settings(device="cpu", compute_type="int8"))

    assert isinstance(backend, WhisperTranscriber)
    assert isinstance(backend, TranscriptionBackend)


def test_elevenlabs_backend_fails_with_an_actionable_message() -> None:
    with pytest.raises(NotImplementedError, match="ELEVENLABS_API_KEY"):
        create_backend(Settings(backend="elevenlabs"))


def test_unknown_backend_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown STT_BACKEND"):
        create_backend(Settings(backend="whisper.cpp"))
