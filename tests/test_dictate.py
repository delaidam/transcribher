import array
import subprocess
import wave
from pathlib import Path

import pytest

from delaida_transcriber import dictate


def _clip(path: Path, amplitude: int, seconds: float = 0.5, rate: int = 16000) -> Path:
    """A mono 16-bit WAV whose RMS is exactly ``amplitude``.

    The samples alternate between +amplitude and -amplitude, which makes the
    level arithmetic exact: 100 is -50 dBFS, 3000 is -21 dBFS, either side of
    the -45 dBFS boundary that separates a quiet room from speech.
    """
    frames = array.array("h", [amplitude, -amplitude] * int(rate * seconds / 2))
    with wave.open(str(path), "wb") as clip:
        clip.setnchannels(1)
        clip.setsampwidth(2)
        clip.setframerate(rate)
        clip.writeframes(frames.tobytes())
    return path


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "İzlediğiniz için teşekkür ederim.",
        "Thanks for watching!",
        "Subscribe to my channel",
    ],
)
def test_known_hallucinations_are_rejected(text: str) -> None:
    """Whisper reports no_speech_prob 0.0 for these, so they must be caught here
    or they get copied to the clipboard after a silent recording."""
    assert dictate._is_hallucination(text) is True


def test_real_speech_is_not_rejected() -> None:
    assert dictate._is_hallucination("Treba mi neki alat s kojim ću ja moći diktirati.") is False


def test_silence_is_detected_from_the_recording_level(tmp_path: Path) -> None:
    assert dictate._looks_like_speech(_clip(tmp_path / "quiet.wav", 100)) is False


def test_speech_level_passes(tmp_path: Path) -> None:
    assert dictate._looks_like_speech(_clip(tmp_path / "loud.wav", 3000)) is True


def test_digital_silence_is_detected(tmp_path: Path) -> None:
    """An all-zero clip must not reach the log, which is undefined at zero."""
    assert dictate._looks_like_speech(_clip(tmp_path / "flat.wav", 0)) is False


def test_unreadable_clip_does_not_block_transcription(tmp_path: Path) -> None:
    """If the audio tells us nothing, transcribe anyway rather than silently drop
    the user's recording."""
    clip = tmp_path / "clip.wav"
    clip.write_bytes(b"x" * 2048)

    assert dictate._looks_like_speech(clip) is True


@pytest.fixture
def linux_desktop(monkeypatch):
    """Exercise the Linux implementations whatever the host platform is.

    Without this the wl-copy tests below quietly pass on Windows by taking the
    Win32 branch and never touching the code they are named after.
    """
    monkeypatch.setattr(dictate, "windows", None)


def test_clipboard_does_not_capture_output(monkeypatch, linux_desktop) -> None:
    """wl-copy forks a daemon that inherits its pipes. Capturing output makes
    subprocess.run block until the timeout and report failure for a copy that
    actually succeeded, so the streams must go to DEVNULL."""
    seen: dict[str, object] = {}

    def fake_run(command, **kwargs):
        seen.update(kwargs)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(dictate.shutil, "which", lambda name: "/usr/bin/wl-copy")
    monkeypatch.setattr(dictate.subprocess, "run", fake_run)

    assert dictate._copy_to_clipboard("zdravo") is True
    assert seen.get("capture_output") is None
    assert seen["stdout"] is subprocess.DEVNULL
    assert seen["stderr"] is subprocess.DEVNULL


def test_clipboard_failure_is_reported_not_raised(monkeypatch, linux_desktop) -> None:
    def fake_run(command, **kwargs):
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(dictate.shutil, "which", lambda name: "/usr/bin/wl-copy")
    monkeypatch.setattr(dictate.subprocess, "run", fake_run)

    assert dictate._copy_to_clipboard("zdravo") is False


def test_clipboard_missing_tool_is_reported(monkeypatch, linux_desktop) -> None:
    monkeypatch.setattr(dictate.shutil, "which", lambda name: None)

    assert dictate._copy_to_clipboard("zdravo") is False


def test_status_reports_idle_when_no_pidfile(monkeypatch, tmp_path: Path, capsys) -> None:
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    monkeypatch.setenv("TEMP", str(tmp_path))
    monkeypatch.setattr("sys.argv", ["delaida-transcriber-dictate", "status"])

    assert dictate.main() == 0
    assert capsys.readouterr().out.strip() == "idle"


def test_stale_pidfile_is_cleared_and_reports_idle(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    """A killed recorder leaves a pidfile behind; it must not wedge the toggle."""
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    monkeypatch.setenv("TEMP", str(tmp_path))
    state = tmp_path / "delaida-transcriber"
    state.mkdir()
    pidfile = state / "recorder.pid"
    pidfile.write_text("999999999", encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["delaida-transcriber-dictate", "status"])

    assert dictate.main() == 0
    assert capsys.readouterr().out.strip() == "idle"
    assert not pidfile.exists()


def test_stop_without_recording_is_an_error(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    monkeypatch.setenv("TEMP", str(tmp_path))
    monkeypatch.setattr("sys.argv", ["delaida-transcriber-dictate", "stop"])

    assert dictate.main() == 1


def test_dictation_forces_a_language(monkeypatch) -> None:
    """Auto-detect mislabels short clips (an 8s Bosnian clip came back as
    Portuguese), so dictation must never pass language=None."""
    from delaida_transcriber.config import Settings

    monkeypatch.delenv("STT_DICTATE_LANGUAGE", raising=False)
    assert Settings().dictate_language == "hr"
    assert Settings(dictate_language="en").dictate_language == "en"
