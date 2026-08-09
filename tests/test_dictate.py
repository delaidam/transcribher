import subprocess
from pathlib import Path

import pytest

from delaida_transcriber import dictate


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


def test_silence_is_detected_from_mean_volume(monkeypatch, tmp_path: Path) -> None:
    clip = tmp_path / "clip.wav"
    clip.write_bytes(b"x" * 2048)

    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(command, 0, "", "mean_volume: -71.2 dB")

    monkeypatch.setattr(dictate.subprocess, "run", fake_run)
    assert dictate._looks_like_speech(clip) is False


def test_speech_level_passes(monkeypatch, tmp_path: Path) -> None:
    clip = tmp_path / "clip.wav"
    clip.write_bytes(b"x" * 2048)

    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(command, 0, "", "mean_volume: -23.4 dB")

    monkeypatch.setattr(dictate.subprocess, "run", fake_run)
    assert dictate._looks_like_speech(clip) is True


def test_unreadable_volume_does_not_block_transcription(monkeypatch, tmp_path: Path) -> None:
    """If the probe tells us nothing, transcribe anyway rather than silently drop
    the user's recording."""
    clip = tmp_path / "clip.wav"
    clip.write_bytes(b"x" * 2048)

    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(command, 1, "", "no volume info here")

    monkeypatch.setattr(dictate.subprocess, "run", fake_run)
    assert dictate._looks_like_speech(clip) is True


def test_clipboard_does_not_capture_output(monkeypatch) -> None:
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


def test_clipboard_failure_is_reported_not_raised(monkeypatch) -> None:
    def fake_run(command, **kwargs):
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(dictate.shutil, "which", lambda name: "/usr/bin/wl-copy")
    monkeypatch.setattr(dictate.subprocess, "run", fake_run)

    assert dictate._copy_to_clipboard("zdravo") is False


def test_clipboard_missing_tool_is_reported(monkeypatch) -> None:
    monkeypatch.setattr(dictate.shutil, "which", lambda name: None)

    assert dictate._copy_to_clipboard("zdravo") is False


def test_status_reports_idle_when_no_pidfile(monkeypatch, tmp_path: Path, capsys) -> None:
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    monkeypatch.setattr("sys.argv", ["delaida-transcriber-dictate", "status"])

    assert dictate.main() == 0
    assert capsys.readouterr().out.strip() == "idle"


def test_stale_pidfile_is_cleared_and_reports_idle(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    """A killed recorder leaves a pidfile behind; it must not wedge the toggle."""
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
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
    monkeypatch.setattr("sys.argv", ["delaida-transcriber-dictate", "stop"])

    assert dictate.main() == 1


def test_dictation_forces_a_language(monkeypatch) -> None:
    """Auto-detect mislabels short clips (an 8s Bosnian clip came back as
    Portuguese), so dictation must never pass language=None."""
    from delaida_transcriber.config import Settings

    monkeypatch.delenv("STT_DICTATE_LANGUAGE", raising=False)
    assert Settings().dictate_language == "hr"
    assert Settings(dictate_language="en").dictate_language == "en"
