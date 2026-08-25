"""Tests for the Win32 half of dictation.

These drive the real desktop -- the real clipboard, real processes -- because
the bugs worth catching here are exactly the ones a mock would paper over.
"""

import os
import subprocess
import sys
import time
import wave
from pathlib import Path

import pytest

if sys.platform != "win32":
    pytest.skip("Win32 desktop integration", allow_module_level=True)

from delaida_transcriber import windows  # noqa: E402 -- must follow the platform skip


def _read_clipboard() -> str:
    """Read the clipboard from outside this process, which is the only proof
    that SetClipboardData handed ownership over rather than leaving a pointer
    that dies with us.

    PowerShell writes its output in the console code page, which turns "Đ" into
    "D" on the way out and would fail this test for a clipboard that is in fact
    correct, so the encoding is forced before anything is printed.
    """
    result = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            "[Console]::OutputEncoding = [Text.Encoding]::UTF8; Get-Clipboard",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    return result.stdout.strip()


def test_asking_whether_a_process_runs_does_not_kill_it() -> None:
    """The reason this module exists at all: os.kill(pid, 0) is a liveness check
    on POSIX and a TerminateProcess on Windows, so the obvious implementation
    would kill the recorder every time the toggle asked about it."""
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        assert windows.is_process_running(child.pid) is True
        assert windows.is_process_running(child.pid) is True
        assert child.poll() is None
    finally:
        child.kill()
        child.wait(timeout=10)

    assert windows.is_process_running(child.pid) is False


def test_this_process_is_running() -> None:
    assert windows.is_process_running(os.getpid()) is True


def test_clipboard_survives_the_process_that_wrote_it(tmp_path: Path) -> None:
    """Bosnian diacritics are the point. clip.exe and Set-Clipboard both route
    the text through a console code page, which is where they get mangled."""
    previous = _read_clipboard()
    text = "Đački šešir žuri – GPT, API, copy paste."
    try:
        assert windows.copy_to_clipboard(text) is True
        assert _read_clipboard() == text
    finally:
        if previous:
            windows.copy_to_clipboard(previous)


def test_recorded_pcm_becomes_a_readable_wav(tmp_path: Path) -> None:
    raw = tmp_path / "recording.pcm"
    raw.write_bytes(b"\x00\x01" * 16000)  # one second at the dictation rate

    written = windows.write_wav(raw, tmp_path / "recording.wav")

    assert written == 32000
    with wave.open(str(tmp_path / "recording.wav")) as clip:
        assert clip.getnchannels() == 1
        assert clip.getsampwidth() == 2
        assert clip.getframerate() == 16000
        assert clip.getnframes() == 16000


def test_waiting_reports_the_microphone_opening(tmp_path: Path) -> None:
    raw = tmp_path / "recording.pcm"
    windows.ready_path(raw).write_text("", encoding="utf-8")

    assert windows.wait_until_recording(raw, timeout=1.0) is None


def test_waiting_reports_a_recorder_that_failed(tmp_path: Path) -> None:
    """The recorder is detached and silent, so the only way a microphone that
    will not open reaches the user is through this file."""
    raw = tmp_path / "recording.pcm"
    windows.error_path(raw).write_text("Error opening RawInputStream", encoding="utf-8")

    assert windows.wait_until_recording(raw, timeout=1.0) == "Error opening RawInputStream"


def test_waiting_gives_up_rather_than_hanging_the_shortcut(tmp_path: Path) -> None:
    started = time.monotonic()

    problem = windows.wait_until_recording(tmp_path / "recording.pcm", timeout=0.3)

    assert problem is not None
    assert time.monotonic() - started < 3.0


def test_toast_script_is_ascii_only() -> None:
    """Windows PowerShell 5.1 reads a script as ANSI unless it carries a BOM, so
    any non-ASCII character in the script body corrupts the parse. All of the
    text the user actually sees travels in the environment instead."""
    windows._TOAST_SCRIPT.encode("ascii")


@pytest.mark.microphone
def test_microphone_capture_produces_audio(tmp_path: Path) -> None:
    """The full capture path, on whatever microphone this machine has.

    Opt-in -- `pytest -m microphone` -- because it is the one test that reaches
    outside the process and can leave a mark. Cycling the capture device on this
    hardware preceded an Intel Smart Sound firmware timeout and the bugcheck
    that followed it, and a test suite should not be able to do that by default.

    Skipped rather than failed when there is no input device: a machine without
    a microphone cannot dictate, but that is not a broken build.
    """
    sounddevice = pytest.importorskip("sounddevice")
    if not any(device["max_input_channels"] > 0 for device in sounddevice.query_devices()):
        pytest.skip("no input device on this machine")

    raw, stop_flag = tmp_path / "recording.pcm", tmp_path / "stop.flag"
    pid = windows.start_recorder(raw, stop_flag)
    assert windows.wait_until_recording(raw) is None

    time.sleep(0.5)
    stop_flag.touch()
    deadline = time.monotonic() + 5
    while windows.is_process_running(pid) and time.monotonic() < deadline:
        time.sleep(0.05)

    assert windows.is_process_running(pid) is False
    assert not windows.error_path(raw).exists()
    # Half a second at 16 kHz, 16-bit mono, minus whatever the last partial
    # block did not cover.
    assert raw.stat().st_size >= 8000
