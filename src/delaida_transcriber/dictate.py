"""Push-to-talk dictation: speak into any window, get text on the clipboard.

Bound to a single desktop shortcut, this is a toggle. The first press starts
recording; the second stops it, transcribes, and copies the text so it can be
pasted anywhere. No browser tab involved.

Deliberately a plain command rather than a daemon: it costs one model load
(~5s) per dictation, which is small next to the ~17s decode, and it keeps the
process model simple enough to bind to a hotkey.

Capture, clipboard and notifications are desktop-specific, so each has two
implementations: PipeWire, Wayland and libnotify on Linux, and the Win32 and
PortAudio equivalents in ``windows``. Everything from the recorded WAV onwards
is shared.
"""

import argparse
import array
import asyncio
import math
import os
import shutil
import signal
import subprocess
import sys
import time
import unicodedata
import wave
from pathlib import Path

from delaida_transcriber.backends import create_backend
from delaida_transcriber.config import DICTATE_SAMPLE_RATE, Settings
from delaida_transcriber.models import FileTranscription

# Imported for its side effects at module scope -- it resolves Win32 entry
# points through ctypes.WinDLL -- so it must stay unimported elsewhere.
if sys.platform == "win32":
    from delaida_transcriber import windows
else:
    windows = None

# Whisper emits these confidently when handed silence or noise -- they come from
# subtitle files in its training data. It does not flag them: no_speech_prob
# stays at 0.0, so they have to be filtered by hand or they land on the
# clipboard. Written in the accent-stripped form that ``_fold`` produces, since
# Turkish "İ".lower() is "i" plus a combining dot and will not match "i".
HALLUCINATIONS = (
    "izlediginiz icin tesekkur ederim",
    "thanks for watching",
    "thank you for watching",
    "subscribe",
    "amara.org",
    "titlovi",
)

# Measured on this hardware: a quiet room records at about -59 dBFS and speech
# well above -40, so the boundary sits comfortably between them.
SILENCE_THRESHOLD_DBFS = -45.0


def _state_dir() -> Path:
    if windows is None:
        base = os.environ.get("XDG_RUNTIME_DIR") or "/tmp"
    else:
        base = os.environ.get("TEMP") or os.environ.get("TMP") or "."
    path = Path(base) / "delaida-transcriber"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _notify(title: str, body: str = "", urgency: str = "normal") -> None:
    if windows is not None:
        # Windows toasts carry no urgency; the wording has to do that work.
        if windows.notify(title, body):
            return
    elif shutil.which("notify-send") is not None:
        subprocess.run(
            [
                "notify-send",
                "--urgency",
                urgency,
                "--app-name",
                "Delaida Transcriber",
                title,
                body,
            ],
            check=False,
        )
        return
    print(f"{title} {body}".strip(), file=sys.stderr)


def _is_running(pid: int) -> bool:
    if windows is not None:
        # Not os.kill: on Windows Python implements it with TerminateProcess,
        # so the POSIX "signal 0 to ask" idiom would kill the recorder.
        return windows.is_process_running(pid)
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


def _looks_like_speech(path: Path) -> bool:
    """Reject recordings that are effectively silent.

    ``no_speech_prob`` is unreliable here (it reads 0.0 on room noise), so the
    check is on the audio itself: a quiet room sits around -60 dBFS, while
    speech is well above -40 dBFS.

    This used to shell out to ffmpeg's volumedetect filter. Measuring the RMS of
    the WAV in-process yields the same figure -- ffmpeg's mean_volume is that
    RMS -- and drops a dependency that Windows does not ship, at a cost of about
    40ms for a minute of audio.
    """
    try:
        with wave.open(str(path), "rb") as clip:
            if clip.getsampwidth() != 2 or clip.getnchannels() != 1:
                return True  # not ours to judge; let Whisper decide
            frames = clip.readframes(clip.getnframes())
    except (wave.Error, OSError):
        return True
    if not frames:
        return False
    samples = array.array("h")
    samples.frombytes(frames)
    mean_square = sum(sample * sample for sample in samples) / len(samples)
    if mean_square <= 0:
        return False
    return 10 * math.log10(mean_square / (32768.0**2)) > SILENCE_THRESHOLD_DBFS


def _fold(text: str) -> str:
    """Lowercase and strip combining accents so matching is diacritic-insensitive."""
    decomposed = unicodedata.normalize("NFKD", text.strip().lower())
    return "".join(character for character in decomposed if not unicodedata.combining(character))


def _is_hallucination(text: str) -> bool:
    folded = _fold(text)
    if not folded:
        return True
    return any(phrase in folded for phrase in HALLUCINATIONS)


def _copy_to_clipboard(text: str) -> bool:
    """Copy to the desktop clipboard, reporting failure rather than raising.

    wl-copy is present but unusable outside a Wayland session (cron, ssh, a
    plain tty), and losing a transcription to that is worse than a warning.
    """
    if windows is not None:
        return windows.copy_to_clipboard(text)
    if shutil.which("wl-copy") is None:
        return False
    try:
        subprocess.run(
            ["wl-copy"],
            input=text,
            text=True,
            check=True,
            # Not capture_output: wl-copy forks a daemon to own the selection,
            # and the child inherits the pipes, so reading them blocks until the
            # timeout and reports failure for a copy that actually succeeded.
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
    except (subprocess.SubprocessError, OSError):
        return False
    return True


def _start(settings: Settings) -> int:
    state = _state_dir()
    recording = state / "recording.wav"
    pidfile = state / "recorder.pid"

    if windows is not None:
        pid = windows.start_recorder(state / "recording.pcm", state / "stop.flag")
        # The notification is the cue to start talking, so it must not go out
        # while PortAudio is still opening: those would be words nobody records.
        problem = windows.wait_until_recording(state / "recording.pcm")
        if problem is not None:
            _notify("Cannot record", problem, urgency="critical")
            return 1
    else:
        if shutil.which("pw-record") is None:
            _notify("Cannot record", "pw-record is not installed.", urgency="critical")
            return 1
        pid = subprocess.Popen(
            [
                "pw-record",
                "--channels=1",
                f"--rate={DICTATE_SAMPLE_RATE}",
                str(recording),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).pid

    pidfile.write_text(str(pid), encoding="utf-8")
    _notify("Recording…", "Press the shortcut again to transcribe.")
    print(f"Recording to {recording} (pid {pid}). Run again to stop.")
    return 0


def _halt_recorder(pid: int) -> Path:
    """Stop the recorder and return the WAV it leaves behind."""
    state = _state_dir()
    recording = state / "recording.wav"

    if windows is None:
        os.kill(pid, signal.SIGTERM)
        for _ in range(50):  # pw-record needs a moment to finalise the WAV header
            if not _is_running(pid):
                break
            time.sleep(0.1)
        return recording

    raw = state / "recording.pcm"
    (state / "stop.flag").touch()
    for _ in range(50):
        if not _is_running(pid):
            break
        time.sleep(0.1)
    else:
        # It ignored the flag. Killing it costs nothing: the PCM on disk is
        # whole, which is the reason the recorder writes PCM and not a WAV.
        windows.terminate_process(pid)

    recording.unlink(missing_ok=True)  # never mistake last time's clip for this one
    if raw.is_file():
        windows.write_wav(raw, recording)
    return recording


def _stop(settings: Settings) -> int:
    state = _state_dir()
    pidfile = state / "recorder.pid"
    pid = int(pidfile.read_text(encoding="utf-8").strip())

    recording = _halt_recorder(pid)
    pidfile.unlink(missing_ok=True)

    if not recording.is_file() or recording.stat().st_size < 1024:
        _notify("Nothing recorded", "The clip was empty.", urgency="critical")
        return 1

    if not _looks_like_speech(recording):
        _notify("No speech detected", "The recording was silent; nothing copied.")
        return 0

    _notify("Transcribing…", "This takes a moment.")
    started = time.time()
    backend = create_backend(settings)
    result: FileTranscription = asyncio.run(
        backend.transcribe_file(recording, language=settings.dictate_language)
    )
    elapsed = time.time() - started

    if _is_hallucination(result.text):
        _notify("No speech detected", "Nothing copied.")
        return 0

    # Always print the text before anything that can fail, so a clipboard problem
    # never costs the user the transcription they just waited for.
    print(result.text)
    preview = result.text if len(result.text) <= 120 else result.text[:117] + "…"

    if not _copy_to_clipboard(result.text):
        _notify(
            f"Transcribed ({elapsed:.0f}s), not copied",
            "Clipboard unavailable; the text was printed to the terminal.",
            urgency="critical",
        )
        return 1

    _notify(f"Copied ({elapsed:.0f}s)", preview)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Toggle push-to-talk dictation; the text lands on the clipboard."
    )
    parser.add_argument(
        "action",
        nargs="?",
        choices=("toggle", "start", "stop", "status"),
        default="toggle",
        help="Defaults to toggle, which is what you bind to a keyboard shortcut.",
    )
    parser.add_argument("--model", help="Override the dictation model.")
    parser.add_argument(
        "--language",
        help="Force a language code. Auto-detection is unreliable on short clips.",
    )
    args = parser.parse_args()

    if windows is not None:
        windows.enable_utf8_output()

    settings = Settings(dictate_model=args.model, dictate_language=args.language)
    # Dictation may run a smaller/faster model than batch transcription.
    settings.model = settings.dictate_model

    pidfile = _state_dir() / "recorder.pid"
    active = False
    if pidfile.is_file():
        try:
            active = _is_running(int(pidfile.read_text(encoding="utf-8").strip()))
        except ValueError:
            active = False
        if not active:
            pidfile.unlink(missing_ok=True)  # stale from a killed recorder

    if args.action == "status":
        print("recording" if active else "idle")
        return 0
    if args.action == "start" or (args.action == "toggle" and not active):
        if active:
            print("Already recording.")
            return 0
        return _start(settings)
    if not active:
        print("Not recording; nothing to stop.")
        return 1
    return _stop(settings)


if __name__ == "__main__":
    raise SystemExit(main())
