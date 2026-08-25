"""Windows implementations of the desktop pieces dictation needs.

Dictation was written against Linux desktop tools -- pw-record to capture,
wl-copy for the clipboard, notify-send for feedback, ffmpeg to measure silence.
None of them exist on Windows, so this module supplies the same services through
Win32 and PortAudio, and ``dictate`` picks between the two at runtime.

Three Windows details drove the shape of this file:

* ``os.kill`` is not a liveness check here. On Windows, Python turns any signal
  other than CTRL_C_EVENT/CTRL_BREAK_EVENT into TerminateProcess, so the POSIX
  idiom ``os.kill(pid, 0)`` would kill the recorder it was meant to ask about.
* The recorder writes headerless PCM, not a WAV, and the parent adds the header
  when it stops the recording. A WAV left behind by a process that was killed
  claims zero frames in its header and reads as empty; raw PCM cannot lose data
  it has already flushed.
* Notification text travels in environment variables. Windows PowerShell 5.1
  reads a ``.ps1`` as ANSI unless it carries a BOM, so a "Snimam..." in the
  script body corrupts the parse, while the environment block is Unicode
  throughout.
"""

import ctypes
import os
import subprocess
import sys
import time
import wave
from ctypes import wintypes
from pathlib import Path

from delaida_transcriber.config import DICTATE_SAMPLE_RATE

# 100 ms of audio. Also the granularity of the stop check, since the recorder
# blocks in ``read`` for exactly this long between glances at the stop flag.
_BLOCK_FRAMES = DICTATE_SAMPLE_RATE // 10

_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_PROCESS_TERMINATE = 0x0001
_STILL_ACTIVE = 259
_CF_UNICODETEXT = 13
_GMEM_MOVEABLE = 0x0002
_CREATE_NO_WINDOW = 0x08000000
# Not combined with CREATE_NO_WINDOW: the two are documented as mutually
# exclusive, and passing both produced a recorder that took seconds to reach its
# first sample. pythonw.exe is a GUI-subsystem binary, so it shows no console on
# its own account anyway.
_DETACHED_PROCESS = 0x00000008

# Toasts need a registered application id to appear at all. PowerShell's own is
# always present on Windows, and borrowing it is the standard workaround for a
# script with no installed identity of its own; the cost is that Settings
# attributes the notification to PowerShell.
_TOAST_APP_ID = r"{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}\WindowsPowerShell\v1.0\powershell.exe"

# Kept ASCII-only on purpose: see the module docstring. Title and body are read
# from the environment and XML-escaped inside PowerShell.
_TOAST_SCRIPT = (
    "[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications,"
    " ContentType = WindowsRuntime] | Out-Null;"
    "[Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom,"
    " ContentType = WindowsRuntime] | Out-Null;"
    "$title = [System.Security.SecurityElement]::Escape($env:DELAIDA_TOAST_TITLE);"
    "$body = [System.Security.SecurityElement]::Escape($env:DELAIDA_TOAST_BODY);"
    "$xml = New-Object Windows.Data.Xml.Dom.XmlDocument;"
    "$xml.LoadXml('<toast><visual><binding template=\"ToastGeneric\">'"
    ' + "<text>$title</text><text>$body</text>"'
    " + '</binding></visual></toast>');"
    "$toast = New-Object Windows.UI.Notifications.ToastNotification $xml;"
    "[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier("
    "$env:DELAIDA_TOAST_APP_ID).Show($toast)"
)

_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
_user32 = ctypes.WinDLL("user32", use_last_error=True)

_kernel32.OpenProcess.restype = wintypes.HANDLE
_kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
_kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
_kernel32.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
_kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
_kernel32.GlobalAlloc.restype = wintypes.HGLOBAL
_kernel32.GlobalAlloc.argtypes = [wintypes.UINT, ctypes.c_size_t]
_kernel32.GlobalLock.restype = wintypes.LPVOID
_kernel32.GlobalLock.argtypes = [wintypes.HGLOBAL]
_kernel32.GlobalUnlock.argtypes = [wintypes.HGLOBAL]
_user32.SetClipboardData.restype = wintypes.HANDLE
_user32.SetClipboardData.argtypes = [wintypes.UINT, wintypes.HANDLE]


def enable_utf8_output() -> None:
    """Stop Bosnian text from crashing a print to a redirected stdout.

    Python picks the ANSI code page for a piped stdout on Windows, and cp1252
    cannot encode "c" with a hacek. Printing the transcript is the last thing
    dictation does before copying it, so without this a transcription the user
    waited for dies at the finish line with UnicodeEncodeError.
    """
    for stream in (sys.stdout, sys.stderr):
        encoding = getattr(stream, "encoding", "") or ""
        if stream is not None and encoding.lower().replace("-", "") != "utf8":
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (AttributeError, OSError):
                pass


def is_process_running(pid: int) -> bool:
    """Whether ``pid`` is a live process, without signalling it.

    A process that exited with code 259 is indistinguishable from a running one
    through this API. That is the documented ambiguity of GetExitCodeProcess,
    and it costs nothing here: the recorder exits 0 or 1.
    """
    handle = _kernel32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return False
    try:
        code = wintypes.DWORD()
        if not _kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
            return False
        return code.value == _STILL_ACTIVE
    finally:
        _kernel32.CloseHandle(handle)


def terminate_process(pid: int) -> None:
    """Kill a recorder that ignored the stop flag. Last resort, not the norm."""
    handle = _kernel32.OpenProcess(_PROCESS_TERMINATE, False, pid)
    if not handle:
        return
    try:
        _kernel32.TerminateProcess(handle, 1)
    finally:
        _kernel32.CloseHandle(handle)


def copy_to_clipboard(text: str) -> bool:
    """Put ``text`` on the clipboard as CF_UNICODETEXT, reporting failure.

    Done through Win32 rather than clip.exe or Set-Clipboard because both route
    the text through a console code page, which is exactly where the Bosnian
    diacritics get lost. The clipboard owns the memory once SetClipboardData
    succeeds, so the allocation is deliberately not freed here.
    """
    payload = text.encode("utf-16-le") + b"\x00\x00"
    if not _user32.OpenClipboard(None):
        return False
    try:
        _user32.EmptyClipboard()
        handle = _kernel32.GlobalAlloc(_GMEM_MOVEABLE, len(payload))
        if not handle:
            return False
        pointer = _kernel32.GlobalLock(handle)
        if not pointer:
            return False
        ctypes.memmove(pointer, payload, len(payload))
        _kernel32.GlobalUnlock(handle)
        return bool(_user32.SetClipboardData(_CF_UNICODETEXT, handle))
    finally:
        _user32.CloseClipboard()


def notify(title: str, body: str = "") -> bool:
    """Show a Windows toast, without waiting for it.

    Loading the WinRT types costs PowerShell a second or more, which would be a
    second of silence between pressing the shortcut and learning that recording
    started. So the process is launched and abandoned; the toast appears when it
    appears.
    """
    environment = dict(os.environ)
    environment["DELAIDA_TOAST_TITLE"] = title
    environment["DELAIDA_TOAST_BODY"] = body
    environment["DELAIDA_TOAST_APP_ID"] = _TOAST_APP_ID
    try:
        subprocess.Popen(
            [
                "powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                _TOAST_SCRIPT,
            ],
            env=environment,
            creationflags=_CREATE_NO_WINDOW,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return False
    return True


def _pythonw() -> str:
    """The interpreter that runs the recorder without flashing up a console."""
    candidate = Path(sys.executable).with_name("pythonw.exe")
    return str(candidate) if candidate.is_file() else sys.executable


def error_path(raw_path: Path) -> Path:
    """Where a recorder that could not start leaves its complaint."""
    return raw_path.with_suffix(".error")


def ready_path(raw_path: Path) -> Path:
    """Marker the recorder drops once the microphone is actually open."""
    return raw_path.with_suffix(".ready")


def start_recorder(raw_path: Path, stop_flag: Path) -> int:
    """Spawn a detached recorder and return its pid.

    Detached because the whole point of the toggle is that the shortcut returns
    immediately and recording continues once this process is gone.
    """
    stop_flag.unlink(missing_ok=True)
    raw_path.unlink(missing_ok=True)
    error_path(raw_path).unlink(missing_ok=True)
    ready_path(raw_path).unlink(missing_ok=True)
    process = subprocess.Popen(
        [_pythonw(), "-m", "delaida_transcriber.windows", str(raw_path), str(stop_flag)],
        creationflags=_DETACHED_PROCESS,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
    )
    return process.pid


def wait_until_recording(raw_path: Path, timeout: float = 5.0) -> str | None:
    """Block until the microphone is live; returns None, or why it is not.

    Without this the shortcut would say "Recording..." while the child was still
    starting Python and opening PortAudio, and the first half-second of speech
    would land nowhere. Measured here, the marker appears about 0.6s after the
    spawn, nearly all of it interpreter startup.
    """
    ready, failure = ready_path(raw_path), error_path(raw_path)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if ready.exists():
            return None
        if failure.exists():
            return failure.read_text(encoding="utf-8").strip()
        time.sleep(0.05)
    return f"The recorder did not open the microphone within {timeout:.0f}s."


def record(raw_path: Path, stop_flag: Path, sample_rate: int = DICTATE_SAMPLE_RATE) -> int:
    """Append microphone audio to ``raw_path`` until ``stop_flag`` appears.

    Runs in the detached child, which has nowhere to report to, so a microphone
    that will not open is written next to the recording for the parent to read
    back and show.
    """
    try:
        import sounddevice
    except (ImportError, OSError) as error:  # PortAudio missing or unloadable
        error_path(raw_path).write_text(str(error), encoding="utf-8")
        return 1
    try:
        with open(raw_path, "wb") as sink:
            with sounddevice.RawInputStream(
                samplerate=sample_rate,
                channels=1,
                dtype="int16",
                blocksize=_BLOCK_FRAMES,
            ) as stream:
                ready_path(raw_path).write_text("", encoding="utf-8")
                while not stop_flag.exists():
                    block, _overflowed = stream.read(_BLOCK_FRAMES)
                    sink.write(bytes(block))
                    sink.flush()
    except Exception as error:  # noqa: BLE001 -- the parent turns this into a notification
        error_path(raw_path).write_text(f"{type(error).__name__}: {error}", encoding="utf-8")
        return 1
    return 0


def write_wav(raw_path: Path, wav_path: Path, sample_rate: int = DICTATE_SAMPLE_RATE) -> int:
    """Wrap recorded PCM in a WAV header; returns the number of audio bytes."""
    pcm = raw_path.read_bytes()
    with wave.open(str(wav_path), "wb") as clip:
        clip.setnchannels(1)
        clip.setsampwidth(2)
        clip.setframerate(sample_rate)
        clip.writeframes(pcm)
    return len(pcm)


if __name__ == "__main__":
    raise SystemExit(record(Path(sys.argv[1]), Path(sys.argv[2])))
