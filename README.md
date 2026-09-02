# Delaida Transcriber

Private local transcription for Bosnian and English OGG, MP3, MP4, and M4A recordings. The
Whisper model runs on the desktop. The project has two interfaces:

- a batch CLI for folders of recordings;
- a small browser interface that works from a phone on the same Wi-Fi.

The model is downloaded by faster-whisper on first use. Nothing is sent to a
cloud transcription service. Hardware detection uses CTranslate2 itself, and
if CUDA initialization fails the app falls back to CPU/int8 automatically.

Transcription is local, always. The step *after* it -- cleaning the transcript
up, summarising it, answering questions about it -- runs on a local model
through Ollama by default, and that is the only configuration the default
install can even reach. Setting `LLM_BACKEND=anthropic` opts that one step into
a hosted model, which means the transcript leaves this machine; it needs a
separate install (`pip install -e ".[hosted]"`) and an API key, and the page
says so on screen the whole time it is in use. It exists because no single
local model both reasons well and holds an hour-long meeting on an 8 GB card.

## Install

```bash
cd ~/Desktop/delaida-transcriber
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
cp .env.example .env
```

On Windows the same three steps go through the `py` launcher, and the commands
land in `.venv\Scripts\` instead of `.venv/bin/`:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\pip install -e ".[dev]"
copy .env.example .env
```

Every `.venv/bin/…` below therefore reads `.venv\Scripts\…` there. Batch
transcription and the phone interface are otherwise identical; dictation is the
one part that differs, and it has its own note below.

The default model is `large-v3-turbo` (~1.6 GB, downloaded on first use). On a
CPU-only machine it transcribes at roughly 1.4x realtime — an 83-second voice
note takes about a minute.

### On an NVIDIA GPU

CTranslate2 needs cuBLAS and cuDNN, which the graphics driver does not include:

```bash
.venv/bin/pip install nvidia-cublas-cu12 "nvidia-cudnn-cu12>=9,<10"
```

Then set `STT_DEVICE=cuda` and `STT_COMPUTE_TYPE=float16` in `.env`.

Windows needs nothing beyond that, though making it so took some doing:
CTranslate2 resolves cuBLAS lazily through a plain `LoadLibrary`, which searches
`PATH` and ignores the directory list `os.add_dll_directory` keeps, so DLLs that
live in site-packages stay invisible and the first GPU operation fails with
`Library cublas64_12.dll is not found` on a machine that is otherwise set up
correctly. Importing the package puts them on `PATH` before anything can load
CTranslate2 — `delaida_transcriber/cuda.py`, a no-op everywhere else.

Measured on an RTX 5050 laptop GPU, a 9.8-second clip through `large-v3-turbo`
takes **4.7s on the GPU against 16.9s on `--cpu`**, both including process start
and model load. The tables below were measured CPU-only: they still rank the
settings correctly, but they are not the wall-clock times you get on a GPU.

### Why these defaults

Measured on an 83-second noisy Bosnian/English phone recording, CPU-only, scored
as word-level agreement with an ElevenLabs Scribe transcript of the same audio:

| Configuration                        | Agreement | Speed | Time for 83s |
| ------------------------------------ | --------- | ----- | ------------ |
| `base` (the old default)             | 23.1%     | 1.1x  | 74s          |
| `large-v3-turbo`                     | 57.9%     | 1.2x  | 67s          |
| `large-v3-turbo` + priming (default) | 60.5%     | 1.4x  | 62s          |
| `large-v3` + priming (`--best`)      | **64.1%** | 0.1x  | 668s         |

`large-v3` is the most accurate, but eleven times slower. Turbo is the default
because a minute is tolerable and eleven is not; pass `--best` (or set
`STT_MODEL=large-v3`) when accuracy matters more than the wait. Priming
(`STT_INITIAL_PROMPT`, `STT_HOTWORDS`) is worth about 2.6 points on its own,
mostly by recovering English technical words.

Measured and rejected, so you need not retry them:

| Idea                                  | Result |
| ------------------------------------- | ------ |
| Disable VAD                           | 37.4% — hallucinates through the silences |
| `beam_size=1` (greedy)                | 37.4%, and slower on long audio |
| `beam_size=10` + `patience=2`         | 57.4%, twice as slow |
| `patience=2`                          | 60.0% — no gain |
| `float32` instead of `int8`           | 52.8%, three times slower |
| Force `language=bs`                   | 48.2% |
| Force `language=hr`                   | 54.4% on turbo; a wash on `large-v3` |
| Disable `condition_on_previous_text`  | worse on turbo, faster on `large-v3` only |
| FFT denoise + speechnorm              | 42.1% |
| FFT denoise + bandpass + loudnorm     | 28.7% |
| Bandpass + dynaudnorm                 | 60.5% — no change |

Two lessons in there. Whisper is trained on noisy real-world audio, so cleaning
the audio up strips information it depends on. And auto-detect beats every
forced language code once priming is in play, because it is the only setting
that can follow a switch into English mid-sentence.

Local Whisper still trails a hosted model noticeably on this kind of audio, and
it sometimes drops negations — turning "nije ovako" into "ovako" and inverting
the meaning. Treat transcripts of anything important as a draft to check.

## Desktop batch use

```bash
.venv/bin/delaida-transcriber recordings/ --language auto --cpu
```

Leave `--language auto`. It beats every forced language code on mixed
Bosnian/English speech, and it is the only setting that handles switching
language mid-sentence. If you must force one, use `--language hr` rather than
`bs` — Whisper decodes Bosnian markedly better under the Croatian code (58.5%
vs 48.2% on the recording above). `--language bs` is the worst option and is
kept only for comparison.

### Output formats

Each input produces `.txt`, `.json` and `.srt` files in
`recordings/transcripts/`. Choose with `--formats`:

```bash
.venv/bin/delaida-transcriber recordings/ --formats srt,vtt
```

`txt` is the plain transcript, `json` adds per-segment and per-word timings,
`srt` is SubRip subtitles, and `vtt` is WebVTT for `<track>` elements in a
browser.

Whisper's own segments run as long as 36 seconds, which is unreadable as a
subtitle, so cues are split at real word boundaries to a maximum of 6 seconds
and 84 characters. That split needs word-level timings, which cost about 5% in
transcription time and are always enabled.

## Dictate into any window (no browser tab)

Press a shortcut, talk, press it again. The text lands on your clipboard, ready
to paste into a chat, an editor, or anything else.

```bash
.venv/bin/delaida-transcriber-dictate
```

The command is a toggle, which is what makes it work as a single hotkey. Run it
once to start recording, again to stop and transcribe. `status` tells you which
state you are in, and `start` / `stop` are available if you prefer two keys.

To bind it in GNOME: **Settings → Keyboard → View and Customize Shortcuts →
Custom Shortcuts → +**, then set the command to the absolute path:

```
/home/delaida/Desktop/delaida-transcriber/.venv/bin/delaida-transcriber-dictate
```

Give it a shortcut such as `Super+D`. Notifications tell you when recording
starts, and what was copied when it finishes.

On Windows, point a shortcut at the venv's `pythonw.exe` — which runs the toggle
without flashing up a console window — and give it a shortcut key:

```powershell
$ws = New-Object -ComObject WScript.Shell
$lnk = $ws.CreateShortcut("$env:APPDATA\Microsoft\Windows\Start Menu\Programs\Delaida Dictation.lnk")
$lnk.TargetPath = "$HOME\Desktop\delaida-transcriber\.venv\Scripts\pythonw.exe"
$lnk.Arguments = "-m delaida_transcriber.dictate"
$lnk.Hotkey = "CTRL+ALT+D"
$lnk.Save()
```

Windows only honours a shortcut's hotkey while the shortcut sits in the Start
Menu or on the Desktop, so it has to be created there rather than anywhere
convenient.

Everything the desktop provides has two implementations behind it: pw-record,
wl-copy and notify-send on Linux, PortAudio, the Win32 clipboard and a toast
notification on Windows. Both capture 16 kHz mono, and everything from the
recorded audio onwards is shared. The Windows recorder writes headerless PCM
that the toggle wraps into a WAV when it stops, so a recorder that has to be
killed still yields the audio it had already captured.

### What to expect

Whisper decodes in fixed 30-second windows, so a short utterance costs nearly as
much as a long one. Practical consequence: **dictate a whole paragraph rather
than a sentence at a time** — you wait about the same either way.

Measured on this CPU, from pressing stop to text on the clipboard:

| `STT_DICTATE_MODEL`      | Wait  | Quality | |
| ------------------------ | ----- | ------- | --------------------------- |
| `base`                   | 2.9s  | 24.1%   | too inaccurate to use       |
| `small`                  | 5.7s  | 46.2%   | the fast option             |
| `medium`                 | 17.4s | 50.8%   | never worth it — see below  |
| `large-v3-turbo`         | 12.7s | 60.5%   | the default                 |

Quality is word-level agreement with a hosted transcript of the same recording.
The default favours accuracy; set `STT_DICTATE_MODEL=small` in `.env` if you
would rather wait half as long and correct more typos.

Those waits are CPU figures. On a GPU the default model turned a 7.3-second
dictation into text in 5.6 seconds end to end, model load included, which is
close enough to make the smaller models pointless.

Two configurations that look tempting and are not: `medium` is slower than
`large-v3-turbo` *and* less accurate, so it is strictly worse; and greedy
decoding (`beam_size=1`) drops accuracy to 37% while actually running slower on
longer audio.

None of this is conversational speed. If you want that, the fix is a hosted
backend rather than a different local model — see `STT_BACKEND`.

Dictation forces a language (`STT_DICTATE_LANGUAGE`, default `hr`) rather than
auto-detecting. Short clips do not give Whisper enough audio to identify the
language: an eight-second Bosnian clip was detected as Portuguese, and silence
as Turkish. Set `STT_DICTATE_MODEL` if you want a smaller, faster model for
dictation than for batch work.

Recordings that are effectively silent are dropped rather than transcribed,
because Whisper answers silence with confident nonsense from its subtitle
training data ("Thanks for watching") and reports no error while doing it.

## Use from your phone

Keep the desktop and phone on the same trusted home network. Start the server:

```bash
.venv/bin/delaida-transcriber-web --host 0.0.0.0 --port 8765 --cpu
```

Find the desktop's local IP:

```bash
hostname -I
```

On the phone, open `http://DESKTOP_IP:8765`, choose an OGG recording, choose
the language, and tap **Transcribe**. The phone uploads the audio to the
desktop; the desktop performs transcription and returns the result.

The phone interface accepts `.ogg`, `.mp3`, `.mp4`, and `.m4a` files. This first phone interface has no login or HTTPS. Do not port-forward it or
expose it to the public internet. Before using it on an untrusted network, the
next security step is authentication plus HTTPS.

## Checks

```bash
.venv/bin/ruff check .
.venv/bin/pytest -q
.venv/bin/delaida-transcriber-web --help
```

`tests/test_windows.py` drives the real desktop — it writes to the clipboard and
restores what was there — and skips itself entirely off Windows. The one test
that opens the microphone is opt-in:

```bash
.venv/bin/pytest -m microphone
```

It is held back because cycling the capture device is not free. Repeatedly
opening and closing it here preceded an Intel Smart Sound firmware timeout and a
`DRIVER_POWER_STATE_FAILURE` bugcheck, and a routine test run should not be able
to take the machine down with it.

## Future phone options

The browser interface is the useful first version because it works on Android
and iPhone without maintaining two native apps. A native app or hosted service
can later reuse the same `/transcribe` API if we need background uploads,
accounts, or access away from home.
