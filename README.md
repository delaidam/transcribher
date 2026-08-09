# Delaida Transcriber

Private local transcription for Bosnian and English OGG, MP3, MP4, and M4A recordings. The
Whisper model runs on the desktop. The project has two interfaces:

- a batch CLI for folders of recordings;
- a small browser interface that works from a phone on the same Wi-Fi.

The model is downloaded by faster-whisper on first use. Nothing is sent to a
cloud transcription service. Hardware detection uses CTranslate2 itself, and
if CUDA initialization fails the app falls back to CPU/int8 automatically.

## Install

```bash
cd ~/Desktop/delaida-transcriber
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
cp .env.example .env
```

The default model is `large-v3-turbo` (~1.6 GB, downloaded on first use). On a
CPU-only machine it transcribes at roughly 1.4x realtime — an 83-second voice
note takes about a minute.

### Why these defaults

Measured on an 83-second noisy Bosnian/English phone recording, CPU-only, scored
as word-level agreement with an ElevenLabs Scribe transcript of the same audio:

| Configuration                        | Agreement | Speed |
| ------------------------------------ | --------- | ----- |
| `base` (the old default)             | 23.1%     | 1.1x  |
| `large-v3`                           | 56.4%     | 0.1x  |
| `large-v3-turbo`                     | 57.9%     | 1.2x  |
| `large-v3-turbo` + priming (current) | **60.5%** | 1.4x  |

Turbo beats `large-v3` on quality *and* is ten times faster, so it is the
default on GPU too. Priming (`STT_INITIAL_PROMPT`, `STT_HOTWORDS`) is worth
about 2.6 points, mostly by recovering English technical words. Widening the
beam, disabling `condition_on_previous_text`, and forcing the `bs` language code
all measured worse and are deliberately not used.

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

Each input produces `.txt` and `.json` files in `recordings/transcripts/`.

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

### What to expect

Whisper decodes in fixed 30-second windows, and one window costs about 17
seconds on this CPU. A five-second utterance therefore takes about as long as a
thirty-second one — budget roughly 15-20 seconds from stopping to pasting. It is
useful for composing a paragraph; it is not yet fast enough to feel like a
conversation.

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

## Future phone options

The browser interface is the useful first version because it works on Android
and iPhone without maintaining two native apps. A native app or hosted service
can later reuse the same `/transcribe` API if we need background uploads,
accounts, or access away from home.
