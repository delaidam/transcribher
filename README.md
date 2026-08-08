# Delaida Transcriber

Private local transcription for Bosnian and English OGG, MP3, and MP4 recordings. The
Whisper model runs on the desktop. The project has two interfaces:

- a batch CLI for folders of recordings;
- a small browser interface that works from a phone on the same Wi-Fi.

The model is downloaded by faster-whisper on first use. Nothing is sent to a
cloud transcription service.

## Install

```bash
cd ~/Desktop/delaida-transcriber
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
cp .env.example .env
```

For a CPU-only machine, the defaults use the `base` model. The first
transcription can take longer while the model downloads and loads.

## Desktop batch use

```bash
.venv/bin/delaida-transcriber recordings/ --language auto --cpu
```

Use `--language bs` for Bosnian or `--language en` for English when known.
Each input produces `.txt` and `.json` files in `recordings/transcripts/`.

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

The phone interface accepts `.ogg`, `.mp3`, and `.mp4` files. This first phone interface has no login or HTTPS. Do not port-forward it or
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
