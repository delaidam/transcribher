"""Environment-backed configuration."""

import os
from pathlib import Path

import ctranslate2
from dotenv import load_dotenv

load_dotenv(override=False)
load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env", override=False)


# Measured on an 83s noisy Bosnian/English phone recording, CPU-only, scored as
# word-level agreement with an ElevenLabs Scribe transcript of the same audio:
#
#   base                      23.1%  at 1.1x realtime  (the old default)
#   large-v3-turbo            57.9%  at 1.2x realtime
#   turbo + priming           60.5%  at 1.4x realtime  (this configuration)
#   large-v3 + priming        64.1%  at 0.1x realtime  (11 minutes per clip)
#
# large-v3 is the most accurate but eleven times slower, so turbo is the default
# and STT_MODEL=large-v3 (or --best) buys the extra 3.6 points when it matters.
#
# Measured and rejected: disabling VAD (37.4% -- it hallucinates through the
# silences), greedy decoding (37.4%), beam 10 with patience 2 (57.4%), patience
# alone (60.0%), float32 instead of int8 (52.8%, and 3x slower), forcing bs
# (48.2%) or hr (54.4%), disabling condition_on_previous_text, and every
# audio-cleanup chain tried -- FFT denoising dropped it to 42.1% and adding
# loudness normalisation to 28.7%. Whisper is trained on noisy audio, so
# cleaning it up removes information the model relies on. The README carries the
# full table.
DEFAULT_MODEL = "large-v3-turbo"

# What --best selects: 3.6 points more accurate, eleven times slower.
BEST_MODEL = "large-v3"

# Priming is worth ~2.6 points, mostly by recovering the English technical words
# that Bosnian-dominant audio otherwise swallows ("just paste in the chat" was
# decoded as "just basically začar" without it).
DEFAULT_INITIAL_PROMPT = (
    "Razgovor na bosanskom jeziku o softveru. Trebam alat kojim mogu diktirati "
    "tekst glasom, pa ga kopirati u chat. Razumiješ? Ovdje, nije, htio, "
    "koncentrirati, zahtjevima."
)
DEFAULT_HOTWORDS = "GPT, API, Claude Code, browser tab, voice text, copy paste, chat, alat"

# Dictation is a different problem from batch transcription. Whisper decodes in
# fixed 30-second windows costing ~17s each on this CPU, so a 5-second utterance
# is as slow as a 30-second one, and language auto-detection needs more audio
# than a short utterance provides -- an 8s Bosnian clip was detected as
# Portuguese, and near-silence as Turkish. So dictation forces a language and
# may use a smaller model than batch work, where clean microphone audio is much
# easier than a noisy phone recording.
DEFAULT_DICTATE_LANGUAGE = "hr"
DICTATE_SAMPLE_RATE = 16000


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default).strip()


def has_cuda() -> bool:
    """Return whether CTranslate2 can actually see a CUDA device."""
    try:
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


class Settings:
    def __init__(
        self,
        model: str | None = None,
        device: str | None = None,
        compute_type: str | None = None,
        host: str | None = None,
        port: int | None = None,
        max_upload_mb: int | None = None,
        initial_prompt: str | None = None,
        hotwords: str | None = None,
        backend: str | None = None,
        dictate_model: str | None = None,
        dictate_language: str | None = None,
        ollama_model: str | None = None,
        ollama_url: str | None = None,
    ) -> None:
        gpu = has_cuda()
        self.backend = backend or _env("STT_BACKEND", "local")
        self.model = model or _env("STT_MODEL", DEFAULT_MODEL)
        self.device = device or _env("STT_DEVICE", "cuda" if gpu else "cpu")
        resolved_compute_type = compute_type or _env(
            "STT_COMPUTE_TYPE", "float16" if gpu else "int8"
        )
        if self.device == "cpu" and "float16" in resolved_compute_type:
            resolved_compute_type = "int8"
        self.compute_type = resolved_compute_type
        self.host = host or _env("HOST", "127.0.0.1")
        self.port = port or int(_env("PORT", "8765"))
        self.max_upload_mb = max_upload_mb or int(_env("MAX_UPLOAD_MB", "100"))
        # Empty string is a meaningful value here: it disables priming.
        self.initial_prompt = (
            initial_prompt
            if initial_prompt is not None
            else _env("STT_INITIAL_PROMPT", DEFAULT_INITIAL_PROMPT)
        )
        self.hotwords = (
            hotwords if hotwords is not None else _env("STT_HOTWORDS", DEFAULT_HOTWORDS)
        )
        # Falls back to the batch model so there is one knob to turn, not two.
        self.dictate_model = dictate_model or _env("STT_DICTATE_MODEL", self.model)
        self.dictate_language = dictate_language or _env(
            "STT_DICTATE_LANGUAGE", DEFAULT_DICTATE_LANGUAGE
        )
        self.ollama_model = ollama_model or _env("OLLAMA_MODEL", "llama3.2:3b")
        self.ollama_url = ollama_url or _env("OLLAMA_URL", "http://127.0.0.1:11434")

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024
