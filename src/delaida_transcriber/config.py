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
#   base            23.1%  at 1.1x realtime   (the old default)
#   large-v3        56.4%  at 0.1x realtime   (9 minutes per clip)
#   large-v3-turbo  57.9%  at 1.2x realtime
#   turbo + priming 60.5%  at 1.4x realtime   (this configuration)
#
# Turbo beats large-v3 on both quality and speed here, so it is the default even
# on GPU. Beam widening, disabling condition_on_previous_text, and forcing the
# "bs" language code all measured worse and are deliberately not used.
DEFAULT_MODEL = "large-v3-turbo"

# Priming is worth ~2.6 points, mostly by recovering the English technical words
# that Bosnian-dominant audio otherwise swallows ("just paste in the chat" was
# decoded as "just basically začar" without it).
DEFAULT_INITIAL_PROMPT = (
    "Razgovor na bosanskom jeziku o softveru. Trebam alat kojim mogu diktirati "
    "tekst glasom, pa ga kopirati u chat. Razumiješ? Ovdje, nije, htio, "
    "koncentrirati, zahtjevima."
)
DEFAULT_HOTWORDS = "GPT, API, Claude Code, browser tab, voice text, copy paste, chat, alat"


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
    ) -> None:
        gpu = has_cuda()
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

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024
