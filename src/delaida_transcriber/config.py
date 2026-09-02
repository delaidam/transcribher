"""Environment-backed configuration."""

import os
import sys
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

# Priming steers what Whisper reaches for, which is exactly why one global
# profile cannot serve every language: the default above is a Bosnian sentence
# plus Bosnian/English jargon, and pointing a Norwegian recording at it is not
# neutral, it is a penalty.
#
# Only Norwegian gets its own profile. Everything else -- including "en" and
# auto-detect -- keeps the default, because that is the configuration the
# README's accuracy table was measured under, and changing it would make every
# number in that table unverifiable. The Bosnian strings above are untouched.
NORWEGIAN_INITIAL_PROMPT = (
    "Samtale på norsk om programvare. Jeg trenger et verktøy som kan skrive ned "
    "det jeg sier, så jeg kan lime det inn i en chat. Skjønner du?"
)
NORWEGIAN_HOTWORDS = "GPT, API, Claude Code, nettleser, tale til tekst, kopier lim inn, chat"

PRIMING_PROFILES: dict[str, tuple[str, str]] = {
    "no": (NORWEGIAN_INITIAL_PROMPT, NORWEGIAN_HOTWORDS),
}

# Dictation is a different problem from batch transcription. Whisper decodes in
# fixed 30-second windows costing ~17s each on this CPU, so a 5-second utterance
# is as slow as a 30-second one, and language auto-detection needs more audio
# than a short utterance provides -- an 8s Bosnian clip was detected as
# Portuguese, and near-silence as Turkish. So dictation forces a language and
# may use a smaller model than batch work, where clean microphone audio is much
# easier than a noisy phone recording.
DEFAULT_DICTATE_LANGUAGE = "hr"
DICTATE_SAMPLE_RATE = 16000

# What batch transcription and the web page use when no language is chosen.
#
# "auto" is the honest default to ship, but it is not always the right one to
# run. Language detection reads the first 30 seconds of audio *after* the VAD
# filter has cut it, and on a 4m47s Bosnian voice note what survived that cut
# was read as English at 37.9% confidence -- while the same audio undetected is
# Serbian at 63.9% and Bosnian at 65.3%. Decoding Bosnian under the English
# token produces exactly the mess you would expect. Setting STT_LANGUAGE=hr
# sidesteps it, and "hr" rather than "bs" because Whisper decodes this language
# family markedly better under the Croatian code.
DEFAULT_LANGUAGE = "auto"

# Ollama loads a model at its own default context, not at the model's maximum,
# and that default is small enough to quietly destroy a long transcript. With
# num_ctx unset, `ollama ps` reports qwen3:8b resident at CONTEXT 4096 against
# the 40960 its model card advertises. What makes it dangerous is how it fails:
# a prompt that does not fit is truncated to roughly half the window rather
# than refused, with no error, no warning, and done_reason still reading
# "stop". The model summarises a fraction of the meeting and reports success.
#
# Measured on this machine, one transcript, only num_ctx changed:
#
#   num_ctx  4096 -> prompt_eval_count  2050
#   num_ctx  8192 -> prompt_eval_count  4098
#   num_ctx 16384 -> prompt_eval_count  8194
#   num_ctx 32768 -> prompt_eval_count 27811   (the whole thing; it fit)
#
# Sizing it: Bosnian speech measured 1.95 tokens per word through qwen3's
# tokenizer (2610 words -> 5081 tokens) and 1.57 through gemma3's. At ~140
# words per minute that is roughly 270 tokens per minute of audio, and the
# window has to hold the transcript *and* num_predict on top.
#
# So 8192 covers about 20 minutes of speech. It is the default because it is
# the largest window that keeps every model here fully on an 8 GB card --
# qwen3:8b measures 5.6 GB at 4096, 6.2 GB at 8192, and 7.8 GB at 16384, where
# it spills 20% to CPU and becomes slower than the smaller window was. For
# hour-long recordings raise this and use a smaller model: gemma3:4b holds
# 32768 in 2.9 GB fully on the GPU. Check `ollama ps` after changing it.
DEFAULT_OLLAMA_NUM_CTX = 8192

# Model load dominates a cold call -- 6.55s of a 7.21s total on this machine --
# so keeping the model resident between refinements is worth far more than
# picking a smaller one. The cost is VRAM held while Whisper may want it.
DEFAULT_OLLAMA_KEEP_ALIVE = "30m"


# Where transcripts are kept. The web path used to hold everything in a
# JavaScript variable, so a reload destroyed the recording along with every
# refinement made from it.
def default_data_dir() -> Path:
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "delaida-transcriber"
    return Path.home() / ".local" / "share" / "delaida-transcriber"


# Off by default, and deliberately so. This project's premise is that recordings
# stay private; quietly accumulating every meeting's audio on disk forever is a
# real change to what the tool does with your data, and it should be something
# you switched on knowing you switched it on -- not something you find out about
# a year later when the folder is 40 GB. Transcripts are saved either way.
DEFAULT_KEEP_AUDIO = False

# Which model does the work *on* a transcript. "ollama" runs on this machine and
# is the default; "anthropic" sends the transcript off it, which is why it has
# to be chosen rather than inherited. See llm_backends.py.
DEFAULT_LLM_BACKEND = "ollama"

# Only consulted when LLM_BACKEND=anthropic. Opus 5 takes no date suffix.
DEFAULT_ANTHROPIC_MODEL = "claude-opus-5"

_TRUE = {"1", "true", "yes", "on", "da"}


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default).strip()


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    return default if raw is None else raw.strip().lower() in _TRUE


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
        language: str | None = None,
        dictate_model: str | None = None,
        dictate_language: str | None = None,
        ollama_model: str | None = None,
        ollama_url: str | None = None,
        ollama_num_ctx: int | None = None,
        ollama_keep_alive: str | None = None,
        data_dir: Path | str | None = None,
        keep_audio: bool | None = None,
        llm_backend: str | None = None,
        anthropic_model: str | None = None,
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
        self.priming_is_explicit = (
            initial_prompt is not None
            or hotwords is not None
            or "STT_INITIAL_PROMPT" in os.environ
            or "STT_HOTWORDS" in os.environ
        )
        self.language = language or _env("STT_LANGUAGE", DEFAULT_LANGUAGE)
        # Falls back to the batch model so there is one knob to turn, not two.
        self.dictate_model = dictate_model or _env("STT_DICTATE_MODEL", self.model)
        self.dictate_language = dictate_language or _env(
            "STT_DICTATE_LANGUAGE", DEFAULT_DICTATE_LANGUAGE
        )
        self.ollama_model = ollama_model or _env("OLLAMA_MODEL", "llama3.2:3b")
        self.ollama_url = ollama_url or _env("OLLAMA_URL", "http://127.0.0.1:11434")
        self.ollama_num_ctx = ollama_num_ctx or int(
            _env("OLLAMA_NUM_CTX", str(DEFAULT_OLLAMA_NUM_CTX))
        )
        self.ollama_keep_alive = ollama_keep_alive or _env(
            "OLLAMA_KEEP_ALIVE", DEFAULT_OLLAMA_KEEP_ALIVE
        )
        self.data_dir = Path(
            data_dir or _env("STT_DATA_DIR", "") or default_data_dir()
        ).expanduser()
        self.keep_audio = (
            keep_audio
            if keep_audio is not None
            else _env_flag("STT_KEEP_AUDIO", DEFAULT_KEEP_AUDIO)
        )
        self.llm_backend = llm_backend or _env("LLM_BACKEND", DEFAULT_LLM_BACKEND)
        self.anthropic_model = anthropic_model or _env(
            "ANTHROPIC_MODEL", DEFAULT_ANTHROPIC_MODEL
        )

    def priming_for(self, language: str | None) -> tuple[str, str]:
        """The initial prompt and hotwords to decode ``language`` with.

        An explicit STT_INITIAL_PROMPT or STT_HOTWORDS wins everywhere -- it is
        set deliberately, so it is not second-guessed per language. Otherwise a
        language with its own profile gets it, and everything else keeps the
        measured default. Auto-detect lands here too: the prompt has to be
        chosen before Whisper has heard the audio, so there is nothing to key on
        and the default is the honest answer.
        """
        if self.priming_is_explicit:
            return self.initial_prompt, self.hotwords
        return PRIMING_PROFILES.get(language or "", (self.initial_prompt, self.hotwords))

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024
