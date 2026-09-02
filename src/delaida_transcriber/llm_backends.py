"""Pluggable backends for the work done *on* a transcript.

The same shape as ``backends.py``, and for the same reason: the CLI, the web app
and the session endpoints all go through ``create_llm_backend``, so a second
implementation slots in without touching any of them.

Ollama is the default and runs on this machine. The hosted backend exists
because there is no single local model that is both the strongest reasoner and
able to hold an hour-long meeting on an 8 GB card -- qwen3:8b spills to CPU past
8192 tokens of context, and the smaller model that fits reasons less well.

Using it sends the transcript to Anthropic. That is a real change to what this
project promises, so it is off by default, chosen per request rather than set
once and forgotten, and named in the page at the moment of use.
"""

from typing import Protocol, runtime_checkable

from delaida_transcriber import local_llm
from delaida_transcriber.config import Settings
from delaida_transcriber.tasks import Task

SUPPORTED_LLM_BACKENDS = {"ollama", "anthropic"}

# Everything hosted is billed per token, so the transcript is not sent twice by
# accident: one request per action, no retries beyond the SDK's own.
HOSTED_MAX_TOKENS = 16000


@runtime_checkable
class LLMBackend(Protocol):
    """Anything that can act on a transcript.

    Deliberately free of ``num_ctx`` and ``keep_alive``: those are Ollama's
    problem, and a hosted model with a million-token window does not have it.
    """

    name: str
    local: bool

    def run_task(
        self,
        text: str,
        task: Task,
        *,
        output_language: str | None = None,
        instruction: str | None = None,
    ) -> dict[str, object]: ...

    def chat(
        self,
        transcript: str,
        history: list[dict[str, str]],
        *,
        output_language: str | None = None,
    ) -> dict[str, object]: ...


class OllamaBackend:
    """The local default. Holds the context settings so callers need not."""

    name = "ollama"
    local = True

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @property
    def model(self) -> str:
        return self.settings.ollama_model

    def run_task(self, text, task, *, output_language=None, instruction=None):
        return local_llm.run_task(
            text,
            task,
            self.settings.ollama_model,
            self.settings.ollama_url,
            num_ctx=self.settings.ollama_num_ctx,
            keep_alive=self.settings.ollama_keep_alive,
            output_language=output_language,
            instruction=instruction,
        )

    def chat(self, transcript, history, *, output_language=None):
        return local_llm.chat(
            transcript,
            history,
            self.settings.ollama_model,
            self.settings.ollama_url,
            num_ctx=self.settings.ollama_num_ctx,
            keep_alive=self.settings.ollama_keep_alive,
            output_language=output_language,
        )


class AnthropicBackend:
    """Hosted. Sends the transcript off this machine -- see the module docstring.

    Reaches for the official SDK rather than hand-rolled HTTP: the Ollama client
    is ``urllib`` because that API is three fields, which is not true here.
    """

    name = "anthropic"
    local = False

    def __init__(self, settings: Settings, client: object | None = None) -> None:
        self.settings = settings
        self.model = settings.anthropic_model
        self._client = client

    def _connect(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError as error:  # pragma: no cover - depends on install
                raise RuntimeError(
                    "Hosted backend nije instaliran. Pokreni: pip install -e '.[hosted]'"
                ) from error
            self._client = anthropic.Anthropic()
        return self._client

    def _send(self, system: str, user: str) -> str:
        client = self._connect()
        try:
            # Streaming because a long transcript plus a large max_tokens can
            # outrun the SDK's HTTP timeout on a single non-streaming call.
            with client.messages.stream(
                model=self.model,
                max_tokens=HOSTED_MAX_TOKENS,
                thinking={"type": "adaptive"},
                system=system,
                messages=[{"role": "user", "content": user}],
            ) as stream:
                message = stream.get_final_message()
        except Exception as error:
            raise RuntimeError(f"Hosted model nije odgovorio: {error}") from error

        text = "".join(
            block.text for block in message.content if getattr(block, "type", "") == "text"
        )
        if not text.strip():
            raise RuntimeError("Hosted model je vratio prazan odgovor.")
        return text.strip()

    def run_task(self, text, task, *, output_language=None, instruction=None):
        prompt = local_llm.build_prompt(
            task, text, output_language=output_language, instruction=instruction
        )
        answer = self._send("", prompt)
        return local_llm.parse_task_answer(answer, task)

    def chat(self, transcript, history, *, output_language=None):
        messages = local_llm.build_messages(
            transcript, history, output_language=output_language
        )
        system = messages[0]["content"]
        # The Messages API takes the system prompt separately rather than as the
        # first turn, so the conversation is rendered into one user message.
        rendered = "\n\n".join(
            f"{'Pitanje' if m['role'] == 'user' else 'Odgovor'}: {m['content']}"
            for m in messages[1:]
        )
        return {"reply": self._send(system, rendered)}


def create_llm_backend(settings: Settings | None = None) -> LLMBackend:
    settings = settings or Settings()
    if settings.llm_backend == "ollama":
        return OllamaBackend(settings)
    if settings.llm_backend == "anthropic":
        return AnthropicBackend(settings)
    supported = ", ".join(sorted(SUPPORTED_LLM_BACKENDS))
    raise ValueError(
        f"Unknown LLM_BACKEND {settings.llm_backend!r}; expected one of: {supported}."
    )


__all__ = [
    "AnthropicBackend",
    "LLMBackend",
    "OllamaBackend",
    "create_llm_backend",
]
