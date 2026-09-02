import json

import pytest

from delaida_transcriber import tasks
from delaida_transcriber.config import Settings
from delaida_transcriber.llm_backends import (
    AnthropicBackend,
    LLMBackend,
    OllamaBackend,
    create_llm_backend,
)

REFINE = tasks.get("refine")
UNIFY = tasks.get("unify")


def test_ollama_is_the_default_and_says_it_is_local() -> None:
    """The page shows whether a request leaves the machine, so the backend has
    to be able to answer that."""
    backend = create_llm_backend(Settings(model="base", device="cpu"))

    assert isinstance(backend, OllamaBackend)
    assert backend.local is True
    assert isinstance(backend, LLMBackend)


def test_the_hosted_backend_is_selected_by_name_and_is_not_local() -> None:
    backend = create_llm_backend(
        Settings(model="base", device="cpu", llm_backend="anthropic")
    )

    assert isinstance(backend, AnthropicBackend)
    assert backend.local is False
    assert backend.model == "claude-opus-5"
    assert isinstance(backend, LLMBackend)


def test_an_unknown_backend_names_the_ones_that_exist() -> None:
    with pytest.raises(ValueError, match="expected one of: anthropic, ollama"):
        create_llm_backend(Settings(model="base", device="cpu", llm_backend="izmisljeno"))


def test_the_local_backend_carries_the_context_settings(monkeypatch) -> None:
    """Callers should not have to know about num_ctx; getting it wrong is silent,
    so exactly one place owns it."""
    seen = {}

    def fake_run_task(text, task, model, base_url, **kwargs):
        seen.update(kwargs | {"model": model, "text": text})
        return {"kind": "text", "task": task.id, "output": "ok"}

    monkeypatch.setattr("delaida_transcriber.local_llm.run_task", fake_run_task)
    settings = Settings(model="base", device="cpu", ollama_num_ctx=16384, ollama_model="m")

    OllamaBackend(settings).run_task("transkript", UNIFY, output_language="no")

    assert seen["num_ctx"] == 16384
    assert seen["keep_alive"] == settings.ollama_keep_alive
    assert seen["output_language"] == "no"
    assert seen["model"] == "m"


# --- the hosted path, without spending anything -----------------------------


class FakeBlock:
    def __init__(self, text: str, type: str = "text") -> None:
        self.text = text
        self.type = type


class FakeMessage:
    def __init__(self, blocks) -> None:
        self.content = blocks


class FakeStream:
    def __init__(self, message, recorder, kwargs) -> None:
        self._message = message
        recorder.append(kwargs)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self._message


class FakeClient:
    """Enough of the Anthropic SDK to check what would be sent."""

    def __init__(self, blocks) -> None:
        self.calls: list[dict] = []
        self.messages = self
        self._blocks = blocks

    def stream(self, **kwargs):
        return FakeStream(FakeMessage(self._blocks), self.calls, kwargs)


def _hosted(blocks) -> tuple[AnthropicBackend, FakeClient]:
    client = FakeClient(blocks)
    settings = Settings(model="base", device="cpu", llm_backend="anthropic")
    return AnthropicBackend(settings, client=client), client


def test_the_hosted_request_uses_the_current_api_shape() -> None:
    """budget_tokens was removed and is rejected outright; adaptive thinking
    replaces it. Streaming because a long transcript with a large max_tokens can
    outrun the SDK's HTTP timeout on a single non-streaming call."""
    backend, client = _hosted([FakeBlock("jedna bilješka")])

    backend.run_task("transkript", UNIFY)

    sent = client.calls[0]
    assert sent["model"] == "claude-opus-5"
    assert sent["thinking"] == {"type": "adaptive"}
    assert "budget_tokens" not in json.dumps(sent["thinking"])
    assert sent["max_tokens"] == 16000


def test_the_hosted_backend_sends_the_same_prompt_as_the_local_one() -> None:
    """Only the transport differs. The rules that keep the model honest are the
    same either way, and that is the point of the seam."""
    backend, client = _hosted([FakeBlock("bilješka")])

    backend.run_task("moj transkript", UNIFY, output_language="no")

    prompt = client.calls[0]["messages"][0]["content"]
    assert "Ne dodaj činjenice" in prompt
    assert "moj transkript" in prompt
    assert "norveški" in prompt


def test_a_hosted_structured_answer_is_parsed_like_a_local_one() -> None:
    answer = json.dumps({"cleaned_text": "uredan", "summary": "kratko"})
    backend, _ = _hosted([FakeBlock(answer)])

    result = backend.run_task("transkript", REFINE)

    assert result["kind"] == "structured"
    assert result["fields"][0]["value"] == "uredan"


def test_hosted_thinking_blocks_are_not_mistaken_for_the_answer() -> None:
    """Adaptive thinking puts reasoning in its own block; only text blocks are
    the reply."""
    backend, _ = _hosted(
        [FakeBlock("razmišljam", type="thinking"), FakeBlock("Stvarni odgovor.")]
    )

    assert backend.run_task("t", UNIFY)["output"] == "Stvarni odgovor."


def test_a_hosted_conversation_keeps_the_rules_in_the_system_prompt() -> None:
    backend, client = _hosted([FakeBlock("Marko, do petka.")])

    result = backend.chat("transkript", [{"role": "user", "content": "ko radi bazu?"}])

    sent = client.calls[0]
    assert "Odgovaraj isključivo na osnovu transkripta" in sent["system"]
    assert "transkript" in sent["system"]
    assert "ko radi bazu?" in sent["messages"][0]["content"]
    assert result == {"reply": "Marko, do petka."}


def test_a_hosted_failure_is_a_runtime_error_the_page_can_show() -> None:
    class Broken(FakeClient):
        def stream(self, **kwargs):
            raise ConnectionError("no route to host")

    backend = AnthropicBackend(
        Settings(model="base", device="cpu", llm_backend="anthropic"), client=Broken([])
    )

    with pytest.raises(RuntimeError, match="Hosted model nije odgovorio"):
        backend.run_task("t", UNIFY)


def test_an_empty_hosted_answer_is_an_error() -> None:
    backend, _ = _hosted([FakeBlock("   ")])

    with pytest.raises(RuntimeError, match="prazan"):
        backend.run_task("t", UNIFY)
