"""Small Ollama client used for optional, fully local work on a transcript.

Prompt building is separated from the HTTP call because one is pure and worth
testing exhaustively, and the other needs a running server. ``build_prompt`` is
where the rules that keep the model honest are guaranteed to reach every task.
"""

import json
import re
from urllib.error import URLError
from urllib.request import Request, urlopen

from delaida_transcriber.tasks import (
    CHAT_SYSTEM,
    OUTPUT_LANGUAGE_NAMES,
    SAFETY_PREAMBLE,
    Task,
)

# A conversation is interactive, so it fails differently from a batch
# refinement: waiting fifteen minutes on a wedged model holds the tab. The
# generate path keeps the longer timeout, this one does not.
CHAT_TIMEOUT = 300

# Beyond this the transcript plus the history stops fitting the context window
# and Ollama truncates it to about half -- silently, which is the failure this
# whole area exists to avoid. Refusing with an explanation is better than
# answering confidently about a conversation the model can no longer see.
MAX_CHAT_MESSAGES = 20

# Bosnian measured 1.95 tokens per word and roughly 3.3 characters per token, so
# characters are a usable proxy for a budget the user can actually be told
# about. Conservative at 3, and num_predict is reserved out of the window.
CHARS_PER_TOKEN = 3
RESERVED_OUTPUT_TOKENS = 2048

# Reasoning models emit their scratchpad before the answer. Constraining the
# response to JSON suppresses it, but the free-text tasks have no such
# constraint, so the block has to come off before the text reaches the page.
_THINKING = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)


def strip_thinking(text: str) -> str:
    """Remove a leading reasoning block, if the model emitted one."""
    return _THINKING.sub("", text, count=1).strip()


def build_prompt(
    task: Task,
    text: str,
    *,
    output_language: str | None = None,
    instruction: str | None = None,
) -> str:
    """Assemble the prompt for one task over one transcript.

    ``instruction`` supplies the request for the free-text ``ask`` task and is
    ignored by every preset that carries its own.
    """
    steps = [SAFETY_PREAMBLE, ""]

    task_instruction = task.instruction or (instruction or "").strip()
    steps.append(f"ZADATAK:\n{task_instruction}")

    # Naming the language works; asking the model to infer it does not. Left to
    # infer, qwen3:8b answered a Bosnian transcript in English for "minutes" and
    # "actions". Three attempts at wording it indirectly were measured: the best
    # one fixed "minutes", did nothing for "actions", and made "refine" worse --
    # it translated and reworded a transcript it was supposed to preserve.
    #
    # So there is no implicit line here. The page sends the language Whisper
    # detected, which turns this into the explicit case that measurably works.
    language_name = OUTPUT_LANGUAGE_NAMES.get(output_language or "", "")
    if output_language and language_name:
        steps.append(f"\nOdgovor napiši na {language_name}, bez obzira na jezik snimka.")

    if task.structured:
        keys = ", ".join(task.keys)
        steps.append(
            f"\nVrati isključivo JSON sa poljima: {keys}. "
            "Polja koja nabrajaju više stvari vrati kao listu stringova."
        )
    else:
        steps.append("\nVrati samo tekst odgovora, bez JSON-a i bez uvoda o tome šta radiš.")

    steps.append(f"\nSIROVI TRANSKRIPT:\n{text}")
    return "\n".join(steps)


def _post(payload: dict, base_url: str, timeout: int, path: str = "/api/generate") -> dict:
    request = Request(
        f"{base_url.rstrip('/')}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (URLError, TimeoutError) as error:
        raise RuntimeError(
            "Ollama nije dostupna ili je obrada istekla. Provjeri da 'ollama serve' radi "
            "i da model postoji."
        ) from error


def run_task(
    text: str,
    task: Task,
    model: str,
    base_url: str,
    *,
    num_ctx: int,
    keep_alive: str,
    output_language: str | None = None,
    instruction: str | None = None,
    timeout: int = 900,
) -> dict[str, object]:
    """Run one task against Ollama and return a payload the page can render.

    ``num_ctx`` and ``keep_alive`` are required rather than defaulted, because a
    default here would be a second place for the context size to live -- and
    getting it wrong is silent. Ollama truncates an over-long prompt to about
    half the window instead of refusing it, so too small a value returns a
    confident answer about part of the recording. Settings owns the number; see
    the measurements on DEFAULT_OLLAMA_NUM_CTX.
    """
    payload: dict[str, object] = {
        "model": model,
        "prompt": build_prompt(
            task, text, output_language=output_language, instruction=instruction
        ),
        "stream": False,
        "keep_alive": keep_alive,
        "options": {
            "temperature": 0,
            "seed": 0,
            "num_predict": 2048,
            "num_ctx": num_ctx,
        },
    }
    if task.structured:
        payload["format"] = "json"

    body = _post(payload, base_url, timeout)
    try:
        answer = body["response"]
    except (KeyError, TypeError) as error:
        raise RuntimeError("Ollama je vratila neispravan odgovor.") from error
    return parse_task_answer(answer, task)


def parse_task_answer(answer: object, task: Task) -> dict[str, object]:
    """Turn a model's raw answer into something the page can render.

    Shared by both backends: the prompt is the same either way, so the shape of
    what comes back is too, and only the transport differs.
    """
    if not task.structured:
        output = strip_thinking(str(answer))
        if not output:
            raise RuntimeError("Ollama je vratila prazan odgovor.")
        return {"kind": "text", "task": task.id, "output": output}

    try:
        result = json.loads(answer)
    except (TypeError, json.JSONDecodeError) as error:
        raise RuntimeError("Ollama je vratila neispravan odgovor.") from error
    if not isinstance(result, dict):
        raise RuntimeError("Ollama je vratila neispravan odgovor.")

    # The first field is the task's primary output; without it there is nothing
    # to show and an empty page would read as success. The rest are filled in
    # rather than demanded, because a smaller model drops one now and then and
    # a partial answer still beats an error.
    primary = task.keys[0]
    if primary not in result:
        raise RuntimeError("Ollama odgovor nema očekivana polja.")

    return {
        "kind": "structured",
        "task": task.id,
        "fields": [
            {"key": key, "label": label, "value": result.get(key, "")}
            for key, label in task.fields
        ],
    }


def chat_budget_chars(num_ctx: int) -> int:
    """How much transcript plus history fits, expressed in characters."""
    return max(0, num_ctx - RESERVED_OUTPUT_TOKENS) * CHARS_PER_TOKEN


def build_messages(
    transcript: str,
    history: list[dict[str, str]],
    *,
    output_language: str | None = None,
) -> list[dict[str, str]]:
    """Put the transcript and the rules in a system message, history after it.

    The system message is assembled here rather than accepted from the client,
    so a browser cannot edit the rules out of the conversation.
    """
    system = CHAT_SYSTEM.format(transcript=transcript)
    language_name = OUTPUT_LANGUAGE_NAMES.get(output_language or "", "")
    if output_language and language_name:
        # Naming it works; asking the model to infer it does not -- see the note
        # in build_prompt.
        system += f"\n\nOdgovaraj na {language_name}."
    return [{"role": "system", "content": system}, *history]


def validate_history(history: object, budget_chars: int) -> list[dict[str, str]]:
    """Check a conversation before it costs a model call.

    Raises ``ValueError`` with something the page can show. Length is checked
    here rather than left to the model because the model does not fail on too
    much input -- Ollama truncates to about half the window and answers anyway.
    """
    if not isinstance(history, list) or not history:
        raise ValueError("Nema pitanja.")
    if len(history) > MAX_CHAT_MESSAGES:
        raise ValueError(
            f"Razgovor je predugačak ({len(history)} poruka, najviše "
            f"{MAX_CHAT_MESSAGES}). Počni novi razgovor o istom transkriptu."
        )

    messages: list[dict[str, str]] = []
    for entry in history:
        if not isinstance(entry, dict):
            raise ValueError("Neispravan oblik poruke.")
        role, content = entry.get("role"), (entry.get("content") or "").strip()
        if role not in ("user", "assistant"):
            raise ValueError(f"Neispravna uloga {role!r}; očekuje se user ili assistant.")
        if not content:
            raise ValueError("Prazna poruka.")
        messages.append({"role": role, "content": content})

    if messages[-1]["role"] != "user":
        raise ValueError("Posljednja poruka mora biti pitanje.")

    used = sum(len(message["content"]) for message in messages)
    if used > budget_chars:
        raise ValueError(
            f"Razgovor i transkript zajedno prelaze ono što model može držati "
            f"({used} znakova, prostor je {budget_chars}). Skrati transkript ili "
            f"povećaj OLLAMA_NUM_CTX."
        )
    return messages


def chat(
    transcript: str,
    history: list[dict[str, str]],
    model: str,
    base_url: str,
    *,
    num_ctx: int,
    keep_alive: str,
    output_language: str | None = None,
    timeout: int = CHAT_TIMEOUT,
) -> dict[str, object]:
    """Answer one turn of a conversation about a transcript."""
    payload: dict[str, object] = {
        "model": model,
        "messages": build_messages(transcript, history, output_language=output_language),
        "stream": False,
        "keep_alive": keep_alive,
        "options": {
            "temperature": 0,
            "seed": 0,
            "num_predict": RESERVED_OUTPUT_TOKENS,
            "num_ctx": num_ctx,
        },
    }
    body = _post(payload, base_url, timeout, path="/api/chat")
    message = body.get("message") if isinstance(body, dict) else None
    if not isinstance(message, dict):
        raise RuntimeError("Ollama je vratila neispravan odgovor.")

    # Measured against qwen3:8b: this endpoint returns reasoning in its own
    # "thinking" field and leaves "content" clean, so reading content is enough.
    # strip_thinking stays as cover for a model that inlines it instead --
    # unlike the generate path, nothing here constrains the shape of the answer.
    reply = strip_thinking(str(message.get("content") or ""))
    if not reply:
        raise RuntimeError("Ollama je vratila prazan odgovor.")
    return {"reply": reply}
