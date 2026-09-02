# Closing the Loop

A plan for making the transcriber the whole workflow — record, transcribe,
refine, ask, and keep — so a transcript never has to be copied into somebody
else's chat window again.

Measured 1 September 2026 on the development machine: RTX 5050 Laptop GPU,
8151 MiB, with gemma3:4b, qwen3:8b, qwen2.5-coder:7b and llama3.2:3b present in
Ollama. Code references are to `fix/upload-limit-web-only`.

## How to read this

Every claim carries how it is known. Nothing here is recalled from general
knowledge about how Ollama or Whisper behave; the measurements were taken on
this machine, against these models.

| Tag              | Means                                      |
| ---------------- | ------------------------------------------ |
| **[measured]**   | Ran it here. Numbers are in the tables.    |
| **[in code]**    | Read at the cited file.                    |
| **[unverified]** | Needs a test before you trust it.          |

## What the measurements found

One of these changes the plan. The refinement step is silently throwing away
most of every long recording, and has been since the feature shipped.

### Long transcripts are truncated before the model ever sees them

`local_llm.py` sets no `num_ctx` in its options block **[in code]**, so Ollama
loads the model at its own default rather than at the model's maximum.
Confirmed with `ollama ps`: `qwen3:8b` loads at **CONTEXT 4096**, even though
its card advertises 40960 **[measured]**.

A 10,800-word Bosnian transcript through `/api/generate`, reading back
`prompt_eval_count`, with only `num_ctx` changed between runs:

| `num_ctx`      | Prompt tokens accepted | Share of transcript | VRAM       | Placement         |
| -------------- | ---------------------- | ------------------- | ---------- | ----------------- |
| 4096 (default) | 2,050                  | 7%                  | 5.6 GB     | 100% GPU          |
| 8192           | 4,098                  | 15%                 | 6.2 GB     | 100% GPU          |
| 16384          | 8,194                  | 29%                 | 7.8 GB     | 20% CPU / 80% GPU |
| 32768          | 27,811                 | 100%                | not measured | —               |

The first three rows are each almost exactly half the window, which is the
tell: **a prompt that does not fit is truncated to roughly half the context
rather than refused.** The fourth row is the transcript's true length — at
32768 it fit, so nothing was cut, which is how we know the other three were
truncations and not measurements of the text.

No error. No warning. `done_reason` comes back normally and the JSON parses
fine. At the shipped default the model reads 7% of a long meeting and reports
success — which is the worst possible failure mode for a tool whose output you
are going to act on.

Setting `num_ctx` is therefore not a tuning nicety. It is a correctness fix,
and it belongs before any feature work.

### How big the window has to be

Measured on realistic Bosnian prose rather than filler: 2,610 words came to
5,081 tokens through qwen3's tokenizer and 4,091 through gemma3's — **1.95 and
1.57 tokens per word** **[measured]**. At a normal speaking rate of ~140 words
per minute that is roughly 270 tokens per minute of audio, and the window has
to hold the transcript *and* the generated answer on top of it.

| Recording length | Transcript tokens | Window needed with 2048 for output |
| ---------------- | ----------------- | ---------------------------------- |
| 10 minutes       | ~2,700            | 8192                               |
| 20 minutes       | ~5,500            | 8192                               |
| 30 minutes       | ~8,200            | 16384                              |
| 60 minutes       | ~16,400           | 32768                              |

Which runs straight into the VRAM ceiling: `qwen3:8b` cannot hold a 30-minute
meeting on this card without spilling to CPU, and `gemma3:4b` holds 32768 in
2.9 GB entirely on the GPU **[measured]**. That is the whole argument for
choosing a model per job rather than picking one.

### The 8B model does not fit this GPU at meeting-length context

Raising `qwen3:8b` to 16k pushed it to 7.8 GB against 8151 MiB of VRAM, and
Ollama spilled 20% of it to CPU **[measured]**. That is the slow path, and it
gets slower once Whisper wants the card back for the next transcription.

`gemma3:4b` at the same 16k context sits at 2.9 GB, entirely on the GPU, and
its card advertises 131072 max context **[measured]**. It leaves roughly 5 GB
free — enough for `large-v3-turbo` to co-reside instead of thrashing.

So there is no single local model that is both the strongest reasoner and the
one that handles a full meeting on this hardware. The plan has to let you pick
per job, which is what the backend seam in phase 4 is for. Chunked map-reduce
summarisation stops being future work and becomes the thing that makes long
meetings work locally at all.

### qwen3's thinking mode does not break the JSON path

This one was expected to be trouble and is not. `qwen3:8b` lists `thinking`
among its capabilities, but a real call through the existing code path —
`format: "json"`, `temperature: 0` — returned clean parseable JSON with no
`<think>` block and no stray reasoning field **[measured]**. No `think: false`
workaround is needed for `/refine`.

That result does not transfer to phase 2. The chat endpoint will not use
`format: "json"`, and whether thinking leaks into `message.content` there is
untested **[unverified]**.

### Supporting measurements

| Model            | Disk   | Max context | Loaded @16k     | Notes                             |
| ---------------- | ------ | ----------- | --------------- | --------------------------------- |
| gemma3:4b        | 3.3 GB | 131,072     | 2.9 GB, all GPU | Best long-transcript candidate    |
| qwen3:8b         | 5.2 GB | 40,960      | 7.8 GB, spills  | Strongest reasoner, short inputs  |
| qwen2.5-coder:7b | 4.7 GB | —           | —               | Code-tuned; wrong tool here       |
| llama3.2:3b      | 2.0 GB | —           | —               | Current default. Weakest present. |

One more number worth carrying into the design: on a cold call, `load_duration`
was 6.55s of a 7.21s total **[measured]**. Model loading dominates perceived
latency far more than generation does, which is the argument for setting
`keep_alive` rather than for picking a smaller model.

### What has not been verified

- **[unverified]** Whether gemma3:4b's Bosnian and Norwegian output is good
  enough. It fits the card; that says nothing about quality. Score it the way
  the README already scores Whisper — word-level agreement against a
  known-good reference.
- **[unverified]** Whether thinking leaks on `/api/chat` without the JSON
  format constraint.
- **[unverified]** How Ollama truncates, from the head or the tail. Worth
  knowing, because it determines whether today's summaries describe the start
  of your meetings or the end of them.

## Before anything else

**The branch is dirty.** Six files carry uncommitted changes on
`fix/upload-limit-web-only`, the `STT_LANGUAGE` work **[in code]**. Phase 1
touches `web.py`, `config.py` and `tests/test_web.py`, all three already
modified. Land or park that first.

**`.env.example` has flipped to CRLF.** The working-tree copy is `ASCII text,
with CRLF line terminators` while the committed version is plain LF
**[measured]**. That is why its diff shows every line rewritten when only a
comment block changed:

```bash
git config core.autocrlf false
```

Then add `* text=auto eol=lf` to `.gitattributes` so it stays fixed.

**`local_llm.py` and `/refine` have no tests at all.** Nothing under `tests/`
matches `refine` or `ollama` **[measured]**. The 57 test lines that arrived
with commit `b987e0c` covered browser recording and upload handling, not the
LLM path.

Phase 1 rewrites exactly this untested code, so its first commit is not a
feature — it is a test for the behaviour that exists today, with a fake LLM
injected. Without it you are refactoring blind, and the failure mode is silent
bad output rather than a stack trace. That requires one structural change up
front: `create_app` must take the LLM as an injectable collaborator, the way it
already takes `service`. Otherwise every test hits a live Ollama.

## Phase 0 — Configuration, not code

*Done. Shipped with tests; verification note at the end of this section.*

Do this first and re-judge the quality afterwards. It may move the output more
than the next two phases combined, and it costs one file.

```
OLLAMA_MODEL=qwen3:8b
OLLAMA_NUM_CTX=8192
OLLAMA_KEEP_ALIVE=30m
```

Two of those three are new settings, so this phase does touch code, but only to
read them:

- `config.py` — add `ollama_num_ctx` (int, default 8192) and
  `ollama_keep_alive` (str, default `"30m"`) to `Settings.__init__`, following
  the existing `_env` pattern.
- `local_llm.py` — pass `num_ctx` in `options`, and `keep_alive` at the top
  level of the payload.
- `.env.example` — document all three with the measured reason, matching the
  file's existing habit of explaining why each default is what it is.

**Why 8192 and not larger.** It is the largest window that keeps every model
here fully on an 8 GB card: `qwen3:8b` measures 6.2 GB at 8192 and 7.8 GB at
16384, where it spills 20% to CPU and becomes slower than the window it was
meant to fix **[measured]**. At 1.95 tokens per word it covers roughly 20
minutes of speech once `num_predict` is reserved — most voice notes, but not a
real meeting. For hour-long recordings the answer is a smaller model with a
bigger window: `gemma3:4b` holds 32768 in 2.9 GB fully on the GPU. Both are in
`.env.example`, the second commented out.

`num_ctx` and `keep_alive` are required keyword arguments on
`refine_transcript` rather than defaulted, because a default there would be a
second place for the number to live and getting it wrong is silent.

### Verified after implementing

- `Settings` reads `qwen3:8b` / `8192` / `30m`, and both values reach the
  request body the code actually builds **[measured]**.
- `prompt_eval_count` moved from 2,050 to 4,098 on the same transcript.
- `ollama ps` reports `CONTEXT 8192`, 6.2 GB, `100% GPU`, and `keep_alive`
  holds the model resident for 30 minutes.
- 21 tests pass, including seven new ones for `local_llm.py`, which had none.

One caveat that came out of the verification: at 8192 the test transcript is
still truncated, because it is a 27,811-token document. The fix removes the
silent 7% ceiling; it does not make every recording fit. Sizing the window to
the recording is now a documented decision rather than an accident.

## Phase 1 — Any instruction, not one prompt

*Done. Six presets, injectable LLM, 71 tests passing.*

Today the app can do exactly one thing to a transcript. This is the phase that
turns it into the open-ended step currently performed in another window.

Files: new `tasks.py`; rewrite `local_llm.py`; edit `web.py` and `config.py`;
new `tests/test_tasks.py` and `tests/test_local_llm.py`; edit
`tests/test_web.py`.

### Presets as data

A new module keeps the prompts out of the transport code and makes them
trivially testable.

```python
@dataclass(frozen=True)
class Task:
    id: str
    label: str          # shown in the UI, Bosnian
    instruction: str    # appended to the shared safety preamble
    structured: bool    # JSON with fields, or free text

TASKS = (
    Task("refine",  "Pročisti i sažmi",             ..., structured=True),
    Task("minutes", "Zapisnik sa sastanka",         ..., structured=True),
    Task("actions", "Šta treba uraditi",            ..., structured=True),
    Task("unify",   "Složi misli u jednu bilješku", ..., structured=False),
    Task("email",   "Napiši kao email",             ..., structured=False),
    Task("ask",     "Pitaj bilo šta",               "",  structured=False),
)
```

`unify` is the one built for the mixed-language case: it takes a transcript
that switches between Bosnian, Norwegian and English and returns one coherent
note in a chosen language. It is the most valuable preset in the list for that
use case, and the one most sensitive to model quality — which is why phase 0
comes first.

### Splitting local_llm.py

The current module fuses prompt-building and HTTP into one function. Separate
them, because one is pure and testable and the other is not.

```python
SAFETY_PREAMBLE = """..."""   # the existing anti-invention rules, verbatim

def build_prompt(task, text, output_language=None, instruction=None) -> str
def run_task(text, task, model, base_url, *, num_ctx, keep_alive, ...) -> dict
```

**Do not lose the safety preamble.** The existing prompt's rules — do not add
facts, change nothing you are not almost certain about, mark unclear passages
`[nejasno]` rather than guessing, preserve the speaker's order and style
**[in code]** — are what make a local model's output trustworthy enough to act
on. They are worth more than any preset. Every task inherits them, `ask`
included and especially `ask`, since a free-text instruction is exactly where a
model starts inventing to satisfy the request.

### Endpoints

- `GET /tasks` — returns the preset list so the page builds its own dropdown
  and the HTML stays dumb.
- `POST /refine` — payload grows to `{text, task, instruction,
  output_language}`, defaulting `task="refine"` so the existing shape keeps
  working and nothing breaks mid-refactor.

An unknown task id is a 400, and `ask` without an instruction is a 400 with a
message that says what to do about it. The project's existing error strings
explain the fix rather than just naming the fault; match that.

### The page

The single "Pročisti" button becomes a task select, a textarea that appears
only for `ask`, an output-language select, and a run button. Structured results
render into the existing field layout; free-text results render into one block.
Add a copy-to-clipboard button — for this workflow it matters more than the
JSON download already there.

### Tests

- `test_tasks.py` — ids unique, every preset has a non-empty instruction except
  `ask`.
- `test_local_llm.py` — `build_prompt` carries the safety preamble for *every*
  task, includes the transcript, and honours `output_language`. Transport
  tested against a monkeypatched `urlopen`, never a live server.
- `test_web.py` — `/tasks` shape; unknown task 400; `ask` without instruction
  400; a happy path through an injected fake LLM.

### What running it against a real model changed

`unify` works, and it is the reason to have built this: a recording that
switches between Bosnian, Norwegian and English came back as one coherent
Bosnian note with nothing dropped **[measured]**. `minutes` correctly wrote
`[nejasno]` rather than guessing at a name it had not been given — the safety
preamble doing its job.

The design changed on one point. **The output language has to be named; the
model will not infer it** **[measured]**. Left to infer, qwen3:8b answered a
Bosnian transcript in English for `minutes` and `actions`. Three indirect
wordings were measured: the best fixed `minutes`, did nothing for `actions`,
and made `refine` *worse* — it translated and reworded a transcript whose whole
job is to preserve what was said. Naming the language explicitly works
reliably, so the page now sends the language Whisper already detected and
`build_prompt` adds no instruction when none is given.

Two limitations recorded rather than papered over:

- **[measured]** `refine` returns its `cleaned_text` faithfully, mixed
  languages and all, but its `summary` and `key_points` may still come back in
  English. Spelling out a per-field language split did not work and degraded
  `unclear_parts` into listing the entire transcript as unclear. An 8B model
  does not hold that many constraints at once. This belongs to phase 4 and a
  stronger backend, not to a longer prompt.
- **[measured]** Prompt wording moves output quality more than expected at this
  model size: two of three phrasings tried produced an invented summary
  (`"Učlanjenje u projekt"` — nothing in the recording was about joining a
  project) despite the safety rules. Treat any preset change as something to
  re-run, not to reason about.

## Phase 2 — Follow-up, not one shot

*Done. `POST /chat`, server-owned system message, 92 tests passing.*

"Make it shorter." "Now in Norwegian." "What did we actually decide about the
deadline?" None of that is possible today, and it is most of what a
conversation with a transcript consists of.

Use Ollama's `/api/chat` rather than `/api/generate`. It takes a message list
natively, so you are not reassembling a conversation into one string and hoping
the model reads it as dialogue.

The server owns the system message and pins two things into it: the transcript,
and the same safety preamble from phase 1. The client sends only the visible
exchange. That keeps the rules un-editable from the browser, which matters if
this ever leaves your own machine.

Two caps, both returning a clear error rather than hanging:

- **History length**, roughly 20 messages. Beyond that the transcript plus
  history exceeds the context window fixed in phase 0, and you are back to
  silent truncation through a different door.
- **Total characters**, checked before the call, because the failure you want
  is a 413 with an explanation, not a three-minute wait that returns confident
  nonsense about the wrong half of the meeting.

Drop the timeout for this path. The existing 900s **[in code]** is defensible
for a long batch refinement; for an interactive reply it means a wedged model
holds the tab for fifteen minutes. 300s is plenty.

**Not in this phase: streaming.** Token-by-token replies need SSE on the server
and an event-source reader on the page, and they are the kind of thing that
quietly doubles the size of the phase. Build it after phase 3 if the wait
bothers you.

### What running it against a real model showed

**The thinking question is settled: `/api/chat` returns reasoning in its own
`thinking` field and leaves `content` clean** **[measured]**, so reading
`content` is enough. `strip_thinking` stays as cover for a model that inlines
it instead, since nothing constrains the shape of an answer here.

The guardrails hold where it matters. Asked "who is responsible for the
database and by when", the model answered correctly from the transcript. Asked
the follow-up "and what if that does not happen in time", it said the
transcript does not say — refusing to invent, while correctly resolving "in
time" against the previous turn **[measured]**. That second answer is the whole
phase in one exchange: it used the history, and it stayed honest.

The caps are expressed in characters rather than tokens because a character
budget is something the page can tell the user about, and the measured ratio
(1.95 tokens/word, ~3.3 characters/token for Bosnian) makes the conversion
safe at a conservative 3. `MAX_CHAT_MESSAGES` is 20; the character budget is
derived from `OLLAMA_NUM_CTX` minus the reserved output, with the transcript
charged against it first, since it sits in the system message on every turn.

## Phase 3 — Somewhere to keep things

*Done. SQLite library, session endpoints, 115 tests passing.*

Everything before this is features. This is the phase that makes the app a
place you keep work rather than a tab you are afraid to refresh.

Right now the web path never touches disk. The transcript lives in a JavaScript
variable **[in code]**, and a reload destroys it along with every refinement
and every message. For a tool meant to hold meeting records, that is the real
gap.

Files: new `store.py`; edit `web.py` and `config.py`; new `tests/test_store.py`.

### SQLite, and why

Use stdlib `sqlite3`. It adds no dependency — which matters in a project that
deliberately writes its Ollama client against `urllib` rather than pulling in
`requests` — it gives atomic writes, and it makes "list my last 30 recordings,
newest first" a query rather than a directory walk with hand-rolled sorting and
partial-write hazards.

Audio does not go in the database. It goes on disk, referenced by path:

```
<data>/sessions.db
<data>/audio/<session_id>.<ext>
```

`STT_DATA_DIR` defaults per platform — `%LOCALAPPDATA%\delaida-transcriber` on
Windows, `~/.local/share/delaida-transcriber` elsewhere. The project already
carries a Windows/Linux split in `dictate.py`; follow the same shape.

### Schema

```sql
CREATE TABLE sessions (
  id                TEXT PRIMARY KEY,      -- uuid4 hex
  created_at        TEXT NOT NULL,         -- ISO-8601 UTC
  title             TEXT NOT NULL,         -- first ~60 chars, editable
  filename          TEXT,
  audio_path        TEXT,                  -- nullable; see below
  language          TEXT,                  -- requested
  detected_language TEXT,
  text              TEXT NOT NULL,
  segments_json     TEXT NOT NULL          -- the full to_dict() payload
);

CREATE TABLE outputs (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id      TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  created_at      TEXT NOT NULL,
  task            TEXT NOT NULL,
  instruction     TEXT,
  output_language TEXT,
  payload_json    TEXT NOT NULL
);

CREATE TABLE messages (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  created_at TEXT NOT NULL,
  role       TEXT NOT NULL,
  content    TEXT NOT NULL
);

CREATE INDEX idx_sessions_created ON sessions(created_at DESC);
```

Storing `segments_json` whole means the SRT and VTT writers keep working
against a stored session exactly as they do against a fresh one — no second
code path, no drift between them.

**Two traps.** `ON DELETE CASCADE` does nothing by default: Python's `sqlite3`
ships with foreign key enforcement disabled, so you must issue
`PRAGMA foreign_keys = ON` on *every* connection. Without it, deleting a
session silently orphans its outputs and messages and the tests still pass.
And FastAPI runs sync handlers in a threadpool, where a SQLite connection
cannot be shared across threads unless you opt out of the check — so open and
close a connection inside each store method rather than holding one on the
instance. For a single-user local app the cost is irrelevant and it removes the
entire class of problem. Wrap the calls in `asyncio.to_thread` from the async
handlers, the way `/refine` already does.

### The store

```python
class SessionStore:
    def __init__(self, path: Path) -> None      # creates tables if absent
    def create(...) -> str
    def get(session_id) -> Session | None
    def list(limit=50, offset=0) -> list[SessionSummary]
    def add_output(session_id, task, payload) -> None
    def add_message(session_id, role, content) -> None
    def messages(session_id) -> list[dict]
    def rename(session_id, title) -> None
    def delete(session_id) -> None              # also unlinks the audio
```

Set `PRAGMA user_version` and put schema creation behind a small `_migrate()`.
It costs a few lines now and saves an awkward afternoon the first time you add
a column.

### Endpoints

| Method | Path                  | Does                                           |
| ------ | --------------------- | ---------------------------------------------- |
| POST   | /transcribe           | Unchanged, but persists and returns session_id |
| GET    | /sessions             | Recent sessions, newest first, paged           |
| GET    | /sessions/{id}        | Transcript, outputs and messages together      |
| PATCH  | /sessions/{id}        | Rename                                         |
| DELETE | /sessions/{id}        | Removes rows and the audio file                |
| POST   | /sessions/{id}/refine | Phase 1, result persisted                      |
| POST   | /sessions/{id}/chat   | Phase 2, both turns persisted                  |

The delete path needs a confirmation step in the UI and must tolerate an
already-missing audio file without raising. It is the only destructive action
in the app.

### Keeping audio is a decision, not a default

`STT_KEEP_AUDIO`, defaulting to false. Transcripts are saved either way; the
original recording is only kept if you say so.

The reason is that this project's entire premise is that recordings stay
private, and quietly accumulating every meeting's audio on disk forever is a
meaningful change to what the tool does with your data. That should be
something you switch on knowing you switched it on, not something you discover
a year later when the folder is 40 GB.

### The page

A list of recent sessions, click to load. Deep-link each one as
`/#<session_id>` so a refresh restores what you were looking at — which is the
specific complaint this whole phase exists to answer.

### Optionally, the CLI

`delaida-transcriber --save` writes batch runs into the same library, so folder
transcription and phone recordings end up in one place rather than two. Small
change, and it is what makes the library feel complete rather than web-only.

### Tests

- `test_store.py` against a `tmp_path` database — round-trip, ordering,
  rename, and an explicit cascade test, since that is the one the missing
  pragma breaks.
- `test_web.py` — `create_app(settings, service=…, llm=…, store=…)` with a
  store on `tmp_path`; transcribe returns an id; the session reads back; delete
  removes it.

### What building it turned up

The end-to-end check is the one that matters, and it passes: upload, refine
with the real local model, ask a question, rename, then **tear the application
down and rebuild it from disk** — title, transcript, outputs, conversation and
listing all come back, and a delete leaves a 404 behind **[measured]**.

Both predicted traps were real. `ON DELETE CASCADE` does nothing without
`PRAGMA foreign_keys = ON` on every connection, and the test that catches it
had to be written deliberately, because orphaned rows are invisible to every
other assertion.

One trap that was **not** predicted, and is worth more than either: because
`create_app` falls back to a store at `settings.data_dir`, the first test run
after wiring it up wrote **four rows into the real session database** under
`%LOCALAPPDATA%`. The tests passed. Nothing looked wrong. The only symptom
would have been somebody else's data appearing in the library later. Fixed with
an autouse fixture in `tests/conftest.py` that points `STT_DATA_DIR` at
`tmp_path` for every test — autouse rather than opt-in precisely because the
failure is silent, and the stray rows were removed.

The lesson generalises: a convenient default that reaches outside the process
is a hazard in tests, and injection alone does not protect you when the
fallback still exists.

## Phase 4 — A choice of model, and Norwegian

*Done. `llm_backends.py`, Norwegian priming, 131 tests passing.*

Two separate things that land together because they serve the same person: the
colleague with thoughts in three languages.

### The seam

Mirror `backends.py` exactly. That file already exists to let a hosted
transcription backend slot in without touching the CLI, the web app, or
dictation **[in code]**; the LLM side deserves the same shape, and copying a
pattern the project already uses costs almost nothing to understand later.

```python
class LLMBackend(Protocol):
    async def complete(self, prompt: str, *, structured: bool) -> str: ...
    async def chat(self, messages: list[dict]) -> str: ...

def create_llm_backend(settings) -> LLMBackend    # LLM_BACKEND=ollama|anthropic
```

Ollama stays the default. The hosted path goes behind an optional extra,
`pip install -e '.[hosted]'`, so the normal install stays lean and works with
no network.

### If you add the hosted path

Use the official `anthropic` SDK rather than hand-rolled HTTP. The project uses
`urllib` for Ollama because Ollama's API is three fields, which is not true
here.

```python
client = anthropic.Anthropic()          # reads ANTHROPIC_API_KEY
response = client.messages.create(
    model="claude-opus-5",
    max_tokens=16000,
    thinking={"type": "adaptive"},
    system=SAFETY_PREAMBLE,
    messages=[{"role": "user", "content": prompt}],
)
```

Three details that are easy to get wrong: `claude-opus-5` is the current model
id and takes no date suffix; `budget_tokens` no longer exists and is rejected
outright, with `thinking={"type": "adaptive"}` replacing it; and for long
transcripts use `client.messages.stream(...)` with `.get_final_message()`,
because a large `max_tokens` on a non-streaming call runs into HTTP timeouts.

A one-hour meeting is comfortably inside a 1M context window, so the chunking
problem that local models force on you does not arise on this path. That is
most of the argument for having it.

**The hosted path must be visible, opt-in, and per-request.** The README's
first paragraph promises that nothing is sent to a cloud service. A hosted LLM
backend makes that promise conditional, and a conditional promise that is not
surfaced in the interface is just a false one. So: off by default; chosen per
request rather than set globally and forgotten; and labelled in the page at the
moment of use — *this request leaves your computer* — not in a settings screen
nobody reopens. Update the README's opening claim in the same commit that adds
the backend, not later.

### Norwegian is blocked in three places

| Where              | Problem                                                              | Fix                                                                         |
| ------------------ | -------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| `service.py`       | `SUPPORTED_LANGUAGE_HINTS` is `{auto, bs, hr, en}`; `no` is rejected  | Add `no`. Whisper's code is `no`; there is no `nb`. `nn` exists for Nynorsk. |
| `web.py`, `cli.py` | No Norwegian option in the select or in `--language` choices          | Add to both                                                                 |
| `config.py`        | Priming is Bosnian-specific and works against Norwegian audio         | Per-language profiles                                                       |

The third is the substantive one. `DEFAULT_INITIAL_PROMPT` is a Bosnian
sentence and `DEFAULT_HOTWORDS` is Bosnian/English jargon **[in code]**. The
README's own measurements show priming is worth about 2.6 points precisely
because it steers what Whisper reaches for. Steering a Norwegian recording
toward Bosnian vocabulary is not neutral — it is a penalty.

Replace the two globals with a mapping from language to profile, falling back
to no priming for languages without one. Keep `STT_INITIAL_PROMPT` working as a
global override so existing `.env` files behave unchanged.

**Keep the Bosnian profile byte-identical.** The README's accuracy table is the
most valuable documentation in this repository — it records what was measured
and what was tried and rejected, so nobody re-runs those experiments. If the
refactor changes the Bosnian priming text even slightly, every number in that
table becomes unverifiable. Move the strings; do not edit them. If you do
change them, re-measure and update the table in the same commit.

One more, easy to miss: dictation forces `hr` **[in code]**, so a colleague
dictating in Norwegian would have it decoded as Croatian. The `--language` flag
on the dictate command already accepts a value; it just needs the widened hint
set to pass validation.

### What was built, and what was not

The seam is `llm_backends.py`, the same shape as `backends.py`. It is
deliberately free of `num_ctx` and `keep_alive` — those are Ollama's problem,
and a hosted model with a million-token window does not have it, so the backend
holds them and callers no longer pass them. Both implementations share
`build_prompt` and `parse_task_answer`: only the transport differs, which is
the whole point.

Norwegian works end to end **[measured]**. A Norwegian recording refined into a
Bosnian note, and a follow-up question answered in Norwegian, both through the
local model. `priming_for("no")` returns the Norwegian profile while `hr` still
returns the original Bosnian one, byte for byte — the README's accuracy table
stays verifiable.

Three things to be clear about:

- **No hosted request was ever sent.** The Anthropic backend is implemented and
  tested against a fake client that records what *would* go out — the current
  API shape (`claude-opus-5`, `thinking: {"type": "adaptive"}`, streaming,
  no `budget_tokens`), the prompt, and the parsing. Actually calling it costs
  money, and that is not a decision to make on someone's behalf.
- **The hosted path is per-configuration, not per-request.** `LLM_BACKEND`
  selects it, and the page shows a banner naming the backend and saying the
  transcript leaves the machine whenever it is not local. Per-request selection
  — a checkbox on each action — is not built.
- **The README's opening claim was rewritten in the same change**, not left for
  later. Transcription is still always local; the sentence now says exactly
  which step can leave and under what conditions.

## Order, and what each phase buys

| # | Phase            | Effort      | What changes for you                                 |
| - | ---------------- | ----------- | ---------------------------------------------------- |
| 0 | Configuration    | done | Long recordings stop being truncated. Quality jumps. |
| 1 | Any instruction  | done | The app can do what you do in the other window.      |
| 2 | Follow-up chat   | done | You can iterate instead of re-running.               |
| 3 | Session storage  | done | Nothing is lost on refresh. It becomes a place.      |
| 4 | Seam + Norwegian | done | Your colleague can actually use it.                  |

All five are built, with 131 tests passing and ruff clean. What is left is not
in this plan: streaming replies, map-reduce chunking for recordings longer than
the context window, `delaida-transcriber --save` for batch runs, and scoring
gemma3:4b's Bosnian against qwen3:8b so the long-meeting configuration can be
recommended on evidence rather than on the fact that it fits.

One thing outside the code still blocks routine work: `pytest` cannot run on
the development machine at all, because PyAV's DLL is blocked by a Windows
Application Control policy **[measured]**. Every test run in this plan used a
stubbed `faster_whisper` to get around it. That policy needs resolving before
the suite is usable normally.

## Risks

- **VRAM contention.** Whisper and the LLM both want the 8 GB card.
  `keep_alive` helps latency and hurts contention; if transcription starts
  crawling after a refinement, that is what happened.
- **Long meetings still exceed local context.** Phase 0 raises the ceiling; it
  does not remove it. Map-reduce chunking — summarise per chunk, then summarise
  the summaries — is the local answer, and it is real work with its own quality
  cost.
- **A local model still trails a hosted one.** The README already says this
  honestly about Whisper. It is at least as true of a 4B or 8B doing
  multilingual synthesis. Phase 4 exists so you can choose per job.
- **Scope.** Phase 3 is bigger than the other four combined. If time is short,
  ship 0–2, use it for a week, then decide whether the storage design here
  still matches how you actually work.
