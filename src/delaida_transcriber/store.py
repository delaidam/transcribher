"""Where recordings, their transcripts, and everything made from them are kept.

SQLite through the standard library rather than a file per session: it adds no
dependency -- which matters in a project that writes its Ollama client against
``urllib`` rather than pulling in ``requests`` -- it gives atomic writes, and it
turns "my last thirty recordings, newest first" into a query instead of a
directory walk with hand-rolled sorting and partial-write hazards.

Audio does not go in the database. It goes on disk beside it, referenced by
name, and only when ``STT_KEEP_AUDIO`` says so.
"""

import json
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

SCHEMA_VERSION = 1

# How much of the transcript becomes the name in the list, before the user
# renames it to something they will recognise later.
TITLE_LENGTH = 60

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
  id                TEXT PRIMARY KEY,
  created_at        TEXT NOT NULL,
  title             TEXT NOT NULL,
  filename          TEXT,
  audio_name        TEXT,
  language          TEXT,
  detected_language TEXT,
  text              TEXT NOT NULL,
  segments_json     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS outputs (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id      TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  created_at      TEXT NOT NULL,
  task            TEXT NOT NULL,
  instruction     TEXT,
  output_language TEXT,
  payload_json    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  created_at TEXT NOT NULL,
  role       TEXT NOT NULL,
  content    TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_sessions_created ON sessions(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_outputs_session ON outputs(session_id);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
"""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def title_from(text: str) -> str:
    """A recognisable name for a recording, taken from what was said in it."""
    first = " ".join(text.split())
    if not first:
        return "Bez teksta"
    return first if len(first) <= TITLE_LENGTH else first[:TITLE_LENGTH].rstrip() + "…"


@dataclass(frozen=True)
class SessionSummary:
    id: str
    created_at: str
    title: str
    detected_language: str | None
    filename: str | None
    has_audio: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "created_at": self.created_at,
            "title": self.title,
            "detected_language": self.detected_language,
            "filename": self.filename,
            "has_audio": self.has_audio,
        }


@dataclass(frozen=True)
class Session:
    id: str
    created_at: str
    title: str
    filename: str | None
    audio_name: str | None
    language: str | None
    detected_language: str | None
    text: str
    segments: dict
    outputs: list[dict] = field(default_factory=list)
    messages: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "created_at": self.created_at,
            "title": self.title,
            "filename": self.filename,
            "has_audio": bool(self.audio_name),
            "requested_language": self.language,
            "text": self.text,
            **self.segments,
            "outputs": self.outputs,
            "messages": self.messages,
        }


class SessionStore:
    """Every method opens its own connection. See ``_connect``."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).expanduser()
        self.audio_dir = self.root / "audio"
        self.path = self.root / "sessions.db"
        self.root.mkdir(parents=True, exist_ok=True)
        self._migrate()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """One connection per call, foreign keys switched on every time.

        Two traps live here. Python's sqlite3 ships with foreign key
        enforcement *off*, so ON DELETE CASCADE does nothing until the pragma
        is issued -- deleting a session would silently orphan its outputs and
        messages, and the tests would still pass. And FastAPI runs sync handlers
        in a threadpool, where a connection cannot be shared across threads
        unless you opt out of the safety check; opening one per call removes
        that whole class of problem, and for a single-user local app the cost is
        irrelevant.
        """
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _migrate(self) -> None:
        with self._connect() as connection:
            connection.executescript(_SCHEMA)
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            if version < SCHEMA_VERSION:
                connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    # --- writing ------------------------------------------------------------

    def create(
        self,
        *,
        text: str,
        segments: dict,
        filename: str | None = None,
        language: str | None = None,
        detected_language: str | None = None,
        audio: bytes | None = None,
        suffix: str = "",
    ) -> str:
        """Save one transcription and return its id.

        ``audio`` is written only when it is given; the caller decides that from
        ``STT_KEEP_AUDIO``. Keeping the file lifecycle here rather than at the
        call site is what lets ``delete`` clean up after itself.
        """
        session_id = uuid.uuid4().hex
        audio_name = None
        if audio:
            self.audio_dir.mkdir(parents=True, exist_ok=True)
            audio_name = f"{session_id}{suffix}"
            (self.audio_dir / audio_name).write_bytes(audio)

        with self._connect() as connection:
            connection.execute(
                "INSERT INTO sessions (id, created_at, title, filename, audio_name,"
                " language, detected_language, text, segments_json)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    session_id,
                    _now(),
                    title_from(text),
                    filename,
                    audio_name,
                    language,
                    detected_language,
                    text,
                    json.dumps(segments, ensure_ascii=False),
                ),
            )
        return session_id

    def add_output(
        self,
        session_id: str,
        *,
        task: str,
        payload: dict,
        instruction: str | None = None,
        output_language: str | None = None,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO outputs (session_id, created_at, task, instruction,"
                " output_language, payload_json) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    session_id,
                    _now(),
                    task,
                    instruction,
                    output_language,
                    json.dumps(payload, ensure_ascii=False),
                ),
            )

    def add_messages(self, session_id: str, messages: list[dict[str, str]]) -> None:
        with self._connect() as connection:
            connection.executemany(
                "INSERT INTO messages (session_id, created_at, role, content)"
                " VALUES (?, ?, ?, ?)",
                [(session_id, _now(), m["role"], m["content"]) for m in messages],
            )

    def rename(self, session_id: str, title: str) -> bool:
        title = " ".join(title.split())[:200]
        if not title:
            raise ValueError("Naslov ne može biti prazan.")
        with self._connect() as connection:
            changed = connection.execute(
                "UPDATE sessions SET title = ? WHERE id = ?", (title, session_id)
            ).rowcount
        return bool(changed)

    def delete(self, session_id: str) -> bool:
        """Remove a session, its outputs and messages, and its audio file."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT audio_name FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
            if row is None:
                return False
            connection.execute("DELETE FROM sessions WHERE id = ?", (session_id,))

        if row["audio_name"]:
            # missing_ok: the row is the record, the file is a cache of it, and
            # a half-deleted session should not be undeletable.
            (self.audio_dir / row["audio_name"]).unlink(missing_ok=True)
        return True

    # --- reading ------------------------------------------------------------

    def list(self, limit: int = 50, offset: int = 0) -> list[SessionSummary]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id, created_at, title, detected_language, filename, audio_name"
                " FROM sessions ORDER BY created_at DESC, rowid DESC LIMIT ? OFFSET ?",
                (max(1, min(limit, 200)), max(0, offset)),
            ).fetchall()
        return [
            SessionSummary(
                id=row["id"],
                created_at=row["created_at"],
                title=row["title"],
                detected_language=row["detected_language"],
                filename=row["filename"],
                has_audio=bool(row["audio_name"]),
            )
            for row in rows
        ]

    def count(self) -> int:
        with self._connect() as connection:
            return connection.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]

    def get(self, session_id: str) -> Session | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
            if row is None:
                return None
            outputs = connection.execute(
                "SELECT created_at, task, instruction, output_language, payload_json"
                " FROM outputs WHERE session_id = ? ORDER BY id",
                (session_id,),
            ).fetchall()
            messages = connection.execute(
                "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id",
                (session_id,),
            ).fetchall()

        return Session(
            id=row["id"],
            created_at=row["created_at"],
            title=row["title"],
            filename=row["filename"],
            audio_name=row["audio_name"],
            language=row["language"],
            detected_language=row["detected_language"],
            text=row["text"],
            segments=json.loads(row["segments_json"]),
            outputs=[
                {
                    "created_at": output["created_at"],
                    "task": output["task"],
                    "instruction": output["instruction"],
                    "output_language": output["output_language"],
                    "payload": json.loads(output["payload_json"]),
                }
                for output in outputs
            ],
            messages=[{"role": m["role"], "content": m["content"]} for m in messages],
        )

    def audio_path(self, session_id: str) -> Path | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT audio_name FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
        if row is None or not row["audio_name"]:
            return None
        path = self.audio_dir / row["audio_name"]
        return path if path.is_file() else None
