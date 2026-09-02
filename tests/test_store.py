import sqlite3

import pytest

from delaida_transcriber.store import SessionStore, title_from


@pytest.fixture
def store(tmp_path) -> SessionStore:
    return SessionStore(tmp_path / "data")


def _create(store: SessionStore, text: str = "prvi snimak o rokovima", **kwargs) -> str:
    return store.create(
        text=text,
        segments={"segments": [], "detected_language": "hr"},
        filename=kwargs.pop("filename", "note.m4a"),
        detected_language=kwargs.pop("detected_language", "hr"),
        **kwargs,
    )


def test_a_transcript_survives_being_written_and_read_back(store) -> None:
    """The web path used to keep this in a JavaScript variable, where a reload
    destroyed it along with everything made from it."""
    session_id = _create(store)

    session = store.get(session_id)

    assert session is not None
    assert session.text == "prvi snimak o rokovima"
    assert session.detected_language == "hr"
    assert session.filename == "note.m4a"
    assert session.segments == {"segments": [], "detected_language": "hr"}


def test_the_title_comes_from_what_was_said(store) -> None:
    assert title_from("  kratko  ") == "kratko"
    assert title_from("") == "Bez teksta"
    long = title_from("riječ " * 40)
    assert len(long) <= 61 and long.endswith("…")

    assert store.get(_create(store)).title == "prvi snimak o rokovima"


def test_sessions_come_back_newest_first(store) -> None:
    first = _create(store, "najstariji")
    second = _create(store, "srednji")
    third = _create(store, "najnoviji")

    listed = [summary.id for summary in store.list()]

    # Same-second timestamps are broken by rowid, so insertion order still wins.
    assert listed == [third, second, first]
    assert store.count() == 3


def test_the_listing_is_paged(store) -> None:
    for index in range(5):
        _create(store, f"snimak {index}")

    assert len(store.list(limit=2)) == 2
    assert len(store.list(limit=2, offset=4)) == 1
    assert store.list(limit=2)[0].id != store.list(limit=2, offset=2)[0].id


def test_outputs_and_messages_hang_off_the_session(store) -> None:
    session_id = _create(store)
    store.add_output(
        session_id, task="unify", payload={"kind": "text", "output": "bilješka"},
        output_language="bs",
    )
    store.add_messages(
        session_id,
        [{"role": "user", "content": "ko radi bazu?"}, {"role": "assistant", "content": "Marko"}],
    )

    session = store.get(session_id)

    assert session.outputs[0]["task"] == "unify"
    assert session.outputs[0]["payload"]["output"] == "bilješka"
    assert session.outputs[0]["output_language"] == "bs"
    assert [m["role"] for m in session.messages] == ["user", "assistant"]
    assert session.messages[1]["content"] == "Marko"


def test_deleting_a_session_takes_its_outputs_and_messages_with_it(store) -> None:
    """Python's sqlite3 ships with foreign keys *off*, so ON DELETE CASCADE does
    nothing until the pragma is issued on the connection. Without this test the
    rows would be orphaned silently and everything else would still pass."""
    session_id = _create(store)
    store.add_output(session_id, task="refine", payload={"kind": "text", "output": "x"})
    store.add_messages(session_id, [{"role": "user", "content": "q"}])
    keep = _create(store, "drugi snimak")

    assert store.delete(session_id) is True

    with sqlite3.connect(store.path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM outputs").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 0
    assert store.get(session_id) is None
    assert store.get(keep) is not None


def test_deleting_something_that_is_not_there_is_not_an_error(store) -> None:
    assert store.delete("nepostojeci") is False


def test_audio_is_written_only_when_it_is_handed_over(store) -> None:
    """Off by default: keeping every meeting's audio forever is a real change to
    what the tool does with your data, so the caller decides."""
    without = _create(store)
    withaudio = _create(store, audio=b"RIFFfake", suffix=".wav")

    assert store.audio_path(without) is None
    assert store.audio_path(withaudio).read_bytes() == b"RIFFfake"
    assert store.get(withaudio).to_dict()["has_audio"] is True
    assert store.list()[1].has_audio is False


def test_deleting_a_session_removes_its_audio(store) -> None:
    session_id = _create(store, audio=b"RIFFfake", suffix=".wav")
    path = store.audio_path(session_id)

    store.delete(session_id)

    assert not path.exists()


def test_a_missing_audio_file_does_not_block_deletion(store) -> None:
    """The row is the record; the file is a cache of it. A half-deleted session
    must not become undeletable."""
    session_id = _create(store, audio=b"RIFFfake", suffix=".wav")
    store.audio_path(session_id).unlink()

    assert store.delete(session_id) is True


def test_renaming(store) -> None:
    session_id = _create(store)

    assert store.rename(session_id, "  Sastanak o bazi  ") is True
    assert store.get(session_id).title == "Sastanak o bazi"
    assert store.rename("nepostojeci", "bilo šta") is False
    with pytest.raises(ValueError):
        store.rename(session_id, "   ")


def test_reopening_the_store_keeps_what_was_there(tmp_path) -> None:
    root = tmp_path / "data"
    session_id = _create(SessionStore(root))

    assert SessionStore(root).get(session_id) is not None


def test_the_schema_is_versioned(store) -> None:
    """Cheap now; saves an awkward afternoon the first time a column is added."""
    with sqlite3.connect(store.path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 1
