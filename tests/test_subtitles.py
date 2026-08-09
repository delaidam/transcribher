from delaida_transcriber.models import (
    FileTranscription,
    TranscriptionSegment,
    TranscriptionWord,
)
from delaida_transcriber.subtitles import to_srt, to_vtt


def _result(*segments: tuple[float, float, str]) -> FileTranscription:
    parts = [TranscriptionSegment(start, end, text) for start, end, text in segments]
    return FileTranscription(" ".join(p.text for p in parts), "hr", 0.9, parts)


def test_srt_matches_the_expected_shape() -> None:
    srt = to_srt(_result((0.519, 4.859, "Neću moći više."), (7.299, 19.039, "Kako ja radim?")))

    assert srt == (
        "1\n"
        "00:00:00,519 --> 00:00:04,859\n"
        "Neću moći više.\n"
        "\n"
        "2\n"
        "00:00:07,299 --> 00:00:19,039\n"
        "Kako ja radim?\n"
    )


def test_vtt_has_the_required_header_and_dot_separator() -> None:
    vtt = to_vtt(_result((0.519, 4.859, "Neću moći više.")))

    assert vtt.startswith("WEBVTT\n\n")
    assert "00:00:00.519 --> 00:00:04.859" in vtt
    assert "," not in vtt.split("-->")[0]


def test_hours_roll_over_correctly() -> None:
    srt = to_srt(_result((3661.5, 3725.25, "Sat vremena kasnije.")))

    assert "01:01:01,500 --> 01:02:05,250" in srt


def test_empty_segments_are_skipped_and_numbering_stays_contiguous() -> None:
    srt = to_srt(_result((0.0, 1.0, "prvi"), (1.0, 2.0, "   "), (2.0, 3.0, "drugi")))

    assert "1\n" in srt and "2\n" in srt
    assert "3\n" not in srt
    assert srt.count("-->") == 2


def test_reversed_or_zero_length_cue_is_clamped() -> None:
    """A cue whose end precedes its start makes players drop the line."""
    srt = to_srt(_result((5.0, 4.0, "obrnuto")))

    assert "00:00:05,000 --> 00:00:05,000" in srt


def test_negative_start_is_clamped_to_zero() -> None:
    assert "00:00:00,000" in to_srt(_result((-0.4, 1.0, "prije nule")))


def test_no_segments_produces_empty_srt_and_bare_vtt() -> None:
    empty = FileTranscription("", None, None, [])

    assert to_srt(empty) == ""
    assert to_vtt(empty) == "WEBVTT\n\n"


def _worded(*words: tuple[float, float, str]) -> FileTranscription:
    parts = [TranscriptionWord(s, e, t) for s, e, t in words]
    segment = TranscriptionSegment(
        parts[0].start, parts[-1].end, " ".join(p.text for p in parts), tuple(parts)
    )
    return FileTranscription(segment.text, "hr", 0.9, [segment])


def test_long_segment_is_split_on_duration() -> None:
    """A 36-second Whisper segment is unusable as a subtitle; word timings let
    it be cut at real boundaries instead of guessed at."""
    words = [(float(i), float(i) + 1.0, f"rijec{i}") for i in range(20)]

    srt = to_srt(_worded(*words))

    assert srt.count("-->") > 1
    for line in [line for line in srt.splitlines() if "-->" in line]:
        start, end = line.split(" --> ")
        to_seconds = lambda t: (  # noqa: E731
            int(t[:2]) * 3600 + int(t[3:5]) * 60 + int(t[6:8]) + int(t[9:]) / 1000
        )
        assert to_seconds(end) - to_seconds(start) <= 6.0


def test_split_uses_real_word_boundary_times() -> None:
    srt = to_srt(_worded((0.0, 1.0, "prvi"), (1.0, 2.0, "drugi"), (10.0, 11.0, "treci")))

    # The third word starts at 10s, well past the 6s budget, so it opens a new
    # cue timed from its own start rather than from where the previous one ended.
    assert "00:00:10,000 --> 00:00:11,000" in srt
    assert "treci" in srt


def test_segment_without_word_timings_is_emitted_whole() -> None:
    srt = to_srt(_result((0.0, 36.0, "jedan dugi segment bez vremena rijeci")))

    assert srt.count("-->") == 1
    assert "00:00:00,000 --> 00:00:36,000" in srt
