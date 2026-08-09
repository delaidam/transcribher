"""Render transcripts as SubRip (.srt) and WebVTT (.vtt) subtitles."""

from delaida_transcriber.models import FileTranscription


def _timestamp(seconds: float, separator: str) -> str:
    """Format seconds as HH:MM:SS<sep>mmm, the shape both formats expect."""
    milliseconds = max(int(round(seconds * 1000)), 0)
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    whole_seconds, milliseconds = divmod(milliseconds, 1000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}{separator}{milliseconds:03d}"


# Whisper returns segments up to ~36 seconds long, which is unreadable as a
# subtitle. These are conventional limits: roughly two lines of forty characters,
# on screen for a few seconds.
MAX_CUE_SECONDS = 6.0
MAX_CUE_CHARS = 84


def _split_segment(segment) -> list[tuple[float, float, str]]:
    """Break one segment into cue-sized pieces at real word boundaries.

    Without word timings there is nothing accurate to split on, so the segment
    is emitted whole rather than guessed at.
    """
    text = segment.text.strip()
    if not segment.words:
        return [(segment.start, max(segment.end, segment.start), text)] if text else []

    pieces: list[tuple[float, float, str]] = []
    current: list[str] = []
    start = segment.words[0].start
    previous_end = segment.words[0].end
    for word in segment.words:
        candidate_length = len(" ".join([*current, word.text]))
        too_long = current and (
            word.end - start > MAX_CUE_SECONDS or candidate_length > MAX_CUE_CHARS
        )
        if too_long:
            pieces.append((start, previous_end, " ".join(current)))
            current, start = [], word.start
        current.append(word.text)
        previous_end = word.end

    if current:
        pieces.append((start, previous_end, " ".join(current)))
    return pieces


def _cues(result: FileTranscription, separator: str) -> list[tuple[str, str, str]]:
    cues = []
    for segment in result.segments:
        for start, end, text in _split_segment(segment):
            if not text.strip():
                continue
            # A zero-length or reversed cue makes players skip the line entirely.
            cues.append(
                (_timestamp(start, separator), _timestamp(max(end, start), separator), text)
            )
    return cues


def to_srt(result: FileTranscription) -> str:
    """Render SubRip. Cues are numbered from 1 over the segments that survive."""
    blocks = [
        f"{number}\n{start} --> {end}\n{text}\n"
        for number, (start, end, text) in enumerate(_cues(result, ","), start=1)
    ]
    return "\n".join(blocks)


def to_vtt(result: FileTranscription) -> str:
    """Render WebVTT, which browsers can load directly with a <track> element."""
    blocks = [f"{start} --> {end}\n{text}\n" for start, end, text in _cues(result, ".")]
    return "WEBVTT\n\n" + "\n".join(blocks)
