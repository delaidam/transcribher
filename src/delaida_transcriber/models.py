"""Stable result models shared by the CLI and phone web interface."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class TranscriptionWord:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class TranscriptionSegment:
    start: float
    end: float
    text: str
    # Word-level timings, when the backend supplies them. Whisper's own segments
    # run as long as 36 seconds, which is unusable as a subtitle cue, so these
    # are what let the subtitle writer split a segment at an accurate boundary.
    words: tuple[TranscriptionWord, ...] = ()


@dataclass(frozen=True)
class FileTranscription:
    text: str
    language: str | None
    language_probability: float | None
    segments: list[TranscriptionSegment]

    def to_dict(self) -> dict[str, object]:
        segments = []
        for segment in self.segments:
            payload = asdict(segment)
            # Omit rather than emit "words": [] for backends without word timings.
            if not payload["words"]:
                del payload["words"]
            segments.append(payload)
        return {
            "text": self.text,
            "detected_language": self.language,
            "language_probability": self.language_probability,
            "segments": segments,
        }
