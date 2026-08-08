"""Stable result models shared by the CLI and phone web interface."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class TranscriptionSegment:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class FileTranscription:
    text: str
    language: str | None
    language_probability: float | None
    segments: list[TranscriptionSegment]

    def to_dict(self) -> dict[str, object]:
        return {
            "text": self.text,
            "detected_language": self.language,
            "language_probability": self.language_probability,
            "segments": [asdict(segment) for segment in self.segments],
        }
