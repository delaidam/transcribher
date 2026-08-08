from delaida_transcriber.models import FileTranscription, TranscriptionSegment


def test_result_serializes_segments() -> None:
    result = FileTranscription("Dobar dan.", "bs", 0.95, [TranscriptionSegment(0, 1, "Dobar dan.")])

    assert result.to_dict() == {
        "text": "Dobar dan.",
        "detected_language": "bs",
        "language_probability": 0.95,
        "segments": [{"start": 0, "end": 1, "text": "Dobar dan."}],
    }
