"""Private Bosnian and English audio transcription."""

# Runs before anything can import CTranslate2, which is the only moment where
# putting the pip-installed CUDA libraries on PATH still helps. A no-op except
# on Windows; see the module for why PATH and not os.add_dll_directory.
from delaida_transcriber.cuda import add_library_path

add_library_path()

from delaida_transcriber.models import FileTranscription, TranscriptionSegment  # noqa: E402
from delaida_transcriber.transcriber import WhisperTranscriber  # noqa: E402

__all__ = ["FileTranscription", "TranscriptionSegment", "WhisperTranscriber"]
