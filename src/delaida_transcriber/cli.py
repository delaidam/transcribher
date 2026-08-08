"""Command-line batch transcription."""

import argparse
import asyncio
import json
from pathlib import Path

from delaida_transcriber.config import Settings
from delaida_transcriber.service import SUPPORTED_SUFFIXES, TranscriptionService
from delaida_transcriber.transcriber import WhisperTranscriber


def _files(path: Path) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix.lower() in SUPPORTED_SUFFIXES else []
    return sorted(
        item
        for item in path.rglob("*")
        if item.is_file() and item.suffix.lower() in SUPPORTED_SUFFIXES
    )


async def _run(args: argparse.Namespace) -> int:
    source = Path(args.input).expanduser().resolve()
    files = _files(source)
    if not files:
        print(f"No supported media files found at {source}")
        return 1

    settings = Settings(
        model=args.model,
        device="cpu" if args.cpu else None,
        compute_type="int8" if args.cpu else None,
    )
    service = TranscriptionService(WhisperTranscriber(settings), settings.max_upload_bytes)
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else (source.parent / "transcripts" if source.is_file() else source / "transcripts")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    for path in files:
        print(f"Transcribing {path}...")
        result = await service.transcribe(path, args.language)
        (output_dir / f"{path.stem}.txt").write_text(result.text + "\n", encoding="utf-8")
        payload = result.to_dict() | {
            "source": str(path),
            "requested_language": args.language,
        }
        (output_dir / f"{path.stem}.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"  output: {output_dir / path.stem}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Transcribe local audio/video files.")
    parser.add_argument("input", help="An .ogg, .mp3, .mp4, or .m4a file or containing folder.")
    parser.add_argument(
        "--language",
        choices=("auto", "bs", "hr", "en"),
        default="auto",
        help="Leave as auto; it beats every forced code on mixed Bosnian/English speech.",
    )
    parser.add_argument("--output-dir", help="Where to write .txt and .json transcripts.")
    parser.add_argument("--model", help="Override the Whisper model.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU transcription.")
    return asyncio.run(_run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
