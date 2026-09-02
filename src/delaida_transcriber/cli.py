"""Command-line batch transcription."""

import argparse
import asyncio
import json
from pathlib import Path

from delaida_transcriber.backends import create_backend
from delaida_transcriber.config import BEST_MODEL, Settings
from delaida_transcriber.service import SUPPORTED_SUFFIXES, TranscriptionService
from delaida_transcriber.subtitles import to_srt, to_vtt

SUPPORTED_FORMATS = ("txt", "json", "srt", "vtt")
DEFAULT_FORMATS = "txt,json,srt"


def _parse_formats(raw: str) -> set[str]:
    requested = {item.strip().lower() for item in raw.split(",") if item.strip()}
    unknown = requested - set(SUPPORTED_FORMATS)
    if unknown:
        raise SystemExit(
            f"Unknown output format(s): {', '.join(sorted(unknown))}. "
            f"Choose from: {', '.join(SUPPORTED_FORMATS)}."
        )
    if not requested:
        raise SystemExit("No output formats requested.")
    return requested


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
        model=args.model or (BEST_MODEL if args.best else None),
        device="cpu" if args.cpu else None,
        compute_type="int8" if args.cpu else None,
    )
    if args.best and not args.model:
        print(f"Using {settings.model} for accuracy; expect roughly 8x the audio duration.")
    service = TranscriptionService(create_backend(settings))
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else (source.parent / "transcripts" if source.is_file() else source / "transcripts")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    formats = _parse_formats(args.formats)
    language = args.language or settings.language

    for path in files:
        print(f"Transcribing {path}...")
        result = await service.transcribe(path, language)

        written = []
        if "txt" in formats:
            (output_dir / f"{path.stem}.txt").write_text(result.text + "\n", encoding="utf-8")
            written.append("txt")
        if "json" in formats:
            payload = result.to_dict() | {
                "source": str(path),
                "requested_language": language,
            }
            (output_dir / f"{path.stem}.json").write_text(
                json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            written.append("json")
        if "srt" in formats:
            (output_dir / f"{path.stem}.srt").write_text(to_srt(result), encoding="utf-8")
            written.append("srt")
        if "vtt" in formats:
            (output_dir / f"{path.stem}.vtt").write_text(to_vtt(result), encoding="utf-8")
            written.append("vtt")

        print(f"  output: {output_dir / path.stem}.{{{','.join(written)}}}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Transcribe local audio/video files.")
    parser.add_argument("input", help="An .ogg, .mp3, .mp4, or .m4a file or containing folder.")
    parser.add_argument(
        "--language",
        choices=("auto", "bs", "hr", "en", "no"),
        default=None,
        help="Defaults to STT_LANGUAGE. Auto-detect wins on short mixed "
        "Bosnian/English speech, but misreads longer recordings; hr is the "
        "code to force, never bs.",
    )
    parser.add_argument("--output-dir", help="Where to write the transcripts.")
    parser.add_argument(
        "--formats",
        default=DEFAULT_FORMATS,
        help=f"Comma-separated output formats from {', '.join(SUPPORTED_FORMATS)} "
        f"(default: {DEFAULT_FORMATS}).",
    )
    parser.add_argument("--model", help="Override the Whisper model.")
    parser.add_argument(
        "--best",
        action="store_true",
        help=f"Use {BEST_MODEL}: a few points more accurate, but roughly 8x the audio duration.",
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU transcription.")
    return asyncio.run(_run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
