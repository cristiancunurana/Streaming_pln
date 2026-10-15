"""Transcribe an authorized audio/video file to timestamped JSONL segments."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path


def safe_id(value: str, label: str) -> str:
    value = value.strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", value):
        raise argparse.ArgumentTypeError(
            f"{label} debe usar solo letras, números, guion o guion bajo"
        )
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Transcribe un audio/video autorizado y guarda una línea JSON por "
            "segmento, con timestamps en segundos."
        )
    )
    parser.add_argument("media", type=Path, help="Ruta al audio o video local")
    parser.add_argument("--streamer-id", required=True, help="ID corto del creador")
    parser.add_argument("--stream-id", required=True, help="ID corto y único del VOD")
    parser.add_argument("--title", required=True, help="Título de la transmisión")
    parser.add_argument("--date", required=True, help="Fecha del VOD: AAAA-MM-DD")
    parser.add_argument("--source-url", required=True, help="URL original del VOD")
    parser.add_argument(
        "--output", type=Path, help="JSONL destino; por defecto data/raw/<stream-id>.jsonl"
    )
    parser.add_argument(
        "--start-sec", type=float, default=0,
        help="Segundo del VOD donde empieza el tramo (por defecto: 0)",
    )
    parser.add_argument(
        "--duration-sec", type=float,
        help="Duración a transcribir en segundos; si se omite, procesa hasta el final",
    )
    parser.add_argument(
        "--offset-sec", type=float, default=0,
        help="Desplazamiento del clip dentro del VOD original (por defecto: 0)",
    )
    parser.add_argument(
        "--model", default="medium", choices=("tiny", "base", "small", "medium", "large-v3"),
        help="Tamaño del modelo Whisper (por defecto: medium; se descarga al primer uso)",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if not args.media.is_file():
        parser.error(f"no encuentro el archivo de audio/video: {args.media}")
    try:
        parsed_date = date.fromisoformat(args.date)
    except ValueError:
        parser.error("--date debe tener el formato AAAA-MM-DD")
    for name in ("streamer_id", "stream_id"):
        try:
            setattr(args, name, safe_id(getattr(args, name), name.replace("_", "-")))
        except argparse.ArgumentTypeError as error:
            parser.error(str(error))
    if not args.source_url.startswith(("https://", "http://")):
        parser.error("--source-url debe ser una URL http o https")
    if args.start_sec < 0:
        parser.error("--start-sec no puede ser negativo")
    if args.duration_sec is not None and args.duration_sec <= 0:
        parser.error("--duration-sec debe ser mayor que cero")
    if args.offset_sec < 0:
        parser.error("--offset-sec no puede ser negativo")

    output = args.output or Path("data/raw") / f"{args.stream_id}.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print(
            "Falta faster-whisper. Instálalo con: pip install -r requirements.txt",
            file=sys.stderr,
        )
        return 2

    print(f"Transcribiendo {args.media.name} con Whisper {args.model} en español...", flush=True)
    model = WhisperModel(args.model, device="cpu", compute_type="int8")
    transcribe_options = {"language": "es", "beam_size": 5, "vad_filter": True}
    if args.duration_sec is not None:
        end_sec = args.start_sec + args.duration_sec
        transcribe_options["clip_timestamps"] = f"{args.start_sec},{end_sec}"
        print(
            f"Tramo seleccionado: {args.start_sec:.1f}–{end_sec:.1f} segundos "
            "del VOD (los timestamps quedan referidos al VOD original)."
        )
    elif args.start_sec > 0:
        parser.error("si usas --start-sec, indica también --duration-sec")

    segments, info = model.transcribe(str(args.media), **transcribe_options)

    count = 0
    with output.open("w", encoding="utf-8") as destination:
        for index, segment in enumerate(segments, start=1):
            text = " ".join(segment.text.split())
            if not text:
                continue
            row = {
                "streamer_id": args.streamer_id,
                "stream_id": args.stream_id,
                "title": args.title,
                "date": parsed_date.isoformat(),
                "source_url": args.source_url,
                "language": info.language or "es",
                "start_sec": round(float(segment.start) + args.offset_sec, 3),
                "end_sec": round(float(segment.end) + args.offset_sec, 3),
                "text": text,
                "transcription_model": f"faster-whisper/{args.model}",
            }
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
            if count % 100 == 0:
                print(f"TRANSCRIBE_SEGMENTS:{count}", flush=True)

    print(f"Listo: {count} segmentos guardados en {output}")
    print("Revisa manualmente nombres propios y tiempos antes de usar el corpus.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
