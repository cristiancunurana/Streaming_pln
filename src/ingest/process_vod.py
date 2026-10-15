"""Get public video metadata, extract an MP3, then transcribe a selected span."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def slug(value: str, fallback: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip()).strip("_-")
    return value or fallback


def metadata_date(info: dict) -> str | None:
    """Prefer the platform's broadcast/release date, then its upload date."""
    for key in ("release_timestamp", "timestamp"):
        value = info.get(key)
        if value:
            return datetime.fromtimestamp(float(value), tz=timezone.utc).date().isoformat()
    value = info.get("upload_date")
    if value and re.fullmatch(r"\d{8}", str(value)):
        return date(int(value[:4]), int(value[4:6]), int(value[6:8])).isoformat()
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "A partir del enlace de un video público o una transmisión guardada, guarda sus metadatos, "
            "extrae el audio MP3 y transcribe un tramo en español."
        )
    )
    parser.add_argument("url", help="Enlace a un video público o transmisión guardada autorizados")
    parser.add_argument("--start-sec", type=float, default=0, help="Inicio del tramo en el VOD")
    parser.add_argument(
        "--duration-sec", type=float, default=900,
        help="Duración a transcribir (por defecto: 900 segundos / 15 minutos)",
    )
    parser.add_argument("--date", help="Fecha de emisión AAAA-MM-DD si no se detecta automáticamente")
    parser.add_argument("--streamer-id", help="ID corto del creador; por defecto se deriva del canal")
    parser.add_argument(
        "--model", default="medium", choices=("tiny", "base", "small", "medium", "large-v3"),
        help="Modelo de transcripción (por defecto: medium)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/raw"),
        help="Carpeta de salida (por defecto: data/raw)",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if not args.url.startswith(("https://", "http://")):
        parser.error("ingresa un enlace http o https")
    if args.start_sec < 0 or args.duration_sec <= 0:
        parser.error("start-sec debe ser >= 0 y duration-sec debe ser > 0")
    if not shutil.which("ffmpeg"):
        parser.error("falta FFmpeg; instálalo para convertir el audio descargado a MP3")

    try:
        import yt_dlp
    except ImportError:
        print("Falta yt-dlp. Instálalo con: %pip install -r requirements.txt", file=sys.stderr)
        return 2

    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    # First inspect the page without downloading media.
    print("STAGE:metadata Leyendo enlace y obteniendo metadatos...", flush=True)
    try:
        with yt_dlp.YoutubeDL({"quiet": True, "noplaylist": True, "no_warnings": True}) as ydl:
            info = ydl.extract_info(args.url, download=False)
    except Exception as error:
        parser.error(f"no pude leer el video desde ese enlace: {error}")
    if info.get("is_live") or info.get("live_status") == "is_live":
        parser.error("este prototipo procesa videos y transmisiones guardadas; no captura directos en vivo")

    stream_id = slug(str(info.get("id") or "vod"), "vod")
    channel_name = str(info.get("channel") or info.get("uploader") or "creador")
    streamer_id = args.streamer_id or slug(
        str(info.get("channel_id") or info.get("uploader_id") or channel_name), "creador"
    )
    title = str(info.get("title") or stream_id)
    source_url = str(info.get("webpage_url") or args.url)
    broadcast_date = args.date or metadata_date(info)
    if broadcast_date:
        try:
            broadcast_date = date.fromisoformat(broadcast_date).isoformat()
        except ValueError:
            parser.error("--date debe tener el formato AAAA-MM-DD")
    else:
        parser.error(
            "la plataforma no entregó una fecha; vuelve a ejecutar con --date AAAA-MM-DD"
        )

    metadata = {
        "streamer_id": streamer_id,
        "channel_name": channel_name,
        "stream_id": stream_id,
        "title": title,
        "date": broadcast_date,
        "source_url": source_url,
        "platform": info.get("extractor_key"),
        "duration_sec": info.get("duration"),
        "audio_offset_sec": args.start_sec,
        "metadata_date_source": "manual" if args.date else "platform metadata",
    }
    metadata_path = output_dir / f"{stream_id}.metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    audio_path = audio_dir / f"{stream_id}.mp3"
    print(f"VOD: {title}")
    print(f"Canal: {channel_name} | fecha registrada: {broadcast_date}")
    print("STAGE:download Descargando el audio y convirtiéndolo a MP3...", flush=True)

    last_percent = [-1]

    def download_progress(event: dict) -> None:
        if event.get("status") != "downloading":
            return
        total = event.get("total_bytes") or event.get("total_bytes_estimate")
        downloaded = event.get("downloaded_bytes")
        if total and downloaded:
            percent = int(100 * downloaded / total)
            if percent != last_percent[0]:
                last_percent[0] = percent
                print(f"DOWNLOAD_PROGRESS:{percent}", flush=True)

    video_duration = info.get("duration")
    actual_duration = args.duration_sec
    if video_duration:
        actual_duration = min(actual_duration, float(video_duration) - args.start_sec)
    if actual_duration <= 0:
        parser.error("el minuto de inicio está después del final del VOD")
    range_end = args.start_sec + actual_duration

    try:
        with tempfile.TemporaryDirectory(prefix=f"{stream_id}-", dir=audio_dir) as temporary_dir:
            download_options = {
                "format": "bestaudio/best",
                "outtmpl": str(Path(temporary_dir) / "source.%(ext)s"),
                "noplaylist": True,
                "overwrites": True,
                "progress_hooks": [download_progress],
                "download_ranges": yt_dlp.utils.download_range_func(
                    None, [(args.start_sec, range_end)]
                ),
                "force_keyframes_at_cuts": True,
            }
            with yt_dlp.YoutubeDL(download_options) as ydl:
                ydl.extract_info(source_url, download=True)
            downloaded_files = [
                path for path in Path(temporary_dir).iterdir()
                if path.is_file() and not path.name.endswith((".part", ".ytdl"))
            ]
            if not downloaded_files:
                parser.error("yt-dlp no dejó un archivo de audio para convertir")
            source_audio = downloaded_files[0]
            print("STAGE:compress Comprimiendo a MP3 mono de 16 kHz y 48 kb/s...", flush=True)
            conversion = subprocess.run(
                [
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-i", str(source_audio), "-vn", "-ac", "1", "-ar", "16000",
                    "-b:a", "48k", "-codec:a", "libmp3lame", str(audio_path),
                ],
                check=False,
            )
            if conversion.returncode != 0:
                parser.error("FFmpeg no pudo convertir el fragmento a MP3")
    except Exception as error:
        parser.error(f"no pude descargar/convertir el audio: {error}")
    if not audio_path.is_file():
        parser.error(f"no encontré el MP3 esperado en {audio_path}")
    print("STAGE:transcription MP3 listo; iniciando transcripción en español...", flush=True)

    transcript_path = output_dir / f"{stream_id}.jsonl"
    transcriber = PROJECT_ROOT / "src" / "ingest" / "transcribe.py"
    command = [
        sys.executable, "-u", str(transcriber), str(audio_path),
        "--streamer-id", streamer_id,
        "--stream-id", stream_id,
        "--title", title,
        "--date", broadcast_date,
        "--source-url", source_url,
        "--output", str(transcript_path),
        "--start-sec", "0",
        "--duration-sec", str(actual_duration),
        "--offset-sec", str(args.start_sec),
        "--model", args.model,
    ]
    print(f"MP3 guardado en: {audio_path}")
    print(f"Transcribiendo {actual_duration / 60:.0f} minutos desde el segundo {args.start_sec:.0f}...")
    result = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    if result.returncode != 0:
        return result.returncode
    print("STAGE:complete Flujo terminado.", flush=True)
    print(f"Metadatos: {metadata_path}")
    print(f"Transcripción: {transcript_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
