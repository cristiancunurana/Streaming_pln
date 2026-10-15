"""Stable creator identity helpers used to scope cross-VOD memory."""

from __future__ import annotations

import re
import unicodedata


def creator_key(metadata: dict) -> str:
    """Return a deterministic creator key, keeping unknown VODs isolated."""
    value = str(metadata.get("streamer_id") or metadata.get("channel_id") or "").strip()
    prefix = "creator"
    if not value:
        value = str(metadata.get("channel_name") or "").strip()
        prefix = "channel"
        if value:
            value = f"{metadata.get('platform') or 'platform-unknown'}-{value}"
    if not value:
        value = str(metadata.get("stream_id") or "unknown").strip()
        prefix = "stream"
    normalized = unicodedata.normalize("NFKD", value.casefold())
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    slug = re.sub(r"[^a-z0-9]+", "-", normalized).strip("-") or "unknown"
    return f"{prefix}-{slug}"


def creator_name(metadata: dict) -> str:
    return str(metadata.get("channel_name") or metadata.get("streamer_id") or "Creador sin identificar")
