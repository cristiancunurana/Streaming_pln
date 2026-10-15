"""Grounded answer generation with an optional local Ollama model."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request


def generate_answer(question: str, evidence: list[dict], model_name: str = "") -> tuple[str | None, str]:
    model_name = model_name or os.environ.get("STREAM_MEMORY_LLM", "").strip()
    if not model_name or not evidence:
        return None, "LLM local no configurado; se usó respuesta extractiva"
    host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
    cited_context = "\n\n".join(
        f"[E{index}] Canal: {row.get('channel_name','')}; directo: {row.get('title','')}; "
        f"fecha: {row.get('date','')}; timestamp: {row.get('start_sec',0)} segundos.\n"
        f"Transcripción literal: {row.get('text','')}"
        for index, row in enumerate(evidence, 1)
    )
    prompt = (
        "Responde en español y en pocas frases. Usa exclusivamente la evidencia entregada. "
        "No completes datos con conocimiento externo. Si la evidencia no responde, dilo. "
        "Toda afirmación factual debe llevar al menos una cita con el formato [E1]. "
        "Distingue una opinión del streamer de una afirmación objetiva.\n\n"
        f"PREGUNTA: {question}\n\nEVIDENCIA:\n{cited_context}\n\nRESPUESTA:"
    )
    payload = json.dumps({"model": model_name, "prompt": prompt, "stream": False,
                          "options": {"temperature": 0.1}}).encode("utf-8")
    request = urllib.request.Request(f"{host}/api/generate", data=payload,
                                     headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            generated = json.loads(response.read().decode("utf-8")).get("response", "").strip()
    except (OSError, urllib.error.URLError, json.JSONDecodeError, TimeoutError):
        return None, "LLM local no respondió; se usó respuesta extractiva"
    valid_citations = {f"[E{index}]" for index in range(1, len(evidence) + 1)}
    if not generated or not any(citation in generated for citation in valid_citations):
        return None, "LLM no devolvió citas válidas; se usó respuesta extractiva"
    return generated, f"RAG local · {model_name} · respuestas limitadas a fragmentos citados"
