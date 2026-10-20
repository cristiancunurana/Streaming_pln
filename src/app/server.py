"""Small local web server for the streaming transcript corpus."""

from __future__ import annotations

import json
import mimetypes
import re
import shutil
import subprocess
import sys
import threading
import uuid
from collections import Counter
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.nlp.process_memory import process_memory, _chunks as make_memory_chunks, SEARCH_INDEX_VERSION
from src.nlp.creators import creator_key, creator_name
from src.nlp.retrieval import query_entities, rank_chunks as rank_hybrid_chunks, temporal_filter
from src.nlp.generation import generate_answer
from src.nlp.answer import compose_answer
from src.nlp.text import (
    TOKEN_RE,
    content_tokens,
    extract_dates,
    normalize,
    search_tokens,
    SEARCH_STOPWORDS_NORMALIZED,
    search_terms,
)
from src.nlp.evaluate_retrieval import evaluate as evaluate_retrieval_questions

RAW = ROOT / "data" / "raw"
MEMORY = ROOT / "data" / "processed" / "memory"
REVIEWS = ROOT / "data" / "processed" / "reviews"
INDEX = ROOT / "src" / "app" / "index.html"
PIPELINE = ROOT / "src" / "ingest" / "process_vod.py"
PORT = 8501
JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()


def tokenize(value: str) -> list[str]:
    """Términos de contenido para estadísticas del corpus (sin palabras vacías)."""
    return content_tokens(value)


def review_candidate_id(stream_id: str, kind: str, row: dict, mention: dict | None = None) -> str:
    """Stable, source-scoped key for a human review annotation."""
    start = (mention or row).get("start_sec", 0)
    if kind == "entity":
        identity = row.get("id", "")
    elif kind == "relation":
        identity = "|".join(str(row.get(key, "")) for key in ("predicate", "object_id", "object"))
    else:
        identity = "|".join(str(row.get(key, "")) for key in ("polarity", "target_id", "target"))
    return f"{stream_id}::{kind}::{float(start):.2f}::{identity}"


def text_analysis(value: str) -> dict:
    raw_tokens = [word.casefold() for word in TOKEN_RE.findall(value)]
    terms = search_tokens(value)
    try:
        from src.nlp.retrieval import _load_query_nlp
        nlp = _load_query_nlp()
        terms = search_terms(value, nlp(value) if nlp else None)
    except Exception:
        pass
    # These are the tokens actually filtered from the lexical retrieval view.
    # Negation, temporal expressions and action words remain searchable.
    removed = [word for word in raw_tokens if normalize(word) in SEARCH_STOPWORDS_NORMALIZED or (len(word) == 1 and not word.isdigit())]
    content_terms = content_tokens(value)
    return {
        "original": value,
        "normalized": normalize(value),
        "removed": removed,
        "terms": terms,
        "content_terms": content_terms,
        "preserved": [word for word in raw_tokens if word in {"no", "nunca", "jamás", "jamas", "hoy", "ayer", "mañana"}],
        "removed_count": len(removed),
        "dates": extract_dates(value),
    }


def load_transmissions() -> list[dict]:
    items = []
    if not RAW.exists():
        return items
    for meta_path in sorted(RAW.glob("*.metadata.json"), key=lambda path: path.stat().st_mtime, reverse=True):
        stream_id = meta_path.name.removesuffix(".metadata.json")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", stream_id):
            continue
        transcript_path = RAW / f"{stream_id}.jsonl"
        audio_path = RAW / "audio" / f"{stream_id}.mp3"
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        segments = []
        if transcript_path.exists():
            for line in transcript_path.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                    if isinstance(row, dict) and row.get("text"):
                        segments.append(row)
                except json.JSONDecodeError:
                    continue
        content_terms: list[str] = []
        total_words = 0
        dates_found: list[str] = []
        for row in segments:
            text = str(row.get("text", ""))
            total_words += len(TOKEN_RE.findall(text))
            content_terms.extend(tokenize(text))
            dates_found.extend(extract_dates(text))
        items.append({
            **metadata,
            "stream_id": stream_id,
            "segment_count": len(segments),
            "transcript_ready": bool(segments),
            "audio_ready": audio_path.is_file(),
            "audio_size": audio_path.stat().st_size if audio_path.is_file() else 0,
            "word_count": len(content_terms),
            "total_word_count": total_words,
            "unique_word_count": len(set(content_terms)),
            "dates_found": list(dict.fromkeys(dates_found)),
            "nlp_preview": text_analysis(str(segments[0].get("text", ""))) if segments else None,
            "transcription_model": segments[0].get("transcription_model", "desconocido") if segments else None,
            "transcript_minutes": round(max(0.0, max((float(row.get("end_sec", 0)) for row in segments), default=0)
                                                - float(metadata.get("audio_offset_sec", 0) or 0)) / 60, 1),
            "transcript_start_sec": min((float(row.get("start_sec", 0)) for row in segments), default=0),
            "transcript_end_sec": max((float(row.get("end_sec", 0)) for row in segments), default=0),
            "memory_ready": memory_index_is_current(stream_id, transcript_path),
            "_segments": segments,
        })
    return items


def memory_index_is_current(stream_id: str, transcript_path: Path) -> bool:
    memory_path = MEMORY / f"{stream_id}.json"
    if not memory_path.is_file() or not transcript_path.is_file() or memory_path.stat().st_mtime < transcript_path.stat().st_mtime:
        return False
    try:
        memory = json.loads(memory_path.read_text(encoding="utf-8"))
        return memory.get("search_index_version") == SEARCH_INDEX_VERSION
    except (OSError, json.JSONDecodeError):
        return False


def make_chunks(segments: list[dict], meta: dict) -> list[dict]:
    return make_memory_chunks(segments, meta)


def corpus_stats(items: list[dict]) -> dict:
    all_segments = [segment for item in items for segment in item["_segments"]]
    words = Counter()
    for row in all_segments:
        words.update(tokenize(str(row.get("text", ""))))
    total_words = sum(
        len(TOKEN_RE.findall(str(row.get("text", "")))) for row in all_segments
    )
    return {
        "transmissions": len(items),
        "segments": len(all_segments),
        "minutes": round(sum(item["transcript_minutes"] for item in items), 1),
        "channels": len({item.get("channel_name") for item in items if item.get("channel_name")}),
        "total_words": total_words,
        "content_words": sum(words.values()),
        "unique_content_words": len(words),
        "topics": [{"word": word, "count": count} for word, count in words.most_common(12)],
    }


def read_fresh_memory(item: dict) -> dict | None:
    path = MEMORY / f"{item['stream_id']}.json"
    transcript = RAW / f"{item['stream_id']}.jsonl"
    if not path.is_file() or not transcript.is_file() or path.stat().st_mtime < transcript.stat().st_mtime:
        return None
    try:
        memory = json.loads(path.read_text(encoding="utf-8"))
        return memory if memory.get("search_index_version") == SEARCH_INDEX_VERSION else None
    except (OSError, json.JSONDecodeError):
        return None


def is_opinion_question(question: str) -> bool:
    folded = normalize(question)
    return any(term in folded for term in
               ("que opina", "que piensa", "que penso", "que le parece", "le gusta", "le disgusta",
                "opinion", "que dijo sobre", "que comento", "recomienda", "le recomienda"))


def is_topic_question(question: str) -> bool:
    folded = normalize(question)
    return any(term in folded for term in
               ("de que hablo mas", "de que se hablo mas", "que tema", "temas principales",
                "que se hablo", "tema principal", "de que trato", "de que trata", "temas trata",
                "de que se trata", "que temas", "de que va"))


class Handler(BaseHTTPRequestHandler):
    server_version = "StreamMemory/1.0"

    def log_message(self, format: str, *args) -> None:
        print(f"[{self.log_date_time_string()}] {format % args}")

    def _json(self, value: object, status: int = 200) -> None:
        payload = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def _body(self) -> dict:
        length = min(int(self.headers.get("Content-Length", "0")), 1_000_000)
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        if path == "/":
            body = INDEX.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path == "/api/transmissions":
            items = load_transmissions()
            for item in items:
                item["creator_key"] = creator_key(item)
                item["creator_name"] = creator_name(item)
            public_items = [{key: val for key, val in item.items() if not key.startswith("_")} for item in items]
            self._json({"items": public_items, "stats": corpus_stats(items)})
            return
        if path == "/api/creators":
            items = load_transmissions()
            groups = {}
            for item in items:
                key = creator_key(item)
                group = groups.setdefault(key, {"key": key, "name": creator_name(item), "transmissions": 0, "memory_ready": 0})
                group["transmissions"] += 1
                group["memory_ready"] += int(item["memory_ready"])
            self._json({"items": sorted(groups.values(), key=lambda row: row["name"].casefold())})
            return
        creator_memory_match = re.fullmatch(r"/api/creator-memory/([A-Za-z0-9_-]+)", path)
        if creator_memory_match:
            selected_key = creator_memory_match.group(1)
            items = [item for item in load_transmissions() if creator_key(item) == selected_key]
            memories = []
            for item in items:
                path = MEMORY / f"{item['stream_id']}.json"
                if not item["memory_ready"] or not path.is_file():
                    continue
                try:
                    memories.append(json.loads(path.read_text(encoding="utf-8")))
                except (OSError, json.JSONDecodeError):
                    continue
            if not memories:
                self._json({"error": "Este creador todavía no tiene transmisiones procesadas para la memoria."}, 404)
                return
            first = memories[0]
            entities = {}
            for memory in memories:
                for entity in memory.get("entities", []):
                    record = entities.setdefault(entity["id"], {**entity, "mentions": []})
                    known = {(row.get("stream_id"), row.get("start_sec"), row.get("evidence")) for row in record["mentions"]}
                    record["mentions"].extend(row for row in entity.get("mentions", []) if (row.get("stream_id"), row.get("start_sec"), row.get("evidence")) not in known)
            relations = [relation for memory in memories for relation in memory.get("relations", [])]
            for relation in relations:
                relation["subject"] = creator_name(first.get("stream", {}))
            opinions = [row for memory in memories for row in memory.get("opinions", [])]
            temporal = [row for memory in memories for row in memory.get("temporal_references", [])]
            chunks = [chunk for memory in memories for chunk in memory.get("chunks", [])]
            term_counts = Counter()
            for memory in memories:
                term_counts.update(memory.get("word_counter", {}).get("content_frequencies", {}))
            fragments = [row for memory in memories for row in memory.get("interpretation", {}).get("key_fragments", [])]
            review_annotations = {}
            for item in items:
                review_path = REVIEWS / f"{item['stream_id']}.json"
                try:
                    review_file = json.loads(review_path.read_text(encoding="utf-8"))
                    review_annotations.update(review_file.get("annotations", {}))
                except (OSError, json.JSONDecodeError):
                    continue
            total_content = sum(memory.get("word_counter", {}).get("content_words", 0) for memory in memories)
            unique_content = len(term_counts)
            word_counter = {
                "total_words": sum(memory.get("word_counter", {}).get("total_words", 0) for memory in memories),
                "content_words": total_content,
                "unique_content_words": unique_content,
                "lexical_diversity": round(unique_content / total_content, 3) if total_content else 0.0,
                "stopwords_removed": sum(memory.get("word_counter", {}).get("stopwords_removed", 0) for memory in memories),
                "fillers_removed": sum(memory.get("word_counter", {}).get("fillers_removed", 0) for memory in memories),
                "content_excluded": sum(memory.get("word_counter", {}).get("content_excluded", 0) for memory in memories),
                "top_content_terms": [{"term": term, "count": count} for term, count in term_counts.most_common(20)],
                "dates": sorted({value for memory in memories for value in memory.get("word_counter", {}).get("dates", [])}),
            }
            merged = {
                "schema_version": 1,
                "stream": {**first.get("stream", {}), "stream_id": "", "title": f"{len(memories)} transmisiones"},
                "creator": first.get("creator", {"key": selected_key, "name": creator_name(first.get("stream", {}))}),
                "streams": [{key: memory.get("stream", {}).get(key) for key in ("stream_id", "channel_name", "title", "date")} for memory in memories],
                "processing": first.get("processing", {}), "chunks": chunks,
                "entities": list(entities.values()), "relations": relations, "opinions": opinions,
                "review_annotations": review_annotations,
                "temporal_references": temporal, "word_counter": word_counter,
                "interpretation": {"method": "lectura extractiva agrupada por creador", "notice": "Mapa limitado a transmisiones con el mismo ID de creador. Cada evidencia conserva su VOD y timestamp.",
                                   "frequent_terms": [{"term": term, "count": count} for term, count in term_counts.most_common(12)],
                                   "key_fragments": fragments[:12]},
            }
            self._json(merged)
            return
        memory_match = re.fullmatch(r"/api/memory/([A-Za-z0-9_-]+)", path)
        if memory_match:
            stream_id = memory_match.group(1)
            memory_path = MEMORY / f"{stream_id}.json"
            if not memory_path.is_file():
                self._json({"error": "Todavía no se ha procesado la memoria de esta transmisión."}, 404)
                return
            try:
                memory = json.loads(memory_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                self._json({"error": "No pude leer el archivo de memoria."}, 500)
                return
            self._json(memory)
            return
        match = re.fullmatch(r"/api/model-runs/([A-Za-z0-9_-]+)", path)
        if match:
            self._json({"runs": self._model_runs(match.group(1))})
            return
        if path == "/api/evaluation":
            report = evaluate_retrieval_questions(5)
            self._json(report)
            return
        match = re.fullmatch(r"/api/jobs/([A-Za-z0-9_-]+)", path)
        if match:
            with JOBS_LOCK:
                job = JOBS.get(match.group(1))
            self._json(job or {"error": "No se encontró ese proceso."}, 200 if job else 404)
            return
        match = re.fullmatch(r"/media/([A-Za-z0-9_-]+)", path)
        if match:
            self._serve_audio(match.group(1))
            return
        self._json({"error": "Ruta no encontrada."}, 404)

    @staticmethod
    def _model_runs(stream_id: str) -> list[dict]:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", stream_id):
            return []
        folder = ROOT / "data" / "processed" / "model_runs" / stream_id
        if not folder.is_dir():
            return []
        runs = []
        for path in sorted(folder.glob("*.jsonl"), key=lambda file: file.stat().st_mtime, reverse=True):
            lines = path.read_text(encoding="utf-8").splitlines()
            valid_count = 0
            sample = []
            for line in lines:
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict) or not row.get("text"):
                        continue
                    valid_count += 1
                    if len(sample) < 3:
                        sample.append(row)
                except json.JSONDecodeError:
                    continue
            runs.append({"model": path.stem, "sample": sample, "segment_count": valid_count,
                         "invalid_lines": sum(bool(line.strip()) for line in lines) - valid_count})
        return runs

    def _serve_audio(self, stream_id: str) -> None:
        audio = RAW / "audio" / f"{stream_id}.mp3"
        if not audio.is_file():
            self._json({"error": "No hay audio para esta transmisión."}, 404)
            return
        size = audio.stat().st_size
        range_header = self.headers.get("Range", "")
        match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header)
        start, end = (int(match.group(1) or 0), int(match.group(2) or size - 1)) if match else (0, size - 1)
        if start >= size:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.end_headers()
            return
        end = min(end, size - 1)
        length = end - start + 1
        self.send_response(206 if match else 200)
        self.send_header("Content-Type", "audio/mpeg")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if match:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with audio.open("rb") as source:
            source.seek(start)
            remaining = length
            try:
                while remaining:
                    data = source.read(min(64 * 1024, remaining))
                    if not data:
                        break
                    self.wfile.write(data)
                    remaining -= len(data)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                # Browsers cancel Range requests when audio is paused or the user seeks.
                # The client is gone; this is a normal end to that partial response.
                return

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            body = self._body()
        except (ValueError, UnicodeDecodeError):
            self._json({"error": "El cuerpo de la solicitud no es JSON válido."}, 400)
            return
        if path == "/api/reviews":
            stream_id = str(body.get("stream_id", ""))
            kind = str(body.get("candidate_type", ""))
            candidate_id = str(body.get("candidate_id", ""))[:500]
            status = str(body.get("status", ""))
            corrected = " ".join(str(body.get("corrected_value", "")).split())[:200]
            if not re.fullmatch(r"[A-Za-z0-9_-]+", stream_id) or kind not in {"entity", "relation", "opinion"}:
                self._json({"error": "El tipo de revisión o la transmisión no son válidos."}, 400)
                return
            if status not in {"confirmed", "rejected", "corrected"} or (status == "corrected" and not corrected):
                self._json({"error": "Elige confirmar, descartar o escribe una corrección."}, 400)
                return
            memory_path = MEMORY / f"{stream_id}.json"
            try:
                memory = json.loads(memory_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                self._json({"error": "Primero construye la memoria de esta transmisión."}, 404)
                return
            valid_ids = set()
            if kind == "entity":
                for entity in memory.get("entities", []):
                    for mention in entity.get("mentions", []):
                        valid_ids.add(review_candidate_id(stream_id, kind, entity, mention))
            else:
                valid_ids = {review_candidate_id(stream_id, kind, row)
                             for row in memory.get("relations" if kind == "relation" else "opinions", [])}
            if candidate_id not in valid_ids:
                self._json({"error": "No encontré ese candidato en la memoria original."}, 404)
                return
            review_path = REVIEWS / f"{stream_id}.json"
            try:
                saved = json.loads(review_path.read_text(encoding="utf-8")) if review_path.is_file() else {"schema_version": 1, "annotations": {}}
            except (OSError, json.JSONDecodeError):
                saved = {"schema_version": 1, "annotations": {}}
            saved.setdefault("annotations", {})[candidate_id] = {
                "candidate_type": kind, "status": status, "corrected_value": corrected if status == "corrected" else "",
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            review_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = review_path.with_suffix(".json.tmp")
            temp_path.write_text(json.dumps(saved, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            temp_path.replace(review_path)
            self._json({"saved": True, "annotation": saved["annotations"][candidate_id]}, 201)
            return
        if path == "/api/ask":
            stream_id = str(body.get("stream_id", ""))
            question = " ".join(str(body.get("question", "")).split())[:500]
            if not question:
                self._json({"error": "Escribe una pregunta."}, 400)
                return
            corpus = [row for row in load_transmissions() if row["_segments"]]
            scope = str(body.get("scope", "selected"))
            candidates = corpus if scope == "corpus" else [row for row in corpus if row["stream_id"] == stream_id]
            if not candidates:
                self._json({"error": "No encontré transcripciones para esta consulta."}, 404)
                return
            detected_entities = query_entities(question)
            chunks, vectors, opinions = [], [], []
            has_vectors = False
            for item in candidates:
                memory = read_fresh_memory(item)
                if memory and memory.get("chunks"):
                    current_chunks = memory["chunks"]
                    current_vectors = memory.get("embedding_vectors")
                    if current_vectors and len(current_vectors) == len(current_chunks):
                        vectors.extend(current_vectors)
                        has_vectors = True
                    else:
                        vectors.extend([None] * len(current_chunks))
                    opinions.extend(memory.get("opinions", []))
                else:
                    current_chunks = make_chunks(item["_segments"], item)
                    vectors.extend([None] * len(current_chunks))
                chunks.extend(current_chunks)
            eligible, temporal_context = temporal_filter(question, chunks)
            if not eligible:
                self._json({"answer": temporal_context or "No hay fragmentos para esa referencia.", "results": [],
                            "method": "PLN · filtro temporal", "query_entities": detected_entities,
                            "temporal_filter": temporal_context})
                return
            dense_vectors = None
            if has_vectors:
                dimension = next(len(vector) for vector in vectors if vector is not None)
                dense_vectors = [vector if vector is not None else [0.0] * dimension for vector in vectors]
                eligible_ids = {id(chunk) for chunk in eligible}
                eligible_indices = [index for index, chunk in enumerate(chunks) if id(chunk) in eligible_ids]
                search_vectors = [dense_vectors[index] for index in eligible_indices]
            else:
                search_vectors = None
            ranked, retrieval_method = rank_hybrid_chunks(question, eligible, search_vectors, top_k=5)
            intent_opinion = is_opinion_question(question)
            if intent_opinion and ranked:
                for result in ranked:
                    matches = [opinion for opinion in opinions
                               if opinion.get("stream_id") == result.get("stream_id")
                               and float(opinion.get("end_sec", opinion.get("start_sec", -1))) >= float(result.get("start_sec", 0))
                               and float(opinion.get("start_sec", -1)) <= float(result.get("end_sec", result.get("start_sec", 0)))
                               and (not detected_entities
                                    or any(entity["text"].casefold() in question.casefold()
                                           and opinion.get("target", "").casefold() == entity["text"].casefold()
                                           for entity in detected_entities))]
                    if matches:
                        result["opinion_candidate"] = matches[0]["polarity"]
                        result["score"] = round(result["score"] + 0.12, 4)
                ranked.sort(key=lambda row: row["score"], reverse=True)
            # Build a useful response to “what was discussed most?” even when the
            # query itself contains no content words after stopword removal.
            all_tokens = Counter(token for chunk in eligible for token in tokenize(str(chunk.get("text", ""))))
            topic_terms = all_tokens.most_common(12) if is_topic_question(question) else []
            if topic_terms:
                topic_ranked, _topic_method = rank_hybrid_chunks(" ".join(word for word, _count in topic_terms[:5]),
                                                                  eligible, search_vectors, top_k=5)
                if topic_ranked:
                    ranked = topic_ranked
                elif not ranked:
                    ranked = sorted(eligible, key=lambda chunk: sum(all_tokens[word] for word in tokenize(chunk["text"])), reverse=True)[:5]
            intent = "opinion" if intent_opinion else "topics" if topic_terms else "retrieval"
            entity_names = [entity["text"] for entity in detected_entities]
            evidence_answer = compose_answer(question, ranked, entities=entity_names, intent=intent,
                                             temporal_context=temporal_context, topic_terms=topic_terms)
            llm_answer, llm_method = generate_answer(question, ranked[:5]) if body.get("use_llm") else (None, "")
            answer = llm_answer or evidence_answer
            method = llm_method if llm_answer else f"{retrieval_method} · respuesta redactada desde la evidencia"
            if llm_method and not llm_answer:
                method += f" · {llm_method}"
            if temporal_context:
                method += f" · {temporal_context}"
            if has_vectors and len(candidates) > 1 and any(vector is None for vector in vectors):
                method += " · algunos directos sin embeddings (BM25 disponible)"
            for index, result in enumerate(ranked, 1):
                result["evidence_id"] = f"E{index}"
            if not ranked:
                self._json({
                    "answer": answer,
                    "results": [],
                    "method": method,
                    "query_entities": detected_entities,
                    "temporal_filter": temporal_context,
                })
                return
            self._json({
                "answer": answer,
                "results": ranked,
                "method": method,
                "query_entities": detected_entities,
                "temporal_filter": temporal_context,
                "intent": intent,
            })
            return
        if path == "/api/evaluation/label":
            question = " ".join(str(body.get("question", "")).split())[:500]
            stream_id = str(body.get("stream_id", ""))
            if not question or not re.fullmatch(r"[A-Za-z0-9_-]+", stream_id):
                self._json({"error": "Falta la pregunta o el identificador del directo."}, 400)
                return
            try:
                start_sec = max(0.0, float(body.get("start_sec", 0)))
                end_sec = max(start_sec, float(body.get("end_sec", start_sec)))
            except (TypeError, ValueError):
                self._json({"error": "El rango temporal no es válido."}, 400)
                return
            gold_path = ROOT / "data" / "evaluation" / "gold_questions.jsonl"
            gold_path.parent.mkdir(parents=True, exist_ok=True)
            rows = []
            if gold_path.is_file():
                for line in gold_path.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        try:
                            rows.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
            scope = "corpus" if body.get("scope") == "corpus" else "selected"
            selected_ids = [] if scope == "corpus" else [stream_id]
            row = next((entry for entry in rows if entry.get("question", "").casefold() == question.casefold()
                        and entry.get("scope", "selected") == scope
                        and sorted(entry.get("stream_ids", [])) == sorted(selected_ids)), None)
            if row is None:
                row = {"question": question, "scope": scope, "stream_ids": selected_ids, "relevant": []}
                rows.append(row)
            judgment = {"stream_id": stream_id, "start_sec": start_sec, "end_sec": end_sec}
            if not any(value.get("stream_id") == stream_id
                       and abs(float(value.get("start_sec", -1)) - start_sec) < 0.01
                       for value in row["relevant"]):
                row["relevant"].append(judgment)
            gold_path.write_text("".join(json.dumps(value, ensure_ascii=False) + "\n" for value in rows), encoding="utf-8")
            self._json({"saved": True, "questions": len(rows), "relevant_ranges": len(row["relevant"])}, 201)
            return
        if path == "/api/activate-transcription":
            stream_id = str(body.get("stream_id", ""))
            model = str(body.get("model", ""))
            if not re.fullmatch(r"[A-Za-z0-9_-]+", stream_id) or not re.fullmatch(r"[A-Za-z0-9_-]+", model):
                self._json({"error": "Identificador o modelo no válido."}, 400)
                return
            candidate = ROOT / "data" / "processed" / "model_runs" / stream_id / f"{model}.jsonl"
            active = RAW / f"{stream_id}.jsonl"
            if not candidate.is_file() or not active.is_file():
                self._json({"error": "No encontré la comparación o la transcripción activa."}, 404)
                return
            try:
                rows = [json.loads(line) for line in candidate.read_text(encoding="utf-8").splitlines() if line.strip()]
                if not rows or any(row.get("stream_id") != stream_id or not row.get("text")
                                   or float(row.get("start_sec", -1)) < 0
                                   or float(row.get("end_sec", -1)) < float(row.get("start_sec", 0))
                                   for row in rows):
                    raise ValueError("La salida no tiene segmentos o timestamps válidos.")
            except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
                self._json({"error": f"No se puede activar esta transcripción: {error}"}, 400)
                return
            backup_dir = ROOT / "data" / "processed" / "backups" / "transcripts" / stream_id
            backup_dir.mkdir(parents=True, exist_ok=True)
            backup = backup_dir / f"{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}.jsonl"
            temporary = active.with_suffix(".jsonl.new")
            try:
                shutil.copy2(active, backup)
                shutil.copy2(candidate, temporary)
                temporary.replace(active)
                (MEMORY / f"{stream_id}.json").unlink(missing_ok=True)
            except OSError as error:
                temporary.unlink(missing_ok=True)
                self._json({"error": f"No pude activar la salida; el respaldo queda en {backup}: {error}"}, 500)
                return
            self._json({"activated": True, "model": model, "segments": len(rows),
                        "backup": str(backup.relative_to(ROOT))}, 200)
            return
        if path == "/api/process-memory":
            stream_id = str(body.get("stream_id", ""))
            item = next((row for row in load_transmissions() if row["stream_id"] == stream_id), None)
            if not item or not item["_segments"]:
                self._json({"error": "Primero necesitas una transcripción para construir la memoria."}, 404)
                return
            job_id = self._new_job()
            threading.Thread(target=self._run_memory_processing, args=(job_id, item), daemon=True).start()
            self._json({"job_id": job_id}, 202)
            return
        if path == "/api/process":
            url = str(body.get("url", "")).strip()
            if not url.startswith(("https://", "http://")):
                self._json({"error": "Pega un enlace http o https."}, 400)
                return
            try:
                start = max(0, min(int(body.get("start_minutes", 0)), 100_000)) * 60
                duration = max(1, min(int(body.get("duration_minutes", 15)), 180)) * 60
                model = str(body.get("model", "medium"))
                stream_date = str(body.get("date", "")).strip()
                streamer_id = str(body.get("streamer_id", "")).strip()
                if model not in {"tiny", "base", "small", "medium", "large-v3"}:
                    raise ValueError
            except (TypeError, ValueError):
                self._json({"error": "Revisa el minuto, la duración y el modelo elegidos."}, 400)
                return
            job_id = uuid.uuid4().hex[:12]
            job = {"id": job_id, "state": "queued", "stage": "En cola", "logs": [], "created_at": datetime.now(timezone.utc).isoformat()}
            with JOBS_LOCK:
                JOBS[job_id] = job
            threading.Thread(target=self._run_pipeline, args=(job_id, url, start, duration, model, stream_date, streamer_id), daemon=True).start()
            self._json({"job_id": job_id}, 202)
            return
        if path == "/api/reprocess":
            stream_id = str(body.get("stream_id", ""))
            item = next((row for row in load_transmissions() if row["stream_id"] == stream_id), None)
            if not item:
                self._json({"error": "No encontré los metadatos de esa transmisión."}, 404)
                return
            model = str(body.get("model", "medium"))
            if model not in {"tiny", "base", "small", "medium", "large-v3"}:
                self._json({"error": "Elige un modelo de Whisper válido."}, 400)
                return
            start = int(float(item.get("audio_offset_sec", 0) or 0))
            stored_duration = max((float(row.get("end_sec", 0)) for row in item["_segments"]), default=0)
            duration = max(60, int(stored_duration - start)) if stored_duration else min(900, max(60, int(float(item.get("duration_sec") or 900) - start)))
            job_id = self._new_job()
            threading.Thread(target=self._run_pipeline, args=(job_id, item["source_url"], start, duration, model), daemon=True).start()
            self._json({"job_id": job_id}, 202)
            return
        if path == "/api/retranscribe":
            stream_id = str(body.get("stream_id", ""))
            model = str(body.get("model", "medium"))
            item = next((row for row in load_transmissions() if row["stream_id"] == stream_id), None)
            audio = RAW / "audio" / f"{stream_id}.mp3"
            if not item or not audio.is_file():
                self._json({"error": "Hace falta el MP3 local y sus metadatos para volver a transcribir."}, 404)
                return
            if model not in {"tiny", "base", "small", "medium", "large-v3"}:
                self._json({"error": "Elige un modelo de Whisper válido."}, 400)
                return
            job_id = self._new_job()
            threading.Thread(target=self._run_retranscription, args=(job_id, item, model), daemon=True).start()
            self._json({"job_id": job_id}, 202)
            return
        self._json({"error": "Ruta no encontrada."}, 404)

    @staticmethod
    def _new_job() -> str:
        job_id = uuid.uuid4().hex[:12]
        with JOBS_LOCK:
            JOBS[job_id] = {"id": job_id, "state": "queued", "stage": "En cola", "logs": [], "created_at": datetime.now(timezone.utc).isoformat()}
        return job_id

    def do_DELETE(self) -> None:
        path = unquote(urlparse(self.path).path)
        transcript_match = re.fullmatch(r"/api/transmissions/([A-Za-z0-9_-]+)/transcript", path)
        full_match = re.fullmatch(r"/api/transmissions/([A-Za-z0-9_-]+)", path)
        match = transcript_match or full_match
        if not match:
            self._json({"error": "Ruta no encontrada."}, 404)
            return
        stream_id = match.group(1)
        targets = [RAW / f"{stream_id}.jsonl"]
        if full_match:
            targets.extend([RAW / f"{stream_id}.metadata.json", RAW / "audio" / f"{stream_id}.mp3"])
            shutil.rmtree(ROOT / "data" / "processed" / "model_runs" / stream_id, ignore_errors=True)
        deleted = []
        for path in targets:
            if path.is_file():
                path.unlink()
                deleted.append(path.name)
        memory_path = MEMORY / f"{stream_id}.json"
        if memory_path.is_file():
            memory_path.unlink()
            deleted.append(memory_path.name)
        review_path = REVIEWS / f"{stream_id}.json"
        if review_path.is_file():
            review_path.unlink()
            deleted.append(review_path.name)
        self._json({"deleted": deleted, "stream_id": stream_id})

    @staticmethod
    def _run_pipeline(job_id: str, url: str, start: int, duration: int, model: str, stream_date: str = "", streamer_id: str = "") -> None:
        command = [sys.executable, "-u", str(PIPELINE), url, "--start-sec", str(start), "--duration-sec", str(duration), "--model", model]
        if stream_date:
            command.extend(["--date", stream_date])
        if streamer_id:
            command.extend(["--streamer-id", streamer_id])
        with JOBS_LOCK:
            JOBS[job_id]["state"] = "running"
            JOBS[job_id]["stage"] = "Leyendo enlace y obteniendo metadatos..."
        try:
            process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            assert process.stdout is not None
            for line in process.stdout:
                line = line.strip()
                with JOBS_LOCK:
                    job = JOBS[job_id]
                    job["logs"] = (job["logs"] + [line])[-60:]
                    if line.startswith("STAGE:"):
                        job["stage"] = line.split(" ", 1)[-1]
                        stage_key = line.split(" ", 1)[0].split(":", 1)[-1]
                        job["progress"] = {"metadata": 5, "download": 10, "compress": 63, "transcription": 68, "complete": 100}.get(stage_key, job.get("progress", 8))
                    elif line.startswith("DOWNLOAD_PROGRESS:"):
                        job["progress"] = 10 + int(int(line.split(":", 1)[1]) * 0.5)
                    elif line.startswith("TRANSCRIBE_SEGMENTS:"):
                        job["stage"] = "Transcribiendo el audio en español…"
            code = process.wait()
            with JOBS_LOCK:
                job = JOBS[job_id]
                job["state"] = "complete" if code == 0 else "error"
                job["stage"] = "Proceso terminado" if code == 0 else "Revisa el registro del proceso"
                job["return_code"] = code
            if code == 0:
                for meta_path in RAW.glob("*.metadata.json"):
                    try:
                        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError):
                        continue
                    if metadata.get("source_url") == url:
                        (MEMORY / f"{meta_path.name.removesuffix('.metadata.json')}.json").unlink(missing_ok=True)
        except OSError as error:
            with JOBS_LOCK:
                JOBS[job_id]["state"] = "error"
                JOBS[job_id]["stage"] = str(error)

    @staticmethod
    def _run_memory_processing(job_id: str, item: dict) -> None:
        stream_id = item["stream_id"]
        def report(stage: str, progress: int) -> None:
            with JOBS_LOCK:
                JOBS[job_id]["stage"] = stage
                JOBS[job_id]["progress"] = progress
        with JOBS_LOCK:
            JOBS[job_id]["state"] = "running"
            JOBS[job_id]["stage"] = "Iniciando el pipeline PLN…"
            JOBS[job_id]["progress"] = 15
        try:
            output = MEMORY / f"{stream_id}.json"
            memory = process_memory(stream_id, item["_segments"], item, output, progress=report)
            with JOBS_LOCK:
                JOBS[job_id]["state"] = "complete"
                JOBS[job_id]["stage"] = "Memoria construida: PLN, trazabilidad y recuperación listos."
                JOBS[job_id]["progress"] = 100
                JOBS[job_id]["result"] = {"chunks": len(memory["chunks"]), "entities": len(memory["entities"]),
                                           "relations": len(memory["relations"]), "opinions": len(memory["opinions"]),
                                           "temporal_references": len(memory["temporal_references"]),
                                           "embeddings": memory["processing"]["embeddings"]["status"]}
        except Exception as error:
            with JOBS_LOCK:
                JOBS[job_id]["state"] = "error"
                JOBS[job_id]["stage"] = f"No se pudo construir la memoria: {error}"
                JOBS[job_id]["progress"] = 0

    @staticmethod
    def _run_retranscription(job_id: str, item: dict, model: str) -> None:
        stream_id = item["stream_id"]
        audio = RAW / "audio" / f"{stream_id}.mp3"
        output = ROOT / "data" / "processed" / "model_runs" / stream_id / f"{model}.jsonl"
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary_output = output.with_suffix(".jsonl.partial")
        temporary_output.unlink(missing_ok=True)
        command = [
            sys.executable, "-u", str(ROOT / "src" / "ingest" / "transcribe.py"), str(audio),
            "--streamer-id", str(item.get("streamer_id", "creador")), "--stream-id", stream_id,
            "--title", str(item.get("title", stream_id)), "--date", str(item.get("date", "2000-01-01")),
            "--source-url", str(item.get("source_url", "https://localhost/")), "--output", str(temporary_output),
            "--offset-sec", str(float(item.get("audio_offset_sec", 0) or 0)), "--model", model,
        ]
        with JOBS_LOCK:
            JOBS[job_id]["state"] = "running"
            JOBS[job_id]["stage"] = f"Probando Whisper {model} con el audio ya guardado…"
            JOBS[job_id]["progress"] = 5
        try:
            process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            assert process.stdout is not None
            for line in process.stdout:
                line = line.strip()
                with JOBS_LOCK:
                    job = JOBS[job_id]
                    job["logs"] = (job["logs"] + [line])[-60:]
                    if line.startswith("TRANSCRIBE_SEGMENTS:"):
                        job["progress"] = min(95, 30 + int(int(line.split(":", 1)[1]) / 10))
            code = process.wait()
            if code == 0:
                temporary_output.replace(output)
            else:
                temporary_output.unlink(missing_ok=True)
            with JOBS_LOCK:
                JOBS[job_id]["state"] = "complete" if code == 0 else "error"
                JOBS[job_id]["stage"] = f"Modelo {model} guardado para comparar" if code == 0 else "Falló la transcripción de prueba"
                JOBS[job_id]["progress"] = 100 if code == 0 else 0
                JOBS[job_id]["return_code"] = code
        except OSError as error:
            with JOBS_LOCK:
                JOBS[job_id]["state"] = "error"
                JOBS[job_id]["stage"] = str(error)


if __name__ == "__main__":
    print(f"Memoria semántica lista en http://127.0.0.1:{PORT}")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
