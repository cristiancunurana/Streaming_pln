"""Build reviewable, timestamped PLN artifacts from an existing transcript."""

from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from src.nlp.creators import creator_key, creator_name


_STOPWORDS_FALLBACK = set("a al algo algunas algunos ante antes como con contra cual cuando de del desde donde dos el ella ellas ellos en entre era eres es esa esas ese eso esas este esto estos estoy fue fueron ha hasta hay la las le les lo los mas más me mi mis mucha muchas mucho muchos muy no nos o os otra otras otro otros para pero poco por porque que quien quienes se ser si sin sobre su sus te tiene tienen todo todos tu tus un una uno unas unos y ya también entonces aquí ahi así pues".split())
try:
    import stopwordsiso

    STOPWORDS = set(stopwordsiso.stopwords("es"))
except ImportError:
    STOPWORDS = _STOPWORDS_FALLBACK
STOPWORDS.difference_update({"no", "nunca", "jamás", "jamas", "tampoco", "mas", "más", "muy", "si", "sí"})
TEMPORAL_WORDS = {
    "hoy", "ayer", "mañana", "anoche", "ahora", "antes", "después", "despues",
    "semana", "mes", "año", "ano", "día", "dia", "lunes", "martes", "miércoles",
    "miercoles", "jueves", "viernes", "sábado", "sabado", "domingo", "pasado", "pasada",
    "próximo", "proxima", "próxima", "proximo", "hace", "durante",
}
SEARCH_ACTION_WORDS = {"decir", "hacer", "hablar", "contar", "mencionar", "comentar", "anunciar", "opinar", "pensar", "jugar", "recomendar", "pasar"}
STOPWORDS.difference_update(TEMPORAL_WORDS | SEARCH_ACTION_WORDS)
FILLER_WORDS = {"eh", "mmm", "osea", "aja", "ajam", "digamos"}
TOKEN_RE = re.compile(r"[a-záéíóúüñ0-9]+", re.IGNORECASE)
SENTENCE_RE = re.compile(r"(?<=[.!?¿])\s+|\n+")
KNOWN_TERMS = {
    "valorant": "WORK", "minecraft": "WORK", "fortnite": "WORK", "league of legends": "WORK",
    "lol": "WORK", "dota": "WORK", "dota 2": "WORK", "twitch": "ORG", "youtube": "ORG",
    "kick": "ORG", "armys": "ORG", "sidereal": "PER", "perú": "LOC",
    "lima": "LOC", "chile": "LOC", "argentina": "LOC",
}
RELATION_PATTERNS = [
    ("recomienda", re.compile(r"\b(recomiendo|recomienda|recomendaría|les recomiendo)\b", re.I)),
    ("critica", re.compile(r"\b(odio|detesto|no me gusta|es malo|está mal|terrible|horrible|critico|critica)\b", re.I)),
    ("aprueba", re.compile(r"\b(me gusta|me encanta|está bueno|es bueno|excelente|buenísimo|amo)\b", re.I)),
    ("juega", re.compile(r"\b(juego|jugamos|jugando|jugar|partida de)\b", re.I)),
    ("habla_sobre", re.compile(r"\b(habla de|hablando de|mencionó|menciona|dijo de|comentó sobre)\b", re.I)),
]
OPINION_PATTERNS = [
    ("negativa", re.compile(r"\b(no me gusta|no me gustó|no me encanta|no me encantó|no es bueno|no está bueno|odio|detesto|horrible|terrible|es malo|está mal|malo|mala|aburrido|aburrida)\b", re.I)),
    ("positiva", re.compile(r"\b((?<!no )me gusta|(?<!no )me gustó|(?<!no )me encanta|(?<!no )me encantó|amo|excelente|buenísimo|buenísima|es bueno|está bueno|chévere)\b", re.I)),
]
TEMPORAL_RE = re.compile(r"\b(hoy|ayer|mañana|anoche|ahora|esta semana|la semana pasada|la próxima semana|el próximo mes|el mes pasado|hace (?:un|una|\d+|dos|tres|cuatro|cinco) (?:días?|semanas?|meses?|años?)|el (?:lunes|martes|miércoles|miercoles|jueves|viernes|sábado|sabado|domingo)|lunes|martes|miércoles|miercoles|jueves|viernes|sábado|sabado|domingo|en 20\d{2}|a las \d{1,2}(?::\d{2})?)\b", re.I)
GENERIC_SPOKEN_SPANS = {"ah", "aja", "ajá", "asi", "así", "bueno", "claro", "chevere", "creci", "crecí", "de verdad", "estoy", "gaaaaa", "gracias", "gente", "hola", "muchisimas", "muchísimas", "mira", "no me gusta", "oye", "papi", "para que día", "para qué día", "que tal", "qué tal", "si", "sí", "solo", "todo", "vale", "verdad", "vamos", "wow", "ya", "yo", "regala", "regalo", "tranquilo", "tranquila", "pico", "papá", "papa"}


def _tokens(text: str) -> list[str]:
    words = [word.casefold() for word in TOKEN_RE.findall(text)]
    return [word for word in words if word not in STOPWORDS and word not in FILLER_WORDS]


def _search_terms(text: str, doc=None) -> list[str]:
    """Index surface forms and Spanish lemmas, keeping time and intent words."""
    surface = _tokens(text)
    lemmas = []
    if doc is not None:
        for token in doc:
            if not token.is_alpha:
                continue
            lemma = token.lemma_.casefold().strip()
            word = token.text.casefold()
            if len(lemma) > 1 and (lemma not in STOPWORDS or word in TEMPORAL_WORDS or lemma in SEARCH_ACTION_WORDS):
                lemmas.append(lemma)
    return list(dict.fromkeys(surface + lemmas))


def _ner_model():
    try:
        import spacy

        for name in ("es_core_news_sm", "es_core_news_md", "es_core_news_lg"):
            try:
                return spacy.load(name), name
            except OSError:
                continue
    except ImportError:
        pass
    return None, "heurística local + vocabulario"


def _entities(text: str, nlp, streamer: str, doc=None) -> list[dict]:
    found: dict[str, tuple[str, str]] = {}
    normalized = text.casefold()
    for term, label in KNOWN_TERMS.items():
        if re.search(rf"(?<!\w){re.escape(term)}(?!\w)", normalized):
            found[term] = (term, label)

    if nlp:
        doc = doc if doc is not None else nlp(text)
        for ent in doc.ents:
            label = {"PER": "PERSON", "PERSON": "PERSON", "ORG": "ORG", "LOC": "PLACE", "GPE": "PLACE", "PRODUCT": "WORK", "WORK_OF_ART": "WORK", "EVENT": "EVENT", "FAC": "PLACE"}.get(ent.label_)
            phrase = " ".join(ent.text.split()).strip(" .,!?¿¡:;()[]{}\"'“”")
            phrase_key = phrase.casefold()
            words = phrase_key.split()
            if not label or len(phrase) < 2 or not words or phrase_key in GENERIC_SPOKEN_SPANS:
                continue
            # Reject spans where the small model swallowed spoken clauses as a name.
            if any(char in phrase for char in "¿?¡!.,;:"):
                continue
            if any(word in FILLER_WORDS for word in words) or words[0] in STOPWORDS:
                continue
            # Small Spanish models often tag sentence openings and short interjections
            # as names. For one-word spans, keep only proper nouns or curated terms.
            alpha_tokens = [token for token in ent if token.is_alpha]
            if len(alpha_tokens) != len(ent) or any(token.pos_ != "PROPN" for token in alpha_tokens):
                continue
            if phrase_key in GENERIC_SPOKEN_SPANS:
                continue
            if any(char.isdigit() for char in phrase):
                continue
            found.setdefault(phrase_key, (phrase, label))
    else:
        # Conservative proper-name heuristic. These are candidates for review, not model certainty.
        for match in re.finditer(r"(?<![.!?])\b(?:[A-ZÁÉÍÓÚÑ][\wáéíóúñ-]*)(?:\s+[A-ZÁÉÍÓÚÑ][\wáéíóúñ-]*){0,2}", text):
            phrase = match.group(0).strip()
            prefix = text[:match.start()].rstrip()
            at_sentence_start = not prefix or prefix[-1] in ".!?¿¡"
            if at_sentence_start and len(phrase.split()) == 1 and phrase.casefold() not in KNOWN_TERMS:
                continue
            words = phrase.split()
            if len(words) == 1 and phrase not in KNOWN_TERMS and not (phrase.isupper() and len(phrase) >= 2):
                continue
            generic_openers = {"ya", "ah", "si", "no", "pero", "bueno", "gente", "claro", "vamos", "entonces", "verdad", "gracias", "quiero", "mira", "ponte", "ahi", "porque", "todo", "muchas", "hola", "saludos"}
            if phrase.casefold() in generic_openers or any(word.casefold() in STOPWORDS or word.casefold() in FILLER_WORDS for word in words):
                continue
            found.setdefault(phrase.casefold(), (phrase, "CANDIDATE"))

    if streamer:
        found[streamer.casefold()] = (streamer, "STREAMER")
    return [{"id": key, "name": value[0], "label": value[1]} for key, value in sorted(found.items(), key=lambda item: item[0])]


def _clean_transcript(segments: list[dict]) -> dict:
    all_tokens: list[str] = []
    removed: list[str] = []
    previews = []
    for row in segments:
        text = " ".join(str(row.get("text", "")).split())
        raw_tokens = [word.casefold() for word in TOKEN_RE.findall(text)]
        kept = [word for word in raw_tokens if word not in STOPWORDS and word not in FILLER_WORDS]
        removed.extend(word for word in raw_tokens if word in STOPWORDS or word in FILLER_WORDS)
        all_tokens.extend(kept)
        previews.append({
            "start_sec": float(row.get("start_sec", 0) or 0),
            "end_sec": float(row.get("end_sec", row.get("start_sec", 0)) or 0),
            "original": text,
            "search_text": " ".join(kept),
            "removed": [word for word in raw_tokens if word in STOPWORDS or word in FILLER_WORDS],
        })
    return {"segments": previews, "term_frequency": Counter(all_tokens).most_common(100), "removed_count": len(removed), "removed_examples": removed[:100]}


def _chunks(segments: list[dict], metadata: dict) -> list[dict]:
    output, batch = [], []
    start = 0.0
    length = 0
    for row in segments:
        text = " ".join(str(row.get("text", "")).split())
        if not text:
            continue
        row_start = float(row.get("start_sec", 0) or 0)
        row_end = float(row.get("end_sec", row_start) or row_start)
        # Keep retrieval evidence short enough to preserve one idea and one precise citation.
        # Whisper's ASR segments are not the same as these PLN retrieval chunks.
        if batch and (len(batch) >= 4 or row_end - start > 30 or length > 360):
            output.append(_make_chunk(batch, start, metadata))
            overlap = batch[-1]
            batch, length = [overlap], len(overlap["text"])
            start = overlap["start_sec"]
        if not batch:
            start = row_start
        batch.append({"start_sec": row_start, "end_sec": row_end, "text": text})
        length += len(text)
    if batch:
        output.append(_make_chunk(batch, start, metadata))
    return output


def _make_chunk(rows: list[dict], start: float, metadata: dict) -> dict:
    return {"start_sec": start, "end_sec": rows[-1]["end_sec"], "text": " ".join(row["text"] for row in rows), "segments": rows,
            "stream_id": metadata["stream_id"], "title": metadata.get("title", ""), "channel_name": metadata.get("channel_name", ""),
            "date": metadata.get("date", ""), "source_url": metadata.get("source_url", "")}


def _relations(chunks: list[dict], nlp, streamer: str) -> list[dict]:
    results, seen = [], set()
    for chunk in chunks:
        spans = chunk.get("segments") or [{"text": chunk["text"], "start_sec": chunk["start_sec"], "end_sec": chunk["end_sec"]}]
        for span in spans:
            for sentence in SENTENCE_RE.split(span["text"]):
                sentence = sentence.strip()
                if not sentence:
                    continue
                entities = _entities(sentence, nlp, "")
                targets = [entity for entity in entities if entity["label"] != "STREAMER"]
                if not targets:
                    continue
                matches = [(predicate, pattern.search(sentence)) for predicate, pattern in RELATION_PATTERNS]
                matches = [(predicate, match) for predicate, match in matches if match]
                # Mere co-occurrence is a mention, not a semantic relation. Only connect
                # entities that occur after an explicit relation cue in the same sentence.
                for predicate, cue in matches:
                    subject = streamer or "streamer (sin identificar)"
                    for target in targets:
                        if target["name"] == subject.casefold():
                            continue
                        occurrence = re.search(rf"(?<!\w){re.escape(target['name'])}(?!\w)", sentence[cue.end():], re.I)
                        if not occurrence:
                            continue
                        key = (subject.casefold(), predicate, target["id"], span["start_sec"], sentence.casefold())
                        if key in seen:
                            continue
                        seen.add(key)
                        results.append({"subject": subject, "predicate": predicate, "object": target["name"], "object_id": target["id"], "object_label": target["label"],
                                        "confidence": 0.62 if nlp else 0.45, "review_status": "needs_review", "evidence": sentence,
                                        "start_sec": span["start_sec"], "end_sec": span["end_sec"], "stream_id": chunk["stream_id"],
                                        "date": chunk["date"], "title": chunk["title"]})
    return results


def _opinion_candidates(chunks: list[dict], streamer: str) -> list[dict]:
    opinions = []
    for chunk in chunks:
        spans = chunk.get("segments") or [{"text": chunk["text"], "start_sec": chunk["start_sec"], "end_sec": chunk["end_sec"]}]
        for span in spans:
            for sentence in SENTENCE_RE.split(span["text"]):
                sentence = sentence.strip()
                if not sentence:
                    continue
                targets = [entity for entity in _entities(sentence, None, "")]
                matches = [(polarity, pattern.search(sentence)) for polarity, pattern in OPINION_PATTERNS]
                matches = [(polarity, match) for polarity, match in matches if match]
                for polarity, cue in matches:
                    for target in targets:
                        if not re.search(rf"(?<!\w){re.escape(target['name'])}(?!\w)", sentence, re.I):
                            continue
                        opinions.append({
                            "subject": streamer or "streamer (sin identificar)",
                            "target": target["name"], "target_id": target["id"], "polarity": polarity,
                            "cue": cue.group(0), "evidence": sentence,
                            "start_sec": span["start_sec"],
                            "end_sec": span["end_sec"],
                            "stream_id": chunk["stream_id"], "date": chunk["date"], "title": chunk["title"],
                            "review_status": "needs_review",
                        })
    return opinions


def _temporal_references(chunks: list[dict]) -> list[dict]:
    references = []
    seen = set()
    for chunk in chunks:
        spans = chunk.get("segments") or [{"text": chunk["text"], "start_sec": chunk["start_sec"], "end_sec": chunk["end_sec"]}]
        for span in spans:
            for match in TEMPORAL_RE.finditer(span["text"]):
                value = match.group(0)
                key = (chunk["stream_id"], value.casefold(), span["start_sec"])
                if key in seen:
                    continue
                seen.add(key)
                references.append({"text": value, "start_sec": span["start_sec"],
                                   "end_sec": span["end_sec"], "stream_id": chunk["stream_id"],
                                   "date": chunk["date"], "title": chunk["title"],
                                   "evidence": span["text"]})
    return references


def process_memory(stream_id: str, segments: list[dict], metadata: dict, output_path: Path, progress=None) -> dict:
    """Create a derived artifact; never edits the source JSONL transcript."""
    model, model_name = _ner_model()
    chunks = _chunks(segments, {**metadata, "stream_id": stream_id})
    if progress:
        progress("Normalizando y segmentando el texto", 15)
    entities_by_id: dict[str, dict] = {}
    for index, chunk in enumerate(chunks):
        doc = model(chunk["text"]) if model else None
        chunk_entities = _entities(chunk["text"], model, str(metadata.get("channel_name", "")), doc=doc)
        chunk["search_text"] = " ".join(_search_terms(chunk["text"], doc))
        chunk["temporal_terms"] = list(dict.fromkeys(match.group(0).casefold() for match in TEMPORAL_RE.finditer(chunk["text"])))
        chunk["entities"] = chunk_entities
        for entity in chunk_entities:
            record = entities_by_id.setdefault(entity["id"], {**entity, "mentions": []})
            record["mentions"].append({"start_sec": chunk["start_sec"], "end_sec": chunk["end_sec"], "evidence": chunk["text"], "stream_id": stream_id})
        if progress and (index % 10 == 0 or index + 1 == len(chunks)):
            progress(f"Detectando entidades: {index + 1}/{len(chunks)} fragmentos", 20 + int(30 * (index + 1) / max(1, len(chunks))))
    relations = _relations(chunks, model, str(metadata.get("channel_name", "")))
    opinions = _opinion_candidates(chunks, str(metadata.get("channel_name", "")))
    temporal_references = _temporal_references(chunks)
    if progress:
        progress("Extrayendo opiniones, relaciones y referencias temporales", 58)
        progress("Cargando el codificador semántico multilingüe", 63)
    from src.nlp.retrieval import EMBEDDING_MODEL_NAME, encode_texts

    vectors, embedding_error = encode_texts([chunk["text"] for chunk in chunks]) if chunks else ([], "")
    embedding_status = "ready" if vectors is not None else "unavailable"
    if progress:
        progress("Generando embeddings semánticos" if vectors is not None else "Embeddings no disponibles; se conservará la búsqueda BM25", 88 if vectors is not None else 80)
    cleaned_transcript = _clean_transcript(segments)
    term_frequency = cleaned_transcript["term_frequency"]
    top_term_weights = dict(term_frequency[:20])
    key_fragments = sorted(
        chunks,
        key=lambda chunk: sum(top_term_weights.get(token, 0) for token in _tokens(chunk["text"])),
        reverse=True,
    )[:5]
    memory = {
        "schema_version": 3,
        "search_index_version": 3,
        "stream": {key: metadata.get(key) for key in ("stream_id", "streamer_id", "channel_name", "title", "date", "source_url", "platform", "audio_offset_sec")},
        "creator": {"key": creator_key({**metadata, "stream_id": stream_id}), "name": creator_name(metadata)},
        "created_at": datetime.now(timezone.utc).isoformat(),
        "processing": {"ner_model": model_name, "relation_method": "solo patrones explícitos; co-ocurrencias no se consideran relaciones", "stopwords_language": "es",
                       "preserves_source_transcript": True, "keeps_negations": True,
                       "embeddings": {"status": embedding_status, "model": EMBEDDING_MODEL_NAME, "error": embedding_error}},
        "preprocessing": cleaned_transcript,
        "interpretation": {
            "method": "lectura extractiva basada en frecuencia y fragmentos; sin LLM",
            "notice": "Los términos frecuentes orientan la lectura, pero no equivalen por sí solos a temas ni a una interpretación humana.",
            "frequent_terms": [{"term": term, "count": count} for term, count in term_frequency[:12]],
            "key_fragments": key_fragments,
        },
        "chunks": chunks,
        "entities": list(entities_by_id.values()),
        "relations": relations,
        "opinions": opinions,
        "temporal_references": temporal_references,
        "embedding_vectors": vectors,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temp_path.write_text(json.dumps(memory, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_path.replace(output_path)
    if progress:
        progress("Memoria semántica guardada", 100)
    return memory

