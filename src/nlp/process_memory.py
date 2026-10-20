"""Build reviewable, timestamped PLN artifacts from an existing transcript."""

from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from src.nlp.creators import creator_key, creator_name
from src.nlp.text import (
    CONTENT_STOPWORDS,
    CONTENT_STOPWORDS_NORMALIZED,
    FILLER_WORDS,
    NOISE_WORDS,
    SEARCH_ACTION_WORDS,
    SENTENCE_RE,
    STOPWORDS,
    STOPWORDS_ONLY_NORMALIZED,
    TEMPORAL_RE,
    TEMPORAL_WORDS,
    TOKEN_RE,
    content_tokens,
    date_tokens,
    extract_dates,
    normalize,
    search_terms,
    search_tokens,
)

# Sube cuando cambia el índice o la estructura; obliga a reconstruir memorias.
SEARCH_INDEX_VERSION = 10

# Vocabulario de dominio revisable. Ayuda a que el NER no dependa solo del
# modelo genérico en nombres de juegos, plataformas y personas del corpus.
KNOWN_TERMS = {
    # Videojuegos / obras
    "valorant": "WORK", "minecraft": "WORK", "fortnite": "WORK", "league of legends": "WORK",
    "lol": "WORK", "dota": "WORK", "dota 2": "WORK", "roblox": "WORK", "gta": "WORK",
    "grand theft auto": "WORK", "among us": "WORK", "counter strike": "WORK",
    # Plataformas y comunidades
    "twitch": "ORG", "youtube": "ORG", "kick": "ORG", "discord": "ORG", "whatsapp": "ORG",
    "telegram": "ORG", "instagram": "ORG", "tiktok": "ORG", "google": "ORG", "wikipedia": "ORG",
    "spotify": "ORG", "twitter": "ORG", "facebook": "ORG",
    "armys": "ORG", "army": "ORG", "bts": "ORG", "kpop": "ORG", "sideral": "ORG", "sidereal": "ORG",
    # Personas y lugares del corpus
    "auronplay": "PERSON", "fabiana sevillano": "PERSON", "fabiana": "PERSON",
    "pereira": "PERSON", "agustín": "PERSON", "agustin": "PERSON", "jefrain": "PERSON",
    "perú": "LOC", "lima": "LOC", "chile": "LOC", "argentina": "LOC", "madrid": "LOC",
    "barcelona": "LOC", "colombia": "LOC", "brasil": "LOC",
}
RELATION_PATTERNS = [
    ("recomienda", re.compile(r"\b(recomiendo|recomienda|recomendaría|les recomiendo|recomendado|deberían probar)\b", re.I)),
    ("critica", re.compile(r"\b(odio|detesto|no me gusta|no me gustó|es malo|está mal|está horrible|terrible|horrible|una porquería|un desastre|critico|critica)\b", re.I)),
    ("aprueba", re.compile(r"\b(me gusta|me gustó|me encanta|me encantó|está bueno|es bueno|excelente|buenísimo|buenísima|amo|increíble|es un crack)\b", re.I)),
    ("juega", re.compile(r"\b(juego|jugamos|jugando|jugar|partida de|estoy jugando|vamos a jugar)\b", re.I)),
    ("habla_sobre", re.compile(r"\b(habla de|hablando de|habló de|mencionó|menciona|dijo de|comentó sobre|se refirió a|trata de)\b", re.I)),
]
OPINION_PATTERNS = [
    ("negativa", re.compile(r"\b(no me gusta|no me gustó|no me encanta|no me encantó|no es bueno|no está bueno|no sirve|odio|detesto|horrible|terrible|asqueroso|asquerosa|una porquería|un desastre|es malo|está mal|malísimo|malísima|aburrido|aburrida|me decepcionó)\b", re.I)),
    ("positiva", re.compile(r"\b(me gusta|me gustó|me encanta|me encantó|amo|excelente|buenísimo|buenísima|es bueno|está bueno|está genial|está increíble|increíble|una belleza|un crack|vale la pena|chévere|me fascina)\b", re.I)),
]
GENERIC_SPOKEN_SPANS = NOISE_WORDS | {
    "asi", "así", "chevere", "creci", "crecí", "de verdad", "estoy", "gaaaaa",
    "muchisimas", "muchísimas", "no me gusta", "papi", "para que día", "para qué día",
    "que tal", "qué tal", "solo", "verdad", "regala", "regalo", "tranquilo", "tranquila",
    "pico", "papá", "papa",
}


# Se conservan los nombres internos como alias del módulo compartido text.py.
def _tokens(text: str) -> list[str]:
    return search_tokens(text)


def _search_terms(text: str, doc=None) -> list[str]:
    """Formas superficiales + lemas en español (ver src/nlp/text.py)."""
    return search_terms(text, doc)


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
    found: dict[str, tuple[str, str, str]] = {}
    normalized = text.casefold()
    for term, label in KNOWN_TERMS.items():
        if re.search(rf"(?<!\w){re.escape(term)}(?!\w)", normalized):
            found[term] = (term, label, "vocabulario PLN revisable")

    if nlp:
        doc = doc if doc is not None else nlp(text)
        for ent in doc.ents:
            label = {"PER": "PERSON", "PERSON": "PERSON", "ORG": "ORG", "LOC": "PLACE", "GPE": "PLACE",
                     "PRODUCT": "WORK", "WORK_OF_ART": "WORK", "EVENT": "EVENT", "FAC": "PLACE"}.get(ent.label_)
            if not label:  # MISC y etiquetas fuera de alcance
                continue
            phrase = " ".join(ent.text.split()).strip(" .,!?¿¡:;()[]{}\"'“”")
            phrase_key = phrase.casefold()
            if len(phrase) < 2 or not phrase_key or phrase_key in GENERIC_SPOKEN_SPANS:
                continue
            if any(char in phrase for char in "¿?¡!.,;:") or any(char.isdigit() for char in phrase):
                continue
            alpha_tokens = [token for token in ent if token.is_alpha]
            if not alpha_tokens or len(alpha_tokens) != len(ent):
                continue
            # Regla 1: interjecciones, muletillas y vocativos nunca son entidades.
            if any(token.lemma_.casefold() in NOISE_WORDS or token.text.casefold() in NOISE_WORDS
                   for token in alpha_tokens):
                continue
            # Regla 2: cada token debe ser nombre propio o sustantivo real; se
            # permiten nexos internos («Universidad de Lima»), no al inicio/fin.
            content_pos = {"PROPN", "NOUN"}
            if any(token.pos_ not in content_pos for token in (alpha_tokens[0], alpha_tokens[-1])):
                continue
            if any(token.pos_ not in content_pos | {"ADP", "CCONJ", "DET"}
                   for token in alpha_tokens[1:-1]):
                continue
            # Regla 3: una entidad de una sola palabra debe ser nombre propio real
            # (evita «bajón», «terracotta», conjugaciones sueltas, etc.).
            if len(alpha_tokens) == 1 and alpha_tokens[0].pos_ != "PROPN":
                continue
            # Regla 4: un span repetido («wow wow») es ruido, no un nombre.
            lemmas = [token.lemma_.casefold() for token in alpha_tokens]
            if len(set(lemmas)) != len(lemmas):
                continue
            found.setdefault(phrase_key, (phrase, label, f"spaCy · {ent.label_}"))
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
            if phrase.casefold() in generic_openers or phrase.casefold() in NOISE_WORDS:
                continue
            if any(word.casefold() in STOPWORDS or word.casefold() in FILLER_WORDS for word in words):
                continue
            found.setdefault(phrase.casefold(), (phrase, "CANDIDATE", "heurística local de nombres"))

    if streamer:
        found[streamer.casefold()] = (streamer, "STREAMER", "metadato del canal")
    return [{"id": key, "name": value[0], "label": value[1], "source": value[2]}
            for key, value in sorted(found.items(), key=lambda item: item[0])]


def _clean_transcript(segments: list[dict]) -> dict:
    all_tokens: list[str] = []
    removed: list[str] = []
    previews = []
    for row in segments:
        text = " ".join(str(row.get("text", "")).split())
        raw_tokens = [word.casefold() for word in TOKEN_RE.findall(text)]
        kept = content_tokens(text)
        removed.extend(word for word in raw_tokens
                       if normalize(word) in CONTENT_STOPWORDS_NORMALIZED or (len(word) == 1 and not word.isdigit()))
        all_tokens.extend(kept)
        previews.append({
            "start_sec": float(row.get("start_sec", 0) or 0),
            "end_sec": float(row.get("end_sec", row.get("start_sec", 0)) or 0),
            "original": text,
            "search_text": " ".join(search_tokens(text) + date_tokens(text)),
            "terms": kept,
            "removed": [word for word in raw_tokens if normalize(word) in CONTENT_STOPWORDS_NORMALIZED],
            "dates": extract_dates(text),
        })
    return {"segments": previews, "term_frequency": Counter(all_tokens).most_common(100),
            "removed_count": len(removed), "removed_examples": removed[:100]}


def _word_counter(segments: list[dict]) -> dict:
    """Contador de palabras y riqueza léxica sobre el texto completo.

    Se cuentan solo palabras de contenido: las palabras vacías, las muletillas y
    los cuantificadores no inflan el vocabulario. Las fechas se conservan aparte
    para que eliminar stopwords no rompa una fecha.
    """
    total_tokens = 0
    content: list[str] = []
    stopwords_removed = 0
    fillers_removed = 0
    content_excluded = 0
    dates: list[str] = []
    for row in segments:
        text = " ".join(str(row.get("text", "")).split())
        raw_tokens = [word.casefold() for word in TOKEN_RE.findall(text)]
        total_tokens += len(raw_tokens)
        content.extend(content_tokens(text))
        stopwords_removed += sum(1 for word in raw_tokens if normalize(word) in STOPWORDS_ONLY_NORMALIZED)
        fillers_removed += sum(1 for word in raw_tokens if word in FILLER_WORDS)
        content_excluded += sum(1 for word in raw_tokens if normalize(word) in CONTENT_STOPWORDS_NORMALIZED)
        dates.extend(extract_dates(text))
    unique = len(set(content))
    return {
        "total_words": total_tokens,
        "content_words": len(content),
        "unique_content_words": unique,
        "lexical_diversity": round(unique / len(content), 3) if content else 0.0,
        "stopwords_removed": stopwords_removed,
        "fillers_removed": fillers_removed,
        "content_excluded": content_excluded,
        "dates": list(dict.fromkeys(dates)),
        "top_content_terms": [{"term": term, "count": count} for term, count in Counter(content).most_common(20)],
        "content_frequencies": dict(Counter(content)),
    }


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
                                        "method": "patrón léxico explícito", "review_status": "needs_review", "evidence": sentence,
                                        "start_sec": span["start_sec"], "end_sec": span["end_sec"], "stream_id": chunk["stream_id"],
                                        "date": chunk["date"], "title": chunk["title"]})
    return results


def _opinion_candidates(chunks: list[dict], streamer: str, nlp=None) -> list[dict]:
    opinions = []
    seen = set()
    for chunk in chunks:
        spans = chunk.get("segments") or [{"text": chunk["text"], "start_sec": chunk["start_sec"], "end_sec": chunk["end_sec"]}]
        for span in spans:
            for sentence in SENTENCE_RE.split(span["text"]):
                sentence = sentence.strip()
                if not sentence:
                    continue
                matches = [(polarity, pattern.search(sentence)) for polarity, pattern in OPINION_PATTERNS]
                matches = [(polarity, match) for polarity, match in matches if match]
                if not matches:
                    continue
                # Las entidades se buscan con el modelo NER real, no con la heurística.
                targets = [entity for entity in _entities(sentence, nlp, "") if entity["label"] != "STREAMER"]
                polarity, cue = matches[0]
                target_rows = targets or [{"name": "el tema tratado", "id": "tema", "label": "TEMA"}]
                for target in target_rows:
                    key = (chunk["stream_id"], span["start_sec"], polarity, target["id"])
                    if key in seen:
                        continue
                    seen.add(key)
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
    opinions = _opinion_candidates(chunks, str(metadata.get("channel_name", "")), model)
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
    word_counter = _word_counter(segments)
    term_frequency = cleaned_transcript["term_frequency"]
    top_term_weights = dict(term_frequency[:20])
    key_fragments = sorted(
        chunks,
        key=lambda chunk: sum(top_term_weights.get(token, 0) for token in _tokens(chunk["text"])),
        reverse=True,
    )[:5]
    memory = {
        "schema_version": 4,
        "search_index_version": SEARCH_INDEX_VERSION,
        "stream": {key: metadata.get(key) for key in ("stream_id", "streamer_id", "channel_name", "title", "date", "source_url", "platform", "audio_offset_sec")},
        "creator": {"key": creator_key({**metadata, "stream_id": stream_id}), "name": creator_name(metadata)},
        "created_at": datetime.now(timezone.utc).isoformat(),
        "processing": {"ner_model": model_name, "relation_method": "solo patrones explícitos; co-ocurrencias no se consideran relaciones",
                       "stopwords_language": "es", "stopwords_categories": "vacías, muletillas, negación, grado y tiempo separadas",
                       "preserves_source_transcript": True, "keeps_negations": True, "protects_dates": True,
                       "lemmatization": "spaCy es_core_news cuando está disponible",
                       "embeddings": {"status": embedding_status, "model": EMBEDDING_MODEL_NAME, "error": embedding_error}},
        "preprocessing": cleaned_transcript,
        "word_counter": word_counter,
        "interpretation": {
            "method": "lectura extractiva basada en frecuencia y fragmentos; sin LLM",
            "notice": "Los términos frecuentes orientan la lectura, pero no equivalen por sí solos a temas ni a una interpretación humana.",
            "frequent_terms": word_counter["top_content_terms"][:12],
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

