"""Hybrid retrieval helpers for the timestamped streaming corpus."""

from __future__ import annotations

import math
import re
from collections import Counter
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from src.nlp.text import (
    FILLER_WORDS,
    FILLER_WORDS_NORMALIZED,
    SEARCH_ACTION_WORDS,
    TEMPORAL_RE,
    TEMPORAL_WORDS_NORMALIZED,
    SEARCH_STOPWORDS_NORMALIZED,
    TOKEN_RE,
    extract_dates,
    normalize,
)

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
_encoder = None
_encoder_error = ""
_query_nlp = None
_query_nlp_loaded = False


def tokenize(text: str) -> list[str]:
    return [word for word in (normalize(token) for token in TOKEN_RE.findall(text))
            if word not in SEARCH_STOPWORDS_NORMALIZED and word not in FILLER_WORDS_NORMALIZED]


def _load_query_nlp():
    global _query_nlp, _query_nlp_loaded
    if _query_nlp_loaded:
        return _query_nlp
    _query_nlp_loaded = True
    try:
        import spacy
        for model_name in ("es_core_news_sm", "es_core_news_md", "es_core_news_lg"):
            try:
                _query_nlp = spacy.load(model_name)
                break
            except OSError:
                continue
    except ImportError:
        pass
    return _query_nlp


def _lemmas(text: str) -> list[str]:
    nlp = _load_query_nlp()
    if nlp is None:
        return []
    return [normalize(token.lemma_) for token in nlp(text)
            if token.is_alpha and len(token.lemma_) > 1
            and (normalize(token.lemma_) not in SEARCH_STOPWORDS_NORMALIZED or normalize(token.text) in TEMPORAL_WORDS_NORMALIZED
                 or token.lemma_.casefold() in SEARCH_ACTION_WORDS)]


def get_encoder():
    """Load the sentence encoder once; retrieval remains usable without it."""
    global _encoder, _encoder_error
    if _encoder is not None:
        return _encoder, ""
    if _encoder_error:
        return None, _encoder_error
    try:
        from sentence_transformers import SentenceTransformer

        _encoder = SentenceTransformer(EMBEDDING_MODEL_NAME)
        return _encoder, ""
    except Exception as error:  # model download or optional dependency can fail
        _encoder_error = f"Embeddings no disponibles: {error}"
        return None, _encoder_error


def encode_texts(texts: list[str], progress=None) -> tuple[list[list[float]] | None, str]:
    encoder, error = get_encoder()
    if encoder is None:
        return None, error
    try:
        vectors = encoder.encode(
            texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True,
            convert_to_numpy=True,
        )
        if progress:
            progress(100)
        return vectors.astype("float32").tolist(), ""
    except Exception as error:  # keep the corpus usable if a local model fails
        return None, f"No se pudieron generar embeddings: {error}"


def _bm25_scores(question: str, chunks: list[dict]) -> list[float]:
    query = set(tokenize(question))
    query.update(_lemmas(question))
    if not query or not chunks:
        return [0.0] * len(chunks)
    missing_indexes = [index for index, chunk in enumerate(chunks) if not chunk.get("search_text")]
    document_lemmas = {}
    nlp = _load_query_nlp() if missing_indexes else None
    if nlp is not None:
        for index, doc in zip(missing_indexes, nlp.pipe([str(chunks[i].get("text", "")) for i in missing_indexes], batch_size=64)):
            document_lemmas[index] = " ".join(normalize(token.lemma_) for token in doc
                if token.is_alpha and len(token.lemma_) > 1
                and (normalize(token.lemma_) not in SEARCH_STOPWORDS_NORMALIZED or normalize(token.text) in TEMPORAL_WORDS_NORMALIZED
                     or token.lemma_.casefold() in SEARCH_ACTION_WORDS))
    docs = []
    for index, chunk in enumerate(chunks):
        text = " ".join(str(chunk.get(key, "")) for key in
                        ("search_text", "text", "title", "channel_name", "date", "date_context"))
        text += " " + document_lemmas.get(index, "")
        docs.append(tokenize(text))
    document_frequency = Counter(token for doc in docs for token in set(doc))
    average_length = sum(map(len, docs)) / max(1, len(docs))
    scores = []
    for doc in docs:
        counts = Counter(doc)
        score = 0.0
        for token in query:
            frequency = counts[token]
            if not frequency:
                continue
            idf = math.log(1 + (len(docs) - document_frequency[token] + 0.5) /
                           (document_frequency[token] + 0.5))
            score += idf * (frequency * 2.2) / (
                frequency + 1.2 * (0.25 + 0.75 * len(doc) / max(average_length, 1)))
        scores.append(score)
    return scores


def rank_chunks(question: str, chunks: list[dict], vectors: list[list[float]] | None = None,
                top_k: int = 5) -> tuple[list[dict], str]:
    query_temporal = {match.group(0).casefold() for match in TEMPORAL_RE.finditer(question)}
    candidate_indexes = list(range(len(chunks)))
    temporal_matches = []
    if query_temporal:
        for index, chunk in enumerate(chunks):
            indexed_terms = set(chunk.get("temporal_terms", []))
            content = str(chunk.get("text", "")).casefold()
            if any(term in indexed_terms or term in content for term in query_temporal):
                temporal_matches.append(index)
        if temporal_matches:
            # A chunk is already a short timestamped context window. Include adjacent
            # windows so the answer can be recovered when the time word is at a boundary.
            candidate_indexes = sorted({nearby for index in temporal_matches
                                        for nearby in (index - 1, index, index + 1)
                                        if 0 <= nearby < len(chunks)})
    bm25 = _bm25_scores(question, chunks)
    dense: list[float] | None = None
    method = "BM25"
    if vectors and len(vectors) == len(chunks):
        encoder, _error = get_encoder()
        if encoder:
            try:
                query_vector = encoder.encode([question], normalize_embeddings=True,
                                              show_progress_bar=False, convert_to_numpy=True)[0]
                dense = [float(sum(float(a) * float(b) for a, b in zip(query_vector, vector)))
                         for vector in vectors]
                method = "Híbrida: BM25 + embedding multilingüe"
            except Exception:
                dense = None
    max_bm25 = max(bm25, default=0.0)
    scored = []
    for index in candidate_indexes:
        chunk = chunks[index]
        lexical = bm25[index] / max_bm25 if max_bm25 else 0.0
        semantic = max(0.0, dense[index]) if dense is not None else 0.0
        score = 0.55 * lexical + 0.45 * semantic if dense is not None else lexical
        chunk_temporal = set(chunk.get("temporal_terms", []))
        if query_temporal and (query_temporal & chunk_temporal or any(
                term in str(chunk.get("text", "")).casefold() for term in query_temporal)):
            score += 0.18
        if score <= 0:
            continue
        row = dict(chunk)
        row["score"] = round(score, 4)
        row["score_lexical"] = round(lexical, 4)
        row["score_semantic"] = round(semantic, 4) if dense is not None else None
        scored.append(row)
    return sorted(scored, key=lambda item: item["score"], reverse=True)[:top_k], method


def query_entities(question: str) -> list[dict]:
    """Entidades de la pregunta con las mismas reglas de limpieza que la memoria."""
    try:
        nlp = _load_query_nlp()
        from src.nlp.process_memory import KNOWN_TERMS
        if nlp is None:
            folded = question.casefold()
            return [
                {"text": term, "label": label, "source": "vocabulario PLN revisable"}
                for term, label in KNOWN_TERMS.items()
                if re.search(rf"(?<!\w){re.escape(term)}(?!\w)", folded)
            ]
        doc = nlp(question)
        allowed = {"PER": "PERSON", "PERSON": "PERSON", "ORG": "ORG", "LOC": "PLACE", "GPE": "PLACE",
                   "PRODUCT": "WORK", "WORK_OF_ART": "WORK", "EVENT": "EVENT", "FAC": "PLACE"}
        from src.nlp.text import NOISE_WORDS
        results = []
        for entity in doc.ents:
            label = allowed.get(entity.label_)
            if not label:
                continue
            text = entity.text.strip(" .,!?¿¡:;\"'")
            label = KNOWN_TERMS.get(text.casefold(), label)
            tokens = [token for token in entity if token.is_alpha]
            if not tokens or any(token.pos_ not in {"PROPN", "NOUN"} for token in tokens):
                continue
            if any(token.lemma_.casefold() in NOISE_WORDS or token.text.casefold() in NOISE_WORDS for token in tokens):
                continue
            if len(tokens) == 1 and tokens[0].pos_ != "PROPN":
                continue
            results.append({"text": text, "label": label, "source": f"spaCy · {entity.label_}"})
        return results
    except ImportError:
        return []


WEEKDAYS = {"lunes": 0, "martes": 1, "miércoles": 2, "miercoles": 2,
            "jueves": 3, "viernes": 4, "sábado": 5, "sabado": 5, "domingo": 6}


def temporal_filter(question: str, chunks: list[dict]) -> tuple[list[dict], str | None]:
    """Filter by VOD date/minute only when the question asks for that metadata."""
    folded = question.casefold()
    iso_date = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", folded)
    local_date = re.search(r"\b(\d{1,2})[/-](\d{1,2})[/-](20\d{2})\b", folded)
    wanted_date = iso_date.group(1) if iso_date else None
    if local_date:
        day, month, year = map(int, local_date.groups())
        try:
            wanted_date = date(year, month, day).isoformat()
        except ValueError:
            wanted_date = None
    asks_emission_date = bool(re.search(r"\b(transmisi[oó]n|vod|video|directo|emisi[oó]n|publicad[oa]|fecha del canal)\b", folded))
    if not wanted_date and asks_emission_date:
        full_dates = [value for value in extract_dates(question)
                      if re.fullmatch(r"20\d{2}-\d{2}-\d{2}", value)]
        wanted_date = full_dates[0] if full_dates else None
        month_day = next((value[-5:] for value in extract_dates(question) if value.startswith("--")), None)
        if not wanted_date and month_day:
            filtered = [chunk for chunk in chunks if str(chunk.get("date", ""))[-5:] == month_day]
            if filtered:
                return filtered, f"Fecha de emisión: {month_day} (año no especificado)"
            return [], f"No hay transmisión con fecha {month_day}"
    if not wanted_date and asks_emission_date:
        relative_day = None
        if re.search(r"\b(hoy)\b", folded):
            relative_day = 0
        elif re.search(r"\b(ayer|anoche)\b", folded):
            relative_day = -1
        elif re.search(r"\b(mañana)\b", folded):
            relative_day = 1
        if relative_day is not None:
            lima_today = datetime.now(ZoneInfo("America/Lima")).date()
            wanted_date = (lima_today + timedelta(days=relative_day)).isoformat()
    if not wanted_date and asks_emission_date:
        for weekday, number in WEEKDAYS.items():
            if re.search(rf"\b{weekday}\b", folded):
                candidates = []
                for chunk in chunks:
                    try:
                        if date.fromisoformat(str(chunk.get("date", ""))[:10]).weekday() == number:
                            candidates.append(chunk)
                    except ValueError:
                        continue
                if candidates:
                    return candidates, f"Día de emisión: {weekday}"
                break
    if wanted_date:
        filtered = [chunk for chunk in chunks if str(chunk.get("date", ""))[:10] == wanted_date]
        if filtered:
            return filtered, f"Fecha de emisión: {wanted_date}"
        return [], f"No hay transmisión con fecha {wanted_date}"
    minute = re.search(r"\bminuto\s+(\d{1,4})(?::(\d{1,2}))?\b", folded)
    if minute:
        seconds = int(minute.group(1)) * 60 + int(minute.group(2) or 0)
        filtered = [chunk for chunk in chunks
                    if float(chunk.get("end_sec", 0)) >= seconds - 60
                    and float(chunk.get("start_sec", 0)) <= seconds + 60]
        if filtered:
            return filtered, f"Alrededor del minuto {minute.group(1)}"
        return [], f"No hay fragmentos alrededor del minuto {minute.group(1)}"
    # A phrase such as «qué dijo ayer» refers to words in the speech, not the
    # publication date of the VOD. Restrict retrieval to matching speech spans
    # and their adjacent short windows; do not silently reinterpret it as a date.
    spoken_temporal = {match.group(0).casefold() for match in TEMPORAL_RE.finditer(question)}
    spoken_dates = set(extract_dates(question))
    if spoken_temporal or spoken_dates:
        matched = []
        for chunk in chunks:
            text = str(chunk.get("text", "")).casefold()
            terms = {str(term).casefold() for term in chunk.get("temporal_terms", [])}
            date_matches = set(extract_dates(text))
            date_match = any(
                candidate in date_matches
                or (candidate.startswith("--") and any(value.endswith(candidate[1:]) for value in date_matches))
                for candidate in spoken_dates
            )
            if (spoken_temporal & terms or any(term in text for term in spoken_temporal) or date_match):
                matched.append(chunk)
        if matched:
            matched_ids = {id(chunk) for chunk in matched}
            indexes = [index for index, chunk in enumerate(chunks) if id(chunk) in matched_ids]
            neighborhood = sorted({nearby for index in indexes for nearby in (index - 1, index, index + 1)
                                   if 0 <= nearby < len(chunks)})
            selected = [chunks[index] for index in neighborhood]
            details = [*sorted(spoken_temporal), *sorted(spoken_dates)]
            return selected, f"Expresión temporal encontrada en el habla: {', '.join(details)}"
    return chunks, None
