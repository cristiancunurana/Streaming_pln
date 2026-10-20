"""Generación de respuestas en lenguaje natural a partir de la evidencia recuperada.

Convierte los fragmentos Top-K en una respuesta redactada (no una lista de
citas crudas): selecciona el fragmento o la oración más informativa, la
introduce con lenguaje natural y añade una cita compacta con creador, directo,
fecha y minuto. No inventa datos: todo lo que afirma proviene de la evidencia.
"""

from __future__ import annotations

from src.nlp.text import SENTENCE_RE, content_tokens, normalize


def format_timestamp(seconds: float) -> str:
    total = max(0, int(float(seconds or 0)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _short_title(title: str, limit: int = 58) -> str:
    title = " ".join(str(title or "").split())
    if len(title) <= limit:
        return title
    return title[: limit - 1].rstrip(" ,.;:") + "…"


def _when(row: dict) -> str:
    """Cita compacta: minuto, directo (recortado) y fecha."""
    minute = format_timestamp(row.get("start_sec", 0))
    title = _short_title(row.get("title", ""))
    date = str(row.get("date", ""))[:10]
    phrase = f"el minuto {minute}"
    if title:
        phrase += f" del directo «{title}»"
    if date:
        phrase += f" ({date})"
    return phrase


def _units(row: dict) -> list[str]:
    """Divide la evidencia en unidades citables: segmentos ASR u oraciones."""
    segments = row.get("segments")
    if segments:
        return [str(segment.get("text", "")).strip() for segment in segments if str(segment.get("text", "")).strip()]
    return [part.strip() for part in SENTENCE_RE.split(str(row.get("text", ""))) if part.strip()]


def _best_sentence(row: dict, query_terms: list[str], entities: list[str]) -> str:
    units = _units(row)
    if not units:
        return str(row.get("text", "")).strip()
    if len(units) == 1:
        return units[0]

    def score(sentence: str) -> float:
        folded = normalize(sentence)
        value = 2.0 * sum(1 for term in query_terms if term in folded)
        value += 3.0 * sum(1 for entity in entities if entity and normalize(entity) in folded)
        value += min(len(sentence), 180) / 1000.0  # premia una idea más completa
        return value

    return max(units, key=score)


def _polarity_phrase(row: dict) -> str:
    polarity = row.get("opinion_candidate")
    if polarity == "positiva":
        return " En el fragmento se percibe una valoración positiva."
    if polarity == "negativa":
        return " En el fragmento se percibe una valoración negativa."
    return ""


def compose_answer(
    question: str,
    ranked: list[dict],
    *,
    entities: list[str] | None = None,
    intent: str = "retrieval",
    temporal_context: str | None = None,
    topic_terms: list[tuple[str, int]] | None = None,
    speaker_default: str = "el creador",
) -> str:
    entities = [entity for entity in (entities or []) if entity]
    if not ranked:
        return ("No encontré evidencia suficiente en las transcripciones para responder. "
                "Prueba con otra formulación o revisa si la fecha o el minuto están en el corpus.")

    query_terms = content_tokens(question)
    speaker = str(ranked[0].get("channel_name") or speaker_default)
    topic = entities[0] if entities else (query_terms[0] if query_terms else "el tema")
    primary = _best_sentence(ranked[0], query_terms, entities)
    primary_when = _when(ranked[0])

    if intent == "topics" and topic_terms:
        words = ", ".join(f"{word} ({count})" for word, count in topic_terms[:5])
        answer = (f"El discurso gira en torno a estos términos por frecuencia: {words}. "
                  f"Un momento representativo es {primary_when}: «{primary}».")
    elif intent == "opinion":
        answer = f"Sobre {topic}, {speaker} se pronuncia en {primary_when}: «{primary}»."
        answer += _polarity_phrase(ranked[0])
    elif entities:
        answer = f"Sobre {topic}, {speaker} lo comenta en {primary_when}: «{primary}»."
    else:
        answer = f"{speaker} lo menciona en {primary_when}: «{primary}»."

    if len(ranked) > 1:
        second = _best_sentence(ranked[1], query_terms, entities)
        if second and normalize(second) != normalize(primary):
            answer += f" También aparece en {_when(ranked[1])}: «{second}»."

    if temporal_context:
        answer += f" ({temporal_context}.)"
    return answer
