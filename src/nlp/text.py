"""Recursos lingüísticos compartidos para el pipeline PLN en español.

Centraliza la normalización, la clasificación de stopwords por categoría, la
tokenización/lematización y la extracción de fechas. Así la búsqueda, el conteo
de palabras y la extracción de entidades usan exactamente las mismas reglas y
las fechas no se rompen al eliminar palabras vacías.
"""

from __future__ import annotations

import re
import unicodedata


def normalize(value: str) -> str:
    """Minúsculas y sin tildes; se usa para comparar, nunca para mostrar."""
    value = unicodedata.normalize("NFKD", str(value).lower())
    return "".join(char for char in value if not unicodedata.combining(char))


# --- Palabras vacías (conectores, artículos, preposiciones...) ---------------
_STOPWORDS_FALLBACK = set(
    "a al algo algunas algunos ante antes como con contra cual cuando de del desde donde dos el ella "
    "ellas ellos en entre era erais eran eras eres es esa esas ese eso esos esta estaba estaban estado "
    "estas este esto estos estoy están está fue fueron ha hasta hay la las le les lo los mas más me mi "
    "mis mucha muchas mucho muchos muy no nos o os otra otras otro otros para pero poco por porque que "
    "quien quienes se ser si sin sobre su sus te tiene tienen todo todos tu tus un una uno unas unos y "
    "ya también entonces aquí ahi así pues".split()
)

try:  # Conjunto amplio ISO si la librería está disponible.
    import stopwordsiso

    _BASE_STOPWORDS = set(stopwordsiso.stopwords("es"))
except ImportError:  # Respaldo reducido y determinista.
    _BASE_STOPWORDS = set(_STOPWORDS_FALLBACK)

# --- Categorías lingüísticas ------------------------------------------------
# Muletillas del habla (no aportan contenido).
FILLER_WORDS = {"eh", "ehh", "em", "mmm", "mmmh", "osea", "aja", "ajam", "ajá", "digamos", "esteee"}

# Negaciones: cambian el sentido; se conservan para buscar, no se cuentan como tema.
NEGATIONS = {"no", "nunca", "jamas", "jamás", "tampoco", "ni", "sin"}

# Cuantificadores y grado: intensifican, pero no son el tema del discurso.
DEGREE_WORDS = {
    "muy", "mas", "más", "tan", "mucho", "muchos", "mucha", "muchas", "poco", "pocos",
    "poca", "pocas", "bastante", "demasiado", "demasiada", "demasiados", "demasiadas",
    "si", "sí",
}

TEMPORAL_WORDS = {
    "hoy", "ayer", "mañana", "anoche", "ahora", "antes", "después", "despues",
    "semana", "mes", "año", "ano", "día", "dia", "lunes", "martes", "miércoles",
    "miercoles", "jueves", "viernes", "sábado", "sabado", "domingo", "pasado", "pasada",
    "próximo", "proxima", "próxima", "proximo", "hace", "durante", "minuto", "minutos",
    "hora", "horas", "segundo", "segundos",
}

SEARCH_ACTION_WORDS = {
    "decir", "hacer", "hablar", "contar", "mencionar", "comentar", "anunciar", "opinar",
    "pensar", "jugar", "recomendar", "pasar", "opina", "dijo", "dice", "comento", "cuenta",
}

# Interjecciones, onomatopeyas y vocativos: nunca son nombres propios.
INTERJECTIONS = {
    "wow", "hola", "eh", "ah", "oh", "uy", "ay", "oye", "oiga", "epale", "ole", "olé",
    "uf", "puf", "guau", "epa", "jaja", "jajaja", "jeje", "jiji", "ajá", "aja", "ajam",
    "mmm", "ey", "chao", "bye", "gracias", "saludos", "buenas", "buenos", "vamos", "dale",
    "ándale", "orale", "órale", "caramba", "jo", "mira", "mire", "oigan", "escuchen",
    "atención", "ya", "bueno", "claro", "ok", "okay", "vale", "listo", "perfecto", "genial",
    "cool", "bro", "man", "gente", "amigos", "amigas", "crack", "cracks", "tío", "tía",
    "pana", "parce", "wey", "güey", "compa", "compas", "chicos", "chicas", "señores",
}

# Palabras comunes que spaCy suele confundir con nombres (sustantivos genéricos).
NOISE_WORDS = INTERJECTIONS | FILLER_WORDS | {
    "dia", "año", "ano", "noche", "tarde", "semana", "mes", "cosa", "cosas", "algo", "nada",
    "todo", "persona", "personas", "tema", "parte", "vez", "veces", "lugar", "momento", "rato",
    "mejor", "peor", "grande", "chico", "chica", "chiquito", "chiquita", "nuevo", "nueva",
    "bueno", "buena", "malo", "mala", "kiwi", "cholito", "orón", "oron", "hermano", "hermana",
    "hermanito", "hermanita", "cuídate", "cuidate", "saludito", "bendiga", "bendiciones", "socio",
    "amiguito", "amiguita", "papito", "mamita", "morocho", "morochito",
}

# --- Conjuntos derivados -----------------------------------------------------
# STOPWORDS = palabras vacías + muletillas. Se usa para descartar ruido.
STOPWORDS = _BASE_STOPWORDS | FILLER_WORDS

# Al buscar SÍ importan la negación, el grado, el tiempo y la acción de la pregunta.
SEARCH_KEEP = NEGATIONS | DEGREE_WORDS | TEMPORAL_WORDS | SEARCH_ACTION_WORDS
SEARCH_STOPWORDS = STOPWORDS - SEARCH_KEEP
SEARCH_STOPWORDS_NORMALIZED = {normalize(word) for word in SEARCH_STOPWORDS}
TEMPORAL_WORDS_NORMALIZED = {normalize(word) for word in TEMPORAL_WORDS}
FILLER_WORDS_NORMALIZED = {normalize(word) for word in FILLER_WORDS}
STOPWORDS_ONLY_NORMALIZED = {normalize(word) for word in STOPWORDS - FILLER_WORDS}

# Al contar vocabulario solo interesa el contenido: fuera funciones, negación,
# grado, tiempo, interjecciones y vocativos (que ensucian el contador).
CONTENT_STOPWORDS = STOPWORDS | DEGREE_WORDS | NEGATIONS | TEMPORAL_WORDS | NOISE_WORDS
CONTENT_STOPWORDS_NORMALIZED = {normalize(word) for word in CONTENT_STOPWORDS}


# --- Tokenización ------------------------------------------------------------
TOKEN_RE = re.compile(r"[a-záéíóúüñ0-9]+", re.IGNORECASE)
WORD_RE = re.compile(r"[A-Za-zÁÉÍÓÚÜÑa-záéíóúüñ][A-Za-zÁÉÍÓÚÜÑa-záéíóúüñ'-]+", re.IGNORECASE)
SENTENCE_RE = re.compile(r"(?<=[.!?¿¡…])\s+|\n+")

MONTHS = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6, "julio": 7,
    "agosto": 8, "septiembre": 9, "setiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12,
}
# Fechas escritas en palabras: «14 de octubre de 2026», «14 de octubre».
DATE_RE = re.compile(
    r"\b(\d{1,2})\s+de\s+(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|setiembre|"
    r"octubre|noviembre|diciembre)(?:\s+de(?:l)?\s+(\d{2,4}))?\b",
    re.IGNORECASE,
)
NUMERIC_DATE_RE = re.compile(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b")
ISO_DATE_RE = re.compile(r"\b(\d{4})[/-](\d{1,2})[/-](\d{1,2})\b")
YEAR_RE = re.compile(r"(?<![-/\d])(?:19|20)\d{2}(?![-/\d])")
TEMPORAL_RE = re.compile(
    r"\b(hoy|ayer|mañana|anoche|ahora|esta semana|la semana pasada|la próxima semana|"
    r"el próximo mes|el mes pasado|hace (?:un|una|\d+|dos|tres|cuatro|cinco) (?:días?|semanas?|meses?|años?)|"
    r"el (?:lunes|martes|miércoles|miercoles|jueves|viernes|sábado|sabado|domingo)|"
    r"lunes|martes|miércoles|miercoles|jueves|viernes|sábado|sabado|domingo|"
    r"en 20\d{2}|a las \d{1,2}(?::\d{2})?)\b",
    re.IGNORECASE,
)


def _fix_year(year: str | None) -> int | None:
    if not year:
        return None
    value = int(year)
    if value < 100:
        value += 2000 if value < 70 else 1900
    return value


def extract_dates(text: str) -> list[str]:
    """Devuelve fechas normalizadas (AAAA-MM-DD) sin romper la expresión original.

    Al proteger las fechas evitamos que quitar stopwords parta «14 de octubre
    del año 2026» en «14 ... octubre ... 2026» sin relación temporal.
    """
    found: list[str] = []
    for match in DATE_RE.finditer(text):
        day = int(match.group(1))
        month = MONTHS[match.group(2).casefold()]
        year = _fix_year(match.group(3))
        if not (1 <= day <= 31 and 1 <= month <= 12):
            continue
        found.append(f"{year:04d}-{month:02d}-{day:02d}" if year else f"--{month:02d}-{day:02d}")
    for match in ISO_DATE_RE.finditer(text):
        year, month, day = int(match.group(1)), int(match.group(2)), int(match.group(3))
        if 1 <= month <= 12 and 1 <= day <= 31:
            found.append(f"{year:04d}-{month:02d}-{day:02d}")
    for match in NUMERIC_DATE_RE.finditer(text):
        day, month, year = int(match.group(1)), int(match.group(2)), _fix_year(match.group(3))
        if 1 <= day <= 31 and 1 <= month <= 12:
            found.append(f"{year:04d}-{month:02d}-{day:02d}")
    found.extend(match.group(0) for match in YEAR_RE.finditer(text))
    return list(dict.fromkeys(found))


def date_tokens(text: str) -> list[str]:
    """Tokens de fecha listos para indexar y contar (p. ej. «2026-10-14»)."""
    return [f"fecha-{value}" for value in extract_dates(text) if not value.startswith("--")]


def filter_tokens(text: str, stopwords: set[str], *, keep_numbers: bool = True) -> list[str]:
    tokens = [token.casefold() for token in TOKEN_RE.findall(str(text))]
    kept = []
    for token in tokens:
        if token in stopwords or normalize(token) in stopwords:
            continue
        if not keep_numbers and token.isdigit():
            continue
        if len(token) == 1 and not token.isdigit():
            continue
        kept.append(token)
    return kept


def search_tokens(text: str) -> list[str]:
    """Formas superficiales para el índice de búsqueda (conserva intención/tiempo)."""
    # Search keys are accent-insensitive ("que" matches "qué"); the original
    # transcript remains untouched and is still used for display and citations.
    return [normalize(token) for token in filter_tokens(text, SEARCH_STOPWORDS_NORMALIZED)]


def content_tokens(text: str) -> list[str]:
    """Palabras de contenido para conteo y vocabulario (sin negación/grado/tiempo)."""
    return [normalize(token) for token in filter_tokens(text, CONTENT_STOPWORDS_NORMALIZED)]


def search_terms(text: str, doc=None) -> list[str]:
    """Formas superficiales + lemas en español, conservando tiempo e intención.

    La lematización unifica variantes flexivas («jugamos», «jugó» -> «jugar»),
    lo que mejora la recuperación sin depender de coincidencias exactas.
    """
    surface = search_tokens(text)
    lemmas: list[str] = []
    if doc is not None:
        for token in doc:
            if not token.is_alpha:
                continue
            lemma = normalize(token.lemma_.strip())
            word = token.text.casefold()
            if len(lemma) > 1 and (lemma not in SEARCH_STOPWORDS_NORMALIZED
                                    or normalize(word) in TEMPORAL_WORDS_NORMALIZED
                                    or lemma in SEARCH_ACTION_WORDS):
                lemmas.append(lemma)
    return list(dict.fromkeys(surface + date_tokens(text) + lemmas))


# Compatibilidad con el nombre anterior.
_tokens = search_tokens
