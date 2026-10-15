# Datos del proyecto

Mantén las fuentes originales en `raw/` y los resultados derivados en `processed/`. No sobrescribas la transcripción original durante la limpieza.

## Alcance recomendado

El corpus puede contener transmisiones de creadores distintos. La búsqueda de biblioteca compara sus textos, pero la memoria de relaciones se mantiene separada por ID de creador/canal; no se debe inferir identidad por nombres parecidos. Si el objetivo experimental es estudiar cambios de opinión de una persona, sí conviene elegir VODs del mismo creador y tema. Para la demo basta un fragmento de 10 a 20 minutos por VOD, siempre que se indique con claridad qué parte se transcribió.

La opción más directa para tener control sobre los datos es usar tus propias transmisiones o pedir permiso a un creador. La herramienta `src/ingest/process_vod.py` recibe el enlace de un VOD terminado, intenta obtener sus metadatos y descarga el audio en MP3 para transcribirlo. Para la demo, transcribe solo 10 a 20 minutos de cada VOD. Registra la autorización o las condiciones de uso junto a la fuente. El prototipo no captura emisiones en vivo.

La API oficial de subtítulos de YouTube permite listar pistas, pero descargar una pista requiere autorización y permisos sobre el video. La API de Twitch tampoco debe asumirse como fuente de transcripciones. No bases el corpus en una API que no te entregue legal y técnicamente el texto. Como alternativa temporal para probar el pipeline, puedes usar un corpus oral abierto en español, aunque ya no representaría transmisiones de streaming; deja esa diferencia clara en el informe.

## Plan para reunir las tres transmisiones

1. Elige si compararás varios creadores o estudiarás a uno solo; delimita un tema o formula preguntas que puedan responderse con el corpus elegido.
2. Identifica tres VODs y confirma que puedes usar su contenido: autorización del creador, contenido propio o términos de reutilización compatibles.
3. Ejecuta `process_vod.py` con el enlace autorizado del VOD. La herramienta guarda el audio MP3 y recoge los metadatos que la página expone.
4. Usa `--start-sec` y `--duration-sec` para procesar un tramo de 10 a 20 minutos. Si la página no entrega la fecha, pásala con `--date AAAA-MM-DD`.
5. Revisa nombres propios y timestamps en una muestra. Los tiempos deben quedar referidos al VOD original.
6. Registra cada VOD en la tabla del corpus y conserva el archivo de origen.

## Formato de entrada sugerido

La herramienta `src/ingest/transcribe.py` genera un archivo JSONL por transmisión. Cada línea es un segmento de transcripción, con los metadatos repetidos para que cada fragmento pueda rastrearse por sí solo:

```json
{"streamer_id":"creador_01","stream_id":"video_001","title":"Título de la transmisión","date":"2026-01-15","source_url":"https://example.org/video","language":"es","start_sec":12.4,"end_sec":17.8,"text":"En este fragmento se conserva lo que dijo el creador."}
```

Usa segundos numéricos para facilitar cálculos. El ejemplo muestra el formato; la herramienta también añade `transcription_model`. Conserva este texto automático como fuente y, si corriges errores, guarda las correcciones aparte para no perder la transcripción original. `source_url` es la referencia de procedencia; reemplaza el ejemplo por la dirección real.

## Registro del corpus

Completa una fila por transmisión antes de procesarla:

| stream_id | streamer_id | título | fecha | idioma | URL | tipo de transcripción | duración aproximada |
|---|---|---|---|---|---|---|---|
| | | | | es | | manual / subtítulos | |

No incluyas credenciales, cookies ni datos personales del chat. Si una transcripción contiene información privada incidental, elimínala del corpus de trabajo.
