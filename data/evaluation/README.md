# Evaluación de recuperación

Este archivo guarda preguntas y fragmentos relevantes anotados manualmente. Es el conjunto de referencia para medir si la memoria recupera evidencia útil; Whisper o un LLM no pueden decidir por sí solos cuál es la respuesta correcta.

## Formato

`gold_questions.jsonl` está vacío al 2026-10-19; por eso la aplicación muestra Precision@5 y Recall@5 como «—». Anota una pregunta por línea después de revisar el audio:

```json
{"question":"¿Qué opinión dio sobre Valorant?","relevant":[{"stream_id":"ID_DEL_VOD","start_sec":642.0,"end_sec":668.0}]}
```

`start_sec` y `end_sec` son los tiempos del VOD original. Marca cada fragmento que responda la pregunta. Si dos directos responden, añade los dos rangos. Las entidades de la pregunta sirven para análisis, pero la anotación humana define el gold standard.

## Flujo de anotación

1. Usa las tres transcripciones activas (`OIKASQeeLdI`, `u2DrPM5UpwI`, `v2900590088`) y revisa una muestra de nombres propios, fechas y timestamps. No vuelvas a descargarlas para evaluar.
2. Escribe al menos diez preguntas variadas: tema, opinión sobre una entidad, recomendación, referencia temporal y búsqueda general.
3. Reproduce el audio y anota los rangos del VOD original que sí contienen la respuesta. No anotes un fragmento solo porque comparta una palabra: debe responder la pregunta.
4. Ejecuta `python -m src.nlp.evaluate_retrieval --k 5` desde la raíz del proyecto.
5. Registra Precision@5 y Recall@5 antes y después de cambiar el recuperador. Revisa también a mano las citas del resultado.

La evaluación busca por BM25 y agrega embeddings cuando la memoria se procesó con `sentence-transformers`. Si el modelo semántico no está disponible, el informe indica que usó BM25. Para evaluar un creador, `stream_ids` debe contener los IDs de sus VODs; para una pregunta de todo el corpus puede ser una lista vacía. El campo `scope` que usa la interfaz distingue ambas opciones.

La interfaz permite guardar evidencia relevante desde los resultados, pero no marca ningún resultado automáticamente como correcto: escucha y confirma la cita antes de anotarla. Las métricas no son válidas como evidencia académica hasta que el conjunto gold esté revisado.
