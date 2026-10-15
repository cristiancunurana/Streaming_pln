# Evaluación de recuperación

Este archivo guarda preguntas y fragmentos relevantes anotados manualmente. Es el conjunto de referencia para medir si la memoria recupera evidencia útil; Whisper o un LLM no pueden decidir por sí solos cuál es la respuesta correcta.

## Formato

Añade una pregunta por línea a `gold_questions.jsonl`:

```json
{"question":"¿Qué opinión dio sobre Valorant?","relevant":[{"stream_id":"ID_DEL_VOD","start_sec":642.0,"end_sec":668.0}]}
```

`start_sec` y `end_sec` son los tiempos del VOD original. Marca cada fragmento que responda la pregunta. Si dos directos responden, añade los dos rangos. Las entidades de la pregunta sirven para análisis, pero la anotación humana define el gold standard.

## Flujo de anotación

1. Transcribe los tres VODs y revisa una muestra de nombres propios, fechas y timestamps.
2. Escribe al menos diez preguntas variadas: tema, opinión sobre una entidad, recomendación, referencia temporal y búsqueda general.
3. Reproduce el audio y anota los rangos que sí contienen la respuesta. No anotes un fragmento porque comparta una palabra: debe responder la pregunta.
4. Ejecuta `python -m src.nlp.evaluate_retrieval --k 5` desde la raíz del proyecto.
5. Registra Precision@5 y Recall@5 antes y después de cambiar el recuperador. Revisa también a mano las citas del resultado.

La evaluación busca por BM25 y agrega embeddings cuando la memoria se procesó con `sentence-transformers`. Si el modelo semántico no está disponible, el informe indica que usó BM25.
