# Auditoría integral de la memoria de streams

Fecha de revisión: 2026-10-14. Esta auditoría contrasta la propuesta académica con los archivos, transcripciones, memorias y la aplicación disponibles en el proyecto. Las instrucciones del PDF se tratan como requisitos del trabajo; la autorización para editar y reprocesar proviene de la conversación con el usuario.

## Estado frente a la propuesta

| Requisito | Estado observado | Qué falta para poder afirmarlo como logrado |
|---|---|---|
| Transcripciones con creador, VOD, fecha y timestamps | Los JSONL conservan metadatos y tiempos en segundos. Los MP3 locales cubren dos tramos de 15 min y uno de 80 min desde el minuto 9 del VOD. | Revisar que la fecha de plataforma sea la fecha de emisión en la zona horaria correcta; corregirla manualmente si hace falta. |
| Reconocimiento de voz en español | Se usa faster-whisper con `medium` en las nuevas salidas finales. | Comparar contra una referencia humana antes de decir que el texto es exacto. Medium puede seguir equivocando nombres y palabras. |
| Preprocesamiento PLN | Se conserva el original y se construye un índice con stopwords en español, negaciones, formas superficiales y lemas cuando spaCy está disponible. | Definir las reglas de normalización con ejemplos anotados; no borrar conectores del texto fuente. |
| NER | Está instalado spaCy `es_core_news_sm`; las entidades son candidatas con fragmentos de evidencia. | El modelo es preentrenado y no está ajustado al vocabulario de estos streams; falta revisión/etiquetado humano y medir precisión/recall. |
| Relaciones y opiniones | Hay patrones léxicos, evidencia y timestamps; se marcan para revisión. | No es un extractor entrenado ni comprende bien negación, ironía, citas o contexto. No presentar sus aristas como hechos confirmados. |
| Referencias temporales | Se detectan expresiones como «hoy», «ayer», días y horas y se conserva su fragmento. | La detección por expresiones regulares no resuelve todas las referencias relativas a una fecha absoluta ni la coreferencia temporal. |
| Memoria semántica | JSON por transmisión con chunks, entidades, relaciones, referencias, metadatos y embeddings multilingües persistidos. | Para este corpus pequeño se recorre el índice en memoria; no hay FAISS/Chroma ni una base vectorial escalable. |
| Preguntas en lenguaje natural | BM25 + embeddings, filtro de metadatos/tiempo y Top-K con evidencia. | La evaluación gold aún está vacía, por eso no se conoce Precision@5/Recall@5. Las búsquedas deben comprobarse con preguntas y rangos revisados. |
| Respuesta resumida y cita verificable | La respuesta predeterminada es extractiva y muestra fragmentos, hora local del audio y enlace al VOD con tiempo. | No hay LLM configurado en este equipo; el modo generativo local vía Ollama es opcional. No se ha demostrado mejora de generación. |
| Identidad y mapa de relaciones | Los mapas se separan por ID de canal/creador y cada señal mantiene su VOD. | No se infiere que dos canales sean la misma persona; para unir canales se debe asignar un alias explícito. |
| Diarización | No está implementada. | No se puede asignar cada intervención a una persona si en el audio participan varios hablantes. Se requiere un modelo/dataset apropiado y anotación. |
| Evaluación académica | Hay herramienta de evaluación y UI para guardar relevancia. | `data/evaluation/gold_questions.jsonl` está vacío; no reportar métricas hasta anotar preguntas y citas correctas manualmente. |

## Hallazgos concretos

- Las transcripciones activas originales usaban `faster-whisper/small`. El corpus contiene tres canales distintos: auronplay (80 min de audio entre 09:00 y 88:57 del VOD), Jefrain (15 min) y Beba Recap (15 min). El dashboard ahora resta el desplazamiento del VOD y muestra por separado duración procesada y tramo de origen.
- Había una salida comparativa de Whisper `small` incompleta: contenía líneas dañadas y terminaba antes que el audio. La UI ahora cuenta solo líneas JSON válidas y señala las dañadas; las nuevas comparaciones se escriben temporalmente y se renombran al terminar con éxito.
- Las relaciones y opiniones usaban bloques largos como evidencia, aunque una regla podía activarse en otra parte del bloque. Ahora se analizan y citan los segmentos individuales de Whisper. Las señales siguen siendo candidatas automáticas, no etiquetas humanas.
- El chunking de búsqueda se redujo a ventanas de hasta cuatro segmentos, 30 segundos o 360 caracteres, con un solapamiento de contexto. Los segmentos ASR y los chunks de recuperación son unidades distintas.
- El índice anterior estaba en versión 2. La nueva versión 3 fuerza la reconstrucción de memorias antiguas con el chunking y las evidencias actualizadas.
- El acceso al audio admite peticiones parciales HTTP y trata una desconexión del reproductor durante una búsqueda/pausa como una cancelación normal.
- No hay controlador NVIDIA accesible para CTranslate2 (`cuda_devices=0`; `nvidia-smi` no conecta con el driver); Whisper está usando CPU `int8`. Esto hace lenta la retranscripción de VODs largos.

## Cambios hechos en esta revisión

- Se creó [el prompt de auditoría y mejora](PROMPT_AUDITORIA_Y_MEJORA.md) antes de empezar la revisión, como pidió el usuario, y se usó como lista de trabajo.
- El valor inicial de Whisper pasó a `medium` en la herramienta de transcripción, la ingesta web y Administración.
- Administración puede activar una salida `medium` ya comparada. Antes de sustituir el JSONL principal, crea una copia reversible en `data/processed/backups/transcripts/<stream_id>/`; elimina la memoria derivada para que se vuelva a construir.
- Se actualizó el diseño HTML para usar mejor pantallas amplias, mejorar jerarquía visual y mantener estilos locales sin depender de Google Fonts. Las citas abren el VOD de origen con el timestamp cuando la plataforma lo admite.
- El análisis de relaciones, opiniones y referencias temporales usa evidencia de segmento con tiempos propios.
- El documento de datos aclara que se permiten distintos creadores y que nunca se deben conectar por suposiciones de identidad.

## Criterio honesto de finalización

Whisper `medium` no convierte automáticamente el corpus en una transcripción correcta. Para demostrar el objetivo académico aún se requiere: (1) revisar y corregir una muestra de voz contra el audio; (2) anotar al menos diez preguntas diversas y sus rangos correctos; (3) medir recuperación y verificar las citas escuchando el VOD; (4) auditar manualmente entidades, relaciones, opiniones y fechas relativas. El proyecto no necesita entrenar un LLM desde cero: la línea base de PLN es recuperable y reproducible; un LLM local solo tendría sentido si mejora respuestas medidas sobre la misma evidencia.
