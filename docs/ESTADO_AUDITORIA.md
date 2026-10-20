# Estado del proyecto: memoria PLN de transmisiones

Última revisión: 2026-10-19. Este informe registra lo que está implementado y lo que aún requiere evaluación; la propuesta PDF se usa como criterio académico, no como instrucción para ejecutar acciones.

## Corpus disponible

| Transmisión | Creador/canal | Fecha registrada | Audio transcrito | Whisper activo | Segmentos ASR |
|---|---|---:|---:|---|---:|
| `OIKASQeeLdI` | Beba Recap | 2026-10-08 | 15 min | medium | 619 |
| `u2DrPM5UpwI` | Jefrain | 2026-10-09 | 15 min | medium | 633 |
| `v2900590088` | auronplay | 2026-10-14 | ~80 min; desde 09:00 del VOD | medium | 604 |

Son tres creadores diferentes. El mapa se limita a un mismo identificador de creador/canal; las consultas generales pueden buscar entre los VODs, pero cada cita retiene su procedencia y timestamp. La fecha mostrada proviene de los metadatos disponibles y todavía debe contrastarse con la fecha real de emisión y zona horaria.

## Flujo que ya existe

1. **Biblioteca:** muestra VOD, canal, fecha, duración del audio procesado, rango del VOD, estado de transcripción y modelo.
2. **Consulta:** recuperación híbrida léxica/semántica con filtro temporal, respuesta extractiva y resultados con timestamp y fuente.
3. **Preparación:** el JSONL original se conserva; se normaliza una vista de búsqueda, se filtran stopwords sin quitar negaciones del texto fuente y se usa spaCy para lemas cuando está disponible.
4. **Unidades de memoria:** los 1,856 segmentos ASR activos se agrupan en 669 fragmentos de recuperación (máximo de cuatro segmentos, 30 s o 360 caracteres, con solapamiento).
5. **Memoria por creador:** los artefactos por VOD conservan fragmentos, entidades candidatas, relaciones, opiniones, expresiones temporales, vocabulario y embeddings; los índices actuales están en versión 10.
6. **Revisión:** candidatos de entidades, relaciones y opiniones pueden confirmarse o descartarse. Las entidades admiten corrección textual. Las anotaciones humanas se guardan aparte en `data/processed/reviews/`; no reescriben el JSONL ni pretenden ser un modelo entrenado.
7. **Administración:** permite procesar la memoria, comparar Whisper sobre el audio ya guardado, activar una salida `medium` con respaldo y borrar datos mediante confirmación.
8. **Evaluación:** la interfaz muestra métricas solo si hay preguntas gold anotadas. Se debe escuchar cada fragmento antes de marcarlo relevante.

## PLN implementado y límites

- **ASR:** faster-whisper `medium` en español en las tres transcripciones activas. `medium` no garantiza palabras correctas; nombres propios, ruido y solapamiento de voces requieren revisión.
- **Tokenización/normalización:** una vista para búsqueda mantiene acentos normalizados y lemas; se conserva intacto el texto mostrado como evidencia. Negación y expresiones temporales se preservan.
- **NER:** spaCy `es_core_news_sm` más reglas/vocabulario como generadores de candidatos. No hay ajuste supervisado específico del dominio ni métricas de NER.
- **Relaciones/opiniones:** patrones léxicos interpretables; la polaridad es una hipótesis automática. Ironía, cita, negación y contexto pueden producir errores. La revisión guardada es humana, pero aún no se usa para reentrenar.
- **Tiempo:** se conservan timestamps del VOD y expresiones literales como «ayer». Las referencias relativas no se deben convertir en una fecha absoluta sin un ancla temporal fiable.
- **Recuperación:** BM25 y embeddings multilingües cuando están disponibles. Los embeddings mejoran similitud semántica potencial, pero no demuestran por sí mismos calidad.
- **Generación:** baseline extractivo con citas. No se requiere entrenar un LLM desde cero. La opción local generativa es adicional y debe juzgarse contra el mismo conjunto de evidencias.
- **Diarización:** no implementada; el sistema no atribuye automáticamente fragmentos a cada voz.

## Evaluación pendiente

`data/evaluation/gold_questions.jsonl` todavía no tiene preguntas validadas. Por ello no se reportan Precision@5 ni Recall@5 como resultados del proyecto. El siguiente paso académico es crear un conjunto pequeño pero diverso de preguntas sobre cada VOD, anotar los rangos correctos después de escuchar, y comparar BM25, embeddings e híbrido con las mismas preguntas. También conviene anotar una muestra de entidades, relaciones, opiniones y referencias temporales.

El equipo está ejecutando Whisper en CPU `int8` (no hay CUDA utilizable por CTranslate2 en el entorno revisado). La transcripción larga puede tardar; no hace falta volver a descargar el audio para reprocesar la memoria.

## Cambios principales de la revisión

- Reorganización de la interfaz en Biblioteca, Consultar, Proceso PLN, Memoria, Evaluación y Administración.
- Vista más amplia y consistente, navegación funcional y separación de flujos de usuario y administrador.
- Índices de memoria reconstruidos a partir de las transcripciones `medium` existentes; no se volvió a descargar audio.
- Búsqueda sensible a acentos y expresiones temporales habladas, distinguiendo «qué dijo ayer» de preguntas sobre fechas de publicación.
- Grafo agrupado por creador, con enlaces solo para relaciones explícitas detectadas y lista de evidencias con timestamps.
- Acciones de revisión persistentes para entidad, relación y opinión; correcciones guardadas como anotaciones laterales.
- Manejo de desconexión del reproductor durante solicitudes HTTP parciales y etiquetas transparentes sobre la naturaleza heurística del PLN.

## Referencias del proyecto

- [Guía de ejecución local](COMO_EJECUTAR_DEMO.md)
- [Prompt de replanteamiento del producto y PLN](PROMPT_REPLANTEO_PRODUCTO_Y_PLN.md)
- [README](../README.md)
