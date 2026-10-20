# Memoria semántica de transmisiones

Prototipo para consultar en lenguaje natural el contenido hablado de transmisiones en español y recuperar fragmentos con su fuente y marca de tiempo.

## Alcance inicial

El prototipo procesa varias transcripciones con marcas de tiempo y metadatos. Las transcripciones nuevas usan Whisper `medium` por defecto. Construye una memoria derivada con normalización, stopwords, fragmentos trazables, entidades candidatas, relaciones explícitas, polaridad, referencias temporales y embeddings multilingües. La consulta combina BM25 con similitud densa cuando el modelo de embeddings está instalado; conserva BM25 como respaldo. Las respuestas se construyen desde fragmentos recuperados y muestran citas reproducibles en el audio. Un LLM local vía Ollama es opcional y queda acotado a las evidencias recuperadas.

La búsqueda puede recorrer toda la biblioteca, pero el mapa de entidades agrupa transmisiones por ID estable de creador. YouTube y otras plataformas suelen aportar ese ID automáticamente. Al agregar un VOD sin ID detectable, asigna el mismo `ID del creador` a todos los videos de una persona y uno distinto a cada persona; sin ese dato, el sistema separa por canal. Cada relación conserva su VOD, fecha y timestamp como evidencia.

La salida siempre incluye fragmentos y citas verificables. Sin LLM local, el sistema usa una respuesta extractiva conservadora. Con Ollama configurado puede sintetizar los fragmentos, con instrucciones de citar la evidencia. La memoria y la respuesta no sustituyen la revisión de transcripciones automáticas.

## Primer objetivo

Construir un corpus pequeño y limpio que permita demostrar el flujo completo. Empieza con 2 o 3 videos o transmisiones guardadas en español, de un tema acotado. Administración extrae los metadatos, descarga un tramo de audio y lo transcribe; revisa la procedencia y corrige manualmente una muestra.

Antes de ampliar el corpus, prepara unas 10 preguntas y marca manualmente qué fragmentos responden cada pregunta. Esto permitirá medir si la búsqueda encuentra la evidencia.

## Flujo automático desde un enlace

`src/ingest/process_vod.py` recibe el enlace de un video público (por ejemplo, cualquier video accesible de YouTube) o una transmisión ya guardada de YouTube, Twitch o Kick. Extrae los metadatos disponibles, descarga el tramo elegido y lo convierte a MP3 mono de 16 kHz y 48 kb/s. Después lo transcribe en español y guarda un segmento JSON por línea. Un tramo de 15 minutos ocupa aproximadamente 5 MB; la descarga puede variar según la plataforma y su formato. No captura directos que siguen en vivo ni usa cookies para acceder a contenido privado.

Instala las dependencias de Python en el entorno virtual que usarás para correr la página. Desde la terminal, después de activar ese entorno:

```bash
python -m pip install -r requirements.txt
```

Si ejecutas el proceso desde una celda de Jupyter, usa `%pip install -r requirements.txt` en esa celda.

Para convertir a MP3 y resolver los retos de descarga de YouTube se necesitan FFmpeg y Node.js instalados en Fedora. Los pasos están en [la guía de ejecución](docs/COMO_EJECUTAR_DEMO.md). `yt-dlp[default]` instala también sus dependencias opcionales de YouTube.

El MP3 reducido y la transcripción del tramo se guardan en `data/raw/`. En una celda de Jupyter, procesa 15 minutos desde el inicio:

```python
%run src/ingest/process_vod.py "https://enlace-del-vod" --start-sec 0 --duration-sec 900
```

Si la página no entrega la fecha o el ID del creador, puedes pasarlos con `--date 2026-10-13` y `--streamer-id creador_01`. Confirma que la fecha detectada corresponde al día de emisión. El MP3, los metadatos y la transcripción quedan en `data/raw/`. El modelo Whisper se descarga al primer uso. Revisa nombres propios y timestamps.

## Demo web

La guía detallada de Fedora, entorno virtual, puerto y archivos generados está en [docs/COMO_EJECUTAR_DEMO.md](docs/COMO_EJECUTAR_DEMO.md).

En la terminal, ejecuta cada línea por separado (sin `\` al final):

```bash
source "/home/ccunurana/Documentos/Aprendizaje Profundo/.venv/bin/activate"
cd "/home/ccunurana/Documentos/Procesamiento de Lenguaje Natural/Proyecto_streaming_pln"
python -m pip install -r requirements.txt
python src/app/server.py
```

Abre `http://localhost:8501` en el navegador. La interfaz permite elegir y escuchar transmisiones, buscar fragmentos con BM25, ver el resumen del corpus y procesar enlaces nuevos. Para apagar el servidor, vuelve a la terminal y presiona `Ctrl+C`. Para YouTube, yt-dlp usa Node.js para resolver los retos de la página. [Requisitos de yt-dlp](https://github.com/yt-dlp/yt-dlp#dependencies)

## Modo manual: transcribir un archivo local

La herramienta `src/ingest/transcribe.py` recibe un archivo local de audio/video y crea un JSONL con un segmento por línea, inicio y fin en segundos y los metadatos del VOD. Úsala solo con material propio o cuyo uso tengas autorizado.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/ingest/transcribe.py "ruta/al/vod.mp4" \
  --streamer-id creador_01 \
  --stream-id valorant_01 \
  --title "Partidas y comentarios" \
  --date 2026-10-13 \
  --source-url "https://plataforma.example/video"
```

Por defecto, el resultado queda en `data/raw/<stream-id>.jsonl`. Whisper `medium` se descarga la primera vez y puede tardar. `small` sirve para pruebas rápidas, pero no es la salida final que buscamos en este corpus. La transcripción automática es un borrador: revisa nombres propios y una muestra de marcas de tiempo antes de analizarla.

Para probar solo 15 minutos de un VOD largo, agrega `--start-sec 1800 --duration-sec 900` (empieza en el minuto 30 y procesa 15 minutos). Los timestamps de salida siguen apuntando al minuto correspondiente del VOD original. Si omites ambos parámetros, se procesa el VOD completo.

## Flujo de PLN implementado

1. Ingesta de transcripción JSONL y metadatos JSON.
2. Normalización ligera para búsqueda (minúsculas y tildes), preservando el texto original.
3. Filtrado de stopwords solo en la representación de búsqueda; negaciones, expresiones temporales y acciones se conservan para no perder intención. El conteo temático usa una lista distinta.
4. Segmentación de búsqueda en ventanas solapadas de hasta cuatro segmentos Whisper, 30 segundos o 360 caracteres, con tiempos del VOD. Los segmentos ASR y los fragmentos PLN son unidades diferentes.
5. Extracción de entidades con NER en español y patrones revisables para relaciones, opiniones y expresiones temporales.
6. Embeddings densos con `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`; si el paquete o el modelo no está disponible, la memoria registra el motivo y el buscador conserva BM25.
7. Recuperación híbrida Top-K: BM25 + similitud de coseno normalizada cuando hay vectores. Distingue fechas de emisión de expresiones temporales mencionadas en el habla; admite filtros de fecha y minuto.
8. Síntesis extractiva con citas por defecto; generación RAG opcional con un LLM local vía Ollama y el contexto limitado a los resultados.
9. **Procesar datos** guarda la memoria, vectores y trazabilidad en `data/processed/memory/`. No modifica el JSONL original.
10. La interfaz muestra las etapas, entidades, opiniones, expresiones temporales, relaciones y evidencia en su timestamp.
11. `data/evaluation/` define las preguntas y rangos gold; `python -m src.nlp.evaluate_retrieval --k 5` calcula Precision@5 y Recall@5.

Para activar el modelo NER español de spaCy en tu entorno virtual, instala los requisitos y luego ejecuta `python -m spacy download es_core_news_sm`. El procesamiento lo detecta automáticamente. Si todavía no está instalado, usa una heurística local y vocabulario conocido; las entidades candidatas y relaciones se deben revisar. Las relaciones se extraen con patrones de texto en esta primera versión, no con un modelo entrenado. No entrenamos una IA desde cero.

El preprocesamiento conserva la transcripción original. La búsqueda normaliza mayúsculas y tildes y filtra conectores frecuentes; mantiene negaciones, acciones y expresiones temporales. El conteo del vocabulario temático usa otro filtro y excluye negación/tiempo solo para contar, nunca para buscar o citar.

La aplicación tiene vistas separadas para **Biblioteca**, **Consultar**, **Proceso PLN**, **Memoria**, **Evaluación** y **Administración**. En Administración se agregan enlaces y se gestionan comparaciones de Whisper. Crear una comparación reutiliza el MP3 guardado; activar una salida revisada guarda primero la transcripción vigente bajo `data/processed/backups/transcripts/` e invalida la memoria para reconstruirla. Los experimentos quedan bajo `data/processed/model_runs/`. Una comparación sin transcripción humana de referencia solo permite inspeccionar salidas; no mide por sí misma precisión de Whisper.

Whisper es el modelo de reconocimiento de voz: convierte audio en texto. `tiny`, `base`, `small`, `medium` y `large-v3` ofrecen distintos compromisos entre velocidad y calidad. BM25 es el buscador textual de esta línea base. Para evaluar Whisper, compara sus salidas del mismo MP3 con una transcripción corregida manualmente; para medir la búsqueda, prepara preguntas y fragmentos relevantes, y calcula Precision@K o Recall@K. La limpieza de stopwords ayuda a visualizar y buscar términos; no corrige errores de Whisper.

## Estructura

```text
data/
  raw/          # transcripciones originales y metadatos
  processed/    # memorias PLN y pruebas de transcripción por modelo
  README.md     # registro de fuentes y reglas para el corpus
src/
  app/          # interfaz HTML y servidor local
  ingest/       # descarga, metadatos y transcripción
notebooks/      # exploración puntual del corpus
docs/           # decisiones y evaluación del proyecto
```

## Orden de trabajo

- [x] Crear un corpus de tres transmisiones con metadatos y tiempos; los VODs actuales pertenecen a tres creadores distintos y se mantienen separados en la memoria.
- [ ] Decidir si el estudio compara varios creadores o si se acota a un dominio temático común; justificar el criterio de muestreo en el informe.
- [x] Validar el formato de entrada y construir fragmentos con trazabilidad.
- [x] Probar NER en español sobre esos fragmentos (spaCy opcional con fallback de candidatos).
- [x] Construir memoria PLN por transmisión y revisar relaciones en un grafo con evidencia temporal.
- [x] Comparar BM25 con embeddings multilingües y búsqueda híbrida.
- [x] Crear el esquema de preguntas de evaluación y el cálculo de Precision@K y Recall@K; falta anotar las preguntas con el corpus real.
- [x] Crear una interfaz HTML con reproductor, dashboard y citas con timestamps.
- [x] Añadir vistas de biblioteca, consulta, proceso PLN, memoria, evaluación y administración, además de anotaciones humanas separadas para candidatos.
- [x] Añadir soporte opcional de respuestas generativas locales acotadas a fragmentos citados; requiere Ollama configurado para activarse.

## Criterio de éxito del prototipo

Para cada pregunta de evaluación, el sistema debe devolver entre los primeros resultados al menos un fragmento marcado manualmente como relevante y mostrar correctamente la transmisión y el timestamp de ese fragmento. Toda respuesta futura debe basarse en esos fragmentos y mostrar su cita.

## Entorno

La línea base local combina BM25 con vectores de `paraphrase-multilingual-MiniLM-L12-v2` cuando están disponibles. El proyecto no entrena un LLM ni el NER con el corpus propio; relaciones, opiniones y algunas entidades son señales automáticas pendientes de revisión humana.
# Streaming_pln
