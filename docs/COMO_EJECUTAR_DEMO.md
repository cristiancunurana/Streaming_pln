# Cómo ejecutar la demo web

Esta guía explica cómo iniciar la aplicación local y procesar un video público o una transmisión guardada que tengas autorización para usar.

## 1. Requisitos del sistema en Fedora

La aplicación usa FFmpeg para comprimir el audio y Node.js para que yt-dlp pueda procesar enlaces de YouTube. La web se sirve con Python y no necesita Streamlit.

Si ya tienes RPM Fusion habilitado, instala los paquetes:

```bash
sudo dnf install ffmpeg nodejs
```

Si Fedora no encuentra `ffmpeg`, habilita RPM Fusion Free e instala la versión completa:

```bash
sudo dnf install "https://mirrors.rpmfusion.org/free/fedora/rpmfusion-free-release-43.noarch.rpm"
sudo dnf install ffmpeg nodejs --allowerasing
```

Confirma que el sistema los encuentra:

```bash
ffmpeg -version
node --version
```

## 2. Instalar dependencias Python

Abre Konsole. Pega estas líneas una por una, sin `\` al final:

```bash
source "/home/ccunurana/Documentos/Aprendizaje Profundo/.venv/bin/activate"
cd "/home/ccunurana/Documentos/Procesamiento de Lenguaje Natural/Proyecto_streaming_pln"
python -m pip install -r requirements.txt
```

La primera línea activa el entorno virtual que ya usas para Aprendizaje Profundo. La segunda entra a la carpeta del proyecto. La tercera instala Whisper, yt-dlp, PyAV compatible y Sentence-Transformers.

Para habilitar el modelo NER español, instala una vez sus datos dentro del mismo entorno virtual:

```bash
python -m spacy download es_core_news_sm
```

Si todavía no instalas el modelo, Administración funciona con una heurística local y vocabulario, identificados como tales en el mapa. Revisa sus candidatos antes de usarlos en resultados del proyecto.

Al pulsar **Procesar datos y construir memoria**, se genera el embedding multilingüe de los fragmentos. La primera ejecución descarga los pesos de Sentence-Transformers; luego la búsqueda combina esos vectores con BM25. Si la descarga o la instalación falla, se guarda el motivo y la consulta sigue funcionando con BM25.

`%pip` solo se usa dentro de una celda Jupyter. En Konsole se escribe `python -m pip`.

## 3. Encender el servidor web

Con el entorno activo y ubicado en la carpeta del proyecto, ejecuta:

```bash
python src/app/server.py
```

Mantén esa terminal abierta mientras uses la página. Abre esta dirección en el navegador:

```text
http://localhost:8501
```

Para apagar el servidor, vuelve a Konsole y presiona `Ctrl+C`. La aplicación abre el puerto local 8501.

## 4. Procesar una transmisión

La página separa **Explorar** y **Administración**. En Explorar, el usuario elige una transmisión, consulta sus metadatos, escucha el MP3 y hace preguntas. En Administración, pega un video público de YouTube o una transmisión guardada compatible de Twitch/Kick. Por defecto procesa 15 minutos desde el comienzo; puedes cambiar inicio, duración y modelo. No captura directos que siguen en vivo.

El flujo que muestra la página es:

1. Lee el enlace y obtiene título, canal, identificador, fecha y duración cuando la plataforma los expone.
2. Descarga el tramo elegido del VOD, no las tres horas completas cuando la plataforma permite seleccionar rangos.
3. Convierte ese audio a MP3 mono de 16 kHz y 48 kb/s. Quince minutos suelen ocupar unos 5 MB.
4. Transcribe el MP3 en español con Whisper y mantiene los timestamps referidos al video original.

En **Pregunta al archivo**, escribe una consulta sobre la transmisión seleccionada o sobre toda la biblioteca. El buscador combina BM25 con similitud semántica cuando la memoria tiene embeddings, lematiza formas verbales y mantiene en el índice expresiones como “hoy” y “ayer”. Así, **“¿Qué dijo ayer?”** busca referencias pronunciadas en el audio; **“¿Qué dijo en la transmisión de ayer?”** filtra por fecha de emisión. La respuesta extractiva cita fragmentos con título, canal, fecha y tiempo. Si quieres generación, instala y ejecuta Ollama con un modelo local y configura `STREAM_MEMORY_LLM` con el nombre exacto del modelo antes de encender el servidor. Activa **Interpretar con LLM local** en la consulta. Si el modelo local no está disponible, se usa la respuesta extractiva y se muestra el método aplicado.

En Administración, **Probar modelo** vuelve a transcribir el MP3 local con el tamaño de Whisper elegido y guarda una salida de comparación en `data/processed/model_runs/`, sin sustituir la transcripción activa. Al revisar la muestra, **Usar como transcripción activa** guarda primero el texto actual en `data/processed/backups/transcripts/`, instala la comparación elegida e invalida la memoria para reconstruirla desde el texto nuevo. Las nuevas transcripciones usan Whisper `medium` por defecto. **Borrar transcripción** conserva el MP3 y los metadatos; **Eliminar registro** borra los tres archivos. **Volver a obtener** usa el enlace de la fuente y descarga otra vez el audio.

En la parte inferior de Administración, elige una transmisión con transcripción y pulsa **Procesar datos y construir memoria** (o **Procesar datos** en su tarjeta). Esta operación no vuelve a descargar ni transcribir: guarda en `data/processed/memory/<stream_id>.json` los fragmentos con tiempos, términos, embeddings, entidades, relaciones, polaridad y referencias temporales. Las relaciones y opiniones son candidatas basadas en reglas y requieren revisión.

El apartado **Memoria por creador** agrupa varios VODs si comparten el ID de creador que entrega la plataforma. Ese ID identifica una cuenta/canal; el sistema no reconoce a la persona por su voz ni deduce que dos canales son del mismo streamer. Para unir al mismo creador entre plataformas o canales, escribe el mismo alias en el campo de agrupación al cargar esos VODs. Las relaciones, opiniones y menciones conservan el identificador del VOD, la fecha y el timestamp para abrir el fragmento correcto. Si borras una transcripción, su memoria derivada también se elimina.

Para evaluar el recuperador, anota preguntas y rangos correctos siguiendo [data/evaluation/README.md](../data/evaluation/README.md). Ejecuta `python -m src.nlp.evaluate_retrieval --k 5` desde la raíz para calcular Precision@5 y Recall@5. Los rangos de referencia deben revisarse escuchando el audio.

La descarga depende de la plataforma y de la disponibilidad del video. La primera transcripción con Whisper puede tardar porque se descarga el modelo seleccionado; `medium` es el valor recomendado para el corpus final. La fecha recuperada puede corresponder a la publicación del video; verifica que sea la fecha de emisión. Si falta o es incorrecta, complétala en Administración antes de procesar.

## 5. Archivos generados

Los resultados se guardan en `data/raw/`:

- `audio/<id-del-vod>.mp3`: tramo comprimido usado para transcribir.
- `<id-del-vod>.metadata.json`: título, canal, fecha, enlace y otros metadatos encontrados.
- `<id-del-vod>.jsonl`: segmentos de transcripción con texto, inicio y fin en segundos.

La transcripción es automática; revisa nombres propios y una muestra de los timestamps antes de usarla para evaluar el sistema.

## Problemas comunes

- **`%pip: no existe ese trabajo`:** estás en Konsole. Usa `python -m pip install -r requirements.txt`.
- **`ffmpeg no encontrado`:** completa la instalación de FFmpeg del paso 1 y reinicia la terminal.
- **No se pudo leer o descargar el enlace:** comprueba que sea un VOD terminado y accesible. La herramienta no entra a contenido privado ni usa cookies.
- **No se detectó la fecha:** en **Administración**, escribe la fecha de emisión antes de procesar.
- **La página no abre:** confirma que el servidor siga corriendo en Konsole y abre `http://localhost:8501`.
