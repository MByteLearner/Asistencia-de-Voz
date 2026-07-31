# MAVI-Robot · Asistente de voz local

Asistente tipo "Alexa" 100% en local con bucle continuo. Cuatro componentes desacoplados.

## Arquitectura

```
MAVI-Robot/
├── main.py                  # Orquestador: bucle principal
├── install.sh               # Script de instalación (requiere sudo)
├── run.sh                   # Lanzador con venv y LD_LIBRARY_PATH
├── requirements.txt         # openwakeword 0.4.x (ver nota Python 3.13)
├── modules/
│   ├── __init__.py
│   ├── wake_word.py         # openwakeword (modelo pre-entrenado "alexa")
│   ├── stt.py               # sounddevice + SpeechRecognition (es-ES)
│   ├── brain.py             # Handler de comandos (if/else)
│   └── tts.py               # espeak-ng --stdout + sounddevice
└── README.md
```

## Componentes

| Módulo           | Librería                | Función                                          |
| ---------------- | ----------------------- | ------------------------------------------------ |
| Wake word        | `openwakeword`          | Escucha pasiva hasta detectar la palabra clave  |
| STT              | `sounddevice` + `SpeechRecognition` | Captura audio + transcripción en español (Google Web Speech, sin API key) |
| Cerebro / Lógica | (stdlib `datetime`, `random`) | Despacha el texto a un comando             |
| TTS              | `espeak-ng` + `sounddevice` | Sintetiza y reproduce la respuesta en voz local |

## Dependencias

- **Python** 3.8+ (probado con 3.13.5)
- **Sistema**: `portaudio19-dev`, `espeak-ng`, `python3-pip` (instaladas por `install.sh` con sudo)
- **Python (pip)**: `openwakeword`, `SpeechRecognition`, `sounddevice`, `numpy`, `soundfile` (en `requirements.txt`)

### ⚠️ Nota sobre Python 3.13 y `openwakeword`

`openwakeword` **0.5.x y 0.6.x** declaran `tflite-runtime` como dependencia obligatoria en Linux, pero `tflite-runtime` no publica wheels para Python 3.13 (su último release soporta hasta 3.12). Por eso `requirements.txt` fija `openwakeword>=0.4.0,<0.5.0`, que es la última rama sin esa dependencia. La API usada por el código (`Model(wakeword_model_paths=...)`, `predict(frame)`) es compatible.

Si necesitas 0.6+ por alguna razón, las opciones son:

- Bajar a Python 3.12 (`conda create -n mavi python=3.12 && conda activate mavi && pip install -r requirements.txt`).
- Compilar `tflite-runtime` desde fuente para 3.13 (no recomendado).
- Esperar a que `tflite-runtime` publique wheels para 3.13.

## Instalación

```bash
chmod +x install.sh run.sh
./install.sh
```

`install.sh` instala: `python3-pip`, `portaudio19-dev`, `espeak-ng` y crea un `venv` con las dependencias de `requirements.txt`.

> Si ya tienes `portaudio` y `espeak-ng` en tu sistema puedes saltarte los `apt-get` y solo crear el venv con `pip install -r requirements.txt`.

## Ejecución

```bash
./run.sh
```

El asistente:

1. Carga el modelo de wake word `alexa` (en inglés, el único que viene pre-entrenado en `openwakeword`).
2. Reproduce "Asistente listo. Di la palabra de activación para comenzar.".
3. Entra en bucle: escucha pasiva → wake word → STT → cerebro → TTS → repite.
4. Sale cuando dices "adiós", "salir", "apágate" o pulsas `Ctrl+C`.

## Comandos disponibles (español)

- `qué hora es` / `qué hora` → dice la hora actual.
- `chiste` / `cuenta un chiste` / `dime un chiste` → cuenta un chiste aleatorio.
- `ayuda` / `qué sabes hacer` → lista los comandos.
- `adiós` / `salir` / `apágate` → cierra el asistente.

## Wake word en español (importante)

`openwakeword` **solo trae modelos pre-entrenados en inglés** (`alexa`, `hey_jarvis`, `hey_mycroft`, `weather`, `timer`). No existe un modelo en español listo para usar. Por eso el sistema arranca con `alexa` por defecto (umbral 0.65 para minimizar falsos positivos).

Tienes dos caminos para tener una wake word en español:

### Opción A · Entrenar un modelo personalizado con `openwakeword`

1. Genera ~5 000 muestras sintéticas de tu wake word en español usando un TTS local (por ejemplo, `piper` con voz `es_ES-davefx-medium`).
2. Recolecta ~30 h de audio negativo (sin la wake word): habla, música, ruido.
3. Ejecuta el notebook de entrenamiento oficial: https://github.com/dscripka/openWakeWord/blob/main/notebooks/automatic_model_training.ipynb
4. Tarda ~1 h en una GPU; con CPU pueden ser varias horas.
5. El notebook genera un archivo `.onnx` que apuntas con `custom_model_path` en `main.py`.

Para que tu modelo sea detectado cuando lo tengas entrenado, edita `main.py`:

```python
wake = WakeWordListener(
    model_name="alexa",
    custom_model_path="ruta/a/mi_modelo_es.onnx",
    threshold=0.65,
)
```

### Opción B · Cambiar el wake word a otro en inglés pre-entrenado

`main.py` acepta cualquiera de los 5 modelos pre-entrenados:

```python
wake = WakeWordListener(model_name="hey_jarvis", threshold=0.65)
```

Opciones: `alexa`, `hey_jarvis`, `hey_mycroft`, `weather` (dice "what's the weather"), `timer` ("set a 10 minute timer").

## Notas técnicas

### ¿Por qué `speechdevice` y no `PyAudio`?

`PyAudio` (la dependencia que `SpeechRecognition` usa por defecto) **no tiene wheels para Python 3.13**, igual que `tflite-runtime`. Compilarlo desde fuente requiere `portaudio19-dev` y herramientas de compilación. Para evitar ese problema, `modules/stt.py` captura audio con `sounddevice` (que ya viene con un wrapper de `ctypes` para `libportaudio` y funciona en 3.13) y luego pasa los bytes crudos a `speech_recognition.AudioData`. La transcripción sigue siendo `recognize_google`, así que no pierdes calidad de STT.

### ¿Por qué `espeak-ng` directo y no `pyttsx3`?

`pyttsx3` usa `espeak` como motor, pero en sistemas con **PipeWire** (el servidor de audio por defecto en Debian 13, Ubuntu 22.04+, Fedora, etc.) `espeak` intenta abrir el dispositivo ALSA directamente y falla con `Device or resource busy` porque PipeWire ya lo está usando. Para evitar instalar `apulse`/`padsp` como wrapper, `modules/tts.py` invoca `espeak-ng --stdout`, captura el WAV resultante y lo reproduce con `sounddevice`, que sí negocia correctamente con PipeWire.

El resultado es el mismo (voz offline en español) pero sin la dependencia de `pyttsx3` y con un pipe de audio limpio.

### Selección del dispositivo de salida de audio

Por defecto, el TTS envía el audio al **sink por defecto de PipeWire/PulseAudio** del usuario. Puedes forzarlo de tres formas:

1. **Variable de entorno** (soportada):
   ```bash
   export MAVI_TTS_DEVICE=2   # por índice
   export MAVI_TTS_DEVICE="ALC236"   # por nombre (subcadena)
   ```

2. **Constructor** de `TextToSpeech` en `main.py`:
   ```python
   tts = TextToSpeech(output_device="2")
   ```

3. **WAVs de respaldo**: cada frase que dice el asistente se guarda en `/tmp/mavi-tts/<uuid>.wav` (WAV 16-bit mono 22050 Hz). Si no oyes nada, reproduce los archivos manualmente con `paplay` o `pw-play` para descartar problemas del sistema de audio:
   ```bash
   paplay /tmp/mavi-tts/*.wav
   ```

Si no oyes nada al ejecutar el asistente, lo más probable es que tu **sink por defecto esté en un volumen muy bajo** o que estés escuchando por un dispositivo distinto al configurado en PipeWire. Compruébalo con:

```bash
pactl get-default-sink          # nombre del sink activo
pactl get-sink-volume @DEFAULT_SINK@   # volumen actual
pactl list sinks short          # todos los sinks disponibles
```

Para cambiar el sink por defecto (ej. al altavoz integrado en vez de auriculares):

```bash
pactl set-default-sink alsa_output.pci-0000_03_00.6.HiFi__Speaker__sink
```

### STT con `recognize_google`

`SpeechRecognition` por defecto usa la `Google Web Speech API` sin API key. **Requiere internet** solo para transcribir (no para activar). El wake word y el TTS sí son 100% offline.

Si quieres STT 100% offline en español, instala `vosk` y un modelo en español:

```bash
pip install vosk
# descarga el modelo desde https://alphacephei.com/vosk/models (vosk-model-small-es-0.42)
```

y reemplaza el `recognize_google` en `modules/stt.py` por `recognize_vosk`. La interfaz es prácticamente idéntica.

### Selección de dispositivo de audio

Por defecto se usa el dispositivo de entrada por defecto del sistema. Para forzar uno concreto, exporta:

```bash
export AUDIOWORKER_DEFAULT_INPUT=2   # índice del dispositivo
```

(o edita `modules/wake_word.py` / `modules/stt.py` pasando el parámetro `device` correspondiente).

## Solución de problemas

| Síntoma                                                           | Causa / solución                                                            |
| ----------------------------------------------------------------- | --------------------------------------------------------------------------- |
| `OSError: PortAudio library not found`                            | Instala `portaudio19-dev` o ejecuta `run.sh` que ajusta `LD_LIBRARY_PATH`.  |
| `tflite-runtime` no se puede instalar                             | Fija `openwakeword<0.5.0` o usa Python 3.12 (ver sección "Nota Python 3.13"). |
| El wake word no se activa nunca                                   | Sube el volumen del micro, baja el ruido ambiente, prueba `threshold=0.5`.  |
| Falsas activaciones continuas                                     | Sube `threshold` a `0.7`–`0.8` o activa `vad_threshold=0.5` en el modelo.   |
| `SpeechRecognition` lanza `RequestError`                          | Sin conexión a internet. Migra a Vosk (ver sección anterior).              |
| `onnxruntime` avisa "CUDAExecutionProvider not available"         | Aviso inocuo. El modelo corre en CPU.                                       |

## Licencia

Código: Apache-2.0 (mismo que `openwakeword`).
Modelos pre-entrenados de `openwakeword`: CC BY-NC-SA 4.0 (no comercial).
