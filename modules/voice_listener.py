"""
modules/voice_listener.py
─────────────────────────
VoiceListener: escucha continua con activación por palabra clave (keyword spotting).

Flujo:
  1. Graba audio del micrófono hasta detectar silencio.
  2. Transcribe con SpeechToText (Google STT, requiere internet).
  3. Comprueba si el texto empieza con una palabra de activación.
  4. Si sí → devuelve el texto del comando (sin la wake word).
  5. Si no → descarta y vuelve al paso 1.

Palabras de activación por defecto: "jarvis", "mavi", "asistente".
Se pueden extender sin modificar la clase (pasar custom_wake_words al constructor).

Compatibilidad:
  - Linux desktop (x86_64)
  - Android / Termux (aarch64)  ← no depende de onnxruntime ni openwakeword
"""

from __future__ import annotations

import logging
from typing import FrozenSet, Optional

from .stt import SpeechToText

logger = logging.getLogger(__name__)

# ─── Palabras de activación predeterminadas ────────────────────────────────────

DEFAULT_WAKE_WORDS: FrozenSet[str] = frozenset({"jarvis", "mavi", "asistente"})


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _strip_wake_word(text: str, wake_words: FrozenSet[str]) -> Optional[str]:
    """Detecta si *text* empieza con alguna wake word y retorna el resto.

    Ejemplos:
        "jarvis qué hora es" → "qué hora es"
        "mavi abre spotify"  → "abre spotify"
        "buenos días"        → None  (no hay wake word)

    Args:
        text: Texto transcripto (ya normalizado a minúsculas y sin espacios extra).
        wake_words: Conjunto de palabras de activación (minúsculas).

    Returns:
        El comando sin la wake word, o None si no se detectó activación.
    """
    for word in wake_words:
        if text == word:
            # Se dijo solo la wake word, sin comando posterior
            return ""
        if text.startswith(word + " "):
            command = text[len(word) :].strip()
            return command
    return None


# ─── Clase principal ──────────────────────────────────────────────────────────


class VoiceListener:
    """Escucha continua con activación por palabra clave (keyword spotting).

    A diferencia del antiguo WakeWordListener (basado en openwakeword/onnxruntime),
    este módulo no requiere un modelo ONNX y es totalmente compatible con ARM64/Termux.

    La detección de activación se realiza en el dominio del texto:
      - El audio se transcribe con el STT existente.
      - Si el texto empieza con una wake word, se procesa el comando.
      - Si no, se descarta silenciosamente y se vuelve a escuchar.

    Attributes:
        wake_words: Conjunto de palabras de activación en minúsculas.
        stt: Instancia de SpeechToText usada para la transcripción.
    """

    def __init__(
        self,
        wake_words: Optional[FrozenSet[str]] = None,
        language: str = "es-ES",
        input_device: Optional[str] = None,
    ) -> None:
        """Inicializa el VoiceListener.

        Args:
            wake_words: Palabras de activación personalizadas. Si es None, se usan
                        las predeterminadas ("jarvis", "mavi", "asistente").
            language: Código de idioma para el STT (p. ej. "es-ES", "en-US").
            input_device: Índice o nombre del dispositivo de entrada. Si es None,
                          se selecciona automáticamente (ver audio_utils.select_input_device).
        """
        self.wake_words: FrozenSet[str] = wake_words or DEFAULT_WAKE_WORDS
        self.stt = SpeechToText(language=language, input_device=input_device)

        logger.info(
            "VoiceListener listo. Wake words: %s | Idioma: %s",
            sorted(self.wake_words),
            language,
        )

    # ── API pública ───────────────────────────────────────────────────────────

    def listen_for_command(self) -> Optional[str]:
        """Escucha una vez, transcribe y valida la palabra de activación.

        Graba audio hasta detectar silencio, lo transcribe y comprueba si el
        texto empieza con una wake word. Si hay activación, devuelve el comando
        (texto sin la wake word). Si no, devuelve None.

        Returns:
            El texto del comando (str, puede ser vacío si solo se dijo la wake word),
            o None si no se detectó activación o si el audio no pudo transcribirse.
        """
        logger.info("Escuchando...")

        text = self.stt.listen_and_transcribe()

        if not text:
            logger.debug("Silencio o audio no reconocido.")
            return None

        logger.info("Texto reconocido: '%s'", text)

        command = _strip_wake_word(text.lower().strip(), self.wake_words)

        if command is None:
            logger.debug("Texto ignorado (sin palabra de activación): '%s'", text)
            return None

        logger.info("Activación detectada. Comando: '%s'", command)
        return command

    def continuous_listen(self):
        """Generador de escucha continua.

        Permite iterar sobre los comandos detectados:

            for command in listener.continuous_listen():
                response = brain.handle(command)
                tts.say(response)

        Yields:
            str: Texto del comando cada vez que se detecta una wake word.
        """
        logger.info(
            "Iniciando escucha continua. Di una de estas palabras para activar: %s",
            ", ".join(sorted(self.wake_words)),
        )
        while True:
            command = self.listen_for_command()
            if command is not None:
                yield command
