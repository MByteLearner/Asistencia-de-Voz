"""
main.py
───────
Punto de entrada del asistente de voz MAVI-Robot.

Flujo principal:
  1. El VoiceListener graba audio continuamente.
  2. Transcribe con Google STT.
  3. Si el texto empieza con una palabra de activación (jarvis / mavi / asistente),
     extrae el comando y lo envía al Brain.
  4. El Brain procesa el comando y retorna una respuesta.
  5. El TTS vocaliza la respuesta.
  6. Vuelta al paso 1.

Compatibilidad:
  - Linux desktop (x86_64)
  - Android / Termux (aarch64)
"""

from __future__ import annotations

import logging
import sys

from modules import Brain, SpeechToText, TextToSpeech, VoiceListener

# ─── Configuración de logging ────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


# ─── Main ─────────────────────────────────────────────────────────────────────


def main() -> int:
    """Inicializa el asistente y entra en el bucle de escucha continua.

    Returns:
        Código de salida (0 = normal, 1 = error de inicialización).
    """
    print("=" * 50)
    print("  MAVI-Robot · Asistente de voz local")
    print("  Activación: jarvis / mavi / asistente")
    print("  STT: Google Speech Recognition (es-ES)")
    print("  TTS: termux-tts-speak | paplay | espeak-ng")
    print("=" * 50)
    print()

    try:
        listener = VoiceListener(language="es-ES")
        brain = Brain()
        tts = TextToSpeech(voice="es", rate=165, volume=1.0)
    except Exception as exc:
        logger.error("Error al inicializar el asistente: %s", exc)
        return 1

    logger.info(
        "Asistente listo. Di una de estas palabras para activarlo: %s",
        ", ".join(sorted(listener.wake_words)),
    )
    tts.say("Asistente listo. Di jarvis, mavi o asistente para comenzar.")

    try:
        # Escucha continua usando el generador de VoiceListener
        for command in listener.continuous_listen():

            # Comando de salida
            if brain.is_exit_command(command):
                logger.info("Comando de salida detectado.")
                tts.say("Hasta luego. Apagando asistente.")
                break

            # Comando sin contenido (solo se dijo la wake word)
            if not command:
                tts.say("Te escucho, ¿en qué puedo ayudarte?")
                continue

            # Procesar comando
            logger.info("Comando recibido: '%s'", command)
            response = brain.handle(command)
            logger.info("Respuesta: '%s'", response)
            tts.say(response)

    except KeyboardInterrupt:
        logger.info("Interrumpido por el usuario (Ctrl+C).")
    finally:
        tts.stop()
        logger.info("Asistente detenido.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
