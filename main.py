import sys

from modules import Brain, SpeechToText, TextToSpeech, WakeWordListener


def main():
    print("==================================================")
    print("  MAVI-Robot · Asistente de voz local")
    print("  Wake word: openwakeword | STT: Google (es-ES)")
    print("  TTS: espeak-ng + sounddevice (offline, PipeWire-friendly)")
    print("==================================================")
    print("Di la palabra de activación y luego tu comando.")
    print("Para salir del programa, di 'adiós' o pulsa Ctrl+C.")
    print()

    wake = WakeWordListener(
        model_name="alexa",
        custom_model_path=None,
        threshold=0.50,
    )
    stt = SpeechToText(language="es-ES")
    brain = Brain()
    tts = TextToSpeech(voice="es", rate=165, volume=1.0, output_device=None)

    tts.say("Asistente listo. Di la palabra de activación para comenzar.")

    try:
        wake.start()
        while True:
            activated = wake.wait_for_activation()
            if not activated:
                break

            wake.pause()
            tts.say("Te escucho.")
            user_text = stt.listen_and_transcribe()
            if not user_text:
                tts.say("No te he oído. Vuelvo a escuchar la palabra de activación.")
                wake.resume()
                continue


            print(f"[main] Usuario dijo: '{user_text}'")

            if brain.is_exit_command(user_text):
                response = "Hasta luego. Apagando asistente."
                tts.say(response)
                break

            response = brain.handle(user_text)
            print(f"[main] Respuesta: '{response}'")
            tts.say(response)
            wake.resume()

    except KeyboardInterrupt:
        print("\n[main] Interrumpido por el usuario.")
    finally:
        wake.stop()
        tts.stop()
        print("[main] Asistente detenido.")


if __name__ == "__main__":
    sys.exit(main())
