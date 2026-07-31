import collections
import os
import sys
from typing import List, Optional

import numpy as np
import openwakeword
import sounddevice as sd


SAMPLE_RATE = 16000
FRAME_DURATION_MS = 80
FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)
CHANNELS = 1
DTYPE = "int16"

DEFAULT_MODEL = "alexa"
DEFAULT_THRESHOLD = 0.65
COOLDOWN_FRAMES = 20
PREDICTION_BUFFER_SIZE = 5

PRETRAINED_MODEL_NAMES = {"alexa", "hey_mycroft", "hey_jarvis", "timer", "weather"}


class WakeWordListener:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        custom_model_path: Optional[str] = None,
        threshold: float = DEFAULT_THRESHOLD,
    ):
        self.threshold = threshold
        self._cooldown = 0
        self._predictions_buffer = collections.deque(
            maxlen=PREDICTION_BUFFER_SIZE
        )

        model_paths: List[str] = []
        if custom_model_path:
            if not os.path.isfile(custom_model_path):
                raise FileNotFoundError(
                    f"Modelo personalizado no encontrado: {custom_model_path}"
                )
            model_paths = [custom_model_path]
            self._active_model_label = os.path.splitext(
                os.path.basename(custom_model_path)
            )[0]
        else:
            if model_name not in PRETRAINED_MODEL_NAMES:
                raise ValueError(
                    f"Nombre de modelo no soportado: '{model_name}'. "
                    f"Usa uno de {sorted(PRETRAINED_MODEL_NAMES)} o pasa "
                    "custom_model_path."
                )
            all_paths = openwakeword.get_pretrained_model_paths()
            model_paths = [p for p in all_paths if model_name in os.path.basename(p)]
            if not model_paths:
                raise RuntimeError(
                    f"No se encontró el modelo '{model_name}' entre los "
                    "pre-entrenados."
                )
            self._active_model_label = os.path.splitext(
                os.path.basename(model_paths[0])
            )[0]

        self.model = openwakeword.Model(wakeword_model_paths=model_paths)

        self._stream: Optional[sd.InputStream] = None
        self._stop_requested = False
        self._paused = False

        print(
            f"[wake_word] Modelo cargado: '{self._active_model_label}' "
            f"(umbral={self.threshold})"
        )

    @property
    def active_model_name(self) -> str:
        return self._active_model_label

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            print(f"[wake_word] estado de audio: {status}", file=sys.stderr)
        audio_frame = np.frombuffer(indata, dtype=np.int16)
        prediction = self.model.predict(audio_frame)
        score = float(prediction.get(self._active_model_label, 0.0))
        self._predictions_buffer.append(score)

    def start(self):
        self._stop_requested = False
        self._paused = False
        self._cooldown = 0
        self._predictions_buffer.clear()
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            blocksize=FRAME_SAMPLES,
            channels=CHANNELS,
            dtype=DTYPE,
            callback=self._audio_callback,
        )
        self._stream.start()
        print(
            f"[wake_word] Escuchando palabra de activación "
            f"'{self._active_model_label}'..."
        )

    def stop(self):
        self._stop_requested = True
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as exc:
                print(
                    f"[wake_word] Error cerrando stream: {exc}", file=sys.stderr
                )
            self._stream = None
        print("[wake_word] Listener detenido.")

    def pause(self):
        if self._stream is not None:
            try:
                self._stream.stop()
            except Exception as exc:
                print(f"[wake_word] Error pausando: {exc}", file=sys.stderr)
        self._paused = True
        self._predictions_buffer.clear()

    def resume(self):
        if self._stream is not None and not self._stop_requested:
            try:
                self._stream.start()
            except Exception as exc:
                print(f"[wake_word] Error reanudando: {exc}", file=sys.stderr)
        self._paused = False
        self._predictions_buffer.clear()

    def detected(self) -> bool:
        if self._stream is None or self._stop_requested:
            return False
        if self._cooldown > 0:
            self._cooldown -= 1
            return False
        if len(self._predictions_buffer) < PREDICTION_BUFFER_SIZE:
            return False
        avg_score = sum(self._predictions_buffer) / len(self._predictions_buffer)
        if avg_score >= self.threshold:
            self._cooldown = COOLDOWN_FRAMES
            self._predictions_buffer.clear()
            print(f"[wake_word] ¡Activado! score={avg_score:.3f}")
            return True
        return False

    def wait_for_activation(self) -> bool:
        try:
            while not self._stop_requested:
                if self.detected():
                    return True
                sd.sleep(20)
        except KeyboardInterrupt:
            self._stop_requested = True
        return False
