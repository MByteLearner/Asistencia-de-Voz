"""
modules/stt.py
──────────────
Speech-to-Text: convierte audio del micrófono en texto.

Usa el motor de Google Speech Recognition (requiere conexión a internet).
Graba hasta detectar silencio o hasta el límite de tiempo máximo.

Compatibilidad:
  - Linux desktop (x86_64)
  - Android / Termux (aarch64) — requiere internet para Google STT
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import sounddevice as sd
import speech_recognition as sr

from .audio_utils import create_input_stream, resample_audio, select_input_device

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "int16"
CHUNK_DURATION = 0.1  # segundos por chunk de grabación

DEFAULT_LANGUAGE = "es-ES"
DEFAULT_MAX_DURATION = 8.0       # segundos máximos de escucha
DEFAULT_SILENCE_DURATION = 0.8   # segundos de silencio para cortar
DEFAULT_CALIBRATION_DURATION = 0.5
DEFAULT_ENERGY_MULTIPLIER = 2.5  # umbral = noise_floor × multiplier
DEFAULT_MIN_ENERGY = 200         # umbral mínimo absoluto


class SpeechToText:
    """Captura audio del micrófono y lo transcribe con Google Speech Recognition.

    Detecta el inicio y fin del habla automáticamente mediante un umbral
    de energía adaptativo calibrado en tiempo real contra el ruido ambiente.

    Attributes:
        language: Código de idioma para el STT (p. ej. "es-ES").
        max_duration: Límite de tiempo máximo de grabación (segundos).
        silence_duration: Silencio necesario para cortar la grabación (segundos).
        energy_multiplier: Factor sobre el ruido ambiente para el umbral de voz.
        min_energy: Energía mínima absoluta para detectar voz.
    """

    def __init__(
        self,
        language: str = DEFAULT_LANGUAGE,
        max_duration: float = DEFAULT_MAX_DURATION,
        silence_duration: float = DEFAULT_SILENCE_DURATION,
        energy_multiplier: float = DEFAULT_ENERGY_MULTIPLIER,
        min_energy: int = DEFAULT_MIN_ENERGY,
        input_device: Optional[str] = None,
    ) -> None:
        self.language = language
        self.max_duration = max_duration
        self.silence_duration = silence_duration
        self.energy_multiplier = energy_multiplier
        self.min_energy = min_energy
        self.recognizer = sr.Recognizer()
        self._last_energy_threshold = self.min_energy

        self._device_index, self._stream_sr, self._need_resample = (
            select_input_device(input_device, SAMPLE_RATE)
        )
        logger.debug(
            "SpeechToText listo | idioma=%s | dispositivo=%s | SR=%d Hz | resample=%s",
            language, self._device_index, self._stream_sr, self._need_resample,
        )

    # ── Energía de audio ─────────────────────────────────────────────────────

    def _measure_energy(self, audio_chunk: np.ndarray) -> float:
        """Calcula la energía media del chunk (valor absoluto promedio de las muestras)."""
        return float(np.abs(audio_chunk.astype(np.float32)).mean())

    # ── Grabación con detección de silencio ──────────────────────────────────

    def _record_until_silence(self) -> bytes:
        """Graba audio hasta detectar silencio o alcanzar max_duration.

        Calibra primero el ruido ambiente y luego escucha hasta que el usuario
        deje de hablar.

        Returns:
            Audio grabado como bytes raw (int16, mono, 16000 Hz).
            Retorna b"" si no se detectó voz.
        """
        max_chunks = int(self.max_duration / CHUNK_DURATION)
        silence_chunks_needed = int(self.silence_duration / CHUNK_DURATION)
        cal_chunks = max(3, int(DEFAULT_CALIBRATION_DURATION / CHUNK_DURATION))

        recorded: list[np.ndarray] = []
        threshold = self._last_energy_threshold
        speech_started = False
        silence_count = 0

        block_samples = int(self._stream_sr * CHUNK_DURATION)
        sd.sleep(400)  # pausa breve para estabilizar el stream

        with create_input_stream(
            device=self._device_index,
            samplerate=self._stream_sr,
            channels=CHANNELS,
            dtype=DTYPE,
            blocksize=block_samples,
        ) as stream:
            # Calibración: medir ruido ambiente
            logger.debug("Calibrando ruido ambiente...")
            cal_energies: list[float] = []
            for _ in range(cal_chunks):
                data, _ = stream.read(block_samples)
                chunk_flat = data.flatten()
                if self._need_resample:
                    chunk_flat = resample_audio(chunk_flat, self._stream_sr, SAMPLE_RATE)
                cal_energies.append(self._measure_energy(chunk_flat))

            noise_floor = float(np.mean(cal_energies))
            threshold = max(noise_floor * self.energy_multiplier, self.min_energy)
            self._last_energy_threshold = threshold
            logger.debug(
                "Ruido ambiente: %.0f | Umbral de voz: %.0f", noise_floor, threshold
            )

            # Grabación
            logger.info("Escuchando comando...")
            for _ in range(max_chunks):
                data, _ = stream.read(block_samples)
                chunk_flat = data.flatten()
                if self._need_resample:
                    chunk_flat = resample_audio(
                        chunk_flat, self._stream_sr, SAMPLE_RATE
                    )
                recorded.append(chunk_flat.reshape(-1, 1).copy())
                energy = self._measure_energy(chunk_flat)

                if energy > threshold:
                    speech_started = True
                    silence_count = 0
                elif speech_started:
                    silence_count += 1
                    if silence_count >= silence_chunks_needed:
                        break

        if not speech_started:
            logger.debug("No se detectó voz en la grabación.")
            return b""

        audio_array = np.concatenate(recorded, axis=0)
        return audio_array.tobytes()

    # ── API pública ───────────────────────────────────────────────────────────

    def listen_and_transcribe(self) -> Optional[str]:
        """Graba audio y transcribe con Google STT.

        Returns:
            Texto reconocido (minúsculas, sin espacios extras), o None si:
              - No se detectó voz.
              - El audio no pudo ser entendido.
              - Hubo un error de red con el servicio de transcripción.
        """
        audio_bytes = self._record_until_silence()
        if not audio_bytes:
            logger.debug("No hay audio que transcribir.")
            return None

        audio_data = sr.AudioData(audio_bytes, SAMPLE_RATE, 2)
        try:
            logger.debug("Transcribiendo con Google STT (%s)...", self.language)
            text = self.recognizer.recognize_google(
                audio_data, language=self.language
            )
            normalized = text.lower().strip()
            logger.info("Texto transcripto: '%s'", normalized)
            return normalized
        except sr.UnknownValueError:
            logger.debug("Google STT no pudo entender el audio.")
            return None
        except sr.RequestError as exc:
            logger.error("Error de red con Google STT: %s", exc)
            return None
