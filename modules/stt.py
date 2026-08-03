from typing import Optional

import numpy as np
import sounddevice as sd
import speech_recognition as sr

from .audio_utils import create_input_stream, resample_audio, select_input_device

SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "int16"
CHUNK_DURATION = 0.1

DEFAULT_LANGUAGE = "es-ES"
DEFAULT_MAX_DURATION = 8.0
DEFAULT_SILENCE_DURATION = 0.8
DEFAULT_CALIBRATION_DURATION = 0.5
DEFAULT_ENERGY_MULTIPLIER = 2.5
DEFAULT_MIN_ENERGY = 200


class SpeechToText:
    def __init__(
        self,
        language: str = DEFAULT_LANGUAGE,
        max_duration: float = DEFAULT_MAX_DURATION,
        silence_duration: float = DEFAULT_SILENCE_DURATION,
        energy_multiplier: float = DEFAULT_ENERGY_MULTIPLIER,
        min_energy: int = DEFAULT_MIN_ENERGY,
        input_device: Optional[str] = None,
    ):
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

    def _measure_energy(self, audio_chunk: np.ndarray) -> float:
        return float(np.abs(audio_chunk.astype(np.float32)).mean())

    def _record_until_silence(self) -> bytes:
        max_chunks = int(self.max_duration / CHUNK_DURATION)
        silence_chunks_needed = int(self.silence_duration / CHUNK_DURATION)
        cal_chunks = max(3, int(DEFAULT_CALIBRATION_DURATION / CHUNK_DURATION))

        recorded: list[np.ndarray] = []
        threshold = self._last_energy_threshold
        speech_started = False
        silence_count = 0

        block_samples = int(self._stream_sr * CHUNK_DURATION)
        sd.sleep(400)

        with create_input_stream(
            device=self._device_index,
            samplerate=self._stream_sr,
            channels=CHANNELS,
            dtype=DTYPE,
            blocksize=block_samples,
        ) as stream:
            print("[stt] Calibrando ruido ambiente...")
            cal_energies: list[float] = []
            for _ in range(cal_chunks):
                data, _ = stream.read(block_samples)
                chunk_flat = data.flatten()
                if self._need_resample:
                    chunk_flat = resample_audio(
                        chunk_flat, self._stream_sr, SAMPLE_RATE
                    )
                cal_energies.append(self._measure_energy(chunk_flat))
            noise_floor = float(np.mean(cal_energies))
            threshold = max(noise_floor * self.energy_multiplier, self.min_energy)
            self._last_energy_threshold = threshold
            print(
                f"[stt] Ruido ambiente: {noise_floor:.0f}, "
                f"umbral voz: {threshold:.0f}"
            )

            print("[stt] Escuchando tu comando...")
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
            return b""

        audio_array = np.concatenate(recorded, axis=0)
        return audio_array.tobytes()

    def listen_and_transcribe(self) -> Optional[str]:
        audio_bytes = self._record_until_silence()
        if not audio_bytes:
            print("[stt] No se detectó voz.")
            return None

        audio_data = sr.AudioData(audio_bytes, SAMPLE_RATE, 2)
        try:
            print(f"[stt] Transcribiendo audio ({self.language})...")
            text = self.recognizer.recognize_google(
                audio_data, language=self.language
            )
            normalized = text.lower().strip()
            print(f"[stt] Texto reconocido: '{normalized}'")
            return normalized
        except sr.UnknownValueError:
            print("[stt] No se pudo entender el audio.")
            return None
        except sr.RequestError as exc:
            print(
                f"[stt] Error de conexión con el servicio de "
                f"transcripción: {exc}"
            )
            return None
