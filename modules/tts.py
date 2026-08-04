"""
modules/tts.py
──────────────
Text-to-Speech: sintetiza y reproduce voz a partir de texto.

Backends (en orden de prioridad):
  1. termux-tts-speak  — motor TTS nativo de Android (solo en Termux)
  2. paplay            — PulseAudio nativo (solo en Linux desktop con PulseAudio)
  3. espeak-ng / espeak → sounddevice  — fallback universal

Compatibilidad:
  - Linux desktop (x86_64) con espeak-ng
  - Android / Termux (aarch64) con termux-tts-speak o espeak-ng
"""

from __future__ import annotations

import io
import logging
import os
import shutil
import subprocess
import sys
import wave
from typing import List, Optional

import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy.signal import resample

logger = logging.getLogger(__name__)

DEFAULT_VOICE = "es"
DEFAULT_RATE_WORDS_PER_MINUTE = 165
DEFAULT_VOLUME = 1.0
DEFAULT_OUTPUT_SAMPLE_RATE = 48000
DEBUG_WAV_DIR = "/tmp/mavi-tts"


# ─── Helpers de dispositivo de salida ─────────────────────────────────────────


def _select_output_sample_rate(device_index: Optional[int]) -> int:
    """Obtiene el sample rate nativo del dispositivo de salida."""
    try:
        dev = sd.query_devices(device_index, kind="output")
        sr = int(dev.get("default_samplerate", DEFAULT_OUTPUT_SAMPLE_RATE))
        if sr > 0:
            return sr
    except Exception:
        pass
    return DEFAULT_OUTPUT_SAMPLE_RATE


def _list_output_devices() -> List[dict]:
    """Lista todos los dispositivos de salida disponibles."""
    devices = sd.query_devices()
    out = []
    for i, d in enumerate(devices):
        if d.get("max_output_channels", 0) > 0:
            out.append({"index": i, **d})
    return out


def _pick_output_device(requested: Optional[str]) -> Optional[int]:
    """Selecciona el dispositivo de salida óptimo.

    Prioridad:
      1. Dispositivo explícitamente solicitado (índice o nombre).
      2. Dispositivo PulseAudio o PipeWire virtual.
      3. Dispositivo "default" de sounddevice (None → PortAudio lo elige).

    Args:
        requested: Índice (str de int) o nombre parcial del dispositivo.

    Returns:
        Índice del dispositivo seleccionado, o None para el default de PortAudio.
    """
    if requested is not None and requested != "":
        try:
            return int(requested)
        except ValueError:
            pass
        requested_lower = requested.lower()
        outs = _list_output_devices()
        for dev in outs:
            if requested_lower in dev["name"].lower():
                logger.debug("Dispositivo de salida por nombre: '%s'", dev["name"])
                return dev["index"]

    # Preferir PulseAudio / PipeWire para evitar bloqueos ALSA
    outs = _list_output_devices()
    for dev in outs:
        name_lower = dev["name"].lower()
        if "pulse" in name_lower or "pipewire" in name_lower:
            logger.debug("Usando dispositivo PulseAudio/PipeWire: '%s'", dev["name"])
            return dev["index"]

    # "default" explícito en la lista de sounddevice
    for dev in outs:
        if "default" in dev["name"].lower():
            logger.debug("Usando dispositivo 'default': '%s'", dev["name"])
            return dev["index"]

    logger.debug("Sin dispositivo de salida explícito; usando default de PortAudio.")
    return None


# ─── Clase principal ──────────────────────────────────────────────────────────


class TextToSpeech:
    """Síntesis y reproducción de voz a partir de texto.

    Detecta automáticamente el mejor backend disponible en el sistema:
    termux-tts-speak (Android), paplay (PulseAudio Linux) o espeak + sounddevice.

    Attributes:
        voice: Código de voz para espeak (ej. "es").
        rate: Velocidad de habla en palabras por minuto.
        volume: Volumen [0.0, 1.0].
    """

    def __init__(
        self,
        voice: str = DEFAULT_VOICE,
        rate: int = DEFAULT_RATE_WORDS_PER_MINUTE,
        volume: float = DEFAULT_VOLUME,
        output_device: Optional[str] = None,
        save_debug_wav: bool = True,
    ) -> None:
        self.voice = voice
        self.rate = rate
        self.volume = max(0.0, min(1.0, volume))
        self.save_debug_wav = save_debug_wav

        # ── Detectar backends disponibles ─────────────────────────────────────
        self._termux_tts: Optional[str] = shutil.which("termux-tts-speak")
        self._paplay: Optional[str] = shutil.which("paplay")
        self._espeak_cmd: Optional[List[str]] = self._find_espeak()

        if self._termux_tts:
            logger.info("TTS backend: termux-tts-speak (Android nativo)")
        elif self._paplay:
            logger.info("TTS backend primario: paplay (PulseAudio)")
        elif self._espeak_cmd:
            # Configurar dispositivo de salida solo si vamos a usar sounddevice
            requested = (
                output_device
                if output_device is not None
                else os.environ.get("MAVI_TTS_DEVICE")
            )
            self._device_index: Optional[int] = _pick_output_device(requested)
            if requested and self._device_index is None:
                logger.warning(
                    "Dispositivo '%s' no encontrado; usando default.", requested
                )
                self._device_index = _pick_output_device(None)

            self._output_sr = _select_output_sample_rate(self._device_index)
            if self._device_index is not None:
                dev = sd.query_devices(self._device_index, kind="output")
                logger.info(
                    "TTS backend: espeak-ng + sounddevice (dispositivo=%d '%s', SR=%d)",
                    self._device_index, dev["name"], self._output_sr,
                )
            else:
                logger.info(
                    "TTS backend: espeak-ng + sounddevice (dispositivo=default, SR=%d)",
                    self._output_sr,
                )
        else:
            raise RuntimeError(
                "No se encontró ningún backend de TTS disponible. "
                "Instala espeak-ng: sudo apt install espeak-ng  "
                "o en Termux: pkg install espeak-ng"
            )

        if self.save_debug_wav:
            try:
                os.makedirs(DEBUG_WAV_DIR, exist_ok=True)
            except Exception:
                pass

    # ── Backend espeak ────────────────────────────────────────────────────────

    @staticmethod
    def _find_espeak() -> Optional[List[str]]:
        """Busca espeak-ng o espeak en el PATH."""
        for name in ("espeak-ng", "espeak"):
            path = shutil.which(name)
            if path:
                return [path]
        return None

    def _synthesize(self, text: str) -> tuple[bytes, int, bytes]:
        """Sintetiza texto con espeak y retorna audio crudo + WAV completo.

        Args:
            text: Texto a sintetizar.

        Returns:
            Tupla (raw_pcm_bytes, sample_rate, full_wav_bytes).

        Raises:
            RuntimeError: Si espeak falla.
        """
        assert self._espeak_cmd is not None
        cmd = self._espeak_cmd + [
            "-v", self.voice,
            "-s", str(self.rate),
            "-a", str(int(self.volume * 200)),
            "--stdout",
            text,
        ]
        proc = subprocess.run(cmd, capture_output=True, check=False)
        if proc.returncode != 0 or not proc.stdout:
            err = proc.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"espeak falló (rc={proc.returncode}): {err}")

        with io.BytesIO(proc.stdout) as bio:
            wav = wave.open(bio, "rb")
            sample_rate = wav.getframerate()
            sample_width = wav.getsampwidth()
            raw = wav.readframes(wav.getnframes())

        if sample_width != 2:
            proc2 = subprocess.run(
                self._espeak_cmd + [
                    "-v", self.voice,
                    "-s", str(self.rate),
                    "-a", str(int(self.volume * 200)),
                    "--stdout",
                    text,
                ],
                capture_output=True, check=True,
            )
            with io.BytesIO(proc2.stdout) as bio2:
                wav2 = wave.open(bio2, "rb")
                raw2 = wav2.readframes(wav2.getnframes())
                return raw2, wav2.getframerate(), proc2.stdout
        return raw, sample_rate, proc.stdout

    def _save_debug_wav(self, raw_bytes: bytes, sample_rate: int) -> None:
        """Guarda el audio sintetizado como WAV en /tmp/mavi-tts (debug)."""
        if not self.save_debug_wav:
            return
        try:
            import uuid
            fname = f"{DEBUG_WAV_DIR}/{uuid.uuid4().hex[:8]}.wav"
            with wave.open(fname, "wb") as wout:
                wout.setnchannels(1)
                wout.setsampwidth(2)
                wout.setframerate(sample_rate)
                wout.writeframes(raw_bytes)
            logger.debug("Debug WAV guardado: %s", fname)
        except Exception:
            pass

    # ── Reproducción ─────────────────────────────────────────────────────────

    def _play_via_termux(self, text: str) -> bool:
        """Reproduce texto usando el TTS nativo de Android (termux-tts-speak).

        termux-tts-speak acepta el texto directamente como argumento,
        por lo que no necesitamos sintetizar WAV.

        Args:
            text: Texto a vocalizar.

        Returns:
            True si la reproducción fue exitosa.
        """
        try:
            subprocess.run(
                [self._termux_tts, text],
                check=True,
                stderr=subprocess.DEVNULL,
            )
            return True
        except Exception as exc:
            logger.warning("termux-tts-speak falló: %s", exc)
            return False

    def _play_via_paplay(self, full_wav: bytes) -> bool:
        """Reproduce WAV usando paplay (PulseAudio).

        Args:
            full_wav: Bytes completos del fichero WAV.

        Returns:
            True si la reproducción fue exitosa.
        """
        try:
            subprocess.run(
                [self._paplay],
                input=full_wav,
                check=True,
                stderr=subprocess.DEVNULL,
            )
            return True
        except Exception as exc:
            logger.warning("paplay falló: %s", exc)
            return False

    def _play_via_sounddevice(self, raw_bytes: bytes, src_sr: int) -> None:
        """Reproduce audio usando sounddevice (PortAudio).

        Args:
            raw_bytes: Muestras int16 crudas del audio.
            src_sr: Sample rate del audio fuente.
        """
        audio_int16 = np.frombuffer(raw_bytes, dtype=np.int16)
        data = audio_int16.astype(np.float32) / 32768.0
        if data.ndim > 1:
            data = data.mean(axis=1)

        if src_sr != self._output_sr:
            new_len = int(round(len(data) * self._output_sr / src_sr))
            data = resample(data, new_len).astype(np.float32)

        if np.abs(data).max() < 1e-6:
            logger.warning("El audio sintetizado es silencio (amplitud ~ 0).")

        if self._device_index is None and sd.default.device[1] < 0:
            logger.error("No hay dispositivo de salida ALSA válido.")
            return

        sd.play(data, self._output_sr, device=self._device_index, blocking=True)

    # ── API pública ───────────────────────────────────────────────────────────

    def say(self, text: str) -> None:
        """Vocaliza el texto dado usando el mejor backend disponible.

        Orden de prioridad:
          1. termux-tts-speak (Android nativo, sin sintetizar WAV).
          2. paplay (PulseAudio — requiere espeak para generar WAV).
          3. sounddevice (PortAudio — fallback universal).

        Args:
            text: Texto a vocalizar.
        """
        if not text:
            return
        logger.info("TTS: '%s'", text)

        # Backend 1: Android nativo
        if self._termux_tts:
            if self._play_via_termux(text):
                return
            # Si falla, intentar con espeak si está disponible
            if not self._espeak_cmd:
                logger.error("termux-tts-speak falló y espeak no está disponible.")
                return

        # Sintetizar con espeak (necesario para paplay y sounddevice)
        if not self._espeak_cmd:
            logger.error("No hay backend de síntesis disponible.")
            return

        try:
            raw_bytes, src_sr, full_wav = self._synthesize(text)
        except Exception as exc:
            logger.error("Error en síntesis de voz: %s", exc)
            return

        self._save_debug_wav(raw_bytes, src_sr)

        # Backend 2: paplay (PulseAudio)
        if self._paplay:
            if self._play_via_paplay(full_wav):
                return
            # Fallback a sounddevice si paplay falla

        # Backend 3: sounddevice (PortAudio)
        try:
            self._play_via_sounddevice(raw_bytes, src_sr)
        except Exception as exc:
            logger.error("Error reproduciendo con sounddevice: %s", exc)

    def stop(self) -> None:
        """Detiene la reproducción de audio (solo aplica a sounddevice)."""
        try:
            sd.stop()
        except Exception:
            pass
