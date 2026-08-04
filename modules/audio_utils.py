"""
modules/audio_utils.py
──────────────────────
Utilidades de audio de bajo nivel.

Funciones para:
  - Seleccionar automáticamente el dispositivo de entrada (micrófono).
  - Crear streams de entrada de sounddevice con reintentos.
  - Remuestrear audio de int16.

Compatibilidad:
  - Linux desktop (x86_64) con ALSA / PipeWire / PulseAudio
  - Android / Termux (aarch64) — pkg install portaudio
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional, Tuple

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)


# ─── Suprimir stderr de C-level (ALSA spam) ───────────────────────────────────


class suppress_c_stderr:
    """Context manager para suprimir la salida de nivel C (ALSA/PortAudio).

    Redirige el descriptor de archivo 2 (stderr del proceso) a /dev/null
    temporalmente para evitar el spam de mensajes de ALSA que no se pueden
    suprimir desde Python.
    """

    def __enter__(self) -> "suppress_c_stderr":
        try:
            self.null_fd = os.open(os.devnull, os.O_WRONLY)
            self.old_stderr = os.dup(2)
            os.dup2(self.null_fd, 2)
            os.close(self.null_fd)
        except Exception:
            pass
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        try:
            os.dup2(self.old_stderr, 2)
            os.close(self.old_stderr)
        except Exception:
            pass


# ─── Remuestreo ───────────────────────────────────────────────────────────────


def resample_audio(audio_int16: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Remuestrea un array 1D de int16 de orig_sr a target_sr (interpolación lineal).

    Args:
        audio_int16: Array de muestras int16.
        orig_sr: Frecuencia de muestreo original (Hz).
        target_sr: Frecuencia de muestreo destino (Hz).

    Returns:
        Array de muestras int16 a target_sr.
    """
    if orig_sr == target_sr or len(audio_int16) == 0:
        return audio_int16
    num_orig = len(audio_int16)
    num_target = int(round(num_orig * target_sr / orig_sr))
    if num_target <= 0:
        return np.array([], dtype=np.int16)
    x_orig = np.linspace(0, num_orig, num_orig, endpoint=False)
    x_target = np.linspace(0, num_orig, num_target, endpoint=False)
    resampled_float = np.interp(x_target, x_orig, audio_int16.astype(np.float32))
    return np.clip(resampled_float, -32768, 32767).astype(np.int16)


# ─── Enumeración de dispositivos ──────────────────────────────────────────────


def list_input_devices() -> List[dict]:
    """Devuelve la lista de dispositivos de entrada disponibles en sounddevice.

    Returns:
        Lista de dicts con la info de cada dispositivo que tiene canales de entrada.
        Incluye la clave extra 'index' con el índice de sounddevice.
    """
    try:
        devices = sd.query_devices()
    except Exception as exc:
        logger.warning("No se pudo consultar dispositivos de audio: %s", exc)
        return []
    out = []
    for i, d in enumerate(devices):
        if d.get("max_input_channels", 0) > 0:
            out.append({"index": i, **d})
    return out


# ─── Selección automática de dispositivo de entrada ───────────────────────────


def select_input_device(
    requested: Optional[str] = None, target_sr: int = 16000
) -> Tuple[Optional[int], int, bool]:
    """Selecciona el mejor dispositivo de entrada disponible.

    Orden de prioridad:
      1. Dispositivo explícitamente solicitado (índice numérico o nombre).
      2. Dispositivos PulseAudio / PipeWire (evitan bloqueos ALSA exclusivos).
      3. Dispositivos externos: headset, USB, micrófono.
      4. Dispositivo predeterminado del sistema.
      5. Primer dispositivo de entrada disponible.

    Args:
        requested: Índice (str de int) o nombre (parcial) del dispositivo deseado.
                   Si es None, se lee la variable de entorno MAVI_INPUT_DEVICE.
        target_sr: Frecuencia de muestreo objetivo (Hz). Por defecto 16000.

    Returns:
        Tupla (device_index, stream_sample_rate, need_resample):
          - device_index: Índice del dispositivo seleccionado, o None para el default.
          - stream_sample_rate: Sample rate al que abrir el stream.
          - need_resample: True si el audio necesita ser remuestreado a target_sr.
    """
    req = requested if requested is not None else os.environ.get("MAVI_INPUT_DEVICE")
    inputs = list_input_devices()

    chosen_index: Optional[int] = None

    # 1. Dispositivo explícitamente solicitado
    if req:
        try:
            idx = int(req)
            if any(d["index"] == idx for d in inputs):
                chosen_index = idx
                logger.debug("Dispositivo de entrada seleccionado por índice: %d", idx)
        except ValueError:
            req_lower = req.lower()
            for d in inputs:
                if req_lower in d["name"].lower():
                    chosen_index = d["index"]
                    logger.debug(
                        "Dispositivo de entrada seleccionado por nombre: '%s'", d["name"]
                    )
                    break

    # 2. Dispositivos PulseAudio / PipeWire (evitan bloqueos ALSA)
    if chosen_index is None:
        for d in inputs:
            name_lower = d["name"].lower()
            if "pulse" in name_lower or "pipewire" in name_lower:
                chosen_index = d["index"]
                logger.debug("Usando dispositivo PulseAudio/PipeWire: '%s'", d["name"])
                break

    # 3. Dispositivos externos: headset, USB, micrófono
    if chosen_index is None:
        for d in inputs:
            name_lower = d["name"].lower()
            if any(k in name_lower for k in ["headset", "usb", "microphone", "mic"]):
                chosen_index = d["index"]
                logger.debug("Usando dispositivo externo: '%s'", d["name"])
                break

    # 4. Default del sistema
    if chosen_index is None:
        try:
            def_idx = sd.default.device[0]
            if def_idx is not None and def_idx >= 0:
                dev_info = sd.query_devices(def_idx)
                if dev_info.get("max_input_channels", 0) > 0:
                    chosen_index = def_idx
                    logger.debug("Usando dispositivo de entrada predeterminado: %d", def_idx)
        except Exception:
            pass

    # 5. Primer dispositivo disponible
    if chosen_index is None and inputs:
        chosen_index = inputs[0]["index"]
        logger.debug(
            "Usando primer dispositivo de entrada disponible: '%s'", inputs[0]["name"]
        )

    if chosen_index is None:
        # Sin dispositivo encontrado: intentar con None (default de PortAudio)
        try:
            def_idx = sd.default.device[0]
            if def_idx is not None and def_idx >= 0:
                dev_info = sd.query_devices(def_idx)
                native_sr = int(dev_info.get("default_samplerate", 44100)) or 44100
                try:
                    sd.check_input_settings(
                        device=None, samplerate=target_sr, channels=1, dtype="int16"
                    )
                    return None, target_sr, False
                except Exception:
                    return None, native_sr, True
        except Exception:
            pass
        logger.warning("No se encontró ningún dispositivo de entrada.")
        return None, target_sr, False

    # Verificar si target_sr es soportado nativamente
    try:
        sd.check_input_settings(
            device=chosen_index, samplerate=target_sr, channels=1, dtype="int16"
        )
        dev_name = sd.query_devices(chosen_index).get("name", str(chosen_index))
        logger.info(
            "Dispositivo de entrada: %d '%s' | SR: %d Hz (nativo)",
            chosen_index, dev_name, target_sr,
        )
        return chosen_index, target_sr, False
    except Exception:
        pass

    dev_info = sd.query_devices(chosen_index)
    native_sr = int(dev_info.get("default_samplerate", 44100)) or 44100
    logger.info(
        "Dispositivo de entrada: %d '%s' | SR nativo: %d Hz (se remuestreará a %d Hz)",
        chosen_index, dev_info.get("name", ""), native_sr, target_sr,
    )
    return chosen_index, native_sr, True


# ─── Creación de stream de entrada ────────────────────────────────────────────


def create_input_stream(
    device: Optional[int],
    samplerate: int,
    channels: int = 1,
    dtype: str = "int16",
    blocksize: Optional[int] = None,
    callback: Optional[callable] = None,
    max_retries: int = 5,
    retry_delay: float = 0.25,
) -> sd.InputStream:
    """Crea y retorna un sd.InputStream con reintentos automáticos.

    Suprime los mensajes de C-level de ALSA/PortAudio y reintenta si el
    dispositivo está temporalmente ocupado (error ALSA típico al abrir
    mientras otro proceso lo usa).

    Args:
        device: Índice del dispositivo, o None para el default.
        samplerate: Frecuencia de muestreo (Hz).
        channels: Número de canales (default 1).
        dtype: Tipo de dato del stream (default "int16").
        blocksize: Tamaño del bloque en muestras (None = default de PortAudio).
        callback: Función callback para modo asíncrono (None = modo síncrono).
        max_retries: Número máximo de intentos.
        retry_delay: Pausa entre reintentos en segundos.

    Returns:
        sd.InputStream listo para usar (no iniciado).

    Raises:
        sd.PortAudioError: Si no se pudo abrir el stream tras todos los intentos.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            kwargs: dict = {
                "device": device,
                "samplerate": samplerate,
                "channels": channels,
                "dtype": dtype,
            }
            if blocksize is not None:
                kwargs["blocksize"] = blocksize
            if callback is not None:
                kwargs["callback"] = callback

            with suppress_c_stderr():
                stream = sd.InputStream(**kwargs)
            return stream

        except sd.PortAudioError as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                logger.debug(
                    "Error al abrir stream (intento %d/%d): %s. Reintentando...",
                    attempt + 1, max_retries, exc,
                )
                sd.sleep(int(retry_delay * 1000))
            else:
                logger.error(
                    "No se pudo abrir el stream de audio tras %d intentos: %s",
                    max_retries, exc,
                )
                raise exc

    if last_exc:
        raise last_exc
