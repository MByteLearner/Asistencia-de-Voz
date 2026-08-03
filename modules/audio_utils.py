import os
import sys
from typing import List, Optional, Tuple

import numpy as np
import sounddevice as sd


class suppress_c_stderr:
    """Context manager to suppress C-level stderr output from ALSA / PortAudio."""

    def __enter__(self):
        try:
            self.null_fd = os.open(os.devnull, os.O_WRONLY)
            self.old_stderr = os.dup(2)
            os.dup2(self.null_fd, 2)
            os.close(self.null_fd)
        except Exception:
            pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            os.dup2(self.old_stderr, 2)
            os.close(self.old_stderr)
        except Exception:
            pass


def resample_audio(audio_int16: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample 1D int16 numpy array from orig_sr to target_sr using linear interpolation."""
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


def list_input_devices() -> List[dict]:
    """Return a list of dicts for all sounddevice devices with max_input_channels > 0."""
    try:
        devices = sd.query_devices()
    except Exception:
        return []
    out = []
    for i, d in enumerate(devices):
        if d.get("max_input_channels", 0) > 0:
            out.append({"index": i, **d})
    return out


def select_input_device(
    requested: Optional[str] = None, target_sr: int = 16000
) -> Tuple[Optional[int], int, bool]:
    """Select a valid working input device index and sample rate instantly without blocking.

    Returns:
        (device_index, stream_sample_rate, need_resample)
    """
    req = requested if requested is not None else os.environ.get("MAVI_INPUT_DEVICE")
    inputs = list_input_devices()

    chosen_index: Optional[int] = None

    # 1. Check explicit requested device index or string match
    if req:
        try:
            idx = int(req)
            if any(d["index"] == idx for d in inputs):
                chosen_index = idx
        except ValueError:
            req_lower = req.lower()
            for d in inputs:
                if req_lower in d["name"].lower():
                    chosen_index = d["index"]
                    break

    # 2. Prioritize external mic / USB / headset input devices
    if chosen_index is None:
        for d in inputs:
            name_lower = d["name"].lower()
            if any(k in name_lower for k in ["headset", "usb", "microphone", "mic"]):
                chosen_index = d["index"]
                break

    # 3. Check system default input device if valid
    if chosen_index is None:
        try:
            def_idx = sd.default.device[0]
            if def_idx is not None and def_idx >= 0:
                dev_info = sd.query_devices(def_idx)
                if dev_info.get("max_input_channels", 0) > 0:
                    chosen_index = def_idx
        except Exception:
            pass

    # 4. Fallback to first available input device
    if chosen_index is None and inputs:
        chosen_index = inputs[0]["index"]

    if chosen_index is None:
        print(
            "[audio_utils] ADVERTENCIA: No se encontró ningún dispositivo de entrada de audio.",
            file=sys.stderr,
        )
        return None, target_sr, False

    # Check if target_sr is supported directly by chosen_index
    try:
        sd.check_input_settings(
            device=chosen_index, samplerate=target_sr, channels=1, dtype="int16"
        )
        return chosen_index, target_sr, False
    except Exception:
        pass

    dev_info = sd.query_devices(chosen_index)
    native_sr = int(dev_info.get("default_samplerate", 44100))
    if native_sr <= 0:
        native_sr = 44100

    return chosen_index, native_sr, True


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
    """Create and return an sd.InputStream, suppressing ALSA C warnings and retrying if ALSA device is temporarily busy."""
    last_exc = None
    for attempt in range(max_retries):
        try:
            kwargs = {
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
                sd.sleep(int(retry_delay * 1000))
            else:
                raise exc
    if last_exc:
        raise last_exc
