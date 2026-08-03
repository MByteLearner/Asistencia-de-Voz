import io
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


DEFAULT_VOICE = "es"
DEFAULT_RATE_WORDS_PER_MINUTE = 165
DEFAULT_VOLUME = 1.0
DEFAULT_OUTPUT_SAMPLE_RATE = 48000
DEBUG_WAV_DIR = "/tmp/mavi-tts"


def _select_output_sample_rate(device_index: Optional[int]) -> int:
    try:
        dev = sd.query_devices(device_index, kind="output")
        sr = int(dev.get("default_samplerate", DEFAULT_OUTPUT_SAMPLE_RATE))
        if sr > 0:
            return sr
    except Exception:
        pass
    return DEFAULT_OUTPUT_SAMPLE_RATE


def _list_output_devices() -> List[dict]:
    devices = sd.query_devices()
    out = []
    for i, d in enumerate(devices):
        if d.get("max_output_channels", 0) > 0:
            out.append({"index": i, **d})
    return out


def _pick_output_device(requested: Optional[str]) -> Optional[int]:
    """Pick the best output device index.

    Priority (when no explicit device is requested):
      1. Analog (ALC236) — laptop speakers that auto-switch to 3.5 mm jack
      2. Any non-HDMI output
      3. sounddevice system default output
    Explicitly skips HDMI/DisplayPort ports which are silent when no display is connected.
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
                return dev["index"]
        return None

    outs = _list_output_devices()
    SILENT_KEYWORDS = ("hdmi", "displayport", "dp")

    # 1. Prefer analog / headset / speaker outputs (non-HDMI)
    for dev in outs:
        name_lower = dev["name"].lower()
        if not any(k in name_lower for k in SILENT_KEYWORDS):
            if any(k in name_lower for k in ("analog", "alc", "headphone", "speaker", "audio")):
                return dev["index"]

    # 2. Any non-HDMI output
    for dev in outs:
        name_lower = dev["name"].lower()
        if not any(k in name_lower for k in SILENT_KEYWORDS):
            return dev["index"]

    # 3. Fallback to sounddevice system default output
    try:
        default_out = sd.default.device[1]
        if default_out is not None and default_out >= 0:
            return default_out
    except Exception:
        pass

    return None


class TextToSpeech:
    def __init__(
        self,
        voice: str = DEFAULT_VOICE,
        rate: int = DEFAULT_RATE_WORDS_PER_MINUTE,
        volume: float = DEFAULT_VOLUME,
        output_device: Optional[str] = None,
        save_debug_wav: bool = True,
    ):
        self.voice = voice
        self.rate = rate
        self.volume = max(0.0, min(1.0, volume))
        self.save_debug_wav = save_debug_wav
        self._espeak_cmd: Optional[List[str]] = self._find_espeak()
        if self._espeak_cmd is None:
            raise RuntimeError(
                "No se encontró 'espeak' ni 'espeak-ng' instalado. "
                "Instala con: sudo apt-get install espeak-ng"
            )

        requested = (
            output_device
            if output_device is not None
            else os.environ.get("MAVI_TTS_DEVICE")
        )
        self._device_index: Optional[int] = _pick_output_device(requested)
        if requested and self._device_index is None:
            print(
                f"[tts] ADVERTENCIA: dispositivo '{requested}' no "
                f"encontrado, usando default.",
                file=sys.stderr,
            )
            self._device_index = _pick_output_device(None)

        self._output_sr = _select_output_sample_rate(self._device_index)

        if self._device_index is not None:
            dev = sd.query_devices(self._device_index, kind="output")
            print(
                f"[tts] Motor: {' '.join(self._espeak_cmd)} "
                f"(voz={voice}, rate={rate}, dispositivo={self._device_index} "
                f"'{dev['name']}', sr={self._output_sr})"
            )
        else:
            print(
                f"[tts] Motor: {' '.join(self._espeak_cmd)} "
                f"(voz={voice}, rate={rate}, dispositivo=default, "
                f"sr={self._output_sr})"
            )

        if self.save_debug_wav:
            try:
                os.makedirs(DEBUG_WAV_DIR, exist_ok=True)
            except Exception:
                pass

    @staticmethod
    def _find_espeak() -> Optional[List[str]]:
        for name in ("espeak-ng", "espeak"):
            path = shutil.which(name)
            if path:
                return [path]
        return None

    def _synthesize(self, text: str) -> tuple[bytes, int]:
        assert self._espeak_cmd is not None
        cmd = self._espeak_cmd + [
            "-v", self.voice,
            "-s", str(self.rate),
            "-a", str(int(self.volume * 200)),
            "--stdout",
            text,
        ]
        proc = subprocess.run(
            cmd, capture_output=True, check=False
        )
        if proc.returncode != 0 or not proc.stdout:
            err = proc.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                f"espeak falló (rc={proc.returncode}): {err}"
            )
        with io.BytesIO(proc.stdout) as bio:
            wav = wave.open(bio, "rb")
            sample_rate = wav.getframerate()
            n_channels = wav.getnchannels()
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
            return proc2.stdout, 22050
        return raw, sample_rate

    def _save_debug_wav(self, raw_bytes: bytes, sample_rate: int):
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
        except Exception:
            pass

    def say(self, text: str):
        if not text:
            return
        print(f"[tts] >> {text}")
        try:
            raw_bytes, src_sr = self._synthesize(text)
            self._save_debug_wav(raw_bytes, src_sr)

            audio_int16 = np.frombuffer(raw_bytes, dtype=np.int16)
            data = audio_int16.astype(np.float32) / 32768.0
            if data.ndim > 1:
                data = data.mean(axis=1)

            if src_sr != self._output_sr:
                new_len = int(round(len(data) * self._output_sr / src_sr))
                data = resample(data, new_len).astype(np.float32)

            if np.abs(data).max() < 1e-6:
                print(
                    "[tts] ADVERTENCIA: el audio sintetizado es silencio.",
                    file=sys.stderr,
                )

            sd.play(
                data,
                self._output_sr,
                device=self._device_index,
                blocking=True,
            )
        except Exception as exc:
            print(f"[tts] Error reproduciendo audio: {exc}", file=sys.stderr)

    def stop(self):
        try:
            sd.stop()
        except Exception:
            pass
