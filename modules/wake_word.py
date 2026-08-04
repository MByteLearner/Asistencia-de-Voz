"""
modules/wake_word.py
────────────────────
MÓDULO DEPRECADO — No usar.

Este módulo fue reemplazado por `modules/voice_listener.py` (VoiceListener)
durante la refactorización para compatibilidad con Android/Termux (ARM64).

Razón: openwakeword y onnxruntime no tienen soporte oficial para Android/ARM64
vía pip, por lo que la instalación fallaba en Termux.

El nuevo sistema de activación funciona por keyword spotting en el texto
transcripto por el STT, sin necesidad de ningún modelo ONNX.

Ver: modules/voice_listener.py
"""

# Este archivo se mantiene vacío para preservar el historial de git.
# El módulo original ha sido reemplazado por VoiceListener en voice_listener.py.
