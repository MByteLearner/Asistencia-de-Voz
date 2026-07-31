#!/usr/bin/env bash
set -euo pipefail

if [[ "${EUID}" -eq 0 ]]; then
  SUDO=""
else
  if ! command -v sudo >/dev/null 2>&1; then
    echo "[install] sudo no encontrado. Si ya eres root ejecuta sin sudo." >&2
    exit 1
  fi
  SUDO="sudo"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "[install] Instalando dependencias de sistema..."
${SUDO} apt-get update
${SUDO} apt-get install -y \
  python3 \
  python3-pip \
  python3-venv \
  portaudio19-dev \
  espeak-ng \
  libespeak-ng1

echo "[install] Creando/activando entorno virtual (.venv)..."
if [[ ! -d ".venv" ]]; then
  python3 -m venv .venv
fi
source .venv/bin/activate

echo "[install] Actualizando pip..."
pip install --upgrade pip

echo "[install] Instalando dependencias Python..."
pip install -r requirements.txt

echo "[install] Verificando instalación..."
python -c "import openwakeword, speech_recognition, pyttsx3, sounddevice; print('OK')"

echo "[install] Listo. Ejecuta: ./run.sh"
