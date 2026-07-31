#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ ! -d ".venv" ]]; then
  echo "[run] No se encontró .venv. Ejecuta ./install.sh primero." >&2
  exit 1
fi

source .venv/bin/activate

if [[ -d "$HOME/.local/lib" ]] && [[ -z "${LD_LIBRARY_PATH:-}" || "${LD_LIBRARY_PATH}" != *"$HOME/.local/lib"* ]]; then
  export LD_LIBRARY_PATH="$HOME/.local/lib:${LD_LIBRARY_PATH:-}"
fi

exec python -u main.py "$@"
