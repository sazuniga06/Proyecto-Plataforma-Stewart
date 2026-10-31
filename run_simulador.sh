#!/usr/bin/env bash
# ==============================================================================
# Lanzador del simulador de la plataforma Stewart 6-SPS
# - Motor de modelado y diseno de control: octave_engine/ (GNU Octave)
# - Interfaz y visualizacion 3D: simulator_gui/ (Python)
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Entorno virtual: dentro del proyecto (.venv) o en la carpeta superior; si no existe, python3 del sistema
VENV_PY="$SCRIPT_DIR/.venv/bin/python"
[ -x "$VENV_PY" ] || VENV_PY="$SCRIPT_DIR/../.venv/bin/python"

if [ ! -x "$VENV_PY" ]; then
    VENV_PY="python3"
fi

exec "$VENV_PY" "$SCRIPT_DIR/simulator_gui/app_simulador_stewart.py" "$@"
