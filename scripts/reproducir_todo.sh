#!/usr/bin/env bash
# ==============================================================================
# Reproduce todos los resultados del proyecto, de principio a fin, desde la raiz:
#   ./scripts/reproducir_todo.sh            (todas las etapas)
#   ./scripts/reproducir_todo.sh --docs     (solo datos, figuras, tablas y PDF)
# Etapas: optimizacion de ganancias,
# identificacion, banco no lineal, validacion del modelo, figuras/tablas y PDF.
# La optimizacion tarda alrededor de una hora y el banco no lineal media hora.
# ==============================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PY=".venv/bin/python"; [ -x "$PY" ] || PY="../.venv/bin/python"; [ -x "$PY" ] || PY="python3"
OCT="octave --no-gui --quiet"
ENG="addpath('octave_engine', 'octave_engine/config', 'octave_engine/kinematics', 'octave_engine/dynamics', 'octave_engine/controllers', 'octave_engine/solvers');"

if [ "${1:-}" != "--docs" ]; then
    echo "[1/5] Optimizacion de ganancias";             "$PY" scripts/optimizar_ganancias.py
    echo "[2/5] Identificacion experimental";           $OCT --eval "$ENG identificacion_experimental(20);"
    echo "[3/5] Banco de pruebas no lineal";            $OCT octave_engine/ENTREGA_1_BENCHMARK_OCTAVE.m
    echo "[4/5] Validacion del modelo dinamico";        $OCT --eval "$ENG validar_modelo_dinamico();"
fi
echo "[5/5] Datos, figuras, tablas y documentos"
$OCT scripts/exportar_datos_informe.m
"$PY" scripts/generar_figuras_informe.py
"$PY" scripts/figuras_modelos.py
"$PY" scripts/resultados_referencias.py
[ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ] && "$PY" scripts/capturas_referencias.py   # requiere entorno grafico
"$PY" scripts/generar_datos_calculo.py
(cd desarrollo_matematico && for i in 1 2; do xelatex -interaction=nonstopmode desarrollo_matematico.tex > /dev/null; done)
for doc in informe/informe presentacion/presentacion; do
    (cd "$(dirname "$doc")" && for i in 1 2; do pdflatex -interaction=nonstopmode "$(basename "$doc").tex" > /dev/null; done)
done
echo "Listo: desarrollo_matematico/desarrollo_matematico.pdf, informe/informe.pdf y presentacion/presentacion.pdf"
