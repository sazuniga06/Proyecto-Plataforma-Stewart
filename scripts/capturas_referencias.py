#!/usr/bin/env python3
"""
capturas_referencias.py
Capturas del simulador (vista 3D + osciloscopios) para cada tipo de movimiento y cada controlador, con la planta
no lineal, las ganancias optimas y ruido de sensor de 0.1 mm. Cada captura muestra los primeros 5 s de la
referencia, desde la pose nominal. Requiere entorno grafico.
Salida: informe/figs/capturas/<referencia>_<pid|ss>.png
Uso, desde la raiz del proyecto:  python scripts/capturas_referencias.py
"""
import os
import sys
import tkinter as tk

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "simulator_gui"))
import app_simulador_stewart as A  # noqa: E402

OUT = os.path.join(ROOT, "informe", "figs", "capturas")
DURACION = 5.0          # ventana del osciloscopio [s]


def main():
    os.makedirs(OUT, exist_ok=True)
    root = tk.Tk()
    app = A.StewartRealtimeSimulatorGUI(root)

    def capturar_todo():
        app.is_running = False
        n = int(round(DURACION / app.engine.Ts))
        for ctrl_label, ctrl in app.CTRL_MODES:
            app.var_ctrl.set(ctrl_label)
            app._on_ctrl_change()
            for ref_label, ref in app.REF_MODES:
                if ref == "manual" or (ref.startswith("bc_") and ctrl == "pid"):   # los casos A-D se muestran con estados
                    continue
                app.var_ref_mode.set(ref_label)
                app._on_ref_change()
                app.var_noise.set(0.1)
                app.is_running = False
                snap = app._advance(n)
                app._autoscale(force=True)
                app._update_visuals_fast(snap)
                path = os.path.join(OUT, f"{ref}_{ctrl}.png")
                app.capturar(path, dpi=110)
                print(f"  {path}")
        root.destroy()

    root.after(400, capturar_todo)
    root.mainloop()


if __name__ == "__main__":
    main()
