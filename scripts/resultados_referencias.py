#!/usr/bin/env python3
"""
resultados_referencias.py
Resultados por tipo de movimiento (todas las referencias del simulador), con la planta no lineal, las ganancias
optimas y ruido de sensor de 0.1 mm, para el informe y la presentacion:
  - Calculo previo de cada referencia (antes de simular): desplazamiento maximo de las patas dl = J0 dq, velocidad
    maxima de pata, aceleracion maxima de la pose y fuerza de prealimentacion que exige el modelo lineal,
        dF_ff = J0^-T (M0 dq'' + K_q dq),   K_q = -M0 A21,
    comparada con la fuerza maxima de la simulacion.
  - Indices de cada referencia con cada controlador (12 s): RMS de posicion y orientacion, error maximo, ITAE, Ju,
    fuerza maxima y porcentaje del tiempo con fuerza saturada.
  - Una figura por referencia: grados de libertad principales (referencia, PID y realimentacion de estados), norma
    del error de posicion y fuerza de la pata 1.
Salidas: informe/tablas/referencias.tex, referencias_calculo.tex y referencias_valores.tex (macros);
         informe/figs/ref_<referencia>.pdf
Uso, desde la raiz del proyecto:  python scripts/resultados_referencias.py
"""
import importlib.util
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "simulator_gui"))
spec = importlib.util.spec_from_file_location("gf", os.path.join(ROOT, "scripts", "generar_figuras_informe.py"))
gf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gf)
import matplotlib.pyplot as plt  # noqa: E402
from octave_bridge import reference_pose  # noqa: E402

T_SIM = 12.0
# (codigo, nombre en las tablas, nombre corto para macros, grados de libertad que se grafican)
REFS = [("sine_z", "Senoidal en $z$", "Sen", (2,)),
        ("coupled_6dof", "Maniobra acoplada 6 GDL", "Seis", (0, 1, 2)),
        ("circle_3d", "Circular con inclinación", "Circ", (0, 1)),
        ("bc_A", "Frontera A", "FA", (2,)), ("bc_B", "Frontera B", "FB", (2,)),
        ("bc_C", "Frontera C", "FC", (2,)), ("bc_D", "Frontera D", "FD", (2,)),
        ("var_fixed", "Extremos fijos (Beltrami)", "Fij", (0, 1, 2)),
        ("min_time", "Tiempo mínimo (bang-bang)", "Tmin", (0, 1, 2)),
        ("var_curve", "Curva terminal", "Curva", (2,))]
DOF = ("x", "y", "z")
COLS = {0: gf.C_X, 1: gf.C_Y, 2: gf.C_Z}


def calculo_previo(e, mode):
    """Cinematica y fuerza de prealimentacion de la referencia, con el modelo lineal nominal."""
    Ts = e.Ts
    t = np.arange(0.0, T_SIM, Ts)
    q = np.array([reference_pose(mode, ti, e.pose0) for ti in t]) - e.pose0
    qd = np.gradient(q, Ts, axis=0)
    qdd = np.gradient(qd, Ts, axis=0)
    J0 = np.asarray(e.planta_J0)
    # Saltos de la referencia: de posicion (> 1 mm en una muestra) y de velocidad (> 2 mm/s en una muestra; una
    # trayectoria suave de estas no cambia mas de 0.3 mm/s por muestra). En un salto la aceleracion teorica es
    # infinita, de modo que la aceleracion y la fuerza se evaluan en los tramos suaves, lejos de los saltos.
    dq = np.linalg.norm(np.diff(q[:, :3], axis=0), axis=1)
    dv = np.linalg.norm(np.diff(qd[:, :3], axis=0), axis=1)
    salto = (dq > 1e-3) | (dv > 2e-3)
    lejos = np.ones(len(t), dtype=bool)
    for k in np.flatnonzero(salto):
        lejos[max(0, k - 3):k + 5] = False
    vel_lados = [np.linalg.norm(qd[min(k + 4, len(t) - 1), :3] - qd[max(k - 3, 0), :3]) for k in np.flatnonzero(dv > 2e-3)]
    dl = q @ J0.T
    dld = qd @ J0.T
    M0, A21 = e.M0, e.A[6:, :6]
    Kq = -M0 @ A21
    F_ff = np.linalg.solve(J0.T, (qdd @ M0.T + q @ Kq.T).T).T
    return {"dl": 1e3 * np.abs(dl).max(), "dld": 1e3 * np.abs(dld[lejos]).max(),
            "qdd": 1e3 * np.linalg.norm(qdd[lejos, :3], axis=1).max(),
            "Fff": np.abs(F_ff[lejos]).max() + e.u0[0],
            "dv": 1e3 * max(vel_lados, default=0.0), "dq": 1e3 * dq.max() if dq.max() > 1e-3 else 0.0}


def simular(e, ctrl, scales, mode):
    d = gf.record(e, ctrl, scales, lambda t: reference_pose(mode, t, e.pose0), lambda k: np.zeros(6), T_SIM)
    m = e.metrics()
    epos = np.linalg.norm(d[:, 7:10] - d[:, 1:4], axis=1)
    m["emax"] = 1e3 * epos.max()
    m["Freg"] = np.abs(d[d[:, 0] > 1.5, 13:19]).max()    # despues del arranque
    return d, m


def figura(code, dofs, data, e):
    fig, axs = plt.subplots(3, 1, figsize=(gf.COL2 * 0.62, 3.1), sharex=True, constrained_layout=True)
    r = data["pid"]
    for j in dofs:
        off = e.h0 if j == 2 else 0.0
        axs[0].plot(r[:, 0], 1e3 * (r[:, 7 + j] - off), "--", color="#555555", lw=1.0)
        for c, ls in (("pid", "-"), ("ss", ":")):
            d = data[c]
            axs[0].plot(d[:, 0], 1e3 * (d[:, 1 + j] - off), ls, color=COLS[j], lw=1.0)
    axs[0].plot([], [], "--", color="#555555", label="referencia")
    axs[0].plot([], [], "-", color="#333333", label="PID")
    axs[0].plot([], [], ":", color="#333333", label="realim. estados")
    for j in dofs:
        axs[0].plot([], [], color=COLS[j], label=f"$\\Delta {DOF[j]}$")
    axs[0].set_ylabel("pose [mm]")
    axs[0].legend(fontsize=5.5, ncol=3 + len(dofs), loc="upper right")
    for c, col, lab in (("pid", gf.C_PID, "PID"), ("ss", gf.C_SS, "realim. estados")):
        d = data[c]
        axs[1].plot(d[:, 0], 1e3 * np.linalg.norm(d[:, 7:10] - d[:, 1:4], axis=1), color=col, lw=0.7, label=lab)
        axs[2].plot(d[:, 0], d[:, 13], color=col, lw=0.6, label=lab)
    axs[1].set_ylabel("$\\|e_p\\|$ [mm]")
    axs[1].legend(fontsize=5.5, ncol=2, loc="upper right")
    axs[2].set_ylabel("$F_1$ [N]")
    axs[2].set_xlabel("tiempo [s]")
    for a in axs:
        a.set_xlim(0, T_SIM)
    gf.save(fig, f"ref_{code}.pdf")


def main():
    e, o, ps = gf.engine_with_gains()
    import scipy.io as sio
    m = sio.loadmat(e.matrices_mat_path, squeeze_me=True)
    e.M0 = np.asarray(m["M0"], dtype=float)
    e.planta_J0 = np.asarray(m["J0"], dtype=float)
    filas, filas_c, macros = [], [], []
    for code, nombre, corto, dofs in REFS:
        c = calculo_previo(e, code)
        data, met = {}, {}
        for ctrl, s in (("pid", ps), ("ss", (1, 1, 1))):
            data[ctrl], met[ctrl] = simular(e, ctrl, s, code)
        figura(code, dofs, data, e)
        for ctrl, lab in (("pid", "PID"), ("ss", "Estados")):
            r = met[ctrl]
            filas.append(f"{nombre if ctrl == 'pid' else ''} & {lab} & {r['RMS_pos']:.3f} & {r['RMS_ang']:.3f} & "
                         f"{r['emax']:.2f} & {r['ITAE']:.4f} & {r['Ju']:.0f} & {r['Fmax']:.1f} & {r['fsat_pct']:.1f} \\\\")
        filas.append("\\midrule")
        salto = (f"{c['dq']:.1f} mm" if c["dq"] else (f"{c['dv']:.1f} mm/s" if c["dv"] else "---"))
        filas_c.append(f"{nombre} & {c['dl']:.2f} & {c['dld']:.1f} & {c['qdd']:.0f} & {salto} & {c['Fff']:.1f} & "
                       f"{met['ss']['Freg']:.1f} \\\\")
        for k, v in (("dl", c["dl"]), ("dld", c["dld"]), ("qdd", c["qdd"]), ("Fff", c["Fff"]), ("dv", c["dv"]),
                     ("dq", c["dq"]), ("Freg", met["ss"]["Freg"])):
            macros.append(f"\\newcommand{{\\Ref{corto}{k}}}{{{v:.2f}}}")
        for ctrl, lab in (("pid", "Pid"), ("ss", "Ss")):
            r = met[ctrl]
            macros += [f"\\newcommand{{\\Ref{corto}{lab}Rms}}{{{r['RMS_pos']:.3f}}}",
                       f"\\newcommand{{\\Ref{corto}{lab}Emax}}{{{r['emax']:.2f}}}",
                       f"\\newcommand{{\\Ref{corto}{lab}Fmax}}{{{r['Fmax']:.1f}}}"]
        print(f"  {code}: PID {met['pid']['RMS_pos']:.3f} mm, estados {met['ss']['RMS_pos']:.3f} mm; "
              f"F_ff {c['Fff']:.1f} N frente a {met['ss']['Freg']:.1f} N (salto dv {c['dv']:.1f} mm/s, dq {c['dq']:.1f} mm)")
    gf.write_tab("referencias.tex", "\n".join(filas[:-1]) + "\n")
    gf.write_tab("referencias_calculo.tex", "\n".join(filas_c) + "\n")
    gf.write_tab("referencias_valores.tex", "\n".join(macros) + "\n")


if __name__ == "__main__":
    main()
