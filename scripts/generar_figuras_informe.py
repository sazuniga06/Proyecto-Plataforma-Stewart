#!/usr/bin/env python3
"""
generar_figuras_informe.py
Genera las figuras (informe/figs/*.pdf) y las tablas (informe/tablas/*.tex) del informe a partir de:
  - informe/datos/*.mat        (scripts/exportar_datos_informe.m, motor de Octave)
  - results/identificacion.mat       (octave_engine/dynamics/identificacion_experimental.m)
  - results/resultados_entrega1_octave.mat (ENTREGA_1_BENCHMARK_OCTAVE.m, planta no lineal)
  - results/ganancias_optimas.json
  - simulador (simulator_gui/octave_bridge.py) para el ensayo normalizado y la Tabla 1
Uso, desde la raiz del proyecto:  python scripts/generar_figuras_informe.py
"""
import json
import os
import shutil
import sys

import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.join(ROOT, "informe")
sys.path.insert(0, os.path.join(ROOT, "simulator_gui"))
from octave_bridge import OctaveRealtimeControlEngine, reference_pose  # noqa: E402

FIG = os.path.join(HERE, "figs")
TAB = os.path.join(HERE, "tablas")
DAT = os.path.join(HERE, "datos")
RES = os.path.join(ROOT, "results")
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)

COL1, COL2 = 3.5, 7.16           # anchos de columna IEEE [in]
C_PID, C_SS, C_REF, C_AUX = "#b5473f", "#1f4e79", "#6b6b6b", "#c0711b"
C_X, C_Y, C_Z = "#b03a2e", "#2e7d32", "#1f4e79"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "axes.grid": True, "grid.color": "#e2e2e2", "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "lines.linewidth": 1.1,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def save(fig, name):
    fig.savefig(os.path.join(FIG, name))
    plt.close(fig)
    print("  figs/" + name)


def write_tab(name, text):
    with open(os.path.join(TAB, name), "w", encoding="utf-8") as f:
        f.write(text)
    print("  tablas/" + name)


def load(name):
    return sio.loadmat(os.path.join(DAT, name), squeeze_me=True)


# ----------------------------------------------------------------------------------------- geometria
def fig_geometria():
    k = load("cinematica.mat")
    B, A = k["B"] * 1e3, k["A"] * 1e3
    fig, ax = plt.subplots(figsize=(COL1 * 0.95, COL1 * 0.82))
    order_b = np.argsort(np.arctan2(B[:, 1], B[:, 0]))
    order_a = np.argsort(np.arctan2(A[:, 1], A[:, 0]))
    ax.fill(*B[order_b].T, color="#d9d9d9", alpha=0.6, lw=0)
    ax.plot(*np.vstack([B[order_b], B[order_b][:1]]).T, color="#4a4a4a", lw=1.0)
    ax.fill(*A[order_a].T, color="#9fb4cc", alpha=0.6, lw=0)
    ax.plot(*np.vstack([A[order_a], A[order_a][:1]]).T, color=C_SS, lw=1.0)
    for i in range(6):
        ax.plot([B[i, 0], A[i, 0]], [B[i, 1], A[i, 1]], color=C_PID, lw=0.9)
        ax.plot(*B[i, :2], "o", color="#4a4a4a", ms=3)
        ax.plot(*A[i, :2], "o", color=C_SS, ms=3)
        ax.annotate(f"$b_{i+1}$", B[i, :2], xytext=(1.12 * B[i, 0], 1.12 * B[i, 1]), ha="center", va="center", fontsize=7)
        ax.annotate(f"$s_{i+1}$", A[i, :2], xytext=(0.72 * A[i, 0], 0.72 * A[i, 1]), ha="center", va="center",
                    fontsize=7, color=C_SS)
    ax.annotate("", xy=(45, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=C_X, lw=0.9))
    ax.annotate("", xy=(0, 45), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=C_Y, lw=0.9))
    ax.text(48, -8, "$x$", color=C_X, fontsize=7)
    ax.text(-10, 46, "$y$", color=C_Y, fontsize=7)
    ax.set_aspect("equal")
    ax.set_xlabel("$x$ [mm]")
    ax.set_ylabel("$y$ [mm]")
    ax.set_xlim(-215, 215)
    ax.set_ylim(-200, 200)
    save(fig, "geometria.pdf")


def fig_condicion():
    k = load("cinematica.mat")
    xs, angs = k["xs"] * 1e3, np.degrees(k["angs"])
    fig, axs = plt.subplots(1, 2, figsize=(COL2 * 0.62, 1.95), constrained_layout=True)
    for ax, data, ext, lab in ((axs[0], k["cond_xy"], xs, ("$x$ [mm]", "$y$ [mm]")),
                               (axs[1], k["cond_rp"], angs, ("$\\phi$ [°]", "$\\theta$ [°]"))):
        im = ax.contourf(ext, ext, data, levels=14, cmap="Blues")
        cs = ax.contour(ext, ext, data, levels=6, colors="#333333", linewidths=0.4)
        ax.clabel(cs, fontsize=6, fmt="%.1f")
        ax.set_xlabel(lab[0])
        ax.set_ylabel(lab[1])
        ax.set_aspect("equal")
        ax.grid(False)
    fig.colorbar(im, ax=axs, shrink=0.85, label="$\\kappa(\\mathbf{J})$")
    save(fig, "condicion_jacobiano.pdf")


# ----------------------------------------------------------------------------------------- identificacion
def fig_identificacion():
    d = sio.loadmat(os.path.join(RES, "identificacion.mat"), squeeze_me=True)
    t2, y2 = d["t2"], d["y2"]
    sel = (t2 >= 4.0) & (t2 <= 7.0)            # 3 s del ensayo de validacion
    fig = plt.figure(figsize=(COL2, 2.05))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1.0], hspace=0.45, wspace=0.28)
    ax = fig.add_subplot(gs[:, 0])
    h = float(d["h_val"])
    ax.plot(t2[sel], 1e3 * y2[0, sel], color="#222222", lw=1.2, label="medida (prefiltrada)")
    ax.plot(t2[sel], 1e3 * d["ysa"][0, sel], "--", color=C_PID, lw=1.0,
            label=f"modelo analítico (FIT {d['FIT_an'][0]:.1f} %)")
    ax.plot(t2[sel], 1e3 * d["ys2"][0, sel], ":", color=C_SS, lw=1.4,
            label=f"ARX prefiltrado (FIT {d['FIT2'][0]:.1f} %)")
    ax.set_xlabel(f"tiempo en el ensayo de validación [s] (predicción a {h:g} s)")
    ax.set_ylabel("$l_1 - l_0$ [mm]")
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo - 0.55 * (hi - lo), hi)           # espacio inferior para la leyenda
    ax.legend(loc="lower left", ncol=2, fontsize=6)
    w = d["wf"]
    a1 = fig.add_subplot(gs[0, 1])
    a2 = fig.add_subplot(gs[1, 1], sharex=a1)
    for G, c, ls, lab in ((d["G_an"], C_PID, "-", "modelo analítico"), (d["G_id"], C_SS, "--", "ARX prefiltrado")):
        a1.semilogx(w, 20 * np.log10(np.abs(G) * 1e3), ls, color=c, label=lab)   # [mm por N]
        a2.semilogx(w, np.degrees(np.unwrap(np.angle(G))), ls, color=c)
    a1.set_ylabel("$|G_{11}|$ [dB, mm/N]")
    a2.set_ylabel("fase [°]")
    a2.set_xlabel("$\\omega$ [rad/s]")
    a1.legend(loc="lower left")
    plt.setp(a1.get_xticklabels(), visible=False)
    save(fig, "identificacion.pdf")

    rows = []
    for i in range(6):
        rows.append(f"{i+1} & {d['VAF1'][i]:.2f} & {d['FIT1'][i]:.2f} & {d['VAF2'][i]:.2f} & {d['FIT2'][i]:.2f} & "
                    f"{1e3*d['RMSE2'][i]:.4f} & {d['a_id'][i]:.5f} & {d['m_id'][i]:.3f} \\\\")
    rows.append("\\midrule")
    rows.append(f"Media & {np.mean(d['VAF1']):.2f} & {np.mean(d['FIT1']):.2f} & {np.mean(d['VAF2']):.2f} & "
                f"{np.mean(d['FIT2']):.2f} & {1e3*np.mean(d['RMSE2']):.4f} & {np.mean(d['a_id']):.5f} & "
                f"{np.mean(d['m_id']):.3f} \\\\")
    rows.append(f"Analítico & --- & --- & {np.mean(d['VAF_an']):.2f} & {np.mean(d['FIT_an']):.2f} & --- & "
                f"{np.mean(d['a_an']):.5f} & {np.mean(d['m_an']):.3f} \\\\")
    write_tab("identificacion.tex", "\n".join(rows) + "\n")


# ----------------------------------------------------------------------------------------- polos
def fig_polos():
    s, l = load("estados.mat"), load("lineal.mat")
    fig, ax = plt.subplots(figsize=(COL1, 2.35))
    ax.axvline(0, color="#999999", lw=0.6)
    pc = l["poles_cont"]
    ax.plot(pc.real, pc.imag, "x", color="#555555", ms=5, mew=1.0, label="lazo abierto")
    ax.plot(s["poles_d"].real, s["poles_d"].imag, "o", mfc="none", color=C_PID, ms=4, label="controlador, diseño")
    ax.plot(s["obs_d"].real, s["obs_d"].imag, "s", mfc="none", color=C_PID, ms=3.5, label="observador, diseño")
    ax.plot(s["poles_o"].real, s["poles_o"].imag, "o", color=C_SS, ms=3.5, label="controlador, óptimo")
    ax.plot(s["obs_o"].real, s["obs_o"].imag, "s", color=C_AUX, ms=3, label="observador, óptimo")
    ax.set_xlabel("Re($s$) [1/s]")
    ax.set_ylabel("Im($s$) [rad/s]")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), fontsize=6.3, ncol=3, columnspacing=0.8, handletextpad=0.3)
    ax.set_xlim(-48, 3)
    save(fig, "polos.pdf")


# ----------------------------------------------------------------------------------------- PID
def lazo_pata(e, Kp, Ki, Kd, w, leg=0):
    """Lazo de una pata con las otras cinco cerradas: L1 = C [G11 - G1r C (I + Grr C)^-1 Gr1]."""
    Ts, al = e.Ts, np.exp(-160.0 * e.Ts)
    B, A = e.B_pts, e.A_pts
    Jq = np.zeros((6, 6))
    for i in range(6):
        li = e.pose0[:3] + A[i] - B[i]
        n = li / np.linalg.norm(li)
        Jq[i] = np.r_[n, np.cross(A[i], n)]
    CL = Jq @ e.Cd
    r = [i for i in range(6) if i != leg]
    L = np.zeros(len(w), complex)
    for k, ww in enumerate(w):
        z = np.exp(1j * ww * Ts)
        G = CL @ np.linalg.solve(z * np.eye(12) - e.Ad, e.Bd)
        C = Kp + Ki * Ts / (z - 1) + Kd * (1 - al) * (z - 1) / (Ts * (z - al))
        L[k] = C * (G[leg, leg] - G[leg, r] @ (C * np.linalg.solve(np.eye(5) + C * G[np.ix_(r, r)], G[r, leg])))
    return L


def fig_pid(e, o):
    w = np.logspace(np.log10(0.5), np.log10(0.98 * np.pi / e.Ts), 1500)
    curves = [((e.Kp_leg[0], e.Ki_leg[0], e.Kd_leg[0]), "diseño", "#999999", ":"),
              ((o["pid"]["Kp"], o["pid"]["Ki"], o["pid"]["Kd"]), "óptimo", C_PID, "-")]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(COL1, 2.4), sharex=True)
    for g, name, col, st in curves:
        L = lazo_pata(e, *g, w)
        a1.semilogx(w, 20 * np.log10(np.abs(L)), st, color=col, label=name)
        a2.semilogx(w, np.degrees(np.angle(L)), st, color=col)
    a1.axhline(0, color="#777777", lw=0.6)
    a2.axhline(-180, color="#777777", lw=0.6)
    a2.axhline(180, color="#777777", lw=0.6)
    a1.set_ylabel("$|L_1|$ [dB]")
    a2.set_ylabel("fase [°]")
    a2.set_xlabel("$\\omega$ [rad/s]")
    a1.legend(loc="upper right", fontsize=6.5)
    save(fig, "pid_frecuencia.pdf")


# ----------------------------------------------------------------------------------------- simulador
def engine_with_gains():
    e = OctaveRealtimeControlEngine(ROOT)
    o = json.load(open(os.path.join(RES, "ganancias_optimas.json"), encoding="utf-8"))
    ps = (o["pid"]["Kp_scale"], o["pid"]["Ki_scale"], o["pid"]["Kd_scale"])
    return e, o, ps


def record(e, ctrl, scales, ref_fn, tau_fn, T):
    rows = []
    e.reset_state(init_z_err_mm=0.0)
    e.rng = np.random.default_rng(0)
    for k in range(int(round(T / e.Ts))):
        q = ref_fn(k * e.Ts)
        out = e.step_realtime(q, tau_fn(k), 0.0, ctrl, *scales, noise_sigma_mm=0.1)
        rows.append(np.r_[out["t"], out["pose"], q, out["F"]])
    return np.array(rows)


def fig_ensayo(e, ps):
    from octave_bridge import TEST_STEP, TEST_STEP_T, TEST_IMPULSE_T, TEST_IMPULSE_N, IMPULSE_DIR, IMPULSE_DUR
    ref = lambda t: e.pose0 + (TEST_STEP if t >= TEST_STEP_T - 1e-9 else 0.0)
    k_imp = int(round(TEST_IMPULSE_T / e.Ts))
    tau = lambda k: TEST_IMPULSE_N * IMPULSE_DIR if k_imp <= k < k_imp + round(IMPULSE_DUR / e.Ts) else np.zeros(6)
    data = {c: record(e, c, s, ref, tau, 5.0) for c, s in (("pid", ps), ("ss", (1, 1, 1)))}
    fig, axs = plt.subplots(3, 1, figsize=(COL1, 3.6), sharex=True, constrained_layout=True)
    r = data["pid"]
    axs[0].plot(r[:, 0], 1e3 * (r[:, 9] - e.h0), "--", color=C_REF, lw=0.9, label="referencia")
    for c, col, lab in (("pid", C_PID, "PID"), ("ss", C_SS, "realim. estados")):
        r = data[c]
        axs[0].plot(r[:, 0], 1e3 * (r[:, 3] - e.h0), color=col, label=lab)
        axs[1].plot(r[:, 0], 1e3 * (r[:, 7] - r[:, 1]), color=col, label=lab)
        axs[2].plot(r[:, 0], r[:, 13], color=col, lw=0.9)
    axs[0].set_ylabel("$\\Delta z$ [mm]")
    axs[1].set_ylabel("$e_x$ [mm]")
    axs[2].set_ylabel("$F_1$ [N]")
    axs[2].set_xlabel("tiempo [s]")
    axs[1].set_ylim(-4, 4)
    axs[0].legend(loc="lower right", ncol=1)
    for ax in axs:
        ax.axvspan(TEST_IMPULSE_T, TEST_IMPULSE_T + IMPULSE_DUR, color="#f1e3d3", lw=0)
    save(fig, "ensayo_normalizado.pdf")


def fig_seguimiento(e, ps):
    o_none = (1, 1, 1)
    ref = lambda t: reference_pose("sine_z", t, e.pose0)
    zero = lambda k: np.zeros(6)
    # PID de diseno vs optimo; realimentacion de estados optima
    curves = [("pid", o_none, "PID diseño", "#bbbbbb", ":"), ("pid", ps, "PID óptimo", C_PID, "-"),
              ("ss", o_none, "realim. estados óptima", C_SS, "-")]
    fig, ax = plt.subplots(figsize=(COL1, 1.85))
    r = record(e, "pid", ps, ref, zero, 6.0)
    ax.plot(r[:, 0], 1e3 * (r[:, 9] - e.h0), "--", color=C_REF, lw=0.9, label="referencia")
    for c, s, lab, col, ls in curves:
        r = record(e, c, s, ref, zero, 6.0)
        ax.plot(r[:, 0], 1e3 * (r[:, 3] - e.h0), ls, color=col, label=lab)
    ax.set_xlabel("tiempo [s]")
    ax.set_ylabel("$\\Delta z$ [mm]")
    ax.legend(loc="lower left", ncol=2, fontsize=6.3)
    ax.set_ylim(-24, 18)
    save(fig, "seguimiento_senoidal.pdf")


def tablas_simulador(e, o, ps):
    # Tabla 1 del protocolo (ensayo normalizado sobre la planta no lineal)
    rows = []
    for c, s, lab in (("pid", ps, "PID"), ("ss", (1, 1, 1), "Estados/observador")):
        r = e.table1_row(c, *s)
        restr = f"$F$ sat. {r['fsat_pct']:.1f}\\,\\%; $|F|_{{\\max}}$ {r['Fmax']:.1f}\\,N; carrera OK"
        rows.append(f"{lab} & {r['RMS_pos']:.3f} / {r['RMS_ang']:.3f} & {r['ITAE']:.4f} & {r['Ju']:.1f} & {restr} & "
                    f"+{r['robustez_pct']:.1f}\\,\\% & {r['cpu_us']:.0f} \\\\")
    write_tab("tabla1.tex", "\n".join(rows) + "\n")


# ----------------------------------------------------------------------------------------- calculo de los optimos
ESCENARIOS = ("Ensayo normalizado", "Carga sostenida", "Senoidal en $z$", "Maniobra 6 GDL", "Circular")


def tabla_calculo(ref, cur, name):
    """Terminos del criterio J por escenario: ITAE y Ju del diseno (0) y del optimo (*)."""
    from octave_bridge import EFFORT_WEIGHT
    rows, j0, js = [], 0.0, 0.0
    for esc, r, c in zip(ESCENARIOS, ref, cur):
        t0 = 1.0 + EFFORT_WEIGHT
        ts = c["ITAE"] / r["ITAE"] + EFFORT_WEIGHT * c["Ju"] / r["Ju"]
        j0, js = j0 + t0, js + ts
        rows.append(f"{esc} & {r['ITAE']:.4f} & {c['ITAE']:.4f} & {r['Ju']:.1f} & {c['Ju']:.1f} & "
                    f"{c['ITAE']/r['ITAE']:.3f} & {c['Ju']/r['Ju']:.3f} & {ts:.3f} \\\\")
    rows.append("\\midrule")
    rows.append(f"\\multicolumn{{7}}{{r}}{{$J = \\sum_s(\\cdot)$ (diseño: {j0:.2f})}} & \\textbf{{{js:.3f}}} \\\\")
    write_tab(name, "\n".join(rows) + "\n")
    return js


def tablas_calculo_optimos(e, o, ps):
    import copy
    import tempfile
    # PID: diseno (factores 1) frente al optimo
    js_pid = tabla_calculo(e.scenario_metrics("pid"), e.scenario_metrics("pid", *ps), "calculo_pid.tex")
    # Realimentacion de estados: matrices de diseno (polos x1, observador x1) frente a las optimas vigentes
    ed = copy.deepcopy(e)
    ed.matrices_mat_path = os.path.join(tempfile.mkdtemp(), "m.mat")
    ed.resynthesize_in_octave(1.0, 1.0)
    js_ss = tabla_calculo(ed.scenario_metrics("ss"), e.scenario_metrics("ss"), "calculo_ss.tex")
    # Verificacion de minimo local (+-10 % en cada parametro), registrada por optimizar_ganancias.py
    vp, vs = o["pid"]["verificacion"], o["estado"]["verificacion"]
    rows = [f"PID & {vp['J_optimo']:.4f} & $K_p$ & {vp['Kp x0.9']:.4f} & {vp['Kp x1.1']:.4f} \\\\",
            f" & & $K_i$ & {vp['Ki x0.9']:.4f} & {vp['Ki x1.1']:.4f} \\\\",
            f" & & $K_d$ & {vp['Kd x0.9']:.4f} & {vp['Kd x1.1']:.4f} \\\\",
            "\\midrule",
            f"Estados & {vs['J_optimo']:.4f} & polos & {vs['polos x0.9']:.4f} & {vs['polos x1.1']:.4f} \\\\",
            f" & & observador & {vs['observador x0.9']:.4f} & {vs['observador x1.1']:.4f} \\\\"]
    write_tab("verificacion.tex", "\n".join(rows) + "\n")
    print(f"  J recalculado: PID {js_pid:.4f} (registrado {o['pid']['J']:.4f}), estados {js_ss:.4f} (registrado {o['estado']['J']:.4f})")


# ----------------------------------------------------------------------------------------- banco no lineal
def banco_no_lineal():
    b = sio.loadmat(os.path.join(RES, "resultados_entrega1_octave.mat"), squeeze_me=True, struct_as_record=False)
    scen = b["results_e1"].scen
    rows = []
    names = {1: "Escalón $z$ 20 mm", 4: "Maniobra 6 GDL", 6: "Escalón + 10 N"}
    for s in scen:
        for c in s.ctrl:
            mt = c.metrics
            lab = "PID" if "PID" in c.name else "Estados"
            rows.append(f"{names[int(s.traj_id)]} & {lab} & {1e3*mt.RMS_pos:.3f} & {np.degrees(mt.RMS_ang):.3f} & "
                        f"{mt.ITAE:.4g} & {mt.Ju:.1f} & {mt.Fmax:.1f} & {int(mt.stroke_sat_count)} \\\\")
        rows.append("\\midrule")
    write_tab("banco_no_lineal.tex", "\n".join(rows[:-1]) + "\n")

    s = [x for x in scen if int(x.traj_id) == 4][0]
    t, qr = s.t, s.q_ref
    fig, axs = plt.subplots(3, 1, figsize=(COL1, 3.0), sharex=True, constrained_layout=True)
    for ax, i, lab, sc in ((axs[0], 0, "$x$ [mm]", 1e3), (axs[1], 2, "$z - h_0$ [mm]", 1e3), (axs[2], 3, "$\\phi$ [°]", None)):
        off = 0.27 if i == 2 else 0.0
        f = (lambda a: (a - off) * 1e3) if sc else (lambda a: np.degrees(a))
        ax.plot(t, f(qr[i]), "--", color=C_REF, lw=0.9, label="referencia")
        for c, col in zip(s.ctrl, (C_PID, C_SS)):
            ax.plot(t, f(c.pose[i]), color=col, label="PID" if "PID" in c.name else "realim. estados")
        ax.set_ylabel(lab)
    axs[0].legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3, fontsize=6.3)
    axs[2].set_xlabel("tiempo [s]")
    save(fig, "banco_6gdl.pdf")


def main():
    print("Generando figuras y tablas:")
    shutil.copy(os.path.join(RES, "captura_simulador.png"), os.path.join(FIG, "captura_simulador.png"))
    fig_geometria()
    fig_condicion()
    fig_identificacion()
    fig_polos()
    e, o, ps = engine_with_gains()
    fig_pid(e, o)
    fig_ensayo(e, ps)
    fig_seguimiento(e, ps)
    tablas_simulador(e, o, ps)
    tablas_calculo_optimos(e, o, ps)
    banco_no_lineal()


if __name__ == "__main__":
    main()
