#!/usr/bin/env python3
"""
generar_datos_calculo.py
Figuras, tablas y valores numericos (valores.tex) del documento desarrollo_matematico.tex, calculados con el mismo
codigo de la simulacion:
  - results/validacion_modelo.mat         (octave_engine/dynamics/validar_modelo_dinamico.m)
  - results/octave_control_matrices.mat  (exportar_matrices_control_octave.m)
  - results/ganancias_optimas.json       (scripts/optimizar_ganancias.py)
Uso, desde la raiz del proyecto:  python scripts/generar_datos_calculo.py
"""
import importlib.util
import os
import sys

import numpy as np
import scipy.io as sio
from scipy.linalg import expm
from scipy.integrate import quad_vec
from scipy.optimize import brentq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.join(ROOT, "desarrollo_matematico")
RES = os.path.join(ROOT, "results")
FIG = os.path.join(HERE, "figs")
TAB = os.path.join(HERE, "tablas")
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)

# Reutiliza las utilidades del informe (estilo de figuras, tablas del criterio) escribiendo en esta carpeta
spec = importlib.util.spec_from_file_location("gf", os.path.join(ROOT, "scripts", "generar_figuras_informe.py"))
gf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gf)
gf.FIG, gf.TAB = FIG, TAB
from octave_bridge import reference_pose  # noqa: E402

OMEGA_F = 160.0                      # polo del filtro derivativo [rad/s]


def num(x, d=3):
    """Numero con signo menos tipografico para LaTeX."""
    return f"{x:.{d}f}"


def bmat(Mx, d=4, tol=None):
    """Cuerpo de un bmatrix de LaTeX (filas separadas por \\\\)."""
    tol = 0.5 * 10**(-d) if tol is None else tol
    return " \\\\ ".join(" & ".join("0" if abs(v) < tol else f"{v:.{d}f}" for v in row) for row in np.atleast_2d(Mx))


def sci(x, d=2):
    m, e = f"{x:.{d}e}".split("e")
    return f"{m}\\times10^{{{int(e)}}}"


def fmt(x, d=3):
    return f"{x:.{d}f}"


def cplx(z, d=2):
    return f"{z.real:.{d}f}" + ("" if abs(z.imag) < 1e-9 else f" \\pm {abs(z.imag):.{d}f}j")


# ----------------------------------------------------------------------------------------- figuras del modelo
def fig_validacion():
    v = sio.loadmat(os.path.join(RES, "validacion_modelo.mat"), squeeze_me=True)
    t, F, L = v["man_t"], v["man_F"], v["man_L"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.4))
    for i in range(6):
        a1.plot(t, L[i] * 1e3, color=gf.C_LEGS[i] if hasattr(gf, "C_LEGS") else None, label=f"pata {i+1}")
        a2.plot(t, F[i])
    a1.set_xlabel("tiempo [s]"); a1.set_ylabel("longitud [mm]")
    a2.set_xlabel("tiempo [s]"); a2.set_ylabel("fuerza [N]")
    a1.legend(ncol=2, fontsize=6)
    gf.save(fig, "validacion_modelo.pdf")


def fig_trayectorias(e):
    import trayectorias_variacionales as tv
    t = np.linspace(0, tv.T_MOV, 400)
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(7.2, 2.3))
    a1.plot(t, 1 - tv.eps_fijos(t), color=gf.C_SS, label="extremos fijos")
    a1.plot(t, 1 - tv.eps_libre(t), color=gf.C_PID, ls="--", label="final libre")
    a1.plot(t, 1 - tv.eps_descuento(t), color=gf.C_AUX, label="descuento")
    dl, Tm = tv.tiempo_minimo(tv.TARGET)
    tt = np.linspace(0, Tm, 300)
    a1.plot(tt, [tv.l_bang_bang(x, dl) / dl for x in tt], color="#555555", ls=":", label="tiempo mínimo")
    a1.set_xlabel("tiempo [s]"); a1.set_ylabel(r"$1 - \varepsilon$"); a1.legend(fontsize=5.5)
    em = np.array([tv.e_multi(x, -tv.TARGET) + tv.TARGET for x in t])
    for k, (lab, sc) in enumerate((("x [mm]", 1e3), ("y [mm]", 1e3), ("z [mm]", 1e3), (r"$\phi$ [°]", 57.2958),
                                   (r"$\theta$ [°]", 57.2958), (r"$\psi$ [°]", 57.2958))):
        a2.plot(t, em[:, k] * sc, label=lab)
    a2.set_xlabel("tiempo [s]"); a2.legend(fontsize=5.5, ncol=2)
    T, c = tv.intercepcion()
    tc = np.linspace(0, 2.2, 400)
    a3.plot(tc, tv.SIN_A * np.sin(tv.SIN_W * tc) * 1e3, color=gf.C_REF, lw=1.4, label=r"$\phi(t)$")
    a3.plot(tc, [tv.z_curva(x) * 1e3 for x in tc], color=gf.C_PID, ls="--", label="referencia")
    a3.plot([T], [tv.z_curva(T) * 1e3], "o", color=gf.C_SS, ms=4)
    a3.set_xlabel("tiempo [s]"); a3.set_ylabel("z [mm]"); a3.legend(fontsize=5.5)
    gf.save(fig, "trayectorias_variacionales.pdf")


def fig_frontera():
    import trayectorias_variacionales as tv
    t = np.linspace(0, tv.T_ONDA, 500)
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    r = np.array([tv.onda(x)[0][0] for x in t]) * 1e3
    ax.plot(t, r, color="#bbbbbb", lw=2.2, label="onda deseada $r(t)$")
    ax.plot(t, tv.kappa() * r, color="#888888", ls=":", lw=1.0, label=r"$\kappa\,r(t)$")
    for caso, col, ls in (("A", gf.C_SS, "-"), ("B", gf.C_PID, "--"), ("C", gf.C_AUX, "-."), ("D", "#2e7d32", ":")):
        x = np.array([tv.x_frontera(caso, ti)[0][0] for ti in t]) * 1e3
        ini, fin = tv.CASOS[caso]
        ax.plot(t, x, color=col, ls=ls, lw=1.3,
                label=f"{caso}: inicio {'conocido' if ini else 'libre'}, final {'conocido' if fin else 'libre'}")
    ax.set_xlabel("tiempo [s]"); ax.set_ylabel("$z$ [mm]")
    ax.legend(fontsize=6, ncol=3, loc="lower left")
    gf.save(fig, "casos_frontera.pdf")


# ----------------------------------------------------------------------------------------- simulacion
REFERENCIAS = (("sine_z", "Senoidal en $z$"), ("coupled_6dof", "Maniobra 6 GDL"), ("circle_3d", "Circular"),
               ("bc_A", "Frontera A: inicio y final conocidos"), ("bc_B", "Frontera B: inicio conocido, final libre"),
               ("bc_C", "Frontera C: inicio libre, final conocido"), ("bc_D", "Frontera D: ambos libres"),
               ("var_fixed", "Variacional: extremos fijos"),
               ("min_time", "Tiempo mínimo (Pontryagin)"), ("var_curve", "Variacional: curva terminal"))


def seguimiento(e, ps):
    def rms(ctrl, scales, mode):
        q0 = e.pose0
        d = gf.record(e, ctrl, scales, lambda t: reference_pose(mode, t, q0), lambda k: np.zeros(6), 12.0)
        err = d[:, 1:4] - d[:, 7:10]
        return np.sqrt(np.mean(np.sum(err**2, axis=1))) * 1e3, d
    rows, data = [], {}
    for mode, lab in REFERENCIAS:
        vals = []
        for c, s in (("pid", ps), ("ss", (1, 1, 1))):
            r, d = rms(c, s, mode)
            vals.append(r)
            if mode == "var_fixed":
                data[c] = d
        rows.append(lab + " & " + " & ".join(f"{v:.2f}" for v in vals) + " \\\\")
    gf.write_tab("seguimiento.tex", "\n".join(rows) + "\n")
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.0, 3.2), sharex=True)
    h0 = e.h0
    d = data["pid"]
    a1.plot(d[:, 0], (d[:, 9] - h0) * 1e3, color=gf.C_REF, lw=1.6, label="referencia variacional (extremos fijos)")
    for c, col, lab in (("pid", gf.C_PID, "PID"), ("ss", gf.C_SS, "realimentación de estados")):
        d = data[c]
        a1.plot(d[:, 0], (d[:, 3] - h0) * 1e3, color=col, lw=0.9, label=lab)
        a2.plot(d[:, 0], (d[:, 9] - d[:, 3]) * 1e3, color=col, lw=0.9)
    a1.set_ylabel(r"$\Delta z$ [mm]"); a1.legend(fontsize=6, ncol=3)
    a2.set_ylabel(r"error en $z$ [mm]"); a2.set_xlabel("tiempo [s]")
    gf.save(fig, "seguimiento_variacional.pdf")


def tabla_corta():
    """Version reducida del calculo del criterio del PID (escenario, cocientes y termino) para la presentacion."""
    rows = []
    for line in open(os.path.join(TAB, "calculo_pid.tex"), encoding="utf-8"):
        c = [x.strip() for x in line.replace("\\\\", "").split("&")]
        if len(c) == 8:
            rows.append(f"{c[0]} & {c[5]} & {c[6]} & {c[7]} \\\\")
        elif "sum_s" in line:
            rows += ["\\midrule", "\\multicolumn{3}{r}{$J$ (diseño: 6.25)} & " + line.split("&")[-1].strip()]
    gf.write_tab("calculo_pid_corta.tex", "\n".join(rows) + "\n")


def tabla1(e, ps):
    rows = []
    for c, s, lab in (("pid", ps, "PID"), ("ss", (1, 1, 1), "Realimentación de estados")):
        r = e.table1_row(c, *s)
        rows.append(f"{lab} & {r['RMS_pos']:.3f} / {r['RMS_ang']:.3f} & {r['ITAE']:.4f} & {r['Ju']:.1f} & "
                    f"+{r['robustez_pct']:.1f}\\,\\% \\\\")
    gf.write_tab("tabla1.tex", "\n".join(rows) + "\n")


# ----------------------------------------------------------------------------------------- valores del control
def jacobiano(e):
    q = e.pose0
    p = q[:3]
    J = np.zeros((6, 6))
    for i in range(6):
        pi_ = e.A_pts[i]          # R = I en la pose nominal
        li = p + pi_ - e.B_pts[i]
        n = li / np.linalg.norm(li)
        J[i] = np.r_[n, np.cross(pi_, n)]
    return J


def modelo_pata(e):
    """Masa y rigidez del modelo desacoplado de una pata en el espacio articular:
    Lambda dl'' = dF + K_l dl, con Lambda = J^-T M0 J^-1 y K_l = -J^-T (M0 A21) J^-1 (rigidez geometrica)."""
    Jq = jacobiano(e)
    Ji = np.linalg.inv(Jq)
    M0 = tv_model()["M0"]
    A21 = e.A[6:, :6]
    Lam = Ji.T @ M0 @ Ji
    Kl = Ji.T @ (M0 @ A21) @ Ji
    return Jq, M0, A21, Lam, Kl


def tv_model():
    import trayectorias_variacionales as tv
    return tv._MODEL


def margenes_lazo(e, Kp, Ki, Kd, leg=0):
    """Margenes exactos del lazo de una pata con las otras cinco cerradas (planta MIMO discreta)."""
    Ts = e.Ts
    al = np.exp(-OMEGA_F * Ts)
    Jq = jacobiano(e)
    CL = Jq @ e.Cd
    w = np.logspace(np.log10(0.5), np.log10(0.98 * np.pi / Ts), 4000)
    r = [i for i in range(6) if i != leg]
    L = np.zeros(len(w), complex)
    for k, ww in enumerate(w):
        z = np.exp(1j * ww * Ts)
        G = CL @ np.linalg.solve(z * np.eye(12) - e.Ad, e.Bd)
        C = Kp + Ki * Ts / (z - 1) + Kd * (1 - al) * (z - 1) / (Ts * (z - al))
        G1 = G[leg, leg] - G[leg, r] @ (C * np.linalg.solve(np.eye(5) + C * G[np.ix_(r, r)], G[r, leg]))
        L[k] = C * G1
    mag, ph = np.abs(L), np.angle(L)
    ic = np.where((mag[:-1] >= 1) & (mag[1:] < 1))[0][-1]
    PM = (np.degrees(np.pi + ph[ic]) + 180) % 360 - 180
    ip = np.where((np.sin(ph[:-1]) * np.sin(ph[1:]) <= 0) & (np.cos(ph[:-1]) < 0))[0]
    GM = np.min(np.abs(20 * np.log10(mag[ip]))) if len(ip) else np.inf
    return w[ic], PM, GM, np.max(1 / np.abs(1 + L))


def valores_modelo(e):
    Jq, M0, A21, Lam, Kl = modelo_pata(e)
    Ts = e.Ts
    v = {"Ts": f"{Ts*1e3:.0f}", "Fmax": f"{e.Fmax:.0f}", "Fzero": num(e.u0[0], 3)}
    v["MzeroMat"] = bmat(M0, 4)
    ev = np.sort(np.linalg.eigvalsh(M0))
    v["MzeroEigs"] = ",\\allowbreak\\ ".join(f"{x:.4f}" for x in ev)
    v["MzeroEigMin"] = num(ev[0], 4)
    v["AtwoOneMat"] = bmat(A21, 2, 5e-3)
    v["BtwoMat"] = bmat(e.B[6:, :], 3, 5e-4)
    lam = np.linalg.eigvals(e.A)
    sig = np.sort(np.unique(np.round(np.abs(lam.real), 2)))[::-1]
    mult = [int(np.sum(np.isclose(np.abs(lam.real), s_, atol=6e-3))) // 2 for s_ in sig]
    v["PolosLA"] = ",\\allowbreak\\ ".join(f"\\pm{s_:.2f}" + (f"\\ (\\times{m})" if m > 1 else "") for s_, m in zip(sig, mult))
    v["SigmaMax"] = num(sig[0], 2)
    eA21 = np.sort(np.linalg.eigvals(A21).real)[::-1]
    v["EigAtwoOne"] = ",\\allowbreak\\ ".join(f"{x:.2f}" for x in eA21)
    # espacio de las patas
    v["LamEigs"] = ",\\allowbreak\\ ".join(f"{x:.3f}" for x in np.sort(np.linalg.eigvalsh(0.5 * (Lam + Lam.T))))
    v["KlEigs"] = ",\\allowbreak\\ ".join(f"{x:.1f}" for x in np.sort(np.linalg.eigvalsh(0.5 * (Kl + Kl.T)))[::-1])
    v["KlMax"] = num(np.max(np.linalg.eigvalsh(0.5 * (Kl + Kl.T))), 1)
    m, k = Lam[0, 0], Kl[0, 0]
    v["mLeg"], v["kLeg"] = num(m, 4), num(k, 2)
    v["mEff"] = num(1 / (Jq @ np.linalg.solve(M0, Jq.T))[0, 0], 4)
    sg = np.sqrt(k / m)
    c = np.cosh(sg * Ts)
    v.update(sigmaLeg=num(sg, 3), cLeg=f"{c:.7f}", gLeg=sci((c - 1) / k, 4), zUnst=f"{np.exp(sg*Ts):.6f}",
             zSt=f"{np.exp(-sg*Ts):.6f}")
    # caso variacional multivariable: K = a^2 diag(M0), W^2 = M0^-1 K
    import trayectorias_variacionales as tv
    w, _ = tv.omega_multi()
    mxt = abs(M0[0, 4])
    v["MxTheta"] = sci(mxt, 2)
    v["muMulti"] = sci(mxt**2 / (M0[0, 0] * M0[4, 4]), 2)
    v["omegaMulti"] = ",\\allowbreak\\ ".join(f"{x:.4f}" for x in np.sort(w))
    # contribucion a diag(M0)
    v["MdiagPlat"] = " & ".join(f"{x:g}" for x in (2, 2, 2, 0.01125, 0.01125, 0.0225))
    v["MdiagPatas"] = " & ".join(f"{x:.4f}" for x in np.diag(M0) - np.array([2, 2, 2, 0.01125, 0.01125, 0.0225]))
    return v, (m, k, sg, c)


def valores_pid(e, o, leg):
    m, k, sg, c = leg
    Ts = e.Ts
    al = np.exp(-OMEGA_F * Ts)
    D = lambda Kd: Kd * (1 - al) / Ts
    Kp, Ki, Kd = o["pid"]["Kp"], o["pid"]["Ki"], o["pid"]["Kd"]
    v = {"KpOpt": fmt(Kp, 1), "KiOpt": fmt(Ki, 1), "KdOpt": fmt(Kd, 2),
         "spOpt": fmt(o["pid"]["Kp_scale"]), "siOpt": fmt(o["pid"]["Ki_scale"]), "sdOpt": fmt(o["pid"]["Kd_scale"]),
         "TiOpt": fmt(Kp / Ki, 3), "TdOpt": fmt(1e3 * Kd / Kp, 2), "JpidOpt": fmt(o["pid"]["J"]),
         "alphaD": f"{al:.4f}", "Dcoef": num((1 - al) / Ts, 2)}
    n2, n1, n0 = Kp + D(Kd), -(1 + al) * Kp + Ki * Ts - 2 * D(Kd), al * Kp - al * Ki * Ts + D(Kd)
    v.update(nTwo=fmt(n2, 1), nOne=fmt(n1, 1), nZero=fmt(n0, 1))
    # margenes exactos (lazo de una pata con las demas cerradas)
    wc, PM, GM, Ms = margenes_lazo(e, Kp, Ki, Kd)
    v.update(wcPid=fmt(wc, 1), PMPid=fmt(PM, 1), GMPid=fmt(GM, 1), MsPid=fmt(Ms, 2))
    # calculo a mano con el modelo desacoplado G(s) = 1/(m s^2 - k) y retardo de medio periodo
    Cs = lambda w: Kp + Ki / (1j * w) + Kd * 1j * w / (1 + 1j * w / OMEGA_F)
    f = lambda w: abs(Cs(w)) / (m * w**2 + k) - 1
    wh = brentq(f, 1.0, 0.98 * np.pi / Ts)
    v.update(wcHand=fmt(wh, 1), GwcMag=sci(1 / (m * wh**2 + k), 3), GwcAng=fmt(-180 - np.degrees(wh * Ts / 2), 1),
             CwcMag=f"{abs(Cs(wh)):.0f}", CwcAng=f"{np.degrees(np.angle(Cs(wh))):+.1f}",
             PMhand=fmt(np.degrees(np.angle(Cs(wh))) - np.degrees(wh * Ts / 2), 1))
    # Routh continuo (sin filtro): m s^3 + Kd s^2 + (Kp - k) s + Ki
    v.update(RouthA=fmt(Kd, 2), RouthB=fmt(Kp - k, 1), RouthC=fmt(Ki, 1), RouthL=f"{Kd*(Kp-k):.0f}", RouthR=f"{m*Ki:.0f}")
    r = np.roots([m, Kd, Kp - k, Ki])
    v["RootsCont"] = ",\\allowbreak\\ ".join(cplx(z_) for z_ in sorted(r, key=lambda z_: z_.real) if z_.imag >= -1e-12)
    # polinomio caracteristico discreto (una pata)
    P1, P2, Pp = np.poly1d([1, -1]), np.poly1d([1, -al]), np.poly1d([1, -2 * c, 1])
    ch = P1 * P2 * Pp * float(k) + np.poly1d([1, 1]) * np.poly1d([n2, n1, n0]) * float(c - 1)
    cc = ch.c / ch.c[0]
    v["charPoly"] = f"z^4 {cc[1]:+.4f}z^3 {cc[2]:+.4f}z^2 {cc[3]:+.4f}z {cc[4]:+.4f}"
    rr = sorted(ch.r, key=lambda z_: -abs(z_))
    v["charRoots"] = ",\\allowbreak\\ ".join(cplx(z_, 4) for z_ in rr if z_.imag >= -1e-12)
    v["rhoSiso"] = fmt(max(abs(ch.r)), 4)
    # lazo MIMO completo: [dx(12), xI(6), e(k-1)(6), d(k-1)(6)]
    Jq = jacobiano(e)
    Ad, Bd, Cd = e.Ad, e.Bd, e.Cd
    E = -Jq @ Cd
    Z, I6 = np.zeros, np.eye(6)
    b = (1 - al) / Ts
    dk = [b * E, Z((6, 6)), -b * I6, al * I6]
    uk = [Kp * E + Kd * dk[0], Ki * I6, Kd * dk[2], Kd * dk[3]]
    Acl = np.block([
        [Ad + Bd @ uk[0], Bd @ uk[1], Bd @ uk[2], Bd @ uk[3]],
        [Ts * E, I6, Z((6, 6)), Z((6, 6))],
        [E, Z((6, 6)), Z((6, 6)), Z((6, 6))],
        [dk[0], dk[1], dk[2], dk[3]]])
    v["rhoPid"] = fmt(max(abs(np.linalg.eigvals(Acl))), 4)
    v["KpBase"], v["KiBase"], v["KdBase"] = (f"{x:.0f}" for x in (e.Kp_leg[0], e.Ki_leg[0], e.Kd_leg[0]))
    return v


def valores_frontera():
    """Constantes, extremos y costo de los cuatro casos de frontera (z en mm)."""
    import trayectorias_variacionales as tv
    sc = np.array([1e3])
    v = {"aOnda": f"{tv.A_ONDA:g}", "TOnda": f"{tv.T_ONDA:g}", "fOnda": f"{tv.F_ONDA:g}", "wOnda": num(tv.W_ONDA, 4),
         "AOnda": f"{tv.AMP_ONDA[0] * 1e3:g}", "rzZero": num(tv.onda(0.0)[0][0] * 1e3, 3),
         "kappaOnda": num(tv.kappa(), 4), "EOnda": sci(np.exp(-tv.A_ONDA * tv.T_ONDA), 3), "TPre": f"{tv.T_PRE:g}"}
    (r0, dr0), (rT, drT) = tv.onda(0.0), tv.onda(tv.T_ONDA)
    v.update(rzT=num(rT[0] * 1e3, 3), drzZero=num(abs(dr0[0]) * 1e3 if abs(dr0[0]) < 1e-9 else dr0[0] * 1e3, 3),
             drzT=num(abs(drT[0]) * 1e3 if abs(drT[0]) < 1e-9 else drT[0] * 1e3, 3))
    rows = []
    for caso, (ini, fin) in tv.CASOS.items():
        C = tv.coef_frontera(caso)
        (x0, d0), (xT, dT) = tv.x_frontera(caso, 0.0), tv.x_frontera(caso, tv.T_ONDA)
        J = tv.costo_frontera(caso) * sc**2
        v[f"CUno{caso}"], v[f"CDos{caso}"] = (num(abs(x) if abs(x) < 5e-5 else x, 4) for x in (C[0, 0] * 1e3, C[1, 0] * 1e3))
        v[f"xZero{caso}"], v[f"xT{caso}"] = num(abs(x0[0]) * 1e3 if abs(x0[0]) < 1e-9 else x0[0] * 1e3, 3), num(abs(xT[0]) * 1e3 if abs(xT[0]) < 1e-9 else xT[0] * 1e3, 3)
        v[f"J{caso}"] = num(J[0], 1)
        cond = ("$x(0) = 0$" if ini else "$\\dot x(0) = 0$") + ", " + ("$x(T) = 0$" if fin else "$\\dot x(T) = 0$")
        rows.append(f"{caso} & {'conocido' if ini else 'libre'} & {'conocido' if fin else 'libre'} & {cond} & "
                    + " / ".join(f"{x + 0.0:.2f}".replace("-0.00", "0.00") for x in x0 * sc) + " & "
                    + " / ".join(f"{x + 0.0:.2f}".replace("-0.00", "0.00") for x in xT * sc)
                    + " & " + " / ".join(f"{x:.1f}" for x in J) + " \\\\")
    gf.write_tab("frontera.tex", "\n".join(rows) + "\n")
    return v


def valores_ident():
    d = sio.loadmat(os.path.join(RES, "identificacion.mat"), squeeze_me=True)
    return {"VAFunoMedia": fmt(np.mean(d["VAF1"]), 2), "FITunoMedia": fmt(np.mean(d["FIT1"]), 1),
            "VAFdosMedia": fmt(np.mean(d["VAF2"]), 2), "FITdosMedia": fmt(np.mean(d["FIT2"]), 1),
            "FITdosMin": fmt(np.min(d["FIT2"]), 1), "FITanMedia": fmt(np.mean(d["FIT_an"]), 1),
            "IdHval": f"{float(d['h_val']):g}", "aIdMin": fmt(np.min(d["a_id"]), 5), "aIdMax": fmt(np.max(d["a_id"]), 5),
            "aAn": fmt(np.mean(d["a_an"]), 5),
            "mIdMin": fmt(np.min(d["m_id"]), 3), "mIdMax": fmt(np.max(d["m_id"]), 3), "mAn": fmt(np.mean(d["m_an"]), 3),
            "IdKp": f"{float(d['Kp_id']):.0f}", "IdKd": f"{float(d['Kd_id']):.0f}",
            "IdLmin": fmt(1e3 * np.min(d["L"]), 1), "IdLmax": fmt(1e3 * np.max(d["L"]), 1),
            "IdSigma": f"{1e3 * float(d.get('sigma_L', 0.0)):g}",
            "mIdDirMin": fmt(np.min(d["m_id_directo"]), 3), "mIdDirMax": fmt(np.max(d["m_id_directo"]), 3),
            "IdTsId": f"{1e3 * float(d['Ts_id']):g}", "IdFc": f"{float(d['fc_id']):g}", "IdD": f"{int(d['D_id'])}",
            "mErrMax": fmt(100 * np.max(np.abs(d["m_id"] / np.mean(d["m_an"]) - 1)), 0)}


def valores_pontryagin(e):
    import trayectorias_variacionales as tv
    Jq, M0, *_ = modelo_pata(e)
    m = 1 / (Jq @ np.linalg.solve(M0, Jq.T))[0, 0]
    dl, T = 0.9509 * 0.015, 2.0
    u0 = 6 * m * dl / T**2
    v = {"dlEn": num(dl * 1e3, 2), "uEnMax": num(u0 * 1e3, 2), "EEn": sci(12 * m**2 * dl**2 / T**3, 3),
         "aMax": f"{tv.A_MAX:g}"}
    dlt, Tm = tv.tiempo_minimo(tv.TARGET)
    v.update(dlTm=num(dlt * 1e3, 2), TMin=num(Tm, 3), tSw=num(Tm / 2, 3), vPico=num(tv.A_MAX * Tm / 2 * 1e3, 1),
             uBang=num(m * tv.A_MAX * 1e3, 1))
    return v


def valores_ss(e, o):
    alpha, beta = o["estado"]["pole_scale"], o["estado"]["obs_scale"]
    pc, po = 3 * alpha * complex(-8.5, 3.0), 3 * beta * complex(-26.0, 6.0)
    v = {"alphaOpt": fmt(alpha, 3), "betaOpt": fmt(beta, 4), "JssOpt": fmt(o["estado"]["J"]),
         "PoloCtrl": cplx(pc), "PoloObs": cplx(po), "tsCtrl": f"{4e3/abs(pc.real):.0f}",
         "cofErr": f"{e.err_Nr_cof:.1e}".replace("e-", "\\times10^{-") + "}",
         "sepErr": f"{e.sep_error:.1e}".replace("e-", "\\times10^{-") + "}",
         "normK": f"{np.linalg.norm(e.K_pi):.3g}".replace("e+0", "\\times10^{") + "}"}
    Ad, Bd, Cd, K, Ki, Ld = e.Ad, e.Bd, e.Cd, e.K_pi, e.Ki_pi, e.Ld
    I12, I6, Z = np.eye(12), np.eye(6), np.zeros
    TS = e.Ts
    Mx, Mh = -K @ Ld @ Cd, -K @ (I12 - Ld @ Cd)
    Acl = np.block([[Ad + Bd @ Mx, Bd @ Ki, Bd @ Mh],
                    [-TS * Cd, I6, Z((6, 12))],
                    [Ad @ Ld @ Cd + Bd @ Mx, Bd @ Ki, Ad @ (I12 - Ld @ Cd) + Bd @ Mh]])
    v["rhoSS"] = fmt(max(abs(np.linalg.eigvals(Acl))), 4)
    Actrl = np.block([[Ad - Bd @ K, Bd @ Ki], [-TS * Cd, I6]])
    v["rhoCtrlBlock"] = fmt(max(abs(np.linalg.eigvals(Actrl))), 4)
    m = sio.loadmat(os.path.join(RES, "octave_control_matrices.mat"), squeeze_me=True)
    des, imp = np.sort_complex(m["obs_poles_disc"]), np.sort_complex(m["obs_poles_impl"])
    v["obsPoleErr"] = f"{np.max(np.abs(des - imp)):.1e}".replace("e-", "\\times10^{-") + "}"
    v["obsZmax"] = fmt(np.max(np.abs(des)), 4)
    dp, dv = np.diag(Ld[:6]), np.diag(Ld[6:])
    v["LdPos"] = f"{dp.min():.3f} y {dp.max():.3f}"
    v["LdVel"] = f"{dv.min():.3f} y {dv.max():.3f}"
    return v


def main():
    print("Generando datos del documento:")
    fig_validacion()
    e, o, ps = gf.engine_with_gains()
    fig_trayectorias(e)
    fig_frontera()
    gf.tablas_calculo_optimos(e, o, ps)
    tabla_corta()
    tabla1(e, ps)
    seguimiento(e, ps)
    v, leg = valores_modelo(e)
    v.update(valores_pid(e, o, leg))
    v.update(valores_ss(e, o))
    v.update(valores_frontera())
    v.update(valores_pontryagin(e))
    v.update(valores_ident())
    with open(os.path.join(HERE, "valores.tex"), "w", encoding="utf-8") as f:
        f.write("% Generado por scripts/generar_datos_calculo.py; no editar a mano\n")
        for k, val in v.items():
            f.write(f"\\newcommand{{\\{k}}}{{{val}}}\n")
    print("  valores.tex")


if __name__ == "__main__":
    main()
