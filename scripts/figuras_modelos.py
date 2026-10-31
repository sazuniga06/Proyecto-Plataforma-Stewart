"""
Figuras de los modelos para el informe (informe/figs/):
  - modelo_lazo_abierto.pdf   respuesta libre de la planta no lineal frente al modelo lineal (inestabilidad)
  - modelo_frecuencia.pdf     valores singulares de G(jw) = C (jwI - A)^-1 B y respuesta de una pata
  - lazo_cerrado_lineal.pdf   escalones en el modelo lineal discreto: realimentacion de estados y PID por pata
  - observador_error.pdf      convergencia del error del estimador actual
Entradas: results/octave_control_matrices.mat (modelo y matrices de diseno en el optimo) y
          results/ganancias_optimas.json (PID optimo). Ejecutar desde la raiz del proyecto.
"""
import json
import os
import sys

import numpy as np
import scipy.io as sio
from scipy.linalg import expm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "simulator_gui"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from planta_no_lineal import PlantaNoLineal  # noqa: E402
import generar_figuras_informe as gf  # noqa: E402  (estilo, colores y save)

plt = gf.plt
m = sio.loadmat(os.path.join(ROOT, "results", "octave_control_matrices.mat"), squeeze_me=True)
A, B, Cy = m["A"], m["B"], m["C"]
Ad, Bd, Cd, Ts = m["Ad"], m["Bd"], m["Cd"], float(m["Ts"])
x0, u0 = np.asarray(m["x0"], float), np.asarray(m["u0"], float)

# ---------------------------------------------------------------- 1. lazo abierto: no lineal frente a lineal
pl = PlantaNoLineal(m)
dx0 = np.zeros(12); dx0[0] = 1e-3; dx0[5] = np.deg2rad(0.2)      # 1 mm en x y 0.2 grados en guinada
t = np.arange(0, 0.5 + Ts / 2, Ts)
xn = np.empty((t.size, 12)); xn[0] = x0 + dx0
for k in range(1, t.size):
    xn[k] = pl.paso_rk4(xn[k - 1], u0, Ts)
xl = np.array([x0 + expm(A * tk) @ dx0 for tk in t])
fig, ax = plt.subplots(1, 2, figsize=(gf.COL1, 1.55))
ax[0].plot(t, 1e3 * (xn[:, 0] - x0[0]), color=gf.C_PID, label="no lineal")
ax[0].plot(t, 1e3 * (xl[:, 0] - x0[0]), "--", color=gf.C_SS, label="lineal")
ax[0].set(xlabel="tiempo [s]", ylabel=r"$\delta x$ [mm]")
ax[1].plot(t, np.rad2deg(xn[:, 5] - x0[5]), color=gf.C_PID)
ax[1].plot(t, np.rad2deg(xl[:, 5] - x0[5]), "--", color=gf.C_SS)
ax[1].set(xlabel="tiempo [s]", ylabel=r"$\delta\psi$ [$^\circ$]")
ax[0].legend(loc="upper left")
fig.tight_layout(w_pad=0.6)
gf.save(fig, "modelo_lazo_abierto.pdf")

# ---------------------------------------------------------------- 2. respuesta en frecuencia del modelo lineal
w = np.logspace(-1, 3, 400)
sv = np.array([np.linalg.svd(Cy @ np.linalg.solve(1j * wk * np.eye(12) - A, B), compute_uv=False) for wk in w])
mleg, kleg = 4.3607, 185.26
Gleg = 1.0 / np.abs(mleg * (1j * w) ** 2 - kleg)
fig, ax = plt.subplots(figsize=(gf.COL1, 1.7))
for i in range(6):
    ax.loglog(w, sv[:, i], color=gf.C_SS, lw=0.9, label=r"$\sigma_i(\mathbf{G}(j\omega))$" if i == 0 else None)
ax.loglog(w, Gleg, "--", color=gf.C_PID, label=r"$|G(j\omega)|$ de una pata")
ax.axvline(7.62, color=gf.C_REF, lw=0.6, ls=":")
ax.set(xlabel=r"$\omega$ [rad/s]", ylabel="ganancia [m/N, rad/(N m)]")
ax.legend(loc="lower left")
gf.save(fig, "modelo_frecuencia.pdf")

# ---------------------------------------------------------------- 3. lazo cerrado lineal discreto
K, Ki, Nr, Ld = m["K_pi"], m["Ki_pi"], m["Nr_pi"], m["Ld"]
N = int(1.0 / Ts) + 1
tt = np.arange(N) * Ts


def escalon_estados(r):
    """Estimador actual + integrador rectangular, como en el informe (sin saturacion: modelo lineal)."""
    x = np.zeros(12); xp = np.zeros(12); z = np.zeros(6); y = np.empty((N, 6)); u = np.empty((N, 6))
    for k in range(N):
        yk = Cd @ x
        xc = xp + Ld @ (yk - Cd @ xp)
        uk = -K @ xc + Ki @ z + Nr @ r
        y[k], u[k] = yk, uk
        z = z + Ts * (r - yk)
        x = Ad @ x + Bd @ uk
        xp = Ad @ xc + Bd @ uk
    return y, u


g = json.load(open(os.path.join(ROOT, "results", "ganancias_optimas.json")))
pid = g.get("pid", g)
Kp = float(pid.get("Kp", 4556.7)); Kip = float(pid.get("Ki", 18445.7)); Kdp = float(pid.get("Kd", 105.43))
sig = np.sqrt(kleg / mleg)
Al = np.array([[0, 1], [sig ** 2, 0]]); Bl = np.array([0, 1 / mleg])
Mz = expm(np.block([[Al, Bl[:, None]], [np.zeros((1, 3))]]) * Ts)
Ald, Bld = Mz[:2, :2], Mz[:2, 2]
alpha = np.exp(-160 * Ts)


def escalon_pid(r):
    x = np.zeros(2); xi = 0.0; d = 0.0; e_ant = r; y = np.empty(N)
    for k in range(N):
        e = r - x[0]
        d = alpha * d + (1 - alpha) * (e - e_ant) / Ts
        F = Kp * e + Kip * xi + Kdp * d
        y[k] = x[0]
        xi += Ts * e; e_ant = e
        x = Ald @ x + Bld * F
    return y


fig, ax = plt.subplots(1, 2, figsize=(gf.COL2, 1.75))
for j, (nombre, amp, esc, col) in enumerate([(r"$x$", 5e-3, 1e3, gf.C_X), (r"$z$", 5e-3, 1e3, gf.C_Z),
                                              (r"$\psi$", np.deg2rad(2), 180 / np.pi, gf.C_Y)]):
    idx = [0, 2, 5][j]
    r = np.zeros(6); r[idx] = amp
    y, _ = escalon_estados(r)
    ax[0].plot(tt, y[:, idx] / amp, color=col, label=nombre)
ax[0].set(xlabel="tiempo [s]", ylabel="salida / referencia", title="Realimentación de estados (MIMO)")
ax[0].legend(loc="lower right")
yl = escalon_pid(1e-3)
ax[1].plot(tt, yl / 1e-3, color=gf.C_PID)
ax[1].set(xlabel="tiempo [s]", ylabel=r"$\delta l/\delta l_r$", title="PID sobre una pata")
fig.tight_layout(w_pad=1.2)
gf.save(fig, "lazo_cerrado_lineal.pdf")

# ---------------------------------------------------------------- 4. error del estimador actual
Ao = Ad - Ld @ Cd @ Ad
e = np.zeros(12); e[6:9] = 0.01; e[9:] = np.deg2rad(1.0)          # error inicial en velocidades
en = np.empty(N)
for k in range(N):
    en[k] = np.linalg.norm(e[6:])
    e = Ao @ e
rho = max(abs(np.linalg.eigvals(Ao)))
fig, ax = plt.subplots(figsize=(gf.COL1, 1.5))
ax.semilogy(tt, en / en[0], color=gf.C_SS, label=r"$\Vert\mathbf{e}_{\dot q}(k)\Vert\,/\,\Vert\mathbf{e}_{\dot q}(0)\Vert$")
ax.semilogy(tt, rho ** np.arange(N), "--", color=gf.C_REF, label=rf"$\rho^k$, $\rho = {rho:.4f}$")
ax.set(xlabel="tiempo [s]", ylabel="error relativo")
ax.legend(loc="upper right")
gf.save(fig, "observador_error.pdf")
