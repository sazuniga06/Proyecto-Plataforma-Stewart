"""
trayectorias_variacionales.py
Trayectorias de referencia optimas obtenidas por calculo de variaciones y por el principio del maximo de Pontryagin,
segun los casos del curso (modulos 02, 03, 04, 06 y 07). Cada maniobra lleva la pose de A a B en T segundos.

Se trabaja con el error normalizado eps(t) = (q(t) - q_B) / (q_A - q_B), que vale 1 en t = 0, de modo que
q(t) = q_B + eps(t) (q_A - q_B). Con el funcional F = 1/2 eps'^2 + 1/2 a^2 eps^2 (esfuerzo de velocidad mas
desviacion respecto al destino):

  extremos fijos  (Caso III, F autonomo -> Beltrami):  eps'' = a^2 eps,  eps(0) = 1, eps(T) = 0
                  eps = sinh(a (T - t)) / sinh(a T)
  estado final libre (transversalidad F_eps'(T) = 0 -> eps'(T) = 0):
                  eps = cosh(a (T - t)) / cosh(a T)
  descuento exponencial (Caso II, L = e^{-alpha t} F):  eps'' - alpha eps' - a^2 eps = 0
                  eps = C1 e^{r1 t} + C2 e^{r2 t},  r_{1,2} = (alpha +- sqrt(alpha^2 + 4 a^2)) / 2
  6 GDL multivariable (modulo 04): F = 1/2 e'^T M0 e' + 1/2 e^T K e,  K = a^2 diag(M0)
                  M0 e'' = K e -> e(t) = sinh(W (T - t)) sinh(W T)^{-1} e(0),  W^2 = M0^{-1} K
  tiempo minimo (Pontryagin, modulo 07): pata como doble integrador l'' = u, |u| <= a_max (aceleracion maxima de
                  pata del proyecto); control bang-bang con una conmutacion en T/2 (Feldbaum: n - 1 = 1)
  casos de frontera (modulos 02 y 03) sobre la onda deseada en z, r(t) = A cos(w t), con
                  F = 1/2 z'^2 + 1/2 a^2 (z - r)^2:  z'' - a^2 z = -a^2 r
                  x = kappa r + C1 e^{-a t} + C2 e^{-a (T - t)},  kappa = a^2 / (a^2 + w^2)
                  A: inicio y final conocidos   x(0) = 0,  x(T) = 0
                  B: inicio conocido, final libre   x(0) = 0,  x'(T) = 0  (F_x'(T) = 0)
                  C: inicio libre, final conocido   x'(0) = 0 (F_x'(0) = 0),  x(T) = 0
                  D: ambos libres   x'(0) = 0,  x'(T) = 0
  curva terminal (modulo 03, caso 3): interceptar la senoidal phi(t) = A sin(w t) con F = 1/2 z'^2 y tiempo
                  final libre. Caso I (F_z = 0) -> z' constante (recta); transversalidad
                  [F + F_z'(phi' - z')]_T = 0 -> z'(T) = 2 phi'(T); con z(T) = phi(T) se obtiene T.
"""
import numpy as np
from scipy.linalg import expm
from scipy.optimize import brentq

T_MOV, T_HOLD = 2.0, 1.0            # traslado y reposo [s]
A_PESO = 2.0                        # a [1/s]: peso de la desviacion frente al esfuerzo de velocidad
ALPHA = 1.0                         # tasa de descuento [1/s]
A_MAX = 0.08                        # aceleracion maxima de pata [m/s^2] (P.constraints.maxLegAccel)
TARGET = np.array([0.010, -0.008, 0.015, np.deg2rad(3.0), np.deg2rad(-3.0), np.deg2rad(4.0)])

_MODEL = {"M0": None, "J0": None}


def configure(M0, J0):
    """Matriz de inercia y Jacobiano nominales (los fija el motor al cargar las matrices de Octave)."""
    _MODEL["M0"] = np.asarray(M0, dtype=float)
    _MODEL["J0"] = np.asarray(J0, dtype=float)


# ----------------------------------------------------------------------------------------- casos escalares
def eps_fijos(t, T=T_MOV, a=A_PESO):
    return np.sinh(a * (T - t)) / np.sinh(a * T)


def eps_libre(t, T=T_MOV, a=A_PESO):
    return np.cosh(a * (T - t)) / np.cosh(a * T)


def raices_descuento(a=A_PESO, alpha=ALPHA):
    d = np.sqrt(alpha**2 + 4 * a**2)
    return (alpha + d) / 2, (alpha - d) / 2


def eps_descuento(t, T=T_MOV, a=A_PESO, alpha=ALPHA):
    r1, r2 = raices_descuento(a, alpha)
    den = np.exp(r1 * T) - np.exp(r2 * T)
    return (np.exp(r1 * T) * np.exp(r2 * t) - np.exp(r2 * T) * np.exp(r1 * t)) / den


# ----------------------------------------------------------------------------------------- 6 GDL
def omega_multi(a=A_PESO):
    M0 = _MODEL["M0"]
    W2 = np.linalg.solve(M0, a**2 * np.diag(np.diag(M0)))
    lam, V = np.linalg.eig(W2)
    return np.sqrt(lam.real), V.real


def e_multi(t, e0, T=T_MOV, a=A_PESO):
    w, V = omega_multi(a)
    c = np.linalg.solve(V, e0)
    return V @ (np.sinh(w * (T - t)) / np.sinh(w * T) * c)


# ----------------------------------------------------------------------------------------- tiempo minimo
def bang_bang(dl, amax=A_MAX):
    """Doble integrador l'' = u, |u| <= amax, del reposo al reposo en dl: conmutacion t1 = T/2, T = 2 sqrt(dl/amax)."""
    T = 2.0 * np.sqrt(dl / amax)
    return T / 2.0, T


def l_bang_bang(t, dl, amax=A_MAX):
    t1, T = bang_bang(dl, amax)
    t = min(max(t, 0.0), T)
    return 0.5 * amax * t**2 if t <= t1 else dl - 0.5 * amax * (T - t)**2


def tiempo_minimo(delta):
    """Elongacion de la pata mas exigida y tiempo minimo T de la maniobra (las demas patas siguen en proporcion)."""
    dl = float(np.max(np.abs(_MODEL["J0"] @ delta)))
    _, T = bang_bang(dl)
    return dl, T


# ----------------------------------------------------------------------------------------- curva terminal
SIN_A, SIN_W = 0.015, 2.0 * np.pi * 0.45      # misma senoidal del generador "Senoidal en z"
Z_START = 0.005                               # altura inicial de la maniobra de intercepcion [m]
T_CICLO_CURVA = 6.0                           # duracion del ciclo: intercepcion y seguimiento [s]


def intercepcion(z0=Z_START, A=SIN_A, w=SIN_W):
    """Tiempo de intercepcion T y pendiente c de la recta optima: z0 + 2 phi'(T) T = phi(T), c = 2 phi'(T)."""
    f = lambda T: z0 + 2 * A * w * np.cos(w * T) * T - A * np.sin(w * T)
    T = brentq(f, 0.2, np.pi / (2 * w))
    return T, 2 * A * w * np.cos(w * T)


def z_curva(t, z0=Z_START):
    T, c = intercepcion(z0)
    return z0 + c * t if t < T else SIN_A * np.sin(SIN_W * t)


# ----------------------------------------------------------------------------------------- casos de frontera
A_ONDA = 2.0                                          # a [1/s] del funcional de seguimiento de la onda
F_ONDA = 0.2                                          # frecuencia de la onda deseada [Hz]
W_ONDA = 2.0 * np.pi * F_ONDA                         # [rad/s]
T_ONDA = 2.5                                          # horizonte [s]: medio periodo, r(0) = A y r(T) = -A
T_PRE = 1.0                                           # posicionamiento previo / regreso [s]
AMP_ONDA = np.array([0.015])                          # amplitud en z [m]
FASE_ONDA = np.array([np.pi / 2])                     # r = A sin(w t + pi/2) = A cos(w t)
GDL_ONDA = (2,)                                       # indice de z en la pose
CASOS = {"A": (True, True), "B": (True, False), "C": (False, True), "D": (False, False)}   # (inicio, final) conocidos


def onda(t):
    """Onda deseada r(t) y su derivada (en z)."""
    ph = W_ONDA * t + FASE_ONDA
    return AMP_ONDA * np.sin(ph), AMP_ONDA * W_ONDA * np.cos(ph)


def kappa(a=A_ONDA, w=W_ONDA):
    return a**2 / (a**2 + w**2)


def coef_frontera(caso, a=A_ONDA, T=T_ONDA):
    """Constantes C1, C2 (por grado de libertad) de x = kappa r + C1 e^{-a t} + C2 e^{-a (T - t)}.
    Extremo conocido: x = 0 (posicion nominal). Extremo libre: condicion natural F_x' = x' = 0."""
    ini, fin = CASOS[caso]
    E, k = np.exp(-a * T), kappa(a)
    (r0, dr0), (rT, drT) = onda(0.0), onda(T)
    fila0 = [1.0, E] if ini else [-a, a * E]                 # h(0) o h'(0), con h = x - kappa r
    filaT = [E, 1.0] if fin else [-a * E, a]
    M = np.array([fila0, filaT])
    rhs = np.vstack([-k * (r0 if ini else dr0), -k * (rT if fin else drT)])
    return np.linalg.solve(M, rhs)                          # 2 x 3: [C1; C2]


def x_frontera(caso, t, a=A_ONDA, T=T_ONDA):
    """Trayectoria optima x*(t) y su derivada (en z) del caso A, B, C o D."""
    C1, C2 = coef_frontera(caso, a, T)
    r, dr = onda(t)
    e1, e2 = np.exp(-a * t), np.exp(-a * (T - t))
    k = kappa(a)
    return k * r + C1 * e1 + C2 * e2, k * dr - a * C1 * e1 + a * C2 * e2


def costo_frontera(caso, a=A_ONDA, T=T_ONDA, n=4001):
    """J* = int_0^T (1/2 x'^2 + 1/2 a^2 (x - r)^2) dt por grado de libertad (regla de Simpson)."""
    t = np.linspace(0.0, T, n)
    x, dx = np.array([x_frontera(caso, ti, a, T) for ti in t]).transpose(1, 0, 2)
    r = np.array([onda(ti)[0] for ti in t])
    f = 0.5 * dx**2 + 0.5 * a**2 * (x - r)**2
    h = t[1] - t[0]
    return h / 3 * (f[0] + f[-1] + 4 * f[1:-1:2].sum(0) + 2 * f[2:-1:2].sum(0))


def pose_frontera(caso, t, pose0):
    """Ciclo: posicionamiento en x*(0) (maniobra de extremos fijos, T_PRE), trayectoria optima (T_ONDA) y regreso."""
    tc = t % (2 * T_PRE + T_ONDA)
    if tc < T_PRE:
        s = 1.0 - eps_fijos(tc, T_PRE)
        val = s * x_frontera(caso, 0.0)[0]
    elif tc < T_PRE + T_ONDA:
        val = x_frontera(caso, tc - T_PRE)[0]
    else:
        val = eps_fijos(tc - T_PRE - T_ONDA, T_PRE) * x_frontera(caso, T_ONDA)[0]
    q = pose0.copy()
    q[list(GDL_ONDA)] += val
    return q


# ----------------------------------------------------------------------------------------- referencia periodica
# Referencias del simulador. Los casos de final libre, descuento y 6 GDL quedan solo como funciones (figuras y
# verificaciones del documento): con estos parametros sus trayectorias difieren menos de 1.5 mm de la de extremos fijos.
MODES = ("var_fixed", "min_time", "var_curve", "bc_A", "bc_B", "bc_C", "bc_D")


def pose(mode, t, pose0, target=None):
    """Ida y vuelta periodica pose0 -> pose0 + target -> pose0 con traslados de T_MOV (o del tiempo minimo)."""
    delta = TARGET if target is None or not np.any(target) else np.asarray(target, dtype=float)
    if mode.startswith("bc_"):
        return pose_frontera(mode[3], t, pose0)
    if mode == "var_curve":
        return pose0 + np.array([0.0, 0.0, z_curva(t % T_CICLO_CURVA), 0.0, 0.0, 0.0])
    if mode == "min_time":
        dl, Tm = tiempo_minimo(delta)
        half = Tm + T_HOLD
        tc = t % (2 * half)
        frac = l_bang_bang(tc, dl) / dl if tc < half else 1.0 - l_bang_bang(tc - half, dl) / dl
        return pose0 + frac * delta
    half = T_MOV + T_HOLD
    tc = t % (2 * half)
    ida = tc < half
    tl = min(tc if ida else tc - half, T_MOV)
    eps = eps_fijos(tl)
    return pose0 + ((1.0 - eps) if ida else eps) * delta
