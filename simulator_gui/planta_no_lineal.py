"""
planta_no_lineal.py
Planta no lineal de la plataforma Stewart 6-SPS para el simulador: la misma dinamica que
octave_engine/dynamics/stewart_dynamic_matrices.m (metodo del momento generalizado, cilindro y piston de
cada pata como cuerpos rigidos, fuerza axial de cada pata como entrada):

    M(q) qddot + C(q, qdot) qdot + g(q) = (J T)' F + tau_d

Se escribe con las velocidades de los cuerpos, u_k = U_k qdot con U_k = J_k T (ecs. de las velocidades del
cilindro, del piston y de la pata en desarrollo_matematico.tex):
    dp_i = v + w x p_i,   w_l = lb_i x dp_i,   v_C = bC (w_l x n_i),   v_S = dp_i - bS (w_l x n_i)
Como w_l es perpendicular a n_i, la inercia axial de cilindro y piston no interviene y su inercia
rotacional actua como el escalar I_perp. Con las columnas de U_k (velocidades para qdot = e_j):
    M = sum_k U_k' I_k U_k,   g = sum_k m_k g U_k' e_z,   C qdot = sum_k U_k' d/dt(I_k U_k) qdot
d/dt(I_k U_k) qdot es la derivada del momento de cada cuerpo con qddot = 0, que se calcula analiticamente
derivando las mismas expresiones (en Octave se obtiene por diferencia central; ambos coinciden a 1e-12).
Los controladores se disenan con la linealizacion; esta planta es la "real" del simulador.
"""
import numpy as np


def rot(rpy):
    cr, sr = np.cos(rpy[0]), np.sin(rpy[0])
    cp, sp = np.cos(rpy[1]), np.sin(rpy[1])
    cy, sy = np.cos(rpy[2]), np.sin(rpy[2])
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def matriz_T(q):
    """nu = T qdot, con omega = E(theta, psi) [roll, pitch, yaw]_dot."""
    p, y = q[4], q[5]
    T = np.eye(6)
    T[3:, 3:] = [[np.cos(y) * np.cos(p), -np.sin(y), 0.0],
                 [np.sin(y) * np.cos(p), np.cos(y), 0.0],
                 [-np.sin(p), 0.0, 1.0]]
    return T


def _S(a):
    """Matrices antisimetricas apiladas de un arreglo (k, 3): S(a) b = a x b (k, 3, 3)."""
    S = np.zeros((a.shape[0], 3, 3))
    S[:, 0, 1], S[:, 0, 2], S[:, 1, 2] = -a[:, 2], a[:, 1], -a[:, 0]
    S[:, 1, 0], S[:, 2, 0], S[:, 2, 1] = a[:, 2], -a[:, 1], a[:, 0]
    return S


def _dot(a, b):
    return np.einsum("...a,...a->...", a, b)


class PlantaNoLineal:
    def __init__(self, m):
        """m: diccionario cargado de results/octave_control_matrices.mat."""
        f = lambda k: float(m[k])
        self.B = np.asarray(m["B_pts"], dtype=float)
        self.A = np.asarray(m["A_pts"], dtype=float)
        self.mP = f("massTop")
        self.IP = np.asarray(m["platformInertiaBody"], dtype=float)
        self.g = f("gravity")
        self.mC, self.mS, self.bC, self.bS = f("legMC"), f("legMS"), f("legBC"), f("legBS")
        self.Iperp = float(np.ravel(m["legIC"])[1] + np.ravel(m["legIS"])[1])

    def matrices(self, q, qd, dm=0.0):
        """M, C qdot, g y J T en (q, qdot). dm: masa adicional sobre la plataforma (carga)."""
        mP = self.mP + dm
        R = rot(q[3:6])
        IPB = R @ self.IP @ R.T
        pi = self.A @ R.T                                   # R s_i
        li = q[:3] + pi - self.B                            # vectores de las patas
        L2 = _dot(li, li)
        L = np.sqrt(L2)
        n = li / L[:, None]
        lb = li / L2[:, None]
        T = matriz_T(q)

        # Velocidades de los cuerpos (6 patas x 3 x 7): columnas 0..5 para qdot = e_j (U_k) y 6 para el qdot actual
        S = _S(np.concatenate([pi, lb, n]))
        Spi, Slb, Sn = S[:6], S[6:12], S[12:]
        nu = T @ np.column_stack([np.eye(6), qd])          # 6 x 7
        V, W = nu[:3], nu[3:]
        dp = V[None] - Spi @ W                              # v + w x p_i
        wl = Slb @ dp                                       # lb x dp
        wxn = -Sn @ wl                                      # w_l x n
        vC, vS = self.bC * wxn, dp - self.bS * wxn

        legs = np.concatenate([np.sqrt(self.mC) * vC[:, :, :6], np.sqrt(self.Iperp) * wl[:, :, :6],
                               np.sqrt(self.mS) * vS[:, :, :6]], axis=1).reshape(-1, 6)
        U_v, U_w = V[:, :6], W[:, :6]
        M = mP * U_v.T @ U_v + U_w.T @ IPB @ U_w + legs.T @ legs
        g = self.g * (mP * U_v[2] + self.mC * vC[:, 2, :6].sum(0) + self.mS * vS[:, 2, :6].sum(0))
        JT = np.einsum("ia,iak->ik", n, dp[:, :, :6])      # Ldot = n_i . dp_i

        # Derivada temporal de las velocidades con qddot = 0 (termino de Coriolis y centripeto)
        cq = np.zeros(6)
        if np.any(qd):
            p_, y_ = q[4], q[5]
            dpit, dyaw = qd[4], qd[5]
            dE = np.array([[-np.sin(y_) * np.cos(p_) * dyaw - np.cos(y_) * np.sin(p_) * dpit, -np.cos(y_) * dyaw, 0.0],
                           [np.cos(y_) * np.cos(p_) * dyaw - np.sin(y_) * np.sin(p_) * dpit, -np.sin(y_) * dyaw, 0.0],
                           [-np.cos(p_) * dpit, 0.0, 0.0]])
            om, dom = W[:, 6], dE @ qd[3:]                  # omega y su derivada con qddot = 0
            So = _S(om[None])[0]
            a_dp = -Spi @ dom - (Spi @ om) @ So.T           # dom x p_i + om x (om x p_i), v_dot = 0
            d6, wl6 = dp[:, :, 6], wl[:, :, 6]
            dlb = d6 / L2[:, None] - 2.0 * li * (_dot(li, d6) / L2**2)[:, None]
            dn = (d6 - n * _dot(n, d6)[:, None]) / L[:, None]
            S2 = _S(np.concatenate([d6, wl6]))
            a_wl = -np.einsum("iab,ib->ia", S2[:6], dlb) + np.einsum("iab,ib->ia", Slb, a_dp)
            a_wxn = -np.einsum("iab,ib->ia", Sn, a_wl) + np.einsum("iab,ib->ia", S2[6:], dn)
            a_vC, a_vS = self.bC * a_wxn, a_dp - self.bS * a_wxn
            dH = So @ (IPB @ om) + IPB @ dom                # d/dt(I_P omega) con qddot = 0
            cq = (U_w.T @ dH + self.mC * np.einsum("iak,ia->k", vC[:, :, :6], a_vC)
                  + self.Iperp * np.einsum("iak,ia->k", wl[:, :, :6], a_wl)
                  + self.mS * np.einsum("iak,ia->k", vS[:, :, :6], a_vS))
        return 0.5 * (M + M.T), cq, g, JT

    def qddot(self, x, F, tau_d=None, dm=0.0):
        """Aceleracion generalizada con fuerzas de pata F [N] y perturbacion generalizada tau_d."""
        M, cq, g, JT = self.matrices(x[:6], x[6:], dm)
        rhs = JT.T @ F - cq - g
        if tau_d is not None:
            rhs = rhs + tau_d
        return np.linalg.solve(M, rhs)

    def paso_rk4(self, x, F, Ts, tau_d=None, dm=0.0):
        """Un paso Ts de RK4 con F constante (retenedor de orden cero), como stewart_simulate_plant.m."""
        f = lambda xx: np.concatenate([xx[6:], self.qddot(xx, F, tau_d, dm)])
        k1 = f(x)
        k2 = f(x + 0.5 * Ts * k1)
        k3 = f(x + 0.5 * Ts * k2)
        k4 = f(x + Ts * k3)
        return x + Ts / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

    def paso(self, x, F, Ts, tau_d=None, dm=0.0):
        """Un paso Ts con F constante (retenedor de orden cero), por el metodo del punto medio (RK2). Con F
        constante la EDO solo tiene la dinamica lenta de la plataforma (|lambda| <= 7.6 1/s, lambda Ts = 0.015),
        de modo que RK2 coincide con RK4 a menos de 0.01 mm en lazo cerrado y cuesta la mitad."""
        k1 = np.concatenate([x[6:], self.qddot(x, F, tau_d, dm)])
        xm = x + 0.5 * Ts * k1
        return x + Ts * np.concatenate([xm[6:], self.qddot(xm, F, tau_d, dm)])

    @staticmethod
    def wrench_a_generalizada(q, wrench):
        """Fuerza y momento externos [Fx, Fy, Fz, Mx, My, Mz] (en {B}) como fuerza generalizada en q: T' w."""
        return matriz_T(q).T @ np.asarray(wrench, dtype=float)


if __name__ == "__main__":
    # Verificacion: M simetrica y definida positiva, y conservacion de la energia en movimiento libre sin gravedad
    import os
    import scipy.io as sio
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    m = sio.loadmat(os.path.join(root, "results", "octave_control_matrices.mat"), squeeze_me=True)
    pl = PlantaNoLineal(m)
    pl.g = 0.0
    x = np.concatenate([m["pose0"], [0.3, -0.2, 0.25, 1.0, 0.8, -1.2]])
    energia = lambda xx: 0.5 * xx[6:] @ pl.matrices(xx[:6], xx[6:])[0] @ xx[6:]
    e0 = energia(x)
    for _ in range(100):
        x = pl.paso_rk4(x, np.zeros(6), 0.002)
    M = pl.matrices(x[:6], x[6:])[0]
    assert np.allclose(M, M.T) and np.linalg.eigvalsh(M).min() > 0
    deriva = abs(energia(x) - e0) / e0
    assert deriva < 1e-9, deriva   # sin el termino de Coriolis la deriva seria del orden de 1e-2
    print(f"OK: M simetrica y definida positiva, deriva relativa de energia en 0.2 s = {deriva:.1e}")
