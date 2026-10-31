#!/usr/bin/env python3
"""
octave_bridge.py
Puente entre la interfaz de simulacion (Python) y el motor de control (octave_engine/).
- Carga las matrices sintetizadas en GNU Octave (`results/octave_control_matrices.mat`).
- Permite redisenar en caliente (`exportar_matrices_control_octave.m`) cuando el usuario
  modifica la ubicacion deseada de los polos de lazo cerrado o del observador de Luenberger.
- Ejecuta el paso de control + observador + planta. La planta es el modelo NO LINEAL completo
  (planta_no_lineal.py, igual a octave_engine/dynamics/stewart_dynamic_matrices.m), con la fuerza axial de cada
  pata como entrada; los controladores y el observador usan el modelo linealizado discreto de diseno.
- Controladores: PID descentralizado en espacio articular ("pid") y realimentacion de estados con
  accion integral + observador de Luenberger ("ss").
- Indices de la Tabla 1 con las mismas definiciones que trajectories/calcular_metricas_tabla1.m.
"""
import json
import multiprocessing as mp
import os
import subprocess
import time
import numpy as np
import trayectorias_variacionales as tv
from planta_no_lineal import PlantaNoLineal
import scipy.io as sio

# Ensayo normalizado (compartido por la interfaz, la Tabla 1 y scripts/optimizar_ganancias.py):
# equilibrio -> escalon de referencia en TEST_STEP_T -> impulso en TEST_IMPULSE_T -> fin en TEST_END_T
TEST_STEP_T, TEST_IMPULSE_T, TEST_END_T = 0.5, 2.5, 5.0
TEST_STEP = np.array([0.008, 0.0, 0.015, 0.0, 0.0, 0.0])          # dx = 8 mm, dz = 15 mm
TEST_IMPULSE_N = 20.0
IMPULSE_DUR = 0.45                                                  # duracion del impulso [s]
IMPULSE_DIR = np.array([0.2, -0.15, -1.0, 0.0175, -0.015, 0.0])    # por newton de |F|: Fz descendente + lateral + par
TEST_NOISE_MM = 0.1
ROBUST_GAINS = (0.8, 1.2)                                           # variacion de la ganancia de la planta (Bd)
LOAD_TAU = np.array([4.0, 0.0, -10.0, 0.0, 0.0, 0.0])               # carga sostenida del escenario 2 [N]
LOAD_MASS = 0.5                                                     # masa adicional del escenario 2 [kg]
TRACKING_MODES = ("sine_z", "coupled_6dof", "circle_3d")           # referencias dinamicas de la interfaz
EFFORT_WEIGHT = 0.25                                                # peso relativo de Ju frente a ITAE en el criterio


def reference_pose(mode: str, t: float, pose0: np.ndarray, manual: np.ndarray = None) -> np.ndarray:
    """Referencia de pose en el instante t. manual = [dx, dy, dz, roll, pitch, yaw] (m, rad) de los deslizadores;
    los generadores dinamicos conservan algunos de esos grados de libertad."""
    m = np.zeros(6) if manual is None else np.asarray(manual, dtype=float)
    if mode == "sine_z":
        return pose0 + np.array([m[0], m[1], 0.015 * np.sin(2.0 * np.pi * 0.45 * t), m[3], m[4], m[5]])
    if mode == "coupled_6dof":
        w = 2.0 * np.pi * 0.35
        return pose0 + np.array([0.012 * np.sin(w * t), 0.012 * np.cos(w * t), 0.014 * np.sin(0.5 * w * t),
                                 np.deg2rad(4.5) * np.sin(w * t), np.deg2rad(4.5) * np.cos(w * t),
                                 np.deg2rad(5.0) * np.sin(0.5 * w * t)])
    if mode == "circle_3d":
        w = 2.0 * np.pi * 0.40
        return pose0 + np.array([0.015 * np.cos(w * t), 0.015 * np.sin(w * t), m[2],
                                 np.deg2rad(3.5) * np.sin(w * t), np.deg2rad(3.5) * np.cos(w * t), m[5]])
    if mode in tv.MODES:
        # trayectorias optimas por calculo de variaciones / Pontryagin (trayectorias_variacionales.py)
        return tv.pose(mode, t, pose0, m)
    return pose0 + m


_ENGINE_FORK = None


def _scenario_fork(i, ctrl_mode, s):
    return _ENGINE_FORK._scenario(i, ctrl_mode, s)


class OctaveRealtimeControlEngine:
    """Paso de control y observador (modelo lineal de diseno) sobre la planta no lineal."""

    def __init__(self, project_root: str = None):
        if project_root is None:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.project_root = project_root
        self.octave_dir = os.path.join(project_root, "octave_engine")
        self.results_dir = os.path.join(project_root, "results")
        os.makedirs(self.results_dir, exist_ok=True)
        self.matrices_mat_path = os.path.join(self.results_dir, "octave_control_matrices.mat")

        if not os.path.exists(self.matrices_mat_path) or "legMC" not in {v[0] for v in sio.whosmat(self.matrices_mat_path)}:
            self.resynthesize_in_octave(1.0, 1.0)   # sin matrices o con un formato anterior
        else:
            self.load_octave_matrices()
        self.rng = np.random.default_rng(0)
        self.plant_gain = 1.0   # factor sobre Bd en la planta (el controlador y el observador usan el nominal)
        self.reset_state()

    # ------------------------------------------------------------------ matrices de diseno
    def resynthesize_in_octave(self, pole_scale: float = 1.0, obs_scale: float = 1.0):
        """Invoca a GNU Octave para recalcular la linealizacion, Kalman, Sylvester y Luenberger."""
        oct_cmd = (
            f"addpath('{self.octave_dir}'); "
            f"exportar_matrices_control_octave({float(pole_scale):.6f}, {float(obs_scale):.6f}, '{self.matrices_mat_path}');"
        )
        proc = subprocess.run(["octave", "--no-gui", "--quiet", "--eval", oct_cmd],
                              cwd=self.octave_dir, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            raise RuntimeError(f"El rediseno no pudo completarse:\n{proc.stderr}\n{proc.stdout}")
        self.load_octave_matrices()

    def load_octave_matrices(self):
        m = sio.loadmat(self.matrices_mat_path, squeeze_me=True)
        self.h0 = float(m["h0"])
        self.B_pts = np.asarray(m["B_pts"], dtype=float)
        self.A_pts = np.asarray(m["A_pts"], dtype=float)
        self.pose0 = np.asarray(m["pose0"], dtype=float).reshape(6)
        self.Lmin = float(m["Lmin"])
        self.Lmax = float(m["Lmax"])
        self.lowerBodyLength = float(m["lowerBodyLength"])
        self.Fmax = float(m["Fmax"])
        self.Ts = float(m["Ts"])

        # Modelo linealizado (continuo y discreto) y pruebas de Kalman
        self.A = np.asarray(m["A"], dtype=float)
        self.B = np.asarray(m["B"], dtype=float)
        self.Ad = np.asarray(m["Ad"], dtype=float)
        self.Bd = np.asarray(m["Bd"], dtype=float)
        self.Cd = np.asarray(m["Cd"], dtype=float)
        self.x0 = np.asarray(m["x0"], dtype=float).reshape(12)
        self.u0 = np.asarray(m["u0"], dtype=float).reshape(6)
        self.rank_Mc = int(m["rank_Mc"])
        self.rank_Mo = int(m["rank_Mo"])

        # Realimentacion de estados con accion integral + observador de Luenberger
        self.K_pi = np.asarray(m["K_pi"], dtype=float)
        self.Ki_pi = np.asarray(m["Ki_pi"], dtype=float)
        self.Nr_pi = np.asarray(m["Nr_pi"], dtype=float)
        self.Ld = np.asarray(m["Ld"], dtype=float)
        self.err_Nr_cof = float(m["err_Nr_cof"])
        self.sep_error = float(m["sep_error"])
        self.pole_scale = float(m["pole_scale"]) if "pole_scale" in m else 1.0
        self.obs_scale = float(m["obs_scale"]) if "obs_scale" in m else 1.0
        self.poles_cl = np.asarray(m["poles_cl"]).reshape(-1)
        self.poles_obs = np.asarray(m["poles_obs"]).reshape(-1)

        # PID descentralizado en espacio articular (ganancias de diseno de controlador_pid.m)
        self.Kp_leg = np.asarray(m["Kp_leg"], dtype=float).reshape(6)
        self.Ki_leg = np.asarray(m["Ki_leg"], dtype=float).reshape(6)
        self.Kd_leg = np.asarray(m["Kd_leg"], dtype=float).reshape(6)
        if "M0" in m:
            tv.configure(m["M0"], m["J0"])   # modelo nominal para las trayectorias 6 GDL y de tiempo minimo
        self.planta = PlantaNoLineal(m)       # planta "real" del simulador

    def load_optimal_gains(self) -> dict:
        """Lee results/ganancias_optimas.json (generado por scripts/optimizar_ganancias.py); {} si no existe."""
        path = os.path.join(self.results_dir, "ganancias_optimas.json")
        if not os.path.exists(path):
            return {}
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    # ------------------------------------------------------------------ cinematica
    @staticmethod
    def _rot(pose):
        cr, sr = np.cos(pose[3]), np.sin(pose[3])
        cp, sp = np.cos(pose[4]), np.sin(pose[4])
        cy, sy = np.cos(pose[5]), np.sin(pose[5])
        return np.array([
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp,     cp * sr,                cp * cr               ]
        ])

    def compute_ik(self, pose: np.ndarray) -> np.ndarray:
        Q = pose[:3][None, :] + self.A_pts @ self._rot(pose).T
        return np.linalg.norm(Q - self.B_pts, axis=1)

    def compute_jacobian(self, pose: np.ndarray) -> np.ndarray:
        r_rel = self.A_pts @ self._rot(pose).T
        D = pose[:3][None, :] + r_rel - self.B_pts
        U = D / np.maximum(np.linalg.norm(D, axis=1, keepdims=True), 1e-9)
        return np.hstack([U, np.cross(r_rel, U)])

    # ------------------------------------------------------------------ ensayos
    def run_standard_test(self, ctrl_mode: str, kp_scale=1.0, ki_scale=1.0, kd_scale=1.0,
                          impulse_N: float = TEST_IMPULSE_N, noise_sigma_mm: float = TEST_NOISE_MM, seed: int = 0) -> dict:
        """Ensayo normalizado completo (sin interfaz): escalon + impulso. Devuelve los indices de la Tabla 1."""
        self.reset_state(init_z_err_mm=0.0)
        self.rng = np.random.default_rng(seed)
        k_step, k_imp = int(round(TEST_STEP_T / self.Ts)), int(round(TEST_IMPULSE_T / self.Ts))
        for k in range(int(round(TEST_END_T / self.Ts))):
            q_ref = self.pose0 + (TEST_STEP if k >= k_step else 0.0)
            tau = impulse_N * IMPULSE_DIR if k_imp <= k < k_imp + round(IMPULSE_DUR / self.Ts) else np.zeros(6)
            out = self.step_realtime(q_ref, tau, 0.0, ctrl_mode, kp_scale, ki_scale, kd_scale,
                                     noise_sigma_mm=noise_sigma_mm)
            if not np.all(np.isfinite(out["pose"])):
                return {"ITAE": np.inf, "Ju": np.inf, "RMS_pos": np.inf}
        return self.metrics()

    def run_load_test(self, ctrl_mode: str, kp_scale=1.0, ki_scale=1.0, kd_scale=1.0,
                      noise_sigma_mm: float = TEST_NOISE_MM, seed: int = 1) -> dict:
        """Escalon de referencia y, desde t = 1 s, carga sostenida (0.5 kg + Fz = -10 N, Fx = 4 N) durante 6 s.
        Evalua el rechazo de perturbaciones constantes, es decir, el aporte de la accion integral."""
        self.reset_state(init_z_err_mm=0.0)
        self.rng = np.random.default_rng(seed)
        tau_load = LOAD_TAU
        k_step, k_load = int(round(TEST_STEP_T / self.Ts)), int(round(1.0 / self.Ts))
        for k in range(int(round(6.0 / self.Ts))):
            q_ref = self.pose0 + (TEST_STEP if k >= k_step else 0.0)
            loaded = k >= k_load
            out = self.step_realtime(q_ref, tau_load if loaded else np.zeros(6), LOAD_MASS if loaded else 0.0,
                                     ctrl_mode, kp_scale, ki_scale, kd_scale, noise_sigma_mm=noise_sigma_mm)
            if not np.all(np.isfinite(out["pose"])):
                return {"ITAE": np.inf, "Ju": np.inf, "RMS_pos": np.inf}
        return self.metrics()

    def run_tracking_test(self, ctrl_mode: str, ref_mode: str, kp_scale=1.0, ki_scale=1.0, kd_scale=1.0,
                          duration: float = 6.0, noise_sigma_mm: float = TEST_NOISE_MM, seed: int = 2) -> dict:
        """Seguimiento de una referencia dinamica de la interfaz (senoidal, maniobra 6 GDL o circular)."""
        self.reset_state(init_z_err_mm=0.0)
        self.rng = np.random.default_rng(seed)
        for k in range(int(round(duration / self.Ts))):
            out = self.step_realtime(reference_pose(ref_mode, k * self.Ts, self.pose0), np.zeros(6), 0.0,
                                     ctrl_mode, kp_scale, ki_scale, kd_scale, noise_sigma_mm=noise_sigma_mm)
            if not np.all(np.isfinite(out["pose"])):
                return {"ITAE": np.inf, "Ju": np.inf, "RMS_pos": np.inf}
        return self.metrics()

    def _scenario(self, i, ctrl_mode, s):
        if i == 0:
            return self.run_standard_test(ctrl_mode, *s)
        if i == 1:
            return self.run_load_test(ctrl_mode, *s)
        return self.run_tracking_test(ctrl_mode, TRACKING_MODES[i - 2], *s)

    def scenario_metrics(self, ctrl_mode, kp_scale=1.0, ki_scale=1.0, kd_scale=1.0):
        """Escenarios del criterio: ensayo normalizado, carga sostenida y las tres referencias dinamicas. Con la
        planta no lineal cada escenario tarda segundos, de modo que los cinco se simulan en paralelo (procesos
        hijos que heredan una copia del motor; los resultados son identicos a los de la ejecucion en serie)."""
        s = (kp_scale, ki_scale, kd_scale)
        n = 2 + len(TRACKING_MODES)
        if os.name != "posix":
            return tuple(self._scenario(i, ctrl_mode, s) for i in range(n))
        global _ENGINE_FORK
        _ENGINE_FORK = self
        with mp.get_context("fork").Pool(n) as pool:
            return tuple(pool.starmap(_scenario_fork, [(i, ctrl_mode, s) for i in range(n)]))

    def tuning_cost(self, ctrl_mode, kp_scale=1.0, ki_scale=1.0, kd_scale=1.0, ref=None) -> float:
        """Criterio de optimizacion de ganancias (ver cost_from_metrics). ref = scenario_metrics() del diseno."""
        return self.cost_from_metrics(self.scenario_metrics(ctrl_mode, kp_scale, ki_scale, kd_scale), ref)

    @staticmethod
    def cost_from_metrics(cur, ref=None) -> float:
        """J = sum_escenarios (ITAE/ITAE_ref + 0.25 * Ju/Ju_ref), con Ju = integral F'F dt (fuerzas de las patas).
        Con las ganancias de referencia, J = 5 * 1.25 = 6.25."""
        ref = ref or cur
        return float(sum(c["ITAE"] / r["ITAE"] + EFFORT_WEIGHT * c["Ju"] / r["Ju"] for c, r in zip(cur, ref)))

    def table1_row(self, ctrl_mode, kp_scale=1.0, ki_scale=1.0, kd_scale=1.0) -> dict:
        """Fila de la Tabla 1 sobre el ensayo normalizado. Robustez: aumento del RMS de posicion en el peor caso
        de una variacion de +-20 % en la ganancia de la planta (Bd), sin redisenar el controlador."""
        nom = self.run_standard_test(ctrl_mode, kp_scale, ki_scale, kd_scale)
        worst = 0.0
        try:
            for g in ROBUST_GAINS:
                self.plant_gain = g
                r = self.run_standard_test(ctrl_mode, kp_scale, ki_scale, kd_scale)
                worst = max(worst, 100.0 * (r["RMS_pos"] / nom["RMS_pos"] - 1.0))
        finally:
            self.plant_gain = 1.0
        nom["robustez_pct"] = worst
        return nom

    # ------------------------------------------------------------------ estado e indices
    def reset_state(self, init_z_err_mm: float = -4.0):
        self.t = 0.0
        self.x = self.x0.copy()
        self.x_hat = np.zeros(12)
        self.x_hat[2] = init_z_err_mm * 1e-3
        self.reset_controller()
        self.u_applied = self.u0.copy()
        self.reset_metrics()

    def reset_controller(self):
        """Borra la memoria del controlador (integradores y derivada filtrada) sin tocar la planta."""
        self.x_int_ss = np.zeros(6)
        self.pid_int = np.zeros(6)
        self.pid_prev_err = np.zeros(6)
        self.pid_d_filt = np.zeros(6)

    def reset_metrics(self):
        """Reinicia los acumuladores de los indices de la Tabla 1."""
        self.m_t0 = self.t
        self.m_n = 0
        self.m_e_pos2 = 0.0     # suma ||e_pos||^2 [m^2]
        self.m_e_ang2 = 0.0     # suma ||e_ang||^2 [rad^2]
        self.m_itae = 0.0       # integral t*||e||_1 dt (e = pose 6-D en m y rad)
        self.m_ju = 0.0         # integral F'F dt [N^2*s]
        self.m_fmax = 0.0
        self.m_fsat = 0         # muestras con alguna fuerza saturada
        self.m_stroke = 0       # muestras con alguna pata fuera de [Lmin, Lmax]
        self.m_cpu = 0.0        # tiempo de calculo de la ley de control (+ observador en "ss") [s]

    def metrics(self) -> dict:
        n = max(self.m_n, 1)
        return {
            "T": self.t - self.m_t0,
            "RMS_pos": 1e3 * np.sqrt(self.m_e_pos2 / n),            # [mm]
            "RMS_ang": np.degrees(np.sqrt(self.m_e_ang2 / n)),      # [deg]
            "ITAE": self.m_itae,
            "Ju": self.m_ju,
            "Fmax": self.m_fmax,
            "fsat_pct": 100.0 * self.m_fsat / n,
            "stroke": self.m_stroke,
            "cpu_us": 1e6 * self.m_cpu / n,
        }

    # ------------------------------------------------------------------ paso de simulacion
    def step_realtime(self, q_ref: np.ndarray, tau_ext: np.ndarray, delta_mass_kg: float,
                      ctrl_mode: str, kp_scale: float = 1.0, ki_scale: float = 1.0, kd_scale: float = 1.0,
                      noise_sigma_mm: float = 0.0) -> dict:
        """
        Avanza un paso Ts (2 ms) de la simulacion en lazo cerrado. La entrada de la planta es la fuerza axial
        de cada pata F [N] sobre la planta no lineal, saturada a +-Fmax.
        ctrl_mode: "pid" (PID articular) o "ss" (realimentacion de estados con accion integral + observador).
        noise_sigma_mm: desviacion estandar del ruido gaussiano del sensor de pose (angulos ~ sigma/0.1 m).
        """
        y_true = self.x[:6].copy()
        sig = float(noise_sigma_mm) * 1e-3
        y_meas = y_true + self.rng.normal(0.0, 1.0, 6) * sig * np.array([1, 1, 1, 10, 10, 10]) if sig > 0 else y_true
        err_q = q_ref - y_meas

        # 1. Observador de Luenberger: correccion con la medicion (se ejecuta siempre para la grafica de z_dot)
        t_obs = time.perf_counter()
        x_hat_corr = self.x_hat + self.Ld @ ((y_meas - self.pose0) - self.Cd @ self.x_hat)
        t_obs = time.perf_counter() - t_obs

        # 2. Ley de control (fuerzas de las patas)
        t_law = time.perf_counter()
        if ctrl_mode == "ss":
            delta_u = (-(kp_scale * self.K_pi) @ x_hat_corr + self.Nr_pi @ (q_ref - self.pose0)
                       + (ki_scale * self.Ki_pi) @ self.x_int_ss)
            u_raw = self.u0 + delta_u
            u_sat = np.clip(u_raw, -self.Fmax, self.Fmax)
            if np.all(np.abs(u_raw) <= self.Fmax):   # anti-windup por integracion condicional
                self.x_int_ss = np.clip(self.x_int_ss + self.Ts * err_q, -0.08, 0.08)
        else:
            err = self.compute_ik(q_ref) - self.compute_ik(y_meas)
            af = np.exp(-160.0 * self.Ts)          # filtro derivativo: polo en wf = 160 rad/s
            self.pid_d_filt = af * self.pid_d_filt + (1 - af) * (err - self.pid_prev_err) / self.Ts
            u_raw = (self.u0 + (kp_scale * self.Kp_leg) * err + (ki_scale * self.Ki_leg) * self.pid_int
                     + (kd_scale * self.Kd_leg) * self.pid_d_filt)
            u_sat = np.clip(u_raw, -self.Fmax, self.Fmax)
            not_sat = (u_raw == u_sat) | (np.sign(err) != np.sign(u_raw))
            self.pid_int = np.clip(self.pid_int + self.Ts * (err * not_sat), -0.08, 0.08)
            self.pid_prev_err = err
        t_law = time.perf_counter() - t_law

        # 3. Fuerza y momento externos (Fx, Fy, Fz, Mx, My, Mz) como fuerza generalizada T' w; la carga adicional
        #    se suma a la masa de la plataforma (peso e inercia)
        tau_total = np.asarray(tau_ext, dtype=float).copy()
        tau_gen = self.planta.wrench_a_generalizada(y_true, tau_total) if np.any(tau_total) else None
        tau_total[2] -= float(delta_mass_kg) * 9.80665          # para la flecha de la vista 3D

        # 4. Planta no lineal (un paso Ts con la fuerza retenida) y tope mecanico inelastico (anula la velocidad
        #    del eje que choca). plant_gain escala la parte de la fuerza que agrega el controlador (robustez).
        F_plant = self.u0 + self.plant_gain * (u_sat - self.u0)
        delta_x_next = self.planta.paso(self.x, F_plant, self.Ts, tau_gen, float(delta_mass_kg)) - self.x0
        lim = np.array([0.045] * 3 + [np.deg2rad(18.0)] * 3)
        hit = np.abs(delta_x_next[:6]) >= lim
        delta_x_next[:6] = np.clip(delta_x_next[:6], -lim, lim)
        delta_x_next[6:12][hit & (np.sign(delta_x_next[6:12]) == np.sign(delta_x_next[:6]))] = 0.0
        self.x = self.x0 + delta_x_next

        # 5. Prediccion del observador (conoce la fuerza comandada, no la perturbacion)
        t_pred = time.perf_counter()
        self.x_hat = self.Ad @ x_hat_corr + self.Bd @ (u_sat - self.u0)
        t_pred = time.perf_counter() - t_pred

        self.u_applied = u_sat
        self.t += self.Ts

        # Indices de la Tabla 1 (calcular_metricas_tabla1.m) sobre la pose real, no la medida
        e = self.x[:6] - q_ref
        L = self.compute_ik(self.x[:6])
        self.m_n += 1
        self.m_e_pos2 += float(e[:3] @ e[:3])
        self.m_e_ang2 += float(e[3:] @ e[3:])
        self.m_itae += (self.t - self.m_t0) * float(np.abs(e).sum()) * self.Ts
        self.m_ju += float(u_sat @ u_sat) * self.Ts
        self.m_fmax = max(self.m_fmax, float(np.abs(u_sat).max()))
        self.m_fsat += int(np.any(np.abs(u_raw) >= self.Fmax))
        self.m_stroke += int(np.any((L < self.Lmin) | (L > self.Lmax)))
        self.m_cpu += t_law + (t_obs + t_pred if ctrl_mode == "ss" else 0.0)

        return {
            "t": self.t,
            "pose": self.x[:6].copy(),
            "qdot": self.x[6:12].copy(),
            "x_hat_abs": (self.x0 + x_hat_corr).copy(),
            "q_ref": q_ref.copy(),
            "F": u_sat.copy(),
            "tau_ext": tau_total.copy(),
        }
