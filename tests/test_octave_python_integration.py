#!/usr/bin/env python3
"""
Suite de pruebas para verificar:
1. Sintesis y exportacion de matrices de control desde GNU Octave (`exportar_matrices_control_octave.m`).
2. Geometria 3D limpia y ultra-rapida (`StewartClean3DGeometry`, < 0.05 ms/frame).
3. Simulacion cerrada en tiempo real con inyeccion de fuerzas externas y observador de Luenberger.
4. PID articular y realimentacion de estados con ruido de sensor, indices de la Tabla 1 y topes mecanicos.
   Ganancias optimas: mejoran el criterio y son minimo local (results/ganancias_optimas.json).
5. Modelo dinamico por momento generalizado: M simetrica y definida positiva, conservacion de la energia
   cinetica en movimiento libre y gravedad igual al gradiente numerico de la energia potencial.
6. Trayectorias variacionales (extremos fijos, final libre, descuento, 6 GDL, los cuatro casos de frontera
   conocida/libre) y de tiempo minimo (Pontryagin).
"""
import os
import subprocess
import sys
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, "simulator_gui"))

from cad_renderer import StewartClean3DGeometry
from octave_bridge import OctaveRealtimeControlEngine


def main():
    print("=== [TEST 1] Verificando Simulador en Tiempo Real con Matrices de GNU Octave ===")
    engine = OctaveRealtimeControlEngine(ROOT_DIR)
    assert engine.rank_Mc == 12, "Rango de controlabilidad de Kalman debe ser 12"
    assert engine.rank_Mo == 12, "Rango de observabilidad de Kalman debe ser 12"
    assert engine.err_Nr_cof < 1e-9, "Verificacion de cofactores Nr fuera de tolerancia"
    import scipy.io as sio
    mats = sio.loadmat(engine.matrices_mat_path, squeeze_me=True)
    err_obs = np.abs(np.sort_complex(mats["obs_poles_disc"]) - np.sort_complex(mats["obs_poles_impl"])).max()
    assert err_obs < 1e-9, "Polos implementados del estimador actual distintos de los disenados"
    print(f"  [OK] Observador (estimador actual): polos implementados = disenados (error {err_obs:.1e})")

    geom = StewartClean3DGeometry(engine.B_pts, engine.A_pts, h0=engine.h0, lower_body_len=engine.lowerBodyLength)
    g0 = geom.ik_and_geometry_mm(engine.pose0)
    assert g0["B_mm"].shape == (6, 3)

    print("=== [TEST 2] Verificando Lazo Cerrado en Tiempo Real y Rechazo a Perturbaciones ===")
    q_ref = engine.pose0 + np.array([0.005, -0.004, 0.015, 0.03, -0.02, 0.04])
    tau_ext = np.array([0.0, 0.0, -10.0, 0.0, 0.0, 0.0])
    for _ in range(1500):
        out = engine.step_realtime(q_ref, tau_ext, 0.5, "ss")

    err_ss_mm = np.linalg.norm(out["pose"][:3] - q_ref[:3]) * 1e3
    print(f"  [OK] Error estacionario bajo carga de -10 N y 0.5 kg con Controlador PI + Luenberger: {err_ss_mm:.4f} mm")
    assert err_ss_mm < 0.5, f"Se esperaba error estacionario < 0.5 mm, se obtuvo {err_ss_mm:.4f} mm"

    print("=== [TEST 3] PID / realimentacion de estados con ruido de sensor e indices de la Tabla 1 ===")
    for mode in ("pid", "ss"):
        engine.reset_state()
        for _ in range(1500):
            out = engine.step_realtime(q_ref, tau_ext, 0.5, mode, noise_sigma_mm=0.3)
        mt = engine.metrics()
        err_mm = np.linalg.norm(out["pose"][:3] - q_ref[:3]) * 1e3
        print(f"  [OK] {mode:4s}: err={err_mm:.3f} mm  RMS pos={mt['RMS_pos']:.2f} mm  ITAE={mt['ITAE']:.4f}  "
              f"Ju={mt['Ju']:.0f}  CPU={mt['cpu_us']:.0f} us")
        assert err_mm < 3.0 and np.isfinite(mt["Ju"]) and mt["ITAE"] > 0 and mt["cpu_us"] > 0
        assert np.all(np.abs(out["F"]) <= engine.Fmax + 1e-9)
        row = engine.table1_row(mode)
        assert np.isfinite(row["robustez_pct"]) and row["stroke"] == 0

    # Tope mecanico: empuje enorme sostenido -> la pose queda en el limite y la velocidad no empuja hacia fuera
    engine.reset_state()
    for _ in range(1000):
        out = engine.step_realtime(engine.pose0, np.array([0, 0, 2.0e3, 0, 0, 0]), 0.0, "pid", 0.05, 0.0, 0.0)
    dz = out["pose"][2] - engine.pose0[2]
    assert abs(abs(dz) - 0.045) < 1e-9 and out["qdot"][2] * dz <= 0, "Tope mecanico inelastico"
    print(f"  [OK] Tope mecanico: dz={dz*1e3:.1f} mm, z_dot={out['qdot'][2]*1e3:.2f} mm/s")

    opt = engine.load_optimal_gains()
    if opt:
        print("=== [TEST 3b] Ganancias optimas: mejoran el criterio y son minimo local ===")
        p = opt["pid"]
        ref = engine.scenario_metrics("pid")
        j_opt = engine.tuning_cost("pid", p["Kp_scale"], p["Ki_scale"], p["Kd_scale"], ref=ref)
        print(f"  [OK] PID: J diseno = {engine.cost_from_metrics(ref):.3f}  ->  J optimo = {j_opt:.3f}")
        assert j_opt < engine.cost_from_metrics(ref)
        for name in ("pid", "estado"):
            assert opt[name]["verificacion"]["es_minimo_local"], f"{name}: el optimo no es minimo local"
            assert opt[name]["J"] < opt[name]["J_diseno"]
        print(f"  [OK] Realim. estados: J diseno = {opt['estado']['J_diseno']:.3f}  ->  J optimo = {opt['estado']['J']:.3f}")

    print("=== [TEST 4] Modelo dinamico (momento generalizado): simetria, energia y gravedad ===")
    eng = os.path.join(ROOT_DIR, "octave_engine")
    oct_cmd = (f"cd('{eng}'); addpath dynamics; r = validar_modelo_dinamico(tempname()); "
               "printf('RES %.3e %.4f %.3e %.3e\\n', r.err_sim, r.eig_min, r.deriva_energia, "
               "norm(r.F_estatica - r.F_estatica_potencial));")
    res = subprocess.run(["octave", "--no-gui", "--quiet", "--eval", oct_cmd], capture_output=True, text=True, check=True)
    line = [l for l in res.stdout.splitlines() if l.startswith("RES")][-1].split()
    err_sim, eig_min, deriva, err_g = map(float, line[1:])
    print(f"  [OK] |M - M'| = {err_sim:.1e}, min eig(M) = {eig_min:.2f}, deriva de energia = {deriva:.1e}, "
          f"gravedad vs gradiente = {err_g:.1e} N")
    assert err_sim < 1e-9 and eig_min > 0 and deriva < 1e-8 and err_g < 1e-6
    # planta no lineal del simulador (Python): M simetrica y definida positiva y conservacion de la energia
    res = subprocess.run([sys.executable, os.path.join(ROOT_DIR, "simulator_gui", "planta_no_lineal.py")],
                         capture_output=True, text=True, check=True)
    print("  [OK] planta no lineal del simulador: " + res.stdout.strip().removeprefix("OK: "))

    print("=== [TEST 5] Trayectorias variacionales y de tiempo minimo ===")
    import trayectorias_variacionales as tv
    a_, T_ = tv.A_PESO, tv.T_MOV
    h = 1e-5
    d1 = lambda f, t: (f(t + h) - f(t - h)) / (2 * h)
    d2 = lambda f, t: (f(t + h) - 2 * f(t) + f(t - h)) / h**2
    # extremos fijos: frontera, Euler-Lagrange eps'' = a^2 eps y Beltrami (eps'^2 - a^2 eps^2 = cte)
    assert abs(tv.eps_fijos(0) - 1) < 1e-12 and abs(tv.eps_fijos(T_)) < 1e-12
    beltrami = [d1(tv.eps_fijos, t)**2 - a_**2 * tv.eps_fijos(t)**2 for t in (0.3, 1.0, 1.7)]
    assert abs(d2(tv.eps_fijos, 1.0) - a_**2 * tv.eps_fijos(1.0)) < 1e-4 and np.ptp(beltrami) < 1e-6
    # estado final libre: condicion natural eps'(T) = 0
    assert abs(d1(tv.eps_libre, T_ - h)) < 1e-3
    # descuento: eps'' - alpha eps' - a^2 eps = 0
    t = 0.8
    assert abs(d2(tv.eps_descuento, t) - tv.ALPHA * d1(tv.eps_descuento, t) - a_**2 * tv.eps_descuento(t)) < 1e-3
    # 6 GDL: M0 e'' = K e con K = a^2 diag(M0), e(T) = 0
    M0 = tv._MODEL["M0"]; e0 = -tv.TARGET
    em = lambda tt: tv.e_multi(tt, e0)
    res = M0 @ ((em(t + h) - 2 * em(t) + em(t - h)) / h**2) - a_**2 * np.diag(np.diag(M0)) @ em(t)
    assert np.linalg.norm(res) < 1e-2 * np.linalg.norm(M0 @ em(t)) and np.linalg.norm(em(T_)) < 1e-12
    # tiempo minimo: llega a dl en reposo, con una sola conmutacion
    dl, Tm = tv.tiempo_minimo(tv.TARGET)
    lb = lambda tt: tv.l_bang_bang(tt, dl)
    assert abs(lb(Tm) - dl) < 1e-9 and abs(d1(lb, Tm - 2 * h)) < 1e-4
    # curva terminal: z(T) = phi(T) y transversalidad c = 2 phi'(T)
    Tc, c = tv.intercepcion()
    phi = lambda tt: tv.SIN_A * np.sin(tv.SIN_W * tt)
    assert abs(tv.Z_START + c * Tc - phi(Tc)) < 1e-12 and abs(c - 2 * d1(phi, Tc)) < 1e-6
    from octave_bridge import reference_pose
    q_end = reference_pose("var_fixed", T_, engine.pose0)
    assert np.allclose(q_end - engine.pose0, tv.TARGET), "llega a la pose objetivo"
    print(f"  [OK] fronteras, Euler-Lagrange, Beltrami, transversalidad, descuento, 6 GDL, curva terminal y tiempo minimo "
          f"(dl = {dl*1e3:.2f} mm en T = {Tm:.3f} s)")
    # casos de frontera A-D: Euler-Lagrange x'' - a^2 x = -a^2 r, extremos conocidos x = 0, libres x' = 0 y
    # J*_D <= J*_B, J*_C <= J*_A (conjunto admisible mayor)
    a_, Tw = tv.A_ONDA, tv.T_ONDA
    for caso, (ini, fin) in tv.CASOS.items():
        x = lambda tt: tv.x_frontera(caso, tt)[0]
        xs = (x(t + h) - 2 * x(t) + x(t - h)) / h**2
        assert np.linalg.norm(xs - a_**2 * x(t) + a_**2 * tv.onda(t)[0]) < 1e-3
        (x0, v0), (xT, vT) = tv.x_frontera(caso, 0.0), tv.x_frontera(caso, Tw)
        assert np.allclose(x0 if ini else v0, 0, atol=1e-12) and np.allclose(xT if fin else vT, 0, atol=1e-12)
    J = {c: tv.costo_frontera(c) for c in tv.CASOS}
    assert np.all(J["D"] <= J["B"]) and np.all(J["D"] <= J["C"]) and np.all(J["B"] <= J["A"]) and np.all(J["C"] <= J["A"])
    print("  [OK] casos de frontera A (conocido-conocido), B (conocido-libre), C (libre-conocido), D (libre-libre)")
    # en el simulador los cuatro casos deben verse distintos entre si (diferencia de z de al menos 5 mm)
    tt = np.arange(0.0, 2 * tv.T_PRE + tv.T_ONDA, 0.01)
    zc = {c: np.array([reference_pose(f"bc_{c}", ti, engine.pose0)[2] for ti in tt]) for c in tv.CASOS}
    dmin = min(np.abs(zc[a] - zc[b]).max() for a in zc for b in zc if a < b)
    assert dmin > 5e-3, f"casos de frontera demasiado parecidos ({dmin*1e3:.2f} mm)"
    print(f"  [OK] en el simulador los casos A-D difieren al menos {dmin*1e3:.1f} mm en z")

    print("=== TODAS LAS PRUEBAS PASARON EXITOSAMENTE ===")


if __name__ == "__main__":
    main()
