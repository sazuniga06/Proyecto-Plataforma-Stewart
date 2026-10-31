#!/usr/bin/env python3
"""
optimizar_ganancias.py
Sintonia optima de los dos controladores, cada uno por separado, sobre cinco escenarios con ruido de
sensor sigma = 0.1 mm (octave_bridge.scenario_metrics):
  1. Ensayo normalizado: escalon de referencia (dx = 8 mm, dz = 15 mm) + impulso de 20 N durante 0.45 s.
  2. Carga sostenida: el mismo escalon y, desde t = 1 s, 0.5 kg + Fz = -10 N + Fx = 4 N durante 6 s.
  3-5. Referencias dinamicas de la interfaz durante 6 s: senoidal en z, maniobra acoplada de 6 GDL y circular.

Criterio (indices de la Tabla 1, normalizados con los del diseno de cada controlador):
    J = sum_escenarios ( ITAE / ITAE_diseno + 0.25 * Ju / Ju_diseno )        J_diseno = 6.25
  el error de seguimiento (ITAE) pesa cuatro veces mas que el esfuerzo (Ju = integral F'F dt, fuerzas de las
  patas, saturadas a +-50 N).

- PID articular: factores (Kp, Ki, Kd) sobre las ganancias de controllers/controlador_pid.m, por
  Nelder-Mead en escala logaritmica desde el diseno. Si existe un optimo anterior, tambien se usa como punto
  inicial: J tiene varios minimos locales.
- Realimentacion de estados con accion integral: factores de velocidad de los polos del controlador y
  del observador, por Nelder-Mead; cada evaluacion redisena K, Ki, L y Nr en Octave (Sylvester).
- Refinamiento final por busqueda de patron (+-20, 10, 5, 2 %): con la saturacion de fuerza J es poco suave y
  Nelder-Mead puede detenerse antes del minimo; la busqueda termina solo cuando una pasada completa a +-10 %
  no mejora J.
- Verificacion: J en el optimo es menor que J con cada parametro desplazado +-10 % (minimo local).

Resultado: results/ganancias_optimas.json y results/octave_control_matrices.mat con el optimo.
La planta de los escenarios es el modelo no lineal (simulator_gui/planta_no_lineal.py); los cinco escenarios
de cada evaluacion se simulan en paralelo.
Uso:  python scripts/optimizar_ganancias.py                    (busqueda completa, varias horas)
      python scripts/optimizar_ganancias.py --desde-previo     (solo desde el optimo anterior, unos 45 min)
"""
import json
import sys
import os
import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "simulator_gui"))
from octave_bridge import OctaveRealtimeControlEngine  # noqa: E402

LOG_LO, LOG_HI = np.log(0.01), np.log(1000.0)


def refine(cost, x, lo, hi):
    """Busqueda de patron multiplicativa: prueba x_i*(1 +- h) y se mueve mientras J mejore. Repite las escalas
    hasta que una pasada completa a h = 10 % no encuentra mejora (minimo local a +-10 %)."""
    x = np.array(x, dtype=float)
    j = cost(x)
    while True:
        moved = False
        for h in (0.2, 0.1, 0.05, 0.02, 0.1):
            improved = True
            while improved:
                improved = False
                for i in range(len(x)):
                    for f in (1 + h, 1 - h):
                        xt = x.copy()
                        xt[i] = np.clip(xt[i] * f, lo, hi)
                        jt = cost(xt)
                        if jt < j - 1e-9:
                            x, j, improved, moved = xt, jt, True, True
        if not moved:
            return x, j


def verify(cost, x_opt, names):
    """J al desplazar cada parametro +-10 %; el optimo debe ser menor que todos."""
    j0 = cost(x_opt)
    out = {"J_optimo": j0}
    for i, name in enumerate(names):
        for f in (0.9, 1.1):
            x = np.array(x_opt, dtype=float)
            x[i] *= f
            out[f"{name} x{f:.1f}"] = cost(x)
    out["es_minimo_local"] = bool(all(v >= j0 - 1e-9 for k, v in out.items() if k not in ("J_optimo",)))
    return out


def optimize_pid(engine, starts=None):
    ref = engine.scenario_metrics("pid")
    cost = lambda s: engine.tuning_cost("pid", *s, ref=ref)
    if starts is None:
        starts = [[1.0, 1.0, 1.0]]
        prev = engine.load_optimal_gains().get("pid")   # optimo de una corrida anterior (J tiene varios valles)
        if prev:
            starts.append([prev["Kp_scale"], prev["Ki_scale"], prev["Kd_scale"]])
    best = None
    for start in starts:
        res = minimize(lambda z: cost(np.exp(np.clip(z, LOG_LO, LOG_HI))), np.log(start), method="Nelder-Mead",
                       options={"xatol": 1e-3, "fatol": 1e-4, "maxiter": 200})
        print(f"  inicio {np.round(start, 3)}: J = {res.fun:.4f}  factores = {np.round(np.exp(res.x), 3)}")
        if best is None or res.fun < best.fun:
            best = res
    s, _ = refine(cost, np.exp(np.clip(best.x, LOG_LO, LOG_HI)), np.exp(LOG_LO), np.exp(LOG_HI))
    print(f"  refinado: J = {cost(s):.4f}  factores = {np.round(s, 3)}")
    ver = verify(cost, s, ["Kp", "Ki", "Kd"])
    std = engine.scenario_metrics("pid", *s)[0]
    return {"Kp_scale": s[0], "Ki_scale": s[1], "Kd_scale": s[2],
            "Kp": float(s[0] * engine.Kp_leg[0]), "Ki": float(s[1] * engine.Ki_leg[0]), "Kd": float(s[2] * engine.Kd_leg[0]),
            "J": ver["J_optimo"], "J_diseno": cost(np.ones(3)), "verificacion": ver,
            "ensayo": {k: std[k] for k in ("RMS_pos", "ITAE", "Ju")},
            "ensayo_diseno": {k: ref[0][k] for k in ("RMS_pos", "ITAE", "Ju")}}


def optimize_state_feedback(engine, x0=(1.0, 1.0)):
    cache = {}

    def metrics_at(ps, os_):
        key = (round(ps, 4), round(os_, 4))
        if key not in cache:
            engine.resynthesize_in_octave(*key)
            cache[key] = engine.scenario_metrics("ss")
        return cache[key]

    ref = metrics_at(1.0, 1.0)

    def cost(f):
        return engine.cost_from_metrics(metrics_at(*f), ref)

    x0 = np.asarray(x0, dtype=float)
    res = minimize(lambda z: cost(np.exp(np.clip(z, np.log(0.2), np.log(5.0)))), np.log(x0), method="Nelder-Mead",
                   options={"xatol": 2e-3, "fatol": 1e-4, "maxiter": 60,
                            "initial_simplex": np.log([x0, x0 * [1.4, 1.0], x0 * [1.0, 1.4]])})
    f = np.round(np.exp(np.clip(res.x, np.log(0.2), np.log(5.0))), 4)
    f, _ = refine(cost, f, 0.2, 5.0)
    f = np.round(f, 4)
    ver = verify(cost, f, ["polos", "observador"])
    print(f"  J = {ver['J_optimo']:.4f}  polos x{f[0]:.3f}, observador x{f[1]:.3f}  ({len(cache)} disenos evaluados)")
    std = metrics_at(*f)[0]
    engine.resynthesize_in_octave(*f)   # deja como matrices por defecto las del optimo
    return {"pole_scale": float(f[0]), "obs_scale": float(f[1]), "J": ver["J_optimo"], "J_diseno": cost((1.0, 1.0)),
            "verificacion": ver, "ensayo": {k: std[k] for k in ("RMS_pos", "ITAE", "Ju")},
            "ensayo_diseno": {k: ref[0][k] for k in ("RMS_pos", "ITAE", "Ju")}}


CRITERIO = ("J = sum(ITAE/ITAE_diseno + 0.25*Ju/Ju_diseno) sobre cinco escenarios (fuerzas de pata saturadas a +-50 N): "
            "ensayo (escalon dx=8 mm, dz=15 mm + impulso 20 N), carga (escalon + 0.5 kg, Fz=-10 N, Fx=4 N, 6 s) y las "
            "referencias senoidal en z, maniobra 6 GDL y circular (6 s cada una); ruido de sensor 0.1 mm.")


def save(engine, pid, est):
    path = os.path.join(engine.results_dir, "ganancias_optimas.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"criterio": CRITERIO, "pid": pid, "estado": est},
                  f, indent=2, ensure_ascii=False, default=float)
    print(f"Guardado en: {path}")


def main():
    engine = OctaveRealtimeControlEngine()
    if "--solo-estados" in sys.argv:
        # Reoptimiza solo la realimentacion de estados (p. ej., tras cambiar el observador) desde el optimo previo
        prev = engine.load_optimal_gains()
        x0 = (prev["estado"]["pole_scale"], prev["estado"]["obs_scale"])
        print(f"Realimentacion de estados desde el optimo previo {x0}:")
        est = optimize_state_feedback(engine, x0)
        print(f"  Estados: minimo local verificado = {est['verificacion']['es_minimo_local']}")
        save(engine, prev["pid"], est)
        return
    prev = engine.load_optimal_gains() if "--desde-previo" in sys.argv else {}
    print("Realimentacion de estados con accion integral (Nelder-Mead sobre factores de polos):")
    est = optimize_state_feedback(engine, (prev["estado"]["pole_scale"], prev["estado"]["obs_scale"]) if prev else (1.0, 1.0))
    print("PID articular (Nelder-Mead):")
    pid = optimize_pid(engine, [[prev["pid"][k] for k in ("Kp_scale", "Ki_scale", "Kd_scale")]] if prev else None)
    print(f"  optimo: Kp = {pid['Kp']:.1f}, Ki = {pid['Ki']:.1f}, Kd = {pid['Kd']:.2f}  J = {pid['J']:.4f}")
    for name, r in (("PID", pid), ("Estados", est)):
        print(f"  {name}: minimo local verificado = {r['verificacion']['es_minimo_local']}")
    save(engine, pid, est)


if __name__ == "__main__":
    main()
