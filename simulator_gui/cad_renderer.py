#!/usr/bin/env python3
"""
cad_renderer.py
Geometria cinematica para la visualizacion 3D de la plataforma Stewart 6-SPS
(base y plataforma hexagonales, cuerpo y vastago de cada actuador).
"""
import numpy as np


class StewartClean3DGeometry:
    """Calcula la cinematica inversa y los segmentos 3D de las 6 patas y placas hexagonales."""

    # Paleta cualitativa (tab10 atenuada); el mismo color identifica la pata y su tension V_i
    LEG_COLORS = ["#3b6ea5", "#d08a2e", "#4f9a5a", "#b5473f", "#7d62a8", "#8a6a4f"]

    def __init__(self, B_pts: np.ndarray, A_pts: np.ndarray, h0: float = 0.270, lower_body_len: float = 0.160):
        # B_pts y A_pts son 6x3 en metros (definidos en config/stewart_params.m)
        self.B_m = np.asarray(B_pts, dtype=float)
        self.A_m = np.asarray(A_pts, dtype=float)
        self.h0 = float(h0)
        self.lower_body_len = float(lower_body_len)

        # Orden ciclico de los 6 vertices para cerrar el hexagono de base y plataforma
        ang_b = np.arctan2(self.B_m[:, 1], self.B_m[:, 0])
        self.base_order = np.argsort(ang_b)
        self.base_loop_idx = np.append(self.base_order, self.base_order[0])

        ang_a = np.arctan2(self.A_m[:, 1], self.A_m[:, 0])
        self.plat_order = np.argsort(ang_a)
        self.plat_loop_idx = np.append(self.plat_order, self.plat_order[0])

    @staticmethod
    def rpy_to_rot(roll: float, pitch: float, yaw: float) -> np.ndarray:
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        return np.array([
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp,     cp * sr,                cp * cr               ]
        ], dtype=float)

    def ik_and_geometry_mm(self, pose: np.ndarray) -> dict:
        """
        Dada la pose q = [x, y, z, roll, pitch, yaw] (m, rad),
        retorna en milimetros las coordenadas de la base, plataforma, cuerpos de los actuadores,
        vastagos y la orientacion R del marco movil.
        """
        tr = np.asarray(pose[:3], dtype=float)
        R = self.rpy_to_rot(float(pose[3]), float(pose[4]), float(pose[5]))

        # Puntos superiores Q_i = tr + R * a_i (6x3 en metros)
        Q_m = tr[None, :] + self.A_m @ R.T
        D_m = Q_m - self.B_m
        L_m = np.linalg.norm(D_m, axis=1)
        U = D_m / np.maximum(L_m[:, None], 1e-9)

        body_len = np.minimum(self.lower_body_len, 0.58 * L_m)
        M_m = self.B_m + body_len[:, None] * U

        B_mm = self.B_m * 1e3
        Q_mm = Q_m * 1e3
        M_mm = M_m * 1e3
        center_mm = tr * 1e3

        return {
            "B_mm": B_mm,
            "Q_mm": Q_mm,
            "M_mm": M_mm,
            "base_loop_mm": B_mm[self.base_loop_idx],
            "plat_loop_mm": Q_mm[self.plat_loop_idx],
            "plat_ordered_mm": Q_mm[self.plat_order],
            "center_mm": center_mm,
            "R": R,
            "L_m": L_m,
            "U": U,
        }
