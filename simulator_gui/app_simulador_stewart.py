#!/usr/bin/env python3
"""
app_simulador_stewart.py
Simulador interactivo de la plataforma Stewart 6-SPS.

Interfaz Tkinter + Matplotlib para evaluar en lazo cerrado dos estrategias: PID descentralizado en
espacio articular y realimentacion de estados con accion integral + observador de Luenberger. Las matrices de diseno se calculan en el motor
de control (octave_engine/) y se cargan a traves de octave_bridge.py.

Notas de implementacion:
- La planta avanza sincronizada con el reloj de pared (tiempo real 1:1).
- Vista 3D con OpenGL (vista_3d.py); si no hay soporte, se usa Matplotlib.
- Osciloscopios con blitting: ejes, rejillas y leyendas se cachean; por cuadro solo se
  redibujan las lineas que cambian.
"""
import copy
import os
import sys
import time
import threading
from collections import deque
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from cad_renderer import StewartClean3DGeometry
from octave_bridge import (OctaveRealtimeControlEngine, TEST_STEP_T, TEST_IMPULSE_T, TEST_END_T,
                           TEST_STEP, IMPULSE_DUR, IMPULSE_DIR, TEST_IMPULSE_N, TEST_NOISE_MM, ROBUST_GAINS,
                           reference_pose)
from vista_3d import create_view

matplotlib.rcParams.update({
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "legend.frameon": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#6b6b6b",
    "xtick.color": "#3a3a3a",
    "ytick.color": "#3a3a3a",
})

# Paleta sobria: texto casi negro, un unico color de acento y grises neutros
BG = "#f3f3f1"
PANEL = "#fbfbfa"
TEXT = "#1c1c1c"
MUTED = "#6b6b6b"
ACCENT = "#1f4e79"
BORDER = "#c9c9c4"
GRID = "#dedede"
# Ejes x, y, z con la convencion RGB de la triada
C_X, C_Y, C_Z = "#b03a2e", "#2e7d32", "#1f4e79"


class StewartRealtimeSimulatorGUI:
    """Ventana principal del simulador."""

    CTRL_MODES = [
        ("PID", "pid"),
        ("Realimentación de estados", "ss"),
    ]
    SHORT_NAMES = {"pid": "PID", "ss": "Realimentación de estados"}

    REF_MODES = [
        ("Manual (deslizadores)", "manual"),
        ("Senoidal en z: ±15 mm, 0.45 Hz", "sine_z"),
        ("Maniobra acoplada de 6 GDL", "coupled_6dof"),
        ("Circular en x–y con inclinación", "circle_3d"),
        ("Frontera A: inicio conocido, final conocido", "bc_A"),
        ("Frontera B: inicio conocido, final libre", "bc_B"),
        ("Frontera C: inicio libre, final conocido", "bc_C"),
        ("Frontera D: inicio y final libres", "bc_D"),
        ("Variacional: extremos fijos (Beltrami)", "var_fixed"),
        ("Tiempo mínimo: bang-bang (Pontryagin)", "min_time"),
        ("Variacional: curva terminal (intercepción de la senoidal)", "var_curve"),
    ]

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Plataforma Stewart 6-SPS — Simulador de control")
        w = min(1600, root.winfo_screenwidth() - 40)
        h = min(900, root.winfo_screenheight() - 80)
        self.root.geometry(f"{w}x{h}")
        self.root.minsize(1200, 720)
        self.root.configure(bg=BG)

        self.engine = OctaveRealtimeControlEngine()
        # Los redisenos hechos desde la interfaz se guardan en un archivo de sesion: el archivo del proyecto
        # (results/octave_control_matrices.mat) conserva siempre las matrices optimas
        import tempfile
        self.engine.matrices_mat_path = os.path.join(tempfile.gettempdir(), "stewart_matrices_sesion.mat")
        self.geom = StewartClean3DGeometry(
            self.engine.B_pts, self.engine.A_pts,
            h0=self.engine.h0, lower_body_len=self.engine.lowerBodyLength
        )

        # Ganancias del PID (iguales en las 6 patas): diseno (controlador_pid.m) y optimo (scripts/optimizar_ganancias.py)
        self.opt = self.engine.load_optimal_gains()
        e = self.engine
        self.pid_base = [float(e.Kp_leg[0]), float(e.Ki_leg[0]), float(e.Kd_leg[0])]
        o = self.opt.get("pid")
        self.pid_opt = [float(o[k]) for k in ("Kp", "Ki", "Kd")] if o else list(self.pid_base)
        self.pid_gain = list(self.pid_opt)

        self.is_running = True
        self.impulse_steps_remaining = 0
        self.impulse_force_vec = np.zeros(6)
        self._test_t0 = None      # instante de inicio del ensayo automatico (None = sin ensayo)
        self._ref_t0 = 0.0        # instante de inicio de la referencia seleccionada (trayectorias desde su comienzo)
        self._ghost = None        # curvas del ensayo anterior para comparacion
        self._frame = 0

        # Ventana del osciloscopio: 5 s (2500 muestras a Ts = 2 ms)
        self.hist_len = int(round(5.0 / self.engine.Ts))
        self.t_buf = deque(np.linspace(-5.0, 0.0, self.hist_len), maxlen=self.hist_len)
        zeros = lambda: deque(np.zeros(self.hist_len), maxlen=self.hist_len)
        self.dx_ref_buf, self.dy_ref_buf, self.dz_ref_buf = zeros(), zeros(), zeros()
        self.dx_buf, self.dy_buf, self.dz_buf = zeros(), zeros(), zeros()
        self.v_bufs = [zeros() for _ in range(6)]
        self.vz_real_buf, self.vz_hat_buf = zeros(), zeros()

        self._setup_styles()
        self._init_vars()
        self._build_ui()
        self._init_plot_artists()
        self._bind_keys()
        self._on_ctrl_change()

        # Blitting: el fondo se recaptura en cada redibujado completo (inicio, cambio de tamano, rotacion 3D)
        self._bg = None
        self.canvas.mpl_connect("draw_event", self._on_full_draw)

        self._t_last = time.perf_counter()
        self._t_acc = 0.0
        self._fps = 0.0
        self._rt = 1.0
        self._last_ctrl = None
        self.root.after(15, self._realtime_loop)

    # ------------------------------------------------------------------ estilos y variables
    def _setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(".", background=PANEL, foreground=TEXT, bordercolor=BORDER,
                        lightcolor=PANEL, darkcolor=BORDER, troughcolor="#e4e4e0")
        style.configure("Main.TFrame", background=BG)
        style.configure("TFrame", background=PANEL)
        style.configure("TLabel", background=PANEL, foreground=TEXT)
        style.configure("Muted.TLabel", background=PANEL, foreground=MUTED)
        style.configure("Value.TLabel", background=PANEL, foreground=TEXT, font=("DejaVu Sans Mono", 9))
        style.configure("TLabelframe", background=PANEL, bordercolor=BORDER)
        style.configure("TLabelframe.Label", background=PANEL, foreground=ACCENT, font=("TkDefaultFont", 9, "bold"))
        style.configure("TNotebook", background=BG, bordercolor=BORDER, tabmargins=(0, 2, 0, 0))
        style.configure("TNotebook.Tab", background="#e6e6e2", foreground=TEXT, padding=(8, 4), bordercolor=BORDER)
        style.map("TNotebook.Tab", background=[("selected", PANEL)], foreground=[("selected", ACCENT)])
        style.configure("TButton", background="#e9e9e5", foreground=TEXT, padding=(8, 4), bordercolor=BORDER)
        style.map("TButton", background=[("active", "#dcdcd7"), ("disabled", "#efefec")],
                  foreground=[("disabled", "#a0a0a0")])
        style.configure("Primary.TButton", background=ACCENT, foreground="#ffffff", bordercolor=ACCENT)
        style.map("Primary.TButton", background=[("active", "#173b5c"), ("disabled", "#8aa1b8")])
        style.configure("Horizontal.TScale", background=PANEL)
        style.configure("Active.TLabel", background=PANEL, foreground=ACCENT, font=("TkDefaultFont", 9, "bold"))
        style.configure("TRadiobutton", background=PANEL, foreground=TEXT)
        style.configure("Compact.TEntry", padding=(2, 0))
        style.configure("Status.TFrame", background="#e6e6e2")
        style.configure("Status.TLabel", background="#e6e6e2", foreground=TEXT, font=("DejaVu Sans Mono", 9))
        style.configure("StatusMuted.TLabel", background="#e6e6e2", foreground=MUTED)

    def _init_vars(self):
        self.var_ctrl = tk.StringVar(value=self.CTRL_MODES[0][0])
        self.var_ref_mode = tk.StringVar(value=self.REF_MODES[0][0])

        self.var_dx, self.var_dy, self.var_dz = tk.DoubleVar(), tk.DoubleVar(), tk.DoubleVar()
        self.var_roll, self.var_pitch, self.var_yaw = tk.DoubleVar(), tk.DoubleVar(), tk.DoubleVar()

        self.var_fx, self.var_fy, self.var_fz = tk.DoubleVar(), tk.DoubleVar(), tk.DoubleVar()
        self.var_mx, self.var_my, self.var_mz = tk.DoubleVar(), tk.DoubleVar(), tk.DoubleVar()
        self.var_dmass = tk.DoubleVar()
        self.var_noise = tk.DoubleVar()
        self.var_impulse = tk.DoubleVar(value=TEST_IMPULSE_N)
        self.var_auto_test = tk.BooleanVar(value=True)
        # Perturbacion aplicada: los deslizadores solo la preparan; se aplica con el boton "Aplicar"
        self.applied_tau = np.zeros(6)
        self.applied_dmass = 0.0
        self._value_commits = []   # confirman lo escrito en los campos de perturbacion aunque no se pulse Enter

        # Ganancias PID absolutas (se aplican en linea) y factores de polos (requieren rediseno)
        self.var_kp = tk.DoubleVar()
        self.var_ki = tk.DoubleVar()
        self.var_kd = tk.DoubleVar()
        for k, var in enumerate((self.var_kp, self.var_ki, self.var_kd)):
            var.set(self.pid_gain[k])
            var.trace_add("write", lambda *_a, k=k, var=var: self.pid_gain.__setitem__(k, float(var.get())))
        self.var_pole_scale = tk.DoubleVar(value=self.engine.pole_scale)
        self.var_obs_scale = tk.DoubleVar(value=self.engine.obs_scale)

        self.var_status = tk.StringVar()
        self.var_msg = tk.StringVar()
        self.var_perf = tk.StringVar()
        self.metric_vars = {k: tk.StringVar(value="—") for k in ("RMS_pos", "RMS_ang", "ITAE", "Ju", "fsat_pct", "cpu_us")}

    # ------------------------------------------------------------------ construccion de la interfaz
    def _slider(self, parent, label_txt, var, vmin, vmax, unit="", fmt="{:+.1f}"):
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=1)
        ttk.Label(row, text=label_txt, width=14).pack(side=tk.LEFT)
        lbl_val = ttk.Label(row, width=10, anchor="e", style="Value.TLabel")
        lbl_val.pack(side=tk.RIGHT)
        sc = ttk.Scale(row, variable=var, from_=vmin, to=vmax, orient=tk.HORIZONTAL)
        sc.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=6)

        def refresh(*_):
            lbl_val.config(text=f"{fmt.format(var.get())} {unit}".rstrip())
        var.trace_add("write", refresh)
        refresh()
        return sc

    def _gain_row(self, parent, label_txt, var, vmin, vmax):
        """Parametro con deslizador y campo numerico editable (Enter)."""
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=1)
        ttk.Label(row, text=label_txt, width=12).pack(side=tk.LEFT)
        txt = tk.StringVar()
        ent = ttk.Entry(row, textvariable=txt, width=10, justify="right", font=("DejaVu Sans Mono", 9), style="Compact.TEntry")
        ent.pack(side=tk.RIGHT, padx=(4, 0))
        sc = ttk.Scale(row, variable=var, from_=vmin, to=vmax, orient=tk.HORIZONTAL)
        sc.pack(side=tk.RIGHT, fill=tk.X, expand=True)

        def show(*_):
            txt.set(f"{var.get():.4f}")

        def commit(_e=None):
            try:
                v = max(vmin, float(txt.get().replace(",", ".")))
            except ValueError:
                return show()
            if v > float(sc.cget("to")):
                sc.configure(to=1.5 * v)
            var.set(v)
            self.root.focus_set()
        var.trace_add("write", show)
        ent.bind("<Return>", commit)
        ent.bind("<FocusOut>", commit)
        show()
        return [sc, ent]

    def _value_row(self, parent, label_txt, var, vmin, vmax, unit, decimals=1):
        """Perturbacion con deslizador y campo numerico (Enter). Admite valores fuera del rango del deslizador,
        que se amplia; si vmin < 0 el valor puede ser negativo y el rango se amplia de forma simetrica."""
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=1)
        ttk.Label(row, text=label_txt, width=10).pack(side=tk.LEFT)
        ttk.Label(row, text=unit, width=4, style="Muted.TLabel").pack(side=tk.RIGHT)
        txt = tk.StringVar()
        ent = ttk.Entry(row, textvariable=txt, width=9, justify="right", font=("DejaVu Sans Mono", 9), style="Compact.TEntry")
        ent.pack(side=tk.RIGHT, padx=(4, 2))
        sc = ttk.Scale(row, variable=var, from_=vmin, to=vmax, orient=tk.HORIZONTAL)
        sc.pack(side=tk.RIGHT, fill=tk.X, expand=True)

        def show(*_):
            txt.set(f"{var.get():.{decimals}f}")

        def commit(_e=None):
            try:
                v = float(txt.get().replace(",", "."))
            except ValueError:
                return show()
            v = v if vmin < 0 else max(0.0, v)
            lim = max(abs(float(sc.cget("from"))), abs(float(sc.cget("to"))))
            if abs(v) > lim:
                new = 1.5 * abs(v)
                sc.configure(from_=-new if vmin < 0 else 0.0, to=new)
            var.set(v)
            self.root.focus_set()
        var.trace_add("write", show)
        ent.bind("<Return>", commit)
        ent.bind("<FocusOut>", commit)
        show()
        self._value_commits.append(commit)
        return sc

    def _build_ui(self):
        status = ttk.Frame(self.root, style="Status.TFrame", padding=(10, 3))
        status.pack(side=tk.BOTTOM, fill=tk.X)
        ttk.Label(status, textvariable=self.var_status, style="Status.TLabel").pack(side=tk.LEFT)
        ttk.Label(status, textvariable=self.var_perf, style="StatusMuted.TLabel").pack(side=tk.RIGHT)
        ttk.Label(status, textvariable=self.var_msg, style="StatusMuted.TLabel").pack(side=tk.RIGHT, padx=16)

        body = ttk.Frame(self.root, style="Main.TFrame")
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=8)

        left = ttk.Frame(body, style="Main.TFrame", width=370)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 8))
        left.pack_propagate(False)
        right = ttk.Frame(body, style="Main.TFrame")
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._build_sidebar(left)
        self._build_plots(right)

    def _build_sidebar(self, parent):
        nb = ttk.Notebook(parent)
        nb.pack(fill=tk.X)
        tab_c = ttk.Frame(nb, padding=8)
        tab_r = ttk.Frame(nb, padding=8)
        tab_f = ttk.Frame(nb, padding=8)
        nb.add(tab_c, text="Controlador")
        nb.add(tab_r, text="Referencia")
        nb.add(tab_f, text="Perturbaciones")

        # Controlador
        cb_ctrl = ttk.Combobox(tab_c, textvariable=self.var_ctrl, values=[x[0] for x in self.CTRL_MODES], state="readonly")
        cb_ctrl.pack(fill=tk.X, pady=(0, 4))
        cb_ctrl.bind("<<ComboboxSelected>>", self._on_ctrl_change)
        test_row = ttk.Frame(tab_c)
        test_row.pack(fill=tk.X, pady=(0, 4))
        ttk.Button(test_row, text="Ejecutar ensayo", style="Primary.TButton", command=self.start_test).pack(side=tk.LEFT)
        ttk.Checkbutton(test_row, text="al cambiar de estrategia", variable=self.var_auto_test).pack(side=tk.LEFT, padx=8)

        o = self.pid_opt
        self.pid_box = pid_box = ttk.LabelFrame(tab_c, text="PID por pata (ganancias óptimas)", padding=(8, 4))
        pid_box.pack(fill=tk.X, pady=(0, 4))
        self.pid_widgets = [
            self._gain_row(pid_box, "Kp [N/m]", self.var_kp, 0.0, 10.0 * o[0]),
            self._gain_row(pid_box, "Ki [N/(m·s)]", self.var_ki, 0.0, 10.0 * o[1]),
            self._gain_row(pid_box, "Kd [N·s/m]", self.var_kd, 0.0, 10.0 * o[2]),
        ]

        ttk.Button(pid_box, text="Restablecer ganancias óptimas", command=self._reset_pid_gains).pack(fill=tk.X, pady=(4, 0))

        self.ss_box = ss_box = ttk.LabelFrame(tab_c, text="Realimentación de estados: asignación de polos", padding=(8, 4))
        ss_box.pack(fill=tk.X)
        self.ss_widgets = [
            self._gain_row(ss_box, "Controlador ×", self.var_pole_scale, 0.1, 5.0),
            self._gain_row(ss_box, "Observador ×", self.var_obs_scale, 0.1, 5.0),
        ]
        self.lbl_design = ttk.Label(ss_box, style="Muted.TLabel", justify=tk.LEFT)
        self.lbl_design.pack(anchor="w", pady=(2, 0))
        ss_btns = ttk.Frame(ss_box)
        ss_btns.pack(fill=tk.X, pady=(4, 0))
        self.btn_redesign = ttk.Button(ss_btns, text="Rediseñar", style="Primary.TButton", command=self.trigger_redesign)
        self.btn_redesign.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 2))
        ttk.Button(ss_btns, text="Matrices", command=self.show_matrices).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(2, 0))
        self._refresh_design_info()

        # Referencia
        ttk.Label(tab_r, text="Generador de referencia").pack(anchor="w")
        cb_ref = ttk.Combobox(tab_r, textvariable=self.var_ref_mode, values=[x[0] for x in self.REF_MODES], state="readonly")
        cb_ref.pack(fill=tk.X, pady=(2, 8))
        cb_ref.bind("<<ComboboxSelected>>", lambda _e: self._on_ref_change())
        self._slider(tab_r, "Δx", self.var_dx, -25.0, 25.0, "mm")
        self._slider(tab_r, "Δy", self.var_dy, -25.0, 25.0, "mm")
        self._slider(tab_r, "Δz", self.var_dz, -25.0, 25.0, "mm")
        self._slider(tab_r, "Roll φ", self.var_roll, -10.0, 10.0, "°")
        self._slider(tab_r, "Pitch θ", self.var_pitch, -10.0, 10.0, "°")
        self._slider(tab_r, "Yaw ψ", self.var_yaw, -12.0, 12.0, "°")
        ttk.Button(tab_r, text="Volver a la pose nominal",
                   command=lambda: self._set_pose_sliders(0, 0, 0, 0, 0, 0)).pack(fill=tk.X, pady=(8, 0))

        # Perturbaciones
        self._value_row(tab_f, "Fx", self.var_fx, -100.0, 100.0, "N", 1)
        self._value_row(tab_f, "Fy", self.var_fy, -100.0, 100.0, "N", 1)
        self._value_row(tab_f, "Fz", self.var_fz, -150.0, 150.0, "N", 1)
        self._value_row(tab_f, "Mx", self.var_mx, -5.0, 5.0, "N·m", 2)
        self._value_row(tab_f, "My", self.var_my, -5.0, 5.0, "N·m", 2)
        self._value_row(tab_f, "Mz", self.var_mz, -5.0, 5.0, "N·m", 2)
        self._value_row(tab_f, "Carga Δm", self.var_dmass, 0.0, 5.0, "kg", 2)
        self._value_row(tab_f, "Ruido σ", self.var_noise, 0.0, 5.0, "mm", 3)
        self._value_row(tab_f, "Impulso", self.var_impulse, 0.0, 200.0, "N", 1)
        row = ttk.Frame(tab_f)
        row.pack(fill=tk.X, pady=(8, 0))
        ttk.Button(row, text="Aplicar fuerza y carga", style="Primary.TButton",
                   command=lambda: (self._commit_values(), self.apply_forces())).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 3))
        ttk.Button(row, text="Anular", command=self.clear_disturbances).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=(3, 0))
        ttk.Button(tab_f, text="Aplicar impulso (0.45 s, magnitud indicada)",
                   command=lambda: (self._commit_values(), self.inject_vertical_impulse())).pack(fill=tk.X, pady=(4, 0))
        self.lbl_applied = ttk.Label(tab_f, style="Muted.TLabel")
        self.lbl_applied.pack(anchor="w", pady=(4, 0))
        self.apply_forces()

        # Indices de desempeno
        m_box = ttk.LabelFrame(parent, text="Índices de desempeño", padding=8)
        m_box.pack(fill=tk.X, pady=(8, 0))
        cells = [("RMS pos", "RMS_pos", "mm"), ("RMS ang", "RMS_ang", "°"), ("ITAE", "ITAE", "SI"),
                 ("Ju", "Ju", "N²·s"), ("F sat", "fsat_pct", "%"), ("CPU", "cpu_us", "µs")]
        for k, (name, key, unit) in enumerate(cells):
            r, c0 = k // 2, 3 * (k % 2)
            ttk.Label(m_box, text=name).grid(row=r, column=c0, sticky="w", padx=(0 if c0 == 0 else 10, 0))
            ttk.Label(m_box, textvariable=self.metric_vars[key], style="Value.TLabel", anchor="e", width=7).grid(
                row=r, column=c0 + 1, sticky="e")
            ttk.Label(m_box, text=unit, style="Muted.TLabel").grid(row=r, column=c0 + 2, sticky="w", padx=(3, 0))
        b_row = ttk.Frame(m_box)
        b_row.grid(row=3, column=0, columnspan=6, sticky="ew", pady=(6, 0))
        ttk.Button(b_row, text="Reiniciar índices", command=self.engine.reset_metrics).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 3))
        ttk.Button(b_row, text="Tabla 1", style="Primary.TButton", command=self.show_table1).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=(3, 0))

        # Simulacion
        sim_row = ttk.Frame(parent, style="Main.TFrame")
        sim_row.pack(fill=tk.X, pady=(8, 0))
        self.btn_pause = ttk.Button(sim_row, text="Pausar", command=self.toggle_pause)
        self.btn_pause.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 3))
        ttk.Button(sim_row, text="Reiniciar", command=self.reset_plant).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=3)
        ttk.Button(sim_row, text="Exportar CSV", command=self.export_csv).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(3, 0))

    def _refresh_design_info(self):
        e = self.engine
        poles = e.poles_cl
        osc = poles[np.abs(poles.imag) > 1e-9]
        integ = poles[np.abs(poles.imag) <= 1e-9]
        obs = e.poles_obs
        txt = (f"Lazo: Re ∈ [{osc.real.min():.1f}, {osc.real.max():.1f}]"
               + (f", integr. [{integ.real.min():.1f}, {integ.real.max():.1f}] s⁻¹" if integ.size else " s⁻¹")
               + f"\nObservador: Re ∈ [{obs.real.min():.1f}, {obs.real.max():.1f}] s⁻¹")
        o = self.opt.get("estado")
        if o:
            txt += f" · óptimo ×{o['pole_scale']:.2f} / ×{o['obs_scale']:.2f}"
        self.lbl_design.config(text=txt)

    def _reset_pid_gains(self):
        """Vuelve a las ganancias optimas del PID (scripts/optimizar_ganancias.py)."""
        for var, v, (sc, _ent) in zip((self.var_kp, self.var_ki, self.var_kd), self.pid_opt, self.pid_widgets):
            sc.configure(to=max(float(sc.cget("to")), 1.5 * v))
            var.set(v)
        self.var_msg.set("PID: ganancias óptimas restablecidas")

    def _ref_mode(self) -> str:
        return dict(self.REF_MODES).get(self.var_ref_mode.get(), "manual")

    def _on_ref_change(self):
        """Al elegir una referencia, la planta se reinicia en la pose nominal y la trayectoria empieza desde su
        comienzo: asi cada caso (p. ej., fronteras A a D) se ve completo y es comparable con los demas."""
        self.root.focus_set()
        self._restart_reference()
        self.var_msg.set(f"Referencia: {self.var_ref_mode.get()}")

    def _restart_reference(self):
        self._test_t0 = None
        self._show_ghost(None)
        self.engine.reset_state(init_z_err_mm=0.0)
        self._set_forces(0, 0, 0, 0, 0, 0)
        self.impulse_steps_remaining = 0
        self._ref_t0 = self.engine.t
        if not self.is_running:
            self.toggle_pause()

    def show_matrices(self):
        """Ventana con las matrices de diseno vigentes (K, Ki, Nr, L) y las pruebas de Kalman."""
        e = self.engine
        win = tk.Toplevel(self.root)
        win.title("Matrices de diseño")
        win.geometry("900x620")
        txt = tk.Text(win, font=("DejaVu Sans Mono", 9), wrap="none", bg="#ffffff", fg=TEXT, relief="flat", padx=10, pady=8)
        sy = ttk.Scrollbar(win, orient="vertical", command=txt.yview)
        sx = ttk.Scrollbar(win, orient="horizontal", command=txt.xview)
        txt.configure(yscrollcommand=sy.set, xscrollcommand=sx.set)
        sy.pack(side=tk.RIGHT, fill=tk.Y)
        sx.pack(side=tk.BOTTOM, fill=tk.X)
        txt.pack(fill=tk.BOTH, expand=True)
        fmt = lambda M: np.array2string(np.asarray(M), precision=3, suppress_small=True, max_line_width=250)
        blocks = [
            f"Modelo linealizado de diseño: n = {e.A.shape[0]}, m = {e.B.shape[1]}, Ts = {e.Ts} s",
            f"rango(Mc) = {e.rank_Mc}, rango(Mo) = {e.rank_Mo}, error de separación = {e.sep_error:.2e}",
            f"Factores de polos vigentes: controlador ×{e.pole_scale:.2f}, observador ×{e.obs_scale:.2f}",
            "", "K (6x12)", fmt(e.K_pi), "", "Ki (6x6)", fmt(e.Ki_pi), "", "Nr (6x6)", fmt(e.Nr_pi),
            "", "L discreta del observador de Luenberger (12x6)", fmt(e.Ld),
            "", "Polos de lazo cerrado (planta aumentada con integradores)", fmt(np.sort_complex(e.poles_cl)),
            "", "Polos del observador", fmt(np.sort_complex(e.poles_obs)),
        ]
        txt.insert("1.0", "\n".join(blocks))
        txt.configure(state="disabled")

    TABLE_COLS = [("est", "Estrategia", 190), ("rms", "RMS pose\npos [mm] / ang [°]", 130), ("itae", "ITAE\n[SI]", 80),
                  ("ju", "Ju\n[N²·s]", 75), ("restr", "Restricciones", 250), ("rob", "Robustez\nΔRMS pos (±20 % Bd)", 140),
                  ("cpu", "Tiempo CPU\n[µs/muestra]", 95)]

    def show_table1(self):
        """Tabla 1 del protocolo: ambas estrategias sobre el ensayo normalizado con las ganancias vigentes."""
        win = tk.Toplevel(self.root)
        win.title("Tabla 1 — Comparación de estrategias")
        win.geometry("1000x330")
        win.configure(bg=PANEL)
        style = ttk.Style()
        style.configure("Table.Treeview", rowheight=34, font=("TkDefaultFont", 9))
        style.configure("Table.Treeview.Heading", font=("TkDefaultFont", 9, "bold"), padding=(4, 6))
        tree = ttk.Treeview(win, columns=[c[0] for c in self.TABLE_COLS], show="headings", height=2, style="Table.Treeview")
        for key, head, w in self.TABLE_COLS:
            tree.heading(key, text=head)
            tree.column(key, width=w, anchor="w" if key in ("est", "restr") else "center")
        tree.pack(fill=tk.X, padx=10, pady=(10, 4))
        pid = self.pid_gain
        e = self.engine
        note = ttk.Label(win, style="Muted.TLabel", justify=tk.LEFT, wraplength=960, text=(
            f"Ensayo normalizado sobre la planta no lineal (Ts = {e.Ts*1e3:.0f} ms): escalón Δx = 8 mm, Δz = 15 mm en "
            f"t = {TEST_STEP_T} s e impulso de {TEST_IMPULSE_N:.0f} N en t = {TEST_IMPULSE_T} s, {TEST_END_T:.0f} s, ruido de "
            f"sensor {TEST_NOISE_MM} mm. Índices como en calcular_metricas_tabla1.m (ITAE con ‖e‖₁ de la pose en m y rad; "
            f"Ju = ∫FᵀF dt, con F las fuerzas de las patas). Restricciones: tiempo con fuerza saturada (±{e.Fmax:.0f} N) y carrera "
            f"de las patas. Robustez: aumento del RMS de posición en el peor caso de ±{100*(ROBUST_GAINS[1]-1):.0f} % en la "
            f"ganancia de la planta sin redisenar. CPU: ley de control (y observador) por muestra.\n"
            f"Ganancias: PID Kp = {pid[0]:.1f}, Ki = {pid[1]:.1f}, Kd = {pid[2]:.2f}; realimentación de estados con polos "
            f"×{e.pole_scale:.3f} y observador ×{e.obs_scale:.3f}."))
        note.pack(fill=tk.X, padx=10)
        btns = ttk.Frame(win)
        btns.pack(fill=tk.X, padx=10, pady=8)
        rows = {}

        def fill():
            tree.delete(*tree.get_children())
            for code in ("pid", "ss"):
                tree.insert("", tk.END, values=rows[code])
            for b in btn_list:
                b.state(["!disabled"])

        def compute():
            eng = copy.deepcopy(self.engine)   # no interfiere con la simulacion en curso
            for code, scales in (("pid", self._pid_scales()), ("ss", (1.0, 1.0, 1.0))):
                r = eng.table1_row(code, *scales)
                restr = (f"F sat {r['fsat_pct']:.1f} % · F máx {r['Fmax']:.1f} N"
                         + (" · carrera OK" if r["stroke"] == 0 else f" · carrera: {r['stroke']} muestras fuera"))
                rows[code] = (self.SHORT_NAMES[code], f"{r['RMS_pos']:.3f} / {r['RMS_ang']:.3f}", f"{r['ITAE']:.4f}",
                              f"{r['Ju']:.1f}", restr, f"+{r['robustez_pct']:.1f} %", f"{r['cpu_us']:.0f}")
            win.after(0, fill)

        def latex():
            lines = ["\\begin{tabular}{lcccp{4.2cm}cc}", "\\hline",
                     "Estrategia & RMS pose (mm / $^\\circ$) & ITAE & $J_u$ (N$^2$s) & Restricciones & Robustez & Tiempo CPU ($\\mu$s) \\\\",
                     "\\hline"]
            for code in ("pid", "ss"):
                v = list(rows[code])
                v[4] = v[4].replace("%", "\\%").replace("·", ";").replace(">", "$>$")
                v[5] = v[5].replace("%", "\\%")
                lines.append(" & ".join(v) + " \\\\")
            lines += ["\\hline", "\\end{tabular}"]
            win.clipboard_clear()
            win.clipboard_append("\n".join(lines))
            self.var_msg.set("Tabla 1 copiada al portapapeles en formato LaTeX")

        def csv():
            path = filedialog.asksaveasfilename(parent=win, title="Exportar Tabla 1", defaultextension=".csv",
                                                initialdir=self.engine.results_dir, initialfile="tabla1.csv",
                                                filetypes=[("CSV", "*.csv")])
            if path:
                import csv as csv_mod
                with open(path, "w", newline="", encoding="utf-8") as f:
                    w = csv_mod.writer(f)
                    w.writerow([h.replace("\n", " ") for _, h, _ in self.TABLE_COLS])
                    for code in ("pid", "ss"):
                        w.writerow(rows[code])
                self.var_msg.set(f"Tabla 1 exportada: {os.path.basename(path)}")

        def recompute():
            for b in btn_list:
                b.state(["disabled"])
            tree.delete(*tree.get_children())
            tree.insert("", tk.END, values=("Calculando…",) + ("",) * 6)
            threading.Thread(target=compute, daemon=True).start()

        btn_list = [ttk.Button(btns, text="Copiar LaTeX", command=latex),
                    ttk.Button(btns, text="Exportar CSV", command=csv),
                    ttk.Button(btns, text="Recalcular", command=recompute)]
        for b in btn_list:
            b.pack(side=tk.LEFT, padx=(0, 6))
        recompute()

    def _on_ctrl_change(self, _event=None):
        if _event is not None:
            if self._ref_mode() != "manual":
                # Con una trayectoria seleccionada, el nuevo controlador la recorre desde el comienzo
                self._restart_reference()
                self.var_msg.set(f"Estrategia activa: {self._short_name(self.var_ctrl.get())} — "
                                 f"referencia {self.var_ref_mode.get()} desde el comienzo")
            elif self.var_auto_test.get():
                self.start_test()
            else:
                # Cambio en vivo: memoria del controlador limpia, sin ensayo pendiente y simulacion corriendo
                self._test_t0 = None
                self.engine.reset_controller()
                self.engine.reset_metrics()
                if not self.is_running:
                    self.toggle_pause()
                self.var_msg.set(f"Estrategia activa: {self._short_name(self.var_ctrl.get())}")
        is_pid = dict(self.CTRL_MODES).get(self.var_ctrl.get(), "pid") == "pid"
        for group, active in ((self.pid_widgets, is_pid), (self.ss_widgets, not is_pid)):
            for w in (w for row in group for w in row):
                w.state(["!disabled"] if active else ["disabled"])
        self.btn_redesign.state(["disabled"] if is_pid else ["!disabled"])
        self.root.focus_set()

    # ------------------------------------------------------------------ acciones
    def _set_pose_sliders(self, dx, dy, dz, r, p, y):
        self.var_ref_mode.set(self.REF_MODES[0][0])
        for var, val in zip((self.var_dx, self.var_dy, self.var_dz, self.var_roll, self.var_pitch, self.var_yaw),
                            (dx, dy, dz, r, p, y)):
            var.set(val)

    def _set_forces(self, fx, fy, fz, mx, my, dm, mz=0.0):
        """Fija deslizadores y perturbacion aplicada a la vez (usado por Anular, Reiniciar y el ensayo)."""
        for var, val in zip((self.var_fx, self.var_fy, self.var_fz, self.var_mx, self.var_my, self.var_mz, self.var_dmass),
                            (fx, fy, fz, mx, my, mz, dm)):
            var.set(val)
        self.apply_forces()

    def _commit_values(self):
        for commit in self._value_commits:
            commit()

    def clear_disturbances(self):
        """Anula todas las perturbaciones: fuerza, momento, carga, impulso en curso y ruido de sensor."""
        self.impulse_steps_remaining = 0
        self.var_noise.set(0.0)
        self._set_forces(0, 0, 0, 0, 0, 0)
        self.var_msg.set("Perturbaciones anuladas: fuerza, momento, carga, impulso y ruido en cero")

    def apply_forces(self):
        """Aplica a la planta la fuerza, el momento y la carga preparados en los deslizadores."""
        self.applied_tau = np.array([v.get() for v in (self.var_fx, self.var_fy, self.var_fz,
                                                        self.var_mx, self.var_my, self.var_mz)], dtype=float)
        self.applied_dmass = float(self.var_dmass.get())
        f, dm = self.applied_tau, self.applied_dmass
        if np.any(f != 0) or dm:
            self.lbl_applied.config(text=f"Aplicado: F = [{f[0]:+.0f}, {f[1]:+.0f}, {f[2]:+.0f}] N, "
                                         f"M = [{f[3]:+.1f}, {f[4]:+.1f}, {f[5]:+.1f}] N·m, Δm = {dm:.1f} kg")
        else:
            self.lbl_applied.config(text="Sin perturbación aplicada")

    def _bind_keys(self):
        # Atajos: Espacio pausa, R reinicia, I impulso, M reinicia indices (ignorados dentro de un Combobox)
        def guard(fn):
            return lambda e: None if isinstance(e.widget, ttk.Combobox) else fn()
        self.root.bind("<space>", guard(self.toggle_pause))
        self.root.bind("<KeyPress-r>", guard(self.reset_plant))
        self.root.bind("<KeyPress-i>", guard(self.inject_vertical_impulse))
        self.root.bind("<KeyPress-m>", guard(self.engine.reset_metrics))
        self.root.bind("<KeyPress-e>", guard(self.start_test))

    def export_csv(self):
        path = filedialog.asksaveasfilename(
            title="Exportar ventana del osciloscopio", defaultextension=".csv",
            initialdir=self.engine.results_dir, initialfile="telemetria_stewart.csv",
            filetypes=[("CSV", "*.csv")])
        if not path:
            return
        cols = [self.t_buf, self.dx_ref_buf, self.dy_ref_buf, self.dz_ref_buf, self.dx_buf, self.dy_buf, self.dz_buf,
                *self.v_bufs, self.vz_real_buf, self.vz_hat_buf]
        header = "t_s,dx_ref_mm,dy_ref_mm,dz_ref_mm,dx_mm,dy_mm,dz_mm,V1,V2,V3,V4,V5,V6,vz_mm_s,vz_hat_mm_s"
        np.savetxt(path, np.column_stack([np.array(c) for c in cols]), delimiter=",", header=header, comments="", fmt="%.6g")
        self.var_msg.set(f"Exportado: {os.path.basename(path)}")

    def inject_vertical_impulse(self):
        """Pulso de fuerza durante 45 pasos = 0.45 s: vertical descendente con componentes lateral y de par."""
        self.impulse_steps_remaining = int(round(IMPULSE_DUR / self.engine.Ts))
        self.impulse_force_vec = float(self.var_impulse.get()) * IMPULSE_DIR

    # Ensayo normalizado (definido en octave_bridge): equilibrio -> escalon -> impulso. Durante el ensayo las
    # graficas se dibujan en tiempo de ensayo (0 a 5 s), alineadas con el ensayo anterior en gris; al terminar
    # la simulacion continua sin pausa con la referencia final del ensayo.

    def start_test(self):
        self.engine.reset_state(init_z_err_mm=0.0)
        self._set_forces(0, 0, 0, 0, 0, 0)
        self.impulse_steps_remaining = 0
        self._test_t0 = self.engine.t
        self._test_impulse_done = False
        self._test_ctrl = self.var_ctrl.get()
        if not self.is_running:
            self.toggle_pause()
        self._show_ghost(self._ghost)
        self.var_msg.set("Ensayo en curso: escalón (Δx = 8 mm, Δz = 15 mm) en t = 0.5 s, impulso en t = 2.5 s")

    def _test_q_ref(self, tt: float) -> np.ndarray:
        return self.engine.pose0 + (TEST_STEP if tt >= TEST_STEP_T else 0.0)

    def _finish_test(self):
        mt = self.engine.metrics()
        prev = self._ghost
        self._ghost = {"name": self._test_ctrl, "dz": np.array(self.dz_buf),
                       "ez": np.array(self.dz_ref_buf) - np.array(self.dz_buf), "RMS": mt["RMS_pos"], "Ju": mt["Ju"]}
        self._test_t0 = None
        if self._ref_mode() == "manual":
            # Continuar sin salto: la referencia manual queda en la pose final del ensayo
            dx, dy, dz = TEST_STEP[:3] * 1e3
            self._set_pose_sliders(dx, dy, dz, 0.0, 0.0, 0.0)
            self._show_ghost(None)
        else:
            self._restart_reference()   # vuelve a la trayectoria seleccionada, desde su comienzo
        msg = (f"Ensayo — {self._short_name(self._ghost['name'])}: RMS pos {mt['RMS_pos']:.3f} mm, Ju {mt['Ju']:.1f} N²·s")
        if prev is not None:
            msg += f"   ·   anterior ({self._short_name(prev['name'])}): RMS pos {prev['RMS']:.3f} mm, Ju {prev['Ju']:.1f} N²·s"
        self.var_msg.set(msg)

    @staticmethod
    def _short_name(label: str) -> str:
        return StewartRealtimeSimulatorGUI.SHORT_NAMES[dict(StewartRealtimeSimulatorGUI.CTRL_MODES)[label]]

    def _show_ghost(self, prev):
        """Muestra en gris el ensayo anterior (otro controlador u otras ganancias) durante el ensayo; None lo oculta."""
        if prev is None:
            self.ghost_dz.set_data([], [])
            self.ghost_ez.set_data([], [])
            self.ghost_dz.set_label("_nolegend_")
            self.ghost_ez.set_label("_nolegend_")
        else:
            self.ghost_dz.set_data(self.t_rel, prev["dz"])
            self.ghost_ez.set_data(self.t_rel, prev["ez"])
            self.ghost_dz.set_label(f"Δz anterior ({self._short_name(prev['name'])})")
            self.ghost_ez.set_label(f"e_z anterior ({self._short_name(prev['name'])})")
        self.ax_pos.legend(loc="upper left", ncol=3 if prev is not None else 4)
        self.ax_err.legend(loc="upper left", ncol=4)
        self._autoscale(force=True)

    def _scope_data(self) -> dict:
        """Series que muestran los osciloscopios. En un ensayo, solo las muestras desde su inicio, colocadas
        en tiempo de ensayo (disparo); fuera de un ensayo, la ventana completa de los ultimos 5 s."""
        ex, ey, ez = self._errors()
        d = {"dx": self.dx_buf, "dy": self.dy_buf, "dz": self.dz_buf,
             "dx_ref": self.dx_ref_buf, "dy_ref": self.dy_ref_buf, "dz_ref": self.dz_ref_buf,
             "ex": ex, "ey": ey, "ez": ez, "vz": self.vz_real_buf, "vz_hat": self.vz_hat_buf}
        d.update({f"V{i}": self.v_bufs[i] for i in range(6)})
        if self._test_t0 is None:
            return {k: np.asarray(v, dtype=float) for k, v in d.items()}
        n = int(np.clip(round((self.engine.t - self._test_t0) / self.engine.Ts), 0, self.hist_len))
        out = {}
        for k, v in d.items():
            a = np.full(self.hist_len, np.nan)
            if n:
                a[:n] = np.asarray(v, dtype=float)[-n:]
            out[k] = a
        return out

    def toggle_pause(self):
        self.is_running = not self.is_running
        self.btn_pause.config(text="Pausar" if self.is_running else "Reanudar")
        if self.is_running:
            self._t_last = time.perf_counter()
            self._t_acc = 0.0

    def reset_plant(self):
        self.engine.reset_state(init_z_err_mm=-5.0)
        self._set_forces(0, 0, 0, 0, 0, 0)
        self._ref_t0 = self.engine.t

    def trigger_redesign(self, on_done=None):
        self.btn_redesign.state(["disabled"])
        p_scale = float(self.var_pole_scale.get())
        o_scale = float(self.var_obs_scale.get())
        self.var_msg.set("Rediseñando controlador y observador…")

        def done(err=None):
            self._on_ctrl_change()
            if err:
                self.var_msg.set("")
                messagebox.showerror("Error en el rediseño", err)
            else:
                self._refresh_design_info()
                self.var_msg.set(f"Rediseño aplicado: polos ×{p_scale:.2f}, observador ×{o_scale:.2f}")
                if on_done:
                    on_done()

        def worker():
            try:
                self.engine.resynthesize_in_octave(p_scale, o_scale)
                self.root.after(0, done)
            except Exception as exc:
                msg = str(exc)
                self.root.after(0, lambda: done(msg))

        threading.Thread(target=worker, daemon=True).start()

    # ------------------------------------------------------------------ graficas
    def _build_plots(self, parent):
        panes = tk.PanedWindow(parent, orient=tk.HORIZONTAL, sashwidth=6, bg=BG, bd=0)
        panes.pack(fill=tk.BOTH, expand=True)

        left = tk.Frame(panes, bg=BORDER, padx=1, pady=1)
        head = tk.Frame(left, bg="#ffffff")
        head.pack(fill=tk.X)
        tk.Label(head, text="Plataforma Stewart 6-SPS", bg="#ffffff", fg=TEXT, font=("TkDefaultFont", 10)).pack(side=tk.LEFT, padx=8, pady=4)
        self.lbl_view = tk.Label(head, bg="#ffffff", fg=MUTED, font=("TkDefaultFont", 8))
        self.lbl_view.pack(side=tk.RIGHT, padx=8)
        self.view3d, self.view_engine = create_view(left, self.geom.LEG_COLORS)
        self.view3d.pack(fill=tk.BOTH, expand=True)
        hint = "arrastrar: rotar · rueda: zoom · doble clic: vista inicial" if self.view_engine == "OpenGL" else ""
        self.lbl_view.config(text=f"{self.view_engine}  {hint}".strip())
        panes.add(left, minsize=360, width=560, stretch="always")

        holder = tk.Frame(panes, bg=BORDER, padx=1, pady=1)
        self.fig = Figure(figsize=(7.0, 8.2), dpi=100, facecolor="#ffffff")
        gs = self.fig.add_gridspec(4, 1, left=0.12, right=0.97, top=0.96, bottom=0.06, hspace=0.55)
        self.ax_pos = self.fig.add_subplot(gs[0])
        self.ax_err = self.fig.add_subplot(gs[1], sharex=self.ax_pos)
        self.ax_volt = self.fig.add_subplot(gs[2], sharex=self.ax_pos)
        self.ax_obs = self.fig.add_subplot(gs[3], sharex=self.ax_pos)
        self.canvas = FigureCanvasTkAgg(self.fig, master=holder)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        panes.add(holder, minsize=420, stretch="always")

    def _init_plot_artists(self):
        # Osciloscopios: ventana fija de 5 s (el extremo derecho es el instante actual). Al terminar un
        # ensayo, el eje coincide con el tiempo del ensayo.
        t_arr = np.linspace(0.0, (self.hist_len - 1) * self.engine.Ts, self.hist_len)
        self.t_rel = t_arr
        z_arr = np.zeros_like(t_arr)

        def style_axis(a, title, ylabel, ylim):
            a.set_xlim(t_arr[0], t_arr[-1])
            a.set_ylim(*ylim)
            a.set_title(title, loc="left", color=TEXT)
            a.set_ylabel(ylabel)
            a.grid(True, color=GRID, linewidth=0.6)
            a.tick_params(labelsize=7)
            if a is not self.ax_obs:
                a.tick_params(labelbottom=False)

        # Pose: medida (continua) y referencia (discontinua)
        self.line_x_ref, = self.ax_pos.plot(t_arr, z_arr, "--", color=C_X, lw=1.0)
        self.line_y_ref, = self.ax_pos.plot(t_arr, z_arr, "--", color=C_Y, lw=1.0)
        self.line_z_ref, = self.ax_pos.plot(t_arr, z_arr, "--", color=C_Z, lw=1.0)
        self.line_dx, = self.ax_pos.plot(t_arr, z_arr, "-", color=C_X, lw=1.4, label="Δx")
        self.line_dy, = self.ax_pos.plot(t_arr, z_arr, "-", color=C_Y, lw=1.4, label="Δy")
        self.line_dz, = self.ax_pos.plot(t_arr, z_arr, "-", color=C_Z, lw=1.6, label="Δz")
        self.ax_pos.plot([], [], "--", color=MUTED, lw=1.0, label="referencia")
        self.ghost_dz, = self.ax_pos.plot([], [], "-", color="#b5b5b5", lw=2.4, zorder=0)
        style_axis(self.ax_pos, "Posición", "Δ [mm]", (-5.0, 20.0))
        self.ax_pos.legend(loc="upper left", ncol=4)

        # Error de seguimiento: aqui se aprecia la reaccion a perturbaciones y el regreso al equilibrio
        self.line_ex, = self.ax_err.plot(t_arr, z_arr, "-", color=C_X, lw=1.3, label="e_x")
        self.line_ey, = self.ax_err.plot(t_arr, z_arr, "-", color=C_Y, lw=1.3, label="e_y")
        self.line_ez, = self.ax_err.plot(t_arr, z_arr, "-", color=C_Z, lw=1.5, label="e_z")
        self.ghost_ez, = self.ax_err.plot([], [], "-", color="#b5b5b5", lw=2.4, zorder=0)
        self.ax_err.axhline(0.0, color=MUTED, lw=0.8)
        style_axis(self.ax_err, "Error de seguimiento  e = q_ref − q", "e [mm]", (-1.0, 1.0))
        self.ax_err.legend(loc="upper left", ncol=3)

        # Fuerzas axiales de las patas (entrada de la planta)
        self.lines_v = []
        for i in range(6):
            lv, = self.ax_volt.plot(t_arr, z_arr, "-", color=self.geom.LEG_COLORS[i], lw=1.1, label=f"F{i+1}")
            self.lines_v.append(lv)
        for v in (self.engine.Fmax, -self.engine.Fmax):
            self.ax_volt.axhline(v, color=MUTED, ls=":", lw=1.0)
        style_axis(self.ax_volt, f"Fuerza en los actuadores (saturación ±{self.engine.Fmax:.0f} N)",
                   "F [N]", (-1.1 * self.engine.Fmax, 1.1 * self.engine.Fmax))
        self.ax_volt.legend(loc="upper right", ncol=6)

        # Observador de Luenberger
        self.line_vz_real, = self.ax_obs.plot(t_arr, z_arr, "-", color=TEXT, lw=1.4, label="ż real")
        self.line_vz_hat, = self.ax_obs.plot(t_arr, z_arr, "--", color="#c0711b", lw=1.4, label="ż estimada")
        style_axis(self.ax_obs, "Observador de Luenberger: velocidad vertical", "ż [mm/s]", (-65.0, 65.0))
        self.ax_obs.set_xlabel("tiempo [s]  (ventana de 5 s; durante un ensayo, tiempo del ensayo)")
        self.ax_obs.legend(loc="upper left", ncol=2)

        # Artistas dinamicos: excluidos del fondo cacheado y redibujados por blitting
        self._anim = [
            self.line_z_ref, self.line_x_ref, self.line_y_ref, self.line_dz, self.line_dx, self.line_dy,
            self.line_ex, self.line_ey, self.line_ez,
            *self.lines_v, self.line_vz_real, self.line_vz_hat,
        ]
        for a in self._anim:
            a.set_animated(True)

    def _autoscale(self, force=False):
        """Ajusta el eje Y de cada osciloscopio a sus datos, con histeresis para no redibujar cada cuadro."""
        # El error se escala con los ultimos 2.6 s: asi un transitorio de escalon antiguo no aplasta la
        # reaccion a una perturbacion reciente (en el ensayo cubre el impulso y la recuperacion)
        d = self._scope_data()
        # Escala del error: en un ensayo que ya paso el impulso, desde t = 2.4 s (impulso y recuperacion, sin el
        # transitorio del escalon); fuera de un ensayo, los ultimos 2.6 s
        if self._test_t0 is not None:
            recent = slice(round(2.4 / self.engine.Ts), None) if self.engine.t - self._test_t0 > 2.6 else slice(None)
        else:
            recent = slice(-round(2.6 / self.engine.Ts), None)
        specs = [
            (self.ax_pos, [d[k] for k in ("dx", "dy", "dz", "dx_ref", "dy_ref", "dz_ref")], 2.0),
            (self.ax_err, [d[k][recent] for k in ("ex", "ey", "ez")], 0.05),
            (self.ax_volt, [d[f"V{i}"] for i in range(6)], 0.5),
            (self.ax_obs, [d["vz"], d["vz_hat"]], 2.0),
        ]
        if self._test_t0 is not None and self._ghost is not None:
            specs[0][1].append(self._ghost["dz"])
            specs[1][1].append(self._ghost["ez"][recent])
        changed = False
        for ax, series, min_span in specs:
            finite = [a[np.isfinite(a)] for a in series]
            finite = [a for a in finite if a.size]
            if not finite:
                continue
            lo = min(float(a.min()) for a in finite)
            hi = max(float(a.max()) for a in finite)
            span = max(hi - lo, min_span)
            mid = 0.5 * (hi + lo)
            t_lo, t_hi = mid - 0.62 * span, mid + 0.62 * span + 0.25 * span  # margen extra arriba para la leyenda
            c_lo, c_hi = ax.get_ylim()
            if force or lo < c_lo or hi > c_hi or (t_hi - t_lo) < 0.45 * (c_hi - c_lo):
                ax.set_ylim(t_lo, t_hi)
                changed = True
        if changed:
            self._bg = None  # el siguiente cuadro hace un dibujo completo y recaptura el fondo

    def _errors(self):
        ex = np.array(self.dx_ref_buf) - np.array(self.dx_buf)
        ey = np.array(self.dy_ref_buf) - np.array(self.dy_buf)
        ez = np.array(self.dz_ref_buf) - np.array(self.dz_buf)
        return ex, ey, ez

    def _on_full_draw(self, _event=None):
        self._bg = self.canvas.copy_from_bbox(self.fig.bbox)
        self._draw_animated()

    def _draw_animated(self):
        for a in self._anim:
            a.axes.draw_artist(a)

    # ------------------------------------------------------------------ simulacion
    def _compute_current_q_ref(self, t_curr: float) -> np.ndarray:
        # Mismas trayectorias que usa el optimizador (octave_bridge.reference_pose)
        ref_mode = dict(self.REF_MODES).get(self.var_ref_mode.get(), "manual")
        manual = np.array([self.var_dx.get() * 1e-3, self.var_dy.get() * 1e-3, self.var_dz.get() * 1e-3,
                           np.deg2rad(self.var_roll.get()), np.deg2rad(self.var_pitch.get()), np.deg2rad(self.var_yaw.get())])
        return reference_pose(ref_mode, t_curr - self._ref_t0, self.engine.pose0, manual)

    def _record(self, snap_out: dict):
        """Guarda una muestra en los buffers circulares del osciloscopio."""
        h0 = self.engine.h0
        self.t_buf.append(snap_out["t"])
        self.dx_ref_buf.append(snap_out["q_ref"][0] * 1e3)
        self.dy_ref_buf.append(snap_out["q_ref"][1] * 1e3)
        self.dz_ref_buf.append((snap_out["q_ref"][2] - h0) * 1e3)
        self.dz_buf.append((snap_out["pose"][2] - h0) * 1e3)
        self.dx_buf.append(snap_out["pose"][0] * 1e3)
        self.dy_buf.append(snap_out["pose"][1] * 1e3)
        for i in range(6):
            self.v_bufs[i].append(snap_out["F"][i])
        self.vz_real_buf.append(snap_out["qdot"][2] * 1e3)
        self.vz_hat_buf.append(snap_out["x_hat_abs"][8] * 1e3)

    def _advance(self, n_steps: int):
        """Avanza n_steps pasos Ts de la planta en lazo cerrado; devuelve la ultima muestra (o None)."""
        ctrl_code = dict(self.CTRL_MODES).get(self.var_ctrl.get(), "pid")
        if ctrl_code != self._last_ctrl:
            self.engine.reset_metrics()  # indices comparables entre controladores
            self._last_ctrl = ctrl_code
        if ctrl_code == "pid":
            kp_s, ki_s, kd_s = self._pid_scales()
        else:
            kp_s = ki_s = kd_s = 1.0  # K, Ki y L se modifican via rediseno de polos
        dmass = self.applied_dmass
        noise = float(self.var_noise.get())
        tau_base = self.applied_tau

        snap_out = None
        for _ in range(n_steps):
            if self._test_t0 is not None:
                tt = self.engine.t - self._test_t0
                if tt >= TEST_END_T - 1e-9:
                    break
                q_ref = self._test_q_ref(tt)
                if tt >= TEST_IMPULSE_T and not self._test_impulse_done:
                    self._test_impulse_done = True
                    self.inject_vertical_impulse()
            else:
                q_ref = self._compute_current_q_ref(self.engine.t)
            tau_ext = tau_base.copy()
            if self.impulse_steps_remaining > 0:
                tau_ext += self.impulse_force_vec
                self.impulse_steps_remaining -= 1
            snap_out = self.engine.step_realtime(q_ref, tau_ext, dmass, ctrl_code, kp_s, ki_s, kd_s,
                                                 noise_sigma_mm=noise)
            self._record(snap_out)
        return snap_out

    def _pid_scales(self):
        return tuple(g / b for g, b in zip(self.pid_gain, self.pid_base))

    def _test_finished(self) -> bool:
        return self._test_t0 is not None and self.engine.t - self._test_t0 >= TEST_END_T - 1e-9

    def _realtime_loop(self):
        now = time.perf_counter()
        dt_wall = now - self._t_last
        self._t_last = now
        if self.is_running:
            # Pasos Ts segun el tiempo de pared transcurrido; con equipo lento se limita a 0.1 s de simulacion
            # por cuadro y se descarta el atraso (camara lenta en lugar de congelarse)
            Ts = self.engine.Ts
            self._t_acc = min(self._t_acc + dt_wall, 0.1)
            n_steps = int(self._t_acc / Ts)
            self._t_acc -= n_steps * Ts
            snap_out = self._advance(n_steps)

            if dt_wall > 0:
                self._fps = 0.9 * self._fps + 0.1 / dt_wall
                self._rt = 0.9 * self._rt + 0.1 * (n_steps * Ts / dt_wall)
            if self._test_finished():
                self._autoscale(force=True)
                self._finish_test()
            elif snap_out is not None:
                self._frame += 1
                if self._frame % 15 == 0:
                    self._autoscale()
            if snap_out is not None:
                self._update_visuals_fast(snap_out)
            self.var_perf.set(f"{self._fps:.0f} FPS · tiempo real ×{self._rt:.2f}")
        else:
            self.var_perf.set("En pausa — Espacio para continuar")

        # El siguiente cuadro se encadena tras las tareas "idle" de Tk (geometria, repintado): si solo se
        # usara after(), el temporizador siempre pendiente las relegaria y la vista 3D podria quedar sin tamano
        self.root.after(15, lambda: self.root.after_idle(self._realtime_loop))

    def capturar(self, path, dpi=150):
        """Imagen compuesta: vista 3D (OpenGL o Matplotlib) a la izquierda y osciloscopios a la derecha."""
        import io
        from PIL import Image
        buf = io.BytesIO()
        self.fig.savefig(buf, format="png", dpi=dpi, facecolor=self.fig.get_facecolor())
        scopes = Image.open(buf).convert("RGB")
        view = self.view3d.snapshot()
        view = view.resize((round(view.width * scopes.height / view.height), scopes.height))
        out = Image.new("RGB", (view.width + scopes.width, scopes.height), "white")
        out.paste(view, (0, 0))
        out.paste(scopes, (view.width, 0))
        out.save(path, optimize=True)

    def _update_visuals_fast(self, snap_out: dict):
        pose, q_ref, tau, t_now = snap_out["pose"], snap_out["q_ref"], snap_out["tau_ext"], snap_out["t"]

        # Escena 3D: plataforma en la pose real, contorno de la pose de referencia y vector de fuerza
        g = self.geom.ik_and_geometry_mm(pose)
        p_ref = self.geom.ik_and_geometry_mm(q_ref)["plat_loop_mm"]
        f_vec = tau[:3]
        f_mag = float(np.linalg.norm(f_vec))
        force_seg = None
        if f_mag > 0.5:
            p_tip = g["center_mm"] + 8.0 * g["R"][:, 2]
            force_seg = (p_tip - f_vec / f_mag * min(110.0, 35.0 + 0.12 * f_mag), p_tip)
        self.view3d.update_scene(g, p_ref, force_seg)

        d = self._scope_data()
        for line, key in ((self.line_z_ref, "dz_ref"), (self.line_x_ref, "dx_ref"), (self.line_y_ref, "dy_ref"),
                          (self.line_dz, "dz"), (self.line_dx, "dx"), (self.line_dy, "dy"),
                          (self.line_ex, "ex"), (self.line_ey, "ey"), (self.line_ez, "ez"),
                          (self.line_vz_real, "vz"), (self.line_vz_hat, "vz_hat")):
            line.set_ydata(d[key])
        for i in range(6):
            self.lines_v[i].set_ydata(d[f"V{i}"])

        # Lecturas en la barra de estado (texto Tk: mas barato que rasterizarlo en Matplotlib)
        r_deg, p_deg, y_deg = np.rad2deg(pose[3:6])
        err_pos_mm = float(np.linalg.norm(pose[:3] - q_ref[:3]) * 1e3)
        self.var_status.set(
            f"t = {t_now:7.2f} s   z = {pose[2]*1e3:6.1f} mm   φ θ ψ = {r_deg:+5.1f} {p_deg:+5.1f} {y_deg:+5.1f}°   "
            f"|e| = {err_pos_mm:6.3f} mm   max|F| = {np.max(np.abs(snap_out['F'])):5.2f} N"
        )
        mt = self.engine.metrics()
        for key, fmt in (("RMS_pos", "{:.3f}"), ("RMS_ang", "{:.3f}"), ("ITAE", "{:.4f}"),
                         ("Ju", "{:.1f}"), ("fsat_pct", "{:.1f}"), ("cpu_us", "{:.0f}")):
            self.metric_vars[key].set(fmt.format(mt[key]))

        if self._bg is None:
            self.canvas.draw()  # primer cuadro: dibujo completo; draw_event captura el fondo
        else:
            self.canvas.restore_region(self._bg)
            self._draw_animated()
            self.canvas.blit(self.fig.bbox)


def main():
    export_shot = None
    if "--export-screenshot" in sys.argv:
        idx = sys.argv.index("--export-screenshot")
        if idx + 1 < len(sys.argv):
            export_shot = sys.argv[idx + 1]

    root = tk.Tk()
    app = StewartRealtimeSimulatorGUI(root)

    if export_shot:
        def capture():
            # Ensayo normalizado con PID articular y luego con realimentacion de estados + integral
            app.is_running = False
            n_test = int(round(TEST_END_T / app.engine.Ts))
            for k, label in enumerate((app.CTRL_MODES[0][0], app.CTRL_MODES[1][0])):
                app.var_ctrl.set(label)
                app.start_test()
                app.is_running = False
                # El segundo ensayo se captura en su ultima muestra, con el primero en gris
                snap_out = app._advance(n_test if k == 0 else n_test - 1)
                if k == 0:
                    app._finish_test()
            app._autoscale(force=True)
            app._update_visuals_fast(snap_out)
            app.capturar(export_shot)
            print(f"Captura guardada en: {export_shot}")
            root.destroy()

        root.after(400, capture)

    root.mainloop()


if __name__ == "__main__":
    main()
