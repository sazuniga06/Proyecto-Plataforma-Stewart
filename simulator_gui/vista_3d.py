#!/usr/bin/env python3
"""
vista_3d.py
Vista 3D de la plataforma Stewart.

- OpenGLView: render por GPU (PyOpenGL + pyopengltk) con actuadores como cilindros sombreados,
  placas iluminadas y camara orbital (arrastrar = rotar, rueda = zoom, doble clic = vista inicial).
- MatplotlibView: alternativa sin OpenGL (mplot3d con blitting), para equipos sin soporte GLX.

create_view() devuelve la primera disponible. Ambas exponen update_scene(), redraw_now() y snapshot().
Todas las coordenadas estan en milimetros, marco de la base con z hacia arriba.
"""
import contextlib
import io
import os
import sys
import tkinter as tk
import numpy as np

# Tk en Linux crea su contexto con GLX (tambien bajo XWayland); PyOpenGL elegiria EGL en Wayland
if sys.platform.startswith("linux"):
    os.environ.setdefault("PYOPENGL_PLATFORM", "glx")

BG_RGB = (1.0, 1.0, 1.0)
PLATE_RGB = (0.62, 0.71, 0.80)
BASE_RGB = (0.55, 0.55, 0.55)
ROD_RGB = (0.72, 0.72, 0.72)
JOINT_RGB = (0.22, 0.22, 0.22)
ACCENT_RGB = (0.12, 0.31, 0.47)
REF_RGB = (0.75, 0.44, 0.11)
FORCE_RGB = (0.56, 0.16, 0.12)
AXIS_RGB = ((0.69, 0.23, 0.18), (0.18, 0.49, 0.20), (0.12, 0.31, 0.47))
BODY_R, ROD_R, JOINT_R, PLATE_T = 13.0, 6.5, 8.0, 8.0


def _hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


try:
    from OpenGL import GL, GLU
    from pyopengltk import OpenGLFrame
    HAVE_GL = True
except Exception:  # sin PyOpenGL/pyopengltk o sin plataforma GL utilizable
    HAVE_GL = False
    OpenGLFrame = tk.Frame


class OpenGLView(OpenGLFrame):
    """Escena 3D con OpenGL (perfil de compatibilidad, iluminacion fija)."""

    TARGET = np.array([0.0, 0.0, 140.0])

    def __init__(self, master, leg_colors, **kw):
        super().__init__(master, **kw)
        self.leg_rgb = [_hex_to_rgb(c) for c in leg_colors]
        self.scene = None
        self._reset_camera()
        self.bind("<ButtonPress-1>", self._on_press)
        self.bind("<B1-Motion>", self._on_drag)
        self.bind("<Double-Button-1>", lambda _e: (self._reset_camera(), self.redraw_now()))
        self.bind("<MouseWheel>", lambda e: self._zoom(0.9 if e.delta > 0 else 1.1))
        self.bind("<Button-4>", lambda _e: self._zoom(0.9))
        self.bind("<Button-5>", lambda _e: self._zoom(1.1))
        # Tras un cambio de tamano se vuelve a dibujar a la nueva resolucion (aunque la simulacion este en pausa)
        self.bind("<Configure>", lambda _e: self.after_idle(self.redraw_now), add="+")

    def tkCreateContext(self):
        # pyopengltk imprime diagnosticos de GLX al crear el contexto; se silencian
        with contextlib.redirect_stdout(io.StringIO()):
            super().tkCreateContext()

    # ---- camara
    def _reset_camera(self):
        self.az, self.el, self.dist = -50.0, 22.0, 820.0

    def _on_press(self, e):
        self._last = (e.x, e.y)

    def _on_drag(self, e):
        dx, dy = e.x - self._last[0], e.y - self._last[1]
        self._last = (e.x, e.y)
        self.az -= 0.4 * dx
        self.el = float(np.clip(self.el + 0.4 * dy, -5.0, 89.0))
        self.redraw_now()

    def _zoom(self, f):
        self.dist = float(np.clip(self.dist * f, 350.0, 2000.0))
        self.redraw_now()

    # ---- API comun
    def update_scene(self, g, ref_loop_mm, force_seg):
        self.scene = (g, ref_loop_mm, force_seg)
        self.redraw_now()

    def redraw_now(self):
        if self.context_created:
            self.tkMakeCurrent()
            self.redraw()
            self.tkSwapBuffers()

    def snapshot(self):
        from PIL import Image
        if not self.context_created:
            self.update()
        # Se lee el buffer trasero recien dibujado, antes del intercambio (despues queda indefinido)
        self.tkMakeCurrent()
        self.redraw()
        w, h = self.winfo_width(), self.winfo_height()
        GL.glReadBuffer(GL.GL_BACK)
        GL.glPixelStorei(GL.GL_PACK_ALIGNMENT, 1)
        data = GL.glReadPixels(0, 0, w, h, GL.GL_RGB, GL.GL_UNSIGNED_BYTE)
        self.tkSwapBuffers()
        return Image.frombytes("RGB", (w, h), data).transpose(Image.FLIP_TOP_BOTTOM)

    # ---- OpenGL
    def initgl(self):
        GL.glClearColor(*BG_RGB, 1.0)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_NORMALIZE)
        GL.glShadeModel(GL.GL_SMOOTH)
        GL.glEnable(GL.GL_LIGHT0)
        GL.glLightfv(GL.GL_LIGHT0, GL.GL_AMBIENT, (0.35, 0.35, 0.35, 1.0))
        GL.glLightfv(GL.GL_LIGHT0, GL.GL_DIFFUSE, (0.75, 0.75, 0.75, 1.0))
        GL.glLightfv(GL.GL_LIGHT0, GL.GL_SPECULAR, (0.25, 0.25, 0.25, 1.0))
        GL.glEnable(GL.GL_COLOR_MATERIAL)
        GL.glColorMaterial(GL.GL_FRONT_AND_BACK, GL.GL_AMBIENT_AND_DIFFUSE)
        GL.glMaterialfv(GL.GL_FRONT_AND_BACK, GL.GL_SPECULAR, (0.3, 0.3, 0.3, 1.0))
        GL.glMaterialf(GL.GL_FRONT_AND_BACK, GL.GL_SHININESS, 32.0)
        GL.glEnable(GL.GL_LINE_SMOOTH)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        if not hasattr(self, "quad"):
            self.quad = GLU.gluNewQuadric()
            GLU.gluQuadricNormals(self.quad, GLU.GLU_SMOOTH)

    def redraw(self):
        w, h = max(self.winfo_width(), 1), max(self.winfo_height(), 1)
        GL.glViewport(0, 0, w, h)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        GLU.gluPerspective(32.0, w / h, 20.0, 6000.0)
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glLoadIdentity()
        az, el = np.radians(self.az), np.radians(self.el)
        dist = self.dist * max(1.0, 0.95 / (w / h))   # panel angosto: alejar la camara para que quepa el ancho
        eye = self.TARGET + dist * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        GLU.gluLookAt(*eye, *self.TARGET, 0.0, 0.0, 1.0)
        GL.glLightfv(GL.GL_LIGHT0, GL.GL_POSITION, (0.4, -0.6, 1.0, 0.0))   # luz direccional fija en la escena

        self._draw_floor()
        if self.scene is None:
            return
        g, ref_loop, force_seg = self.scene
        B, M, Q, c, R = g["B_mm"], g["M_mm"], g["Q_mm"], g["center_mm"], g["R"]

        GL.glEnable(GL.GL_LIGHTING)
        self._plate(g["base_loop_mm"][:-1], np.array([0.0, 0.0, 1.0]), BASE_RGB)
        for i in range(6):
            self._cylinder(B[i], M[i], BODY_R, self.leg_rgb[i])
            self._cylinder(M[i] - 0.3 * BODY_R * (M[i] - B[i]) / np.linalg.norm(M[i] - B[i]), Q[i], ROD_R, ROD_RGB)
            self._sphere(B[i], JOINT_R, JOINT_RGB)
            self._sphere(Q[i], JOINT_R * 0.8, JOINT_RGB)
        self._plate(g["plat_ordered_mm"], R[:, 2], PLATE_RGB)
        if force_seg is not None:
            p0, p1 = force_seg
            d = (p1 - p0) / np.linalg.norm(p1 - p0)
            self._cylinder(p0, p1 - 16.0 * d, 3.2, FORCE_RGB)
            self._cylinder(p1 - 16.0 * d, p1, 8.0, FORCE_RGB, r_top=0.0)
        GL.glDisable(GL.GL_LIGHTING)

        # Contornos, referencia y triada (sin iluminacion)
        self._line_loop(g["plat_loop_mm"] + 0.5 * R[:, 2], ACCENT_RGB, 2.0)
        GL.glEnable(GL.GL_LINE_STIPPLE)
        GL.glLineStipple(2, 0x0F0F)
        self._line_loop(ref_loop, REF_RGB, 1.6)
        GL.glDisable(GL.GL_LINE_STIPPLE)
        GL.glLineWidth(2.5)
        GL.glBegin(GL.GL_LINES)
        for k in range(3):
            GL.glColor3f(*AXIS_RGB[k])
            GL.glVertex3f(*(c + 1.0 * R[:, 2]))
            GL.glVertex3f(*(c + 1.0 * R[:, 2] + 50.0 * R[:, k]))
        GL.glEnd()

    def _draw_floor(self):
        GL.glDisable(GL.GL_LIGHTING)
        GL.glLineWidth(1.0)
        GL.glColor3f(0.88, 0.88, 0.86)
        GL.glBegin(GL.GL_LINES)
        for v in np.arange(-250.0, 251.0, 50.0):
            GL.glVertex3f(v, -250.0, -0.5)
            GL.glVertex3f(v, 250.0, -0.5)
            GL.glVertex3f(-250.0, v, -0.5)
            GL.glVertex3f(250.0, v, -0.5)
        GL.glEnd()
        # Ejes del marco de la base
        GL.glLineWidth(2.0)
        GL.glBegin(GL.GL_LINES)
        for k in range(3):
            e = np.zeros(3)
            e[k] = 60.0
            GL.glColor3f(*AXIS_RGB[k])
            GL.glVertex3f(-230.0, -230.0, 0.0)
            GL.glVertex3f(*(np.array([-230.0, -230.0, 0.0]) + e))
        GL.glEnd()

    def _line_loop(self, pts, rgb, width):
        GL.glLineWidth(width)
        GL.glColor3f(*rgb)
        GL.glBegin(GL.GL_LINE_LOOP)
        for p in pts:
            GL.glVertex3f(*p)
        GL.glEnd()

    def _plate(self, poly, normal, rgb):
        """Placa hexagonal de espesor PLATE_T bajo el poligono dado (caras superior, inferior y laterales)."""
        n = normal / np.linalg.norm(normal)
        top, bot = poly, poly - PLATE_T * n
        GL.glColor3f(*rgb)
        for face, nn in ((top, n), (bot[::-1], -n)):
            GL.glBegin(GL.GL_POLYGON)
            GL.glNormal3f(*nn)
            for p in face:
                GL.glVertex3f(*p)
            GL.glEnd()
        ctr = poly.mean(axis=0)
        GL.glBegin(GL.GL_QUADS)
        for i in range(len(poly)):
            j = (i + 1) % len(poly)
            side_n = 0.5 * (top[i] + top[j]) - ctr
            GL.glNormal3f(*side_n)
            for p in (top[i], top[j], bot[j], bot[i]):
                GL.glVertex3f(*p)
        GL.glEnd()

    def _cylinder(self, p0, p1, r, rgb, r_top=None):
        d = p1 - p0
        length = float(np.linalg.norm(d))
        if length < 1e-6:
            return
        d /= length
        GL.glColor3f(*rgb)
        GL.glPushMatrix()
        GL.glTranslatef(*p0)
        axis = np.cross([0.0, 0.0, 1.0], d)
        s = np.linalg.norm(axis)
        if s > 1e-9:
            GL.glRotatef(np.degrees(np.arctan2(s, d[2])), *(axis / s))
        elif d[2] < 0:
            GL.glRotatef(180.0, 1.0, 0.0, 0.0)
        GLU.gluCylinder(self.quad, r, r if r_top is None else r_top, length, 18, 1)
        GL.glPopMatrix()

    def _sphere(self, p, r, rgb):
        GL.glColor3f(*rgb)
        GL.glPushMatrix()
        GL.glTranslatef(*p)
        GLU.gluSphere(self.quad, r, 12, 8)
        GL.glPopMatrix()


class MatplotlibView(tk.Frame):
    """Alternativa sin OpenGL: mplot3d con blitting (el fondo se cachea; solo se redibujan las lineas)."""

    def __init__(self, master, leg_colors, **kw):
        super().__init__(master, **kw)
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        self.fig = Figure(figsize=(5.5, 6.0), dpi=100, facecolor="#ffffff")
        ax = self.ax = self.fig.add_subplot(111, projection="3d")
        self.fig.subplots_adjust(left=0.0, right=1.0, bottom=0.02, top=0.98)
        ax.set_xlim(-200, 200)
        ax.set_ylim(-200, 200)
        ax.set_zlim(0, 340)
        ax.set_box_aspect((1.0, 1.0, 0.82))
        ax.view_init(elev=23, azim=-45)
        ax.set_xlabel("x [mm]", labelpad=2)
        ax.set_ylabel("y [mm]", labelpad=2)
        ax.set_zlabel("z [mm]", labelpad=2)
        ax.tick_params(labelsize=7, pad=0)
        self.leg_colors = leg_colors
        self.base_drawn = False
        self.patch = Poly3DCollection([np.zeros((6, 3))], facecolors="#9fb4cc", edgecolors="none", alpha=0.35)
        ax.add_collection3d(self.patch)
        mk = lambda *a, **k: ax.plot([], [], [], *a, **k)[0]
        self.plat = mk("-", color="#1f4e79", lw=2.4)
        self.ref = mk("--", color="#c0711b", lw=1.2)
        self.low = [mk("-", color=c, lw=4.5, solid_capstyle="round") for c in leg_colors]
        self.up = [mk("-", color="#9a9a9a", lw=2.2, solid_capstyle="round") for _ in leg_colors]
        self.force = mk("-", color="#8e2a1f", lw=3.0)
        self.anim = [self.patch, self.plat, self.ref, *self.low, *self.up, self.force]
        for a in self.anim:
            a.set_animated(True)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._bg = None
        self.canvas.mpl_connect("draw_event", self._on_draw)

    def _on_draw(self, _e=None):
        self._bg = self.canvas.copy_from_bbox(self.fig.bbox)
        self._draw_anim()

    def _draw_anim(self):
        self.patch.do_3d_projection()
        for a in self.anim:
            self.ax.draw_artist(a)

    def update_scene(self, g, ref_loop_mm, force_seg):
        if not self.base_drawn:
            b = g["base_loop_mm"]
            self.ax.plot(b[:, 0], b[:, 1], b[:, 2], "-", color="#4a4a4a", lw=2.2)
            self.base_drawn = True
            self._bg = None
        B, M, Q = g["B_mm"], g["M_mm"], g["Q_mm"]
        self.patch.set_verts([g["plat_ordered_mm"]])
        p = g["plat_loop_mm"]
        self.plat.set_data_3d(p[:, 0], p[:, 1], p[:, 2])
        self.ref.set_data_3d(ref_loop_mm[:, 0], ref_loop_mm[:, 1], ref_loop_mm[:, 2])
        for i in range(6):
            self.low[i].set_data_3d(*[[B[i, k], M[i, k]] for k in range(3)])
            self.up[i].set_data_3d(*[[M[i, k], Q[i, k]] for k in range(3)])
        if force_seg is None:
            self.force.set_data_3d([], [], [])
        else:
            self.force.set_data_3d(*[[force_seg[0][k], force_seg[1][k]] for k in range(3)])
        self.redraw_now()

    def redraw_now(self):
        if self._bg is None:
            self.canvas.draw()
        else:
            self.canvas.restore_region(self._bg)
            self._draw_anim()
            self.canvas.blit(self.fig.bbox)

    def snapshot(self):
        from PIL import Image
        self.canvas.draw()
        return Image.frombuffer("RGBA", self.canvas.get_width_height(), self.canvas.buffer_rgba()).convert("RGB")


def create_view(master, leg_colors):
    """OpenGL si esta disponible; si no, Matplotlib. Devuelve (vista, nombre_del_motor)."""
    if HAVE_GL and os.environ.get("STEWART_NO_GL") != "1":
        return OpenGLView(master, leg_colors, width=560, height=600), "OpenGL"
    return MatplotlibView(master, leg_colors), "Matplotlib"
