"""
4D CartPole Dual-View Renderer
===============================
Left panel  : real-time 3D OpenGL view (X, Y-up, Z) — cart on XZ plane, pole tilts.
Right panel : 2D auxiliary view of the 4th dimension (W axis + theta_w).

Together the two panels let you observe all four spatial dimensions
of the CartPole4DEnv.

Install:
    pip install pygame PyOpenGL PyOpenGL_accelerate numpy

Usage:
    from cartpole4d_env      import CartPole4DEnv
    from cartpole3d_renderer  import CartPole4DRenderer

    env      = CartPole4DEnv(use_discrete=True)
    renderer = CartPole4DRenderer(env)

    obs, _ = env.reset()
    renderer.reset_stats()

    done = False
    while not done:
        action = your_model.predict(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        if not renderer.render(reward=reward):
            break
        done = terminated or truncated

    renderer.close()
"""

import numpy as np
import pygame
from pygame.locals import DOUBLEBUF, OPENGL, QUIT, KEYDOWN, K_ESCAPE
from OpenGL.GL import *
from OpenGL.GLU import *


# ── Colours (RGB 0-1) ────────────────────────────────────────────────
C_BG       = (0.05, 0.07, 0.10)
C_BG_AUX   = (0.06, 0.05, 0.10)
C_GRID     = (0.15, 0.20, 0.25)
C_PLATFORM = (0.12, 0.22, 0.30)
C_CART     = (0.10, 0.55, 0.80)
C_POLE     = (1.00, 0.42, 0.21)
C_TIP      = (1.00, 0.85, 0.20)
C_LIMIT    = (0.85, 0.15, 0.25)
C_GOOD     = (0.22, 1.00, 0.40)
C_WARN     = (1.00, 0.75, 0.10)
C_BAD      = (1.00, 0.25, 0.35)
C_DIM      = (0.45, 0.60, 0.70)
C_DIVIDER  = (0.20, 0.55, 0.85)
C_TRACK    = (0.18, 0.28, 0.38)
C_AUX_POLE = (0.90, 0.35, 0.95)
C_AUX_TIP  = (1.00, 0.70, 1.00)
C_AUX_CART = (0.30, 0.70, 0.95)


SPLIT_RATIO = 0.65          # left panel gets 65 % of window width


class CartPole4DRenderer:

    def __init__(self, env, width: int = 1200, height: int = 650,
                 title: str = "CartPole 4D — XYZ + W"):
        self.env    = env
        self.width  = width
        self.height = height
        self.main_w = int(width * SPLIT_RATIO)
        self.aux_w  = width - self.main_w

        pygame.init()
        pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL)
        pygame.display.set_caption(title)

        self._setup_gl()
        self._setup_font()

        self.cam_yaw    = 35.0
        self.cam_pitch  = 28.0
        self.cam_dist   = 16.0
        self._mouse_dn  = False
        self._last_pos  = (0, 0)
        self._auto_orbit = True

        self.clock      = pygame.time.Clock()
        self.frame      = 0
        self.episode    = 0
        self.cum_reward = 0.0

    # ── GL setup ────────────────────────────────────────────────────

    def _setup_gl(self):
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_LINE_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        glShadeModel(GL_SMOOTH)

        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_LIGHT1)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)

        glLightfv(GL_LIGHT0, GL_POSITION, [ 6.0, 12.0,  6.0, 1.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE,  [ 1.0,  0.95, 0.9, 1.0])
        glLightfv(GL_LIGHT0, GL_AMBIENT,  [ 0.2,  0.25, 0.3, 1.0])
        glLightfv(GL_LIGHT1, GL_POSITION, [-5.0,  6.0, -5.0, 1.0])
        glLightfv(GL_LIGHT1, GL_DIFFUSE,  [ 0.3,  0.5,  0.7, 1.0])
        glLightfv(GL_LIGHT1, GL_AMBIENT,  [ 0.0,  0.0,  0.0, 1.0])

    def _setup_font(self):
        pygame.font.init()
        self._font_lg = pygame.font.SysFont("monospace", 16, bold=True)
        self._font_sm = pygame.font.SysFont("monospace", 13)
        self._font_xs = pygame.font.SysFont("monospace", 11)

    # ── Public API ───────────────────────────────────────────────────

    def render(self, reward: float = 0.0, fps: int = 50) -> bool:
        """Call once per env.step(). Returns False if window closed."""
        self.cum_reward += reward
        self.frame      += 1

        if not self._handle_events():
            return False

        if self._auto_orbit:
            self.cam_yaw += 0.12

        self._draw_3d_panel()
        self._draw_aux_panel()
        self._draw_divider()

        pygame.display.flip()
        self.clock.tick(fps)
        return True

    def reset_stats(self):
        self.episode    += 1
        self.frame       = 0
        self.cum_reward  = 0.0

    def close(self):
        pygame.quit()

    # ── Events ───────────────────────────────────────────────────────

    def _handle_events(self) -> bool:
        for e in pygame.event.get():
            if e.type == QUIT:
                return False
            if e.type == KEYDOWN and e.key == K_ESCAPE:
                return False
            if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                self._mouse_dn   = True
                self._auto_orbit = False
                self._last_pos   = pygame.mouse.get_pos()
            if e.type == pygame.MOUSEBUTTONUP and e.button == 1:
                self._mouse_dn = False
            if e.type == pygame.MOUSEMOTION and self._mouse_dn:
                mx, my = pygame.mouse.get_pos()
                dx = mx - self._last_pos[0]
                dy = my - self._last_pos[1]
                self.cam_yaw   += dx * 0.4
                self.cam_pitch  = np.clip(self.cam_pitch - dy * 0.4, 5, 85)
                self._last_pos  = (mx, my)
            if e.type == pygame.MOUSEWHEEL:
                self.cam_dist = np.clip(self.cam_dist - e.y * 0.6, 5, 30)
        return True

    # ==================================================================
    #  LEFT PANEL — 3D Scene  (X, Y-up, Z)
    # ==================================================================

    def _draw_3d_panel(self):
        glViewport(0, 0, self.main_w, self.height)
        glScissor(0, 0, self.main_w, self.height)
        glEnable(GL_SCISSOR_TEST)

        glClearColor(*C_BG, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        aspect = self.main_w / self.height
        gluPerspective(50, aspect, 0.1, 100)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        yr  = np.radians(self.cam_yaw)
        pr  = np.radians(self.cam_pitch)
        cx  = self.cam_dist * np.cos(pr) * np.sin(yr)
        cy  = self.cam_dist * np.sin(pr)
        cz  = self.cam_dist * np.cos(pr) * np.cos(yr)
        gluLookAt(cx, cy, cz,  0, 1, 0,  0, 1, 0)

        state  = self.env.state
        cx_pos = float(state[0])
        cz_pos = float(state[2])
        th_x   = float(state[6])
        th_z   = float(state[8])

        glEnable(GL_LIGHTING)
        self._draw_grid()
        self._draw_platform()
        self._draw_limit_box()
        self._draw_cart(cx_pos, cz_pos)
        self._draw_pole(cx_pos, cz_pos, th_x, th_z)

        self._draw_hud_3d()

        glDisable(GL_SCISSOR_TEST)

    # -- 3D primitives ------------------------------------------------

    def _draw_grid(self):
        glDisable(GL_LIGHTING)
        glColor3f(*C_GRID)
        glLineWidth(1.0)
        N = 12
        glBegin(GL_LINES)
        for i in range(-N, N + 1):
            glVertex3f(i, 0, -N); glVertex3f(i, 0,  N)
            glVertex3f(-N, 0, i); glVertex3f( N, 0, i)
        glEnd()
        glEnable(GL_LIGHTING)

    def _draw_platform(self):
        xt = self.env.x_threshold
        zt = self.env.z_threshold
        glColor3f(*C_PLATFORM)
        glPushMatrix()
        glScalef(xt * 2, 0.05, zt * 2)
        self._solid_cube()
        glPopMatrix()

        glDisable(GL_LIGHTING)
        glColor3f(0.0, 0.65, 1.0)
        glLineWidth(2.0)
        y = 0.03
        glBegin(GL_LINE_LOOP)
        glVertex3f(-xt, y, -zt); glVertex3f( xt, y, -zt)
        glVertex3f( xt, y,  zt); glVertex3f(-xt, y,  zt)
        glEnd()
        glEnable(GL_LIGHTING)

    def _draw_limit_box(self):
        glDisable(GL_LIGHTING)
        glColor4f(*C_LIMIT, 0.25)
        xt = self.env.x_threshold
        zt = self.env.z_threshold
        h  = 0.5
        for sx, sz in [(-1,-1),( 1,-1),( 1, 1),(-1, 1)]:
            glPushMatrix()
            glTranslatef(sx * xt, h / 2, sz * zt)
            glScalef(0.08, h, 0.08)
            self._solid_cube()
            glPopMatrix()
        glEnable(GL_LIGHTING)

    def _draw_cart(self, cx: float, cz: float):
        glColor3f(*C_CART)
        glPushMatrix()
        glTranslatef(cx, 0.20, cz)
        glScalef(0.65, 0.28, 0.65)
        self._solid_cube()
        glPopMatrix()

        glColor3f(0.12, 0.12, 0.18)
        for wx, wz in [(-0.25, -0.25), (0.25, -0.25),
                       (-0.25,  0.25), (0.25,  0.25)]:
            glPushMatrix()
            glTranslatef(cx + wx, 0.06, cz + wz)
            glScalef(0.10, 0.10, 0.10)
            self._solid_cube()
            glPopMatrix()

    def _draw_pole(self, cx: float, cz: float, th_x: float, th_z: float):
        pole_len = self.env.length * 2

        glPushMatrix()
        glTranslatef(cx, 0.34, cz)
        glRotatef(np.degrees(th_x), 0, 0, -1)
        glRotatef(np.degrees(th_z), 1, 0,  0)

        glColor3f(*C_POLE)
        glPushMatrix()
        glTranslatef(0, pole_len / 2, 0)
        glScalef(0.06, pole_len, 0.06)
        self._solid_cube()
        glPopMatrix()

        glColor3f(*C_TIP)
        glPushMatrix()
        glTranslatef(0, pole_len, 0)
        glScalef(0.12, 0.12, 0.12)
        self._solid_cube()
        glPopMatrix()

        glPopMatrix()

    def _solid_cube(self):
        v = [
            ( 0.5,  0.5, -0.5), ( 0.5, -0.5, -0.5),
            (-0.5, -0.5, -0.5), (-0.5,  0.5, -0.5),
            ( 0.5,  0.5,  0.5), ( 0.5, -0.5,  0.5),
            (-0.5, -0.5,  0.5), (-0.5,  0.5,  0.5),
        ]
        faces = [
            ((0,1,2,3), ( 0, 0,-1)), ((4,5,6,7), ( 0, 0, 1)),
            ((0,4,5,1), ( 1, 0, 0)), ((3,7,6,2), (-1, 0, 0)),
            ((0,3,7,4), ( 0, 1, 0)), ((1,5,6,2), ( 0,-1, 0)),
        ]
        glBegin(GL_QUADS)
        for indices, normal in faces:
            glNormal3fv(normal)
            for i in indices:
                glVertex3fv(v[i])
        glEnd()

    # -- 3D HUD -------------------------------------------------------

    def _draw_hud_3d(self):
        state = self.env.state
        x    = float(state[0])
        z    = float(state[2])
        th_x = float(np.degrees(state[6]))
        th_z = float(np.degrees(state[8]))
        lim  = float(np.degrees(self.env.angle_threshold))

        def acol(deg):
            r = abs(deg) / lim
            if r < 0.5: return C_GOOD
            if r < 0.8: return C_WARN
            return C_BAD

        lines = [
            ("CARTPOLE 4D  [XYZ]",          C_DIM,       True),
            ("",                             C_DIM,       False),
            (f"Episode  {self.episode}",     C_DIM,       False),
            (f"Step     {self.frame}",       C_DIM,       False),
            (f"Reward   {self.cum_reward:.1f}", C_GOOD,   False),
            ("",                             C_DIM,       False),
            (f"Cart X   {x:+.3f}",           C_DIM,       False),
            (f"Cart Z   {z:+.3f}",           C_DIM,       False),
            (f"Theta X  {th_x:+.1f}\u00b0",  acol(th_x),  False),
            (f"Theta Z  {th_z:+.1f}\u00b0",  acol(th_z),  False),
            ("",                             C_DIM,       False),
            ("ESC quit | Drag orbit",        C_DIM,       False),
        ]

        self._blit_text_panel(lines, x_off=8, y_off=self.height - 8,
                              panel_w=210, viewport_w=self.main_w)

    # ==================================================================
    #  RIGHT PANEL — 2D Auxiliary (W dimension)
    # ==================================================================

    def _draw_aux_panel(self):
        glViewport(self.main_w, 0, self.aux_w, self.height)
        glScissor(self.main_w, 0, self.aux_w, self.height)
        glEnable(GL_SCISSOR_TEST)

        glClearColor(*C_BG_AUX, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, self.aux_w, 0, self.height, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        state = self.env.state
        w_pos = float(state[4])
        th_w  = float(state[10])
        w_thr = self.env.w_threshold

        self._draw_aux_scene(w_pos, th_w, w_thr)
        self._draw_aux_hud(w_pos, th_w)

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glDisable(GL_SCISSOR_TEST)

    def _draw_aux_scene(self, w_pos, th_w, w_thr):
        """Draw the 2D side-view of the W axis inside the aux panel."""
        pw = self.aux_w
        ph = self.height

        margin   = 40
        track_y  = ph * 0.35
        usable_w = pw - 2 * margin
        scale    = usable_w / (2 * w_thr)

        def w_to_px(w):
            return margin + (w + w_thr) * scale

        # ── track ────────────────────────────────────────────────────
        glColor3f(*C_TRACK)
        glLineWidth(2.0)
        glBegin(GL_LINES)
        glVertex2f(margin, track_y)
        glVertex2f(pw - margin, track_y)
        glEnd()

        # ── boundary markers ────────────────────────────────────────
        glColor4f(*C_LIMIT, 0.7)
        glLineWidth(2.0)
        for bound in (-w_thr, w_thr):
            bx = w_to_px(bound)
            glBegin(GL_LINES)
            glVertex2f(bx, track_y - 30)
            glVertex2f(bx, track_y + 100)
            glEnd()

        # tick marks every 1.0
        glColor3f(*C_GRID)
        glLineWidth(1.0)
        tick = -int(w_thr)
        while tick <= int(w_thr):
            tx = w_to_px(tick)
            glBegin(GL_LINES)
            glVertex2f(tx, track_y - 5)
            glVertex2f(tx, track_y + 5)
            glEnd()
            tick += 1

        # ── cart ────────────────────────────────────────────────────
        cart_cx = w_to_px(np.clip(w_pos, -w_thr, w_thr))
        cart_hw = 18
        cart_hh = 12
        glColor3f(*C_AUX_CART)
        glBegin(GL_QUADS)
        glVertex2f(cart_cx - cart_hw, track_y)
        glVertex2f(cart_cx + cart_hw, track_y)
        glVertex2f(cart_cx + cart_hw, track_y + cart_hh * 2)
        glVertex2f(cart_cx - cart_hw, track_y + cart_hh * 2)
        glEnd()

        # wheels
        glColor3f(0.12, 0.12, 0.18)
        for ox in (-12, 12):
            glBegin(GL_QUADS)
            cx = cart_cx + ox
            glVertex2f(cx - 4, track_y - 4)
            glVertex2f(cx + 4, track_y - 4)
            glVertex2f(cx + 4, track_y + 4)
            glVertex2f(cx - 4, track_y + 4)
            glEnd()

        # ── pole ────────────────────────────────────────────────────
        pole_base_x = cart_cx
        pole_base_y = track_y + cart_hh * 2
        pole_px_len = ph * 0.28
        pole_tip_x  = pole_base_x + pole_px_len * np.sin(th_w)
        pole_tip_y  = pole_base_y + pole_px_len * np.cos(th_w)

        glColor3f(*C_AUX_POLE)
        glLineWidth(4.0)
        glBegin(GL_LINES)
        glVertex2f(pole_base_x, pole_base_y)
        glVertex2f(pole_tip_x,  pole_tip_y)
        glEnd()

        # tip dot
        glColor3f(*C_AUX_TIP)
        tip_r = 6
        glBegin(GL_TRIANGLE_FAN)
        glVertex2f(pole_tip_x, pole_tip_y)
        for a in np.linspace(0, 2 * np.pi, 16):
            glVertex2f(pole_tip_x + tip_r * np.cos(a),
                       pole_tip_y + tip_r * np.sin(a))
        glEnd()

        # ── angle arc indicator ─────────────────────────────────────
        arc_r = pole_px_len * 0.3
        ang_lim = self.env.angle_threshold
        ratio = min(abs(th_w) / ang_lim, 1.0)
        if ratio < 0.5:
            arc_col = C_GOOD
        elif ratio < 0.8:
            arc_col = C_WARN
        else:
            arc_col = C_BAD
        glColor3f(*arc_col)
        glLineWidth(2.0)
        glBegin(GL_LINE_STRIP)
        for a in np.linspace(0, th_w, 20):
            glVertex2f(pole_base_x + arc_r * np.sin(a),
                       pole_base_y + arc_r * np.cos(a))
        glEnd()

    def _draw_aux_hud(self, w_pos, th_w):
        th_w_deg = np.degrees(th_w)
        lim      = np.degrees(self.env.angle_threshold)
        r        = abs(th_w_deg) / lim
        if r < 0.5:
            col = C_GOOD
        elif r < 0.8:
            col = C_WARN
        else:
            col = C_BAD

        lines = [
            ("4TH DIMENSION  [W]",         C_DIM,  True),
            ("",                            C_DIM,  False),
            (f"Cart W   {w_pos:+.3f}",      C_DIM,  False),
            (f"Theta W  {th_w_deg:+.1f}\u00b0", col, False),
        ]

        self._blit_text_panel(lines, x_off=8, y_off=self.height - 8,
                              panel_w=self.aux_w - 16,
                              viewport_w=self.aux_w)

    # ==================================================================
    #  Divider line between panels
    # ==================================================================

    def _draw_divider(self):
        glViewport(0, 0, self.width, self.height)

        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)

        glMatrixMode(GL_PROJECTION)
        glPushMatrix(); glLoadIdentity()
        glOrtho(0, self.width, 0, self.height, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix(); glLoadIdentity()

        glColor3f(*C_DIVIDER)
        glLineWidth(2.0)
        dx = self.main_w
        glBegin(GL_LINES)
        glVertex2f(dx, 0)
        glVertex2f(dx, self.height)
        glEnd()

        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)

    # ==================================================================
    #  Shared text-blitting helper
    # ==================================================================

    def _blit_text_panel(self, lines, x_off, y_off,
                         panel_w, viewport_w):
        """Render a list of (text, colour, bold) lines as a translucent
        panel using Pygame font → OpenGL texture."""
        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)

        glMatrixMode(GL_PROJECTION)
        glPushMatrix(); glLoadIdentity()
        glOrtho(0, viewport_w, self.height, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix(); glLoadIdentity()

        panel_h = 14 + len(lines) * 20
        px = x_off
        py = y_off - panel_h

        glColor4f(0.04, 0.07, 0.11, 0.78)
        glBegin(GL_QUADS)
        glVertex2f(px,            py)
        glVertex2f(px + panel_w,  py)
        glVertex2f(px + panel_w,  py + panel_h)
        glVertex2f(px,            py + panel_h)
        glEnd()

        surf = pygame.Surface((panel_w - 4, panel_h), pygame.SRCALPHA)
        surf.fill((0, 0, 0, 0))
        for i, (text, color, bold) in enumerate(lines):
            if not text:
                continue
            font = self._font_lg if bold else self._font_sm
            r, g, b = [int(c * 255) for c in color]
            surf.blit(font.render(text, True, (r, g, b)), (6, 4 + i * 20))

        tex_data = pygame.image.tostring(surf, "RGBA", True)
        tid = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tid)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA,
                     surf.get_width(), surf.get_height(),
                     0, GL_RGBA, GL_UNSIGNED_BYTE, tex_data)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        glEnable(GL_TEXTURE_2D)
        glColor4f(1, 1, 1, 1)
        w, h = surf.get_size()
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0); glVertex2f(px,     py)
        glTexCoord2f(1, 0); glVertex2f(px + w, py)
        glTexCoord2f(1, 1); glVertex2f(px + w, py + h)
        glTexCoord2f(0, 1); glVertex2f(px,     py + h)
        glEnd()
        glDisable(GL_TEXTURE_2D)
        glDeleteTextures([tid])

        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)


# ── Demo (random policy) ─────────────────────────────────────────────
if __name__ == "__main__":
    from cartpole4d_env import CartPole4DEnv

    env      = CartPole4DEnv(use_discrete=True)
    renderer = CartPole4DRenderer(env)

    for ep in range(20):
        obs, _ = env.reset()
        renderer.reset_stats()
        done = False

        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            if not renderer.render(reward=reward):
                renderer.close()
                exit()
            done = terminated or truncated

    renderer.close()
