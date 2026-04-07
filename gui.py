"""
Интерактивный GUI для визуализации перехода к хаосу
в обобщённом отображении параболы x_{n+1} = 1 - mu * |x_n|^z.

Шесть вкладок:
  1. Бифуркационная диаграмма
  2. Паутинная диаграмма
  3. Показатель Ляпунова
  4. Константы Фейгенбаума
  5. Скейлинг и самоподобие
  6. Динамика (анимация в реальном времени)
"""

import numpy as np
import matplotlib
try:
    matplotlib.use("TkAgg")
except ImportError:
    pass
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

import core


COLORS = {
    "bg": "#1e1e2e",
    "panel": "#2a2a3d",
    "text": "#cdd6f4",
    "accent1": "#89b4fa",
    "accent2": "#f38ba8",
    "accent3": "#a6e3a1",
    "accent4": "#fab387",
    "accent5": "#cba6f7",
    "grid": "#45475a",
    "btn_active": "#89b4fa",
    "btn_inactive": "#313244",
    "btn_active_text": "#1e1e2e",
    "scatter": "#89dceb",
    "cobweb": "#f9e2af",
    "bisect": "#f38ba8",
    "play_btn": "#a6e3a1",
    "pause_btn": "#fab387",
    "reset_btn": "#f38ba8",
    "trail_old": "#585b70",
}

TAB_NAMES = [
    "Бифуркация",
    "Паутина",
    "Ляпунов",
    "Фейгенбаум",
    "Скейлинг",
    "Динамика",
]

DYN_TAB = 5


class ChaosApp:
    def __init__(self):
        self.current_tab = 0

        self.z_val = 2.0
        self.mu_val = 1.5
        self.n_iter_cobweb = 80
        self.speed_val = 10

        self._bif_cache = {}
        self._lyap_cache = {}
        self._feig_cache = {}
        self._feig_vs_z_cache = None
        self._scaling_cache = {}

        self._timer = None
        self._anim_running = False
        self._anim_x = 0.1
        self._anim_xs = [0.1]
        self._anim_cobweb_lines = []
        self._anim_ts_line = None
        self._anim_dot = None
        self._anim_iter_text = None

        self.fig = plt.figure(figsize=(14, 8.5), facecolor=COLORS["bg"])
        self.fig.canvas.manager.set_window_title(
            "Переход к хаосу в обобщённом отображении параболы"
        )

        self.gs = GridSpec(
            12, 12,
            figure=self.fig,
            left=0.07, right=0.97, top=0.92, bottom=0.13,
            hspace=0.4, wspace=0.35,
        )

        self._create_tab_buttons()
        self._create_sliders()
        self._create_axes()

        self._show_tab(0)

    # ------------------------------------------------------------------ layout
    def _create_tab_buttons(self):
        self.tab_buttons = []
        btn_width = 0.145
        btn_gap = 0.004
        total = len(TAB_NAMES) * btn_width + (len(TAB_NAMES) - 1) * btn_gap
        x_start = 0.5 - total / 2.0

        self.tab_indicators = []
        for i, name in enumerate(TAB_NAMES):
            x_pos = x_start + i * (btn_width + btn_gap)
            ax = self.fig.add_axes(
                [x_pos, 0.949, btn_width, 0.04]
            )
            btn = Button(
                ax, name,
                color=COLORS["btn_inactive"],
                hovercolor=COLORS["btn_active"],
            )
            btn.label.set_color(COLORS["text"])
            btn.label.set_fontsize(9.5)
            btn.label.set_fontweight("normal")
            btn.on_clicked(self._make_tab_callback(i))
            self.tab_buttons.append((ax, btn))

            ind_ax = self.fig.add_axes(
                [x_pos, 0.945, btn_width, 0.004]
            )
            ind_ax.set_xticks([])
            ind_ax.set_yticks([])
            for spine in ind_ax.spines.values():
                spine.set_visible(False)
            ind_ax.set_facecolor(COLORS["bg"])
            self.tab_indicators.append(ind_ax)

    def _make_tab_callback(self, idx):
        def callback(event):
            self._show_tab(idx)
        return callback

    def _create_sliders(self):
        ax_z = self.fig.add_axes(
            [0.15, 0.05, 0.30, 0.025], facecolor=COLORS["panel"]
        )
        self.slider_z = Slider(
            ax_z, "z", 1.5, 6.0, valinit=self.z_val,
            valstep=0.1, color=COLORS["accent1"],
        )
        self.slider_z.label.set_color(COLORS["text"])
        self.slider_z.valtext.set_color(COLORS["text"])
        self.slider_z.on_changed(self._on_z_changed)

        ax_mu = self.fig.add_axes(
            [0.15, 0.02, 0.30, 0.025], facecolor=COLORS["panel"]
        )
        self.slider_mu = Slider(
            ax_mu, "μ", 0.01, 2.0, valinit=self.mu_val,
            valstep=0.01, color=COLORS["accent2"],
        )
        self.slider_mu.label.set_color(COLORS["text"])
        self.slider_mu.valtext.set_color(COLORS["text"])
        self.slider_mu.on_changed(self._on_mu_changed)

        ax_n = self.fig.add_axes(
            [0.60, 0.05, 0.30, 0.025], facecolor=COLORS["panel"]
        )
        self.slider_n = Slider(
            ax_n, "Итераций", 10, 200, valinit=self.n_iter_cobweb,
            valstep=5, color=COLORS["accent3"],
        )
        self.slider_n.label.set_color(COLORS["text"])
        self.slider_n.valtext.set_color(COLORS["text"])
        self.slider_n.on_changed(self._on_n_changed)

        ax_speed = self.fig.add_axes(
            [0.60, 0.05, 0.30, 0.025], facecolor=COLORS["panel"]
        )
        self.slider_speed = Slider(
            ax_speed, "Скорость", 1, 50, valinit=self.speed_val,
            valstep=1, color=COLORS["accent5"],
        )
        self.slider_speed.label.set_color(COLORS["text"])
        self.slider_speed.valtext.set_color(COLORS["text"])
        self.slider_speed.on_changed(self._on_speed_changed)

        bw = 0.085
        bg = 0.008
        bx_start = 0.60
        by = 0.018

        ax_play = self.fig.add_axes([bx_start, by, bw, 0.03])
        self.btn_play = Button(ax_play, "▶ Play",
                               color=COLORS["btn_inactive"],
                               hovercolor=COLORS["play_btn"])
        self.btn_play.label.set_color(COLORS["play_btn"])
        self.btn_play.label.set_fontsize(9)
        self.btn_play.on_clicked(self._on_play)

        ax_pause = self.fig.add_axes([bx_start + bw + bg, by, bw, 0.03])
        self.btn_pause = Button(ax_pause, "❚❚ Пауза",
                                color=COLORS["btn_inactive"],
                                hovercolor=COLORS["pause_btn"])
        self.btn_pause.label.set_color(COLORS["pause_btn"])
        self.btn_pause.label.set_fontsize(9)
        self.btn_pause.on_clicked(self._on_pause)

        ax_reset = self.fig.add_axes([bx_start + 2 * (bw + bg), by, bw, 0.03])
        self.btn_reset = Button(ax_reset, "⟲ Сброс",
                                color=COLORS["btn_inactive"],
                                hovercolor=COLORS["reset_btn"])
        self.btn_reset.label.set_color(COLORS["reset_btn"])
        self.btn_reset.label.set_fontsize(9)
        self.btn_reset.on_clicked(self._on_reset)

        self.slider_axes = {
            "z": ax_z, "mu": ax_mu, "n": ax_n, "speed": ax_speed,
        }
        self.dyn_btn_axes = [ax_play, ax_pause, ax_reset]

    def _create_axes(self):
        self.ax_main = self.fig.add_subplot(self.gs[0:11, 0:12])
        self._style_ax(self.ax_main)

        self.ax_left = self.fig.add_subplot(self.gs[0:11, 0:6])
        self._style_ax(self.ax_left)

        self.ax_right = self.fig.add_subplot(self.gs[0:11, 6:12])
        self._style_ax(self.ax_right)

        self.ax_panels = []
        for k in range(3):
            ax = self.fig.add_subplot(self.gs[0:11, k * 4:(k + 1) * 4])
            self._style_ax(ax)
            self.ax_panels.append(ax)

        self.ax_dyn_left = self.fig.add_subplot(self.gs[0:11, 0:6])
        self._style_ax(self.ax_dyn_left)

        self.ax_dyn_right = self.fig.add_subplot(self.gs[0:11, 6:12])
        self._style_ax(self.ax_dyn_right)

        self._hide_all_axes()

    def _style_ax(self, ax):
        ax.set_facecolor(COLORS["panel"])
        ax.tick_params(colors=COLORS["text"], labelsize=9)
        ax.xaxis.label.set_color(COLORS["text"])
        ax.yaxis.label.set_color(COLORS["text"])
        ax.title.set_color(COLORS["text"])
        for spine in ax.spines.values():
            spine.set_color(COLORS["grid"])
        ax.grid(True, color=COLORS["grid"], alpha=0.3, linewidth=0.5)

    def _hide_all_axes(self):
        for ax in ([self.ax_main, self.ax_left, self.ax_right,
                     self.ax_dyn_left, self.ax_dyn_right] + self.ax_panels):
            ax.set_visible(False)

    # ----------------------------------------------------------- tab switching
    def _show_tab(self, idx):
        if self.current_tab == DYN_TAB and idx != DYN_TAB:
            self._stop_animation()

        self.current_tab = idx

        for i, (bax, btn) in enumerate(self.tab_buttons):
            if i == idx:
                bax.set_facecolor(COLORS["btn_active"])
                btn.label.set_color(COLORS["btn_active_text"])
                btn.label.set_fontweight("bold")
                self.tab_indicators[i].set_facecolor(COLORS["btn_active"])
            else:
                bax.set_facecolor(COLORS["btn_inactive"])
                btn.label.set_color(COLORS["text"])
                btn.label.set_fontweight("normal")
                self.tab_indicators[i].set_facecolor(COLORS["bg"])

        self._hide_all_axes()

        is_dyn = idx == DYN_TAB
        slider_visibility = {
            0: {"z": True,  "mu": False, "n": False, "speed": False},
            1: {"z": True,  "mu": True,  "n": True,  "speed": False},
            2: {"z": True,  "mu": False, "n": False, "speed": False},
            3: {"z": True,  "mu": False, "n": False, "speed": False},
            4: {"z": True,  "mu": False, "n": False, "speed": False},
            5: {"z": True,  "mu": True,  "n": False, "speed": True},
        }
        for key, visible in slider_visibility[idx].items():
            ax = self.slider_axes[key]
            ax.set_visible(visible)
            ax.set_zorder(10 if visible else -10)

        for bax in self.dyn_btn_axes:
            bax.set_visible(is_dyn)
            bax.set_zorder(10 if is_dyn else -10)

        draw_funcs = [
            self._draw_bifurcation,
            self._draw_cobweb,
            self._draw_lyapunov,
            self._draw_feigenbaum,
            self._draw_scaling,
            self._draw_dynamics,
        ]
        draw_funcs[idx]()
        self.fig.canvas.draw_idle()

    # --------------------------------------------------------- slider handlers
    def _on_z_changed(self, val):
        self.z_val = round(val, 1)
        if self.current_tab == DYN_TAB:
            self._reset_dynamics()
        else:
            self._show_tab(self.current_tab)

    def _on_mu_changed(self, val):
        self.mu_val = round(val, 2)
        if self.current_tab == 1:
            self._show_tab(1)
        elif self.current_tab == DYN_TAB:
            self._reset_dynamics()

    def _on_n_changed(self, val):
        self.n_iter_cobweb = int(val)
        if self.current_tab == 1:
            self._show_tab(1)

    def _on_speed_changed(self, val):
        self.speed_val = int(val)
        if self._timer is not None and self._anim_running:
            self._timer.interval = max(10, 1000 // self.speed_val)

    # ================================================= TAB 1: Bifurcation
    def _draw_bifurcation(self):
        ax = self.ax_main
        ax.set_visible(True)
        ax.cla()
        self._style_ax(ax)

        z = self.z_val
        cache_key = round(z, 1)
        if cache_key not in self._bif_cache:
            self._bif_cache[cache_key] = core.bifurcation_data(
                z, mu_min=0.0, mu_max=2.0, n_mu=1200, n_skip=400, n_plot=200,
            )
        mu_data, x_data = self._bif_cache[cache_key]

        ax.scatter(mu_data, x_data, s=0.02, c=COLORS["scatter"], alpha=0.5, linewidths=0)
        ax.set_xlim(0, 2)
        ax.set_ylim(-1.5, 1.5)
        ax.set_xlabel("μ", fontsize=12)
        ax.set_ylabel("x", fontsize=12)
        ax.set_title(
            f"Бифуркационная диаграмма   x_{{n+1}} = 1 − μ·|x_n|^{z:.1f}",
            fontsize=13, pad=10,
        )

    # ================================================= TAB 2: Cobweb
    def _draw_cobweb(self):
        ax = self.ax_main
        ax.set_visible(True)
        ax.cla()
        self._style_ax(ax)

        z = self.z_val
        mu = self.mu_val
        n_iter = self.n_iter_cobweb

        xs = np.linspace(-1.3, 1.3, 500)
        ys = core.generalized_map(xs, mu, z)
        ax.plot(xs, ys, color=COLORS["accent1"], linewidth=2, label=f"f(x) = 1 − {mu:.2f}·|x|^{z:.1f}")
        ax.plot(xs, xs, color=COLORS["bisect"], linewidth=1.5, linestyle="--", label="y = x")

        cx, cy = core.cobweb_data(mu, z, x0=0.1, n_iter=n_iter)
        ax.plot(cx, cy, color=COLORS["cobweb"], linewidth=0.7, alpha=0.85)

        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.3, 1.3)
        ax.set_xlabel("x_n", fontsize=12)
        ax.set_ylabel("x_{n+1}", fontsize=12)
        ax.set_title(
            f"Паутинная диаграмма   μ = {mu:.2f},  z = {z:.1f}",
            fontsize=13, pad=10,
        )
        ax.legend(loc="upper right", fontsize=9, facecolor=COLORS["panel"],
                  edgecolor=COLORS["grid"], labelcolor=COLORS["text"])

    # ================================================= TAB 3: Lyapunov
    def _draw_lyapunov(self):
        ax = self.ax_main
        ax.set_visible(True)
        ax.cla()
        self._style_ax(ax)

        z = self.z_val
        cache_key = round(z, 1)
        if cache_key not in self._lyap_cache:
            mu_arr = np.linspace(0.01, 2.0, 1000)
            lyap = core.lyapunov_exponent(mu_arr, z, n_iter=800, n_skip=300)
            self._lyap_cache[cache_key] = (mu_arr, lyap)
        mu_arr, lyap = self._lyap_cache[cache_key]

        pos = lyap >= 0
        neg = lyap < 0
        ax.fill_between(mu_arr, lyap, 0, where=pos, color=COLORS["accent2"], alpha=0.35, label="λ > 0 (хаос)")
        ax.fill_between(mu_arr, lyap, 0, where=neg, color=COLORS["accent3"], alpha=0.35, label="λ < 0 (порядок)")
        ax.plot(mu_arr, lyap, color=COLORS["accent1"], linewidth=0.8)
        ax.axhline(0, color=COLORS["text"], linewidth=1, linestyle="--", alpha=0.6)

        ax.set_xlim(0, 2)
        ax.set_ylim(min(-3, np.nanmin(lyap) - 0.3), max(1.5, np.nanmax(lyap) + 0.3))
        ax.set_xlabel("μ", fontsize=12)
        ax.set_ylabel("λ (показатель Ляпунова)", fontsize=12)
        ax.set_title(
            f"Показатель Ляпунова   z = {z:.1f}",
            fontsize=13, pad=10,
        )
        ax.legend(loc="lower left", fontsize=9, facecolor=COLORS["panel"],
                  edgecolor=COLORS["grid"], labelcolor=COLORS["text"])

    # ================================================= TAB 4: Feigenbaum
    def _draw_feigenbaum(self):
        self.ax_left.set_visible(True)
        self.ax_right.set_visible(True)
        self.ax_left.cla()
        self.ax_right.cla()
        self._style_ax(self.ax_left)
        self._style_ax(self.ax_right)

        z = self.z_val
        cache_key = round(z, 1)
        if cache_key not in self._feig_cache:
            bp = core.find_bifurcation_points(z, n_bifurcations=7)
            ds = core.feigenbaum_deltas(bp)
            als = core.feigenbaum_alphas(z, bp)
            self._feig_cache[cache_key] = (bp, ds, als)
        bp, ds, als = self._feig_cache[cache_key]

        ax = self.ax_left
        ns = np.arange(1, len(ds) + 1)
        if len(ds) > 0:
            ax.plot(ns, ds, "o-", color=COLORS["accent1"], markersize=7, linewidth=2, label="δ_n")
            if len(ds) > 1:
                ax.axhline(ds[-1], color=COLORS["accent1"], linestyle=":", alpha=0.5)
        if len(als) > 0:
            ns_a = np.arange(1, len(als) + 1)
            ax.plot(ns_a, als, "s-", color=COLORS["accent2"], markersize=7, linewidth=2, label="α_n")
            if len(als) > 1:
                ax.axhline(als[-1], color=COLORS["accent2"], linestyle=":", alpha=0.5)

        ax.set_xlabel("n", fontsize=11)
        ax.set_ylabel("Значение", fontsize=11)
        ax.set_title(f"Сходимость δ_n и α_n   (z = {z:.1f})", fontsize=12, pad=8)
        ax.legend(loc="best", fontsize=9, facecolor=COLORS["panel"],
                  edgecolor=COLORS["grid"], labelcolor=COLORS["text"])

        if len(bp) > 0:
            txt_lines = ["Точки бифуркаций:"]
            for i, m in enumerate(bp):
                txt_lines.append(f"  μ_{i + 1} = {m:.8f}")
            if len(ds) > 0:
                txt_lines.append(f"\nδ → {ds[-1]:.4f}")
            if len(als) > 0:
                txt_lines.append(f"α → {als[-1]:.4f}")
            ax.text(
                0.98, 0.02, "\n".join(txt_lines),
                transform=ax.transAxes, fontsize=8,
                verticalalignment="bottom", horizontalalignment="right",
                color=COLORS["text"], alpha=0.8,
                bbox=dict(boxstyle="round,pad=0.4", facecolor=COLORS["bg"], alpha=0.7),
            )

        ax2 = self.ax_right
        if self._feig_vs_z_cache is None:
            z_arr = np.array([2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0])
            d_arr, a_arr = core.feigenbaum_constants_vs_z(z_arr, n_bif=6)
            self._feig_vs_z_cache = (z_arr, d_arr, a_arr)
        z_arr, d_arr, a_arr = self._feig_vs_z_cache

        mask_d = ~np.isnan(d_arr)
        mask_a = ~np.isnan(a_arr)
        if np.any(mask_d):
            ax2.plot(z_arr[mask_d], d_arr[mask_d], "o-", color=COLORS["accent1"],
                     markersize=7, linewidth=2, label="δ(z)")
        if np.any(mask_a):
            ax2.plot(z_arr[mask_a], a_arr[mask_a], "s-", color=COLORS["accent2"],
                     markersize=7, linewidth=2, label="α(z)")

        ax2.axhline(4.6692, color=COLORS["accent1"], linestyle="--", alpha=0.4, label="δ = 4.6692 (z=2)")
        ax2.axhline(2.5029, color=COLORS["accent2"], linestyle="--", alpha=0.4, label="α = 2.5029 (z=2)")

        ax2.set_xlabel("z (порядок экстремума)", fontsize=11)
        ax2.set_ylabel("Значение константы", fontsize=11)
        ax2.set_title("Зависимость констант Фейгенбаума от z", fontsize=12, pad=8)
        ax2.legend(loc="best", fontsize=8, facecolor=COLORS["panel"],
                   edgecolor=COLORS["grid"], labelcolor=COLORS["text"])

    # ================================================= TAB 5: Scaling
    def _draw_scaling(self):
        for ax in self.ax_panels:
            ax.set_visible(True)
            ax.cla()
            self._style_ax(ax)

        z = self.z_val
        cache_key = round(z, 1)

        if cache_key not in self._scaling_cache:
            regions = core.scaling_zoom_regions(z, n_zooms=3)
            panels_data = []
            for k in range(3):
                if k < len(regions):
                    mu_lo, mu_hi, x_lo, x_hi = regions[k]
                else:
                    mu_lo, mu_hi, x_lo, x_hi = 0.0, 2.0, -1.5, 1.5
                n_mu = 800 if k == 0 else 600
                mu_data, x_data = core.bifurcation_data(
                    z, mu_min=mu_lo, mu_max=mu_hi, n_mu=n_mu, n_skip=500, n_plot=200,
                )
                panels_data.append((mu_data, x_data, mu_lo, mu_hi, x_lo, x_hi))
            self._scaling_cache[cache_key] = panels_data

        panels_data = self._scaling_cache[cache_key]
        titles = ["Полный вид", "Зум ×1", "Зум ×2"]

        for k, ax in enumerate(self.ax_panels):
            mu_data, x_data, mu_lo, mu_hi, x_lo, x_hi = panels_data[k]

            ax.scatter(mu_data, x_data, s=0.05, c=COLORS["scatter"], alpha=0.5, linewidths=0)
            ax.set_xlim(mu_lo, mu_hi)
            ax.set_ylim(x_lo, x_hi)
            ax.set_xlabel("μ", fontsize=10)
            if k == 0:
                ax.set_ylabel("x", fontsize=10)
            ax.set_title(f"{titles[k]}   (z = {z:.1f})", fontsize=11, pad=6)

            if k > 0:
                rect_color = COLORS["accent4"] if k == 1 else COLORS["accent5"]
                ax_prev = self.ax_panels[k - 1]
                r = Rectangle(
                    (mu_lo, x_lo), mu_hi - mu_lo, x_hi - x_lo,
                    linewidth=1.5, edgecolor=rect_color, facecolor="none",
                    linestyle="--",
                )
                ax_prev.add_patch(r)

    # ================================================= TAB 6: Dynamics
    def _prepare_dyn_axes(self):
        """Подготавливает оси для анимации: f(x), биссектриса, оси временного ряда."""
        ax_l = self.ax_dyn_left
        ax_r = self.ax_dyn_right
        ax_l.set_visible(True)
        ax_r.set_visible(True)
        ax_l.cla()
        ax_r.cla()
        self._style_ax(ax_l)
        self._style_ax(ax_r)

        z = self.z_val
        mu = self.mu_val

        xs = np.linspace(-1.3, 1.3, 500)
        ys = core.generalized_map(xs, mu, z)
        ax_l.plot(xs, ys, color=COLORS["accent1"], linewidth=2,
                  label=f"f(x) = 1 − {mu:.2f}·|x|^{z:.1f}")
        ax_l.plot(xs, xs, color=COLORS["bisect"], linewidth=1.5,
                  linestyle="--", label="y = x")
        ax_l.set_xlim(-1.3, 1.3)
        ax_l.set_ylim(-1.3, 1.3)
        ax_l.set_xlabel("x", fontsize=11)
        ax_l.set_ylabel("f(x)", fontsize=11)
        ax_l.set_title(
            f"Паутина (анимация)   μ={mu:.2f}, z={z:.1f}",
            fontsize=12, pad=8,
        )
        ax_l.legend(loc="upper right", fontsize=8, facecolor=COLORS["panel"],
                    edgecolor=COLORS["grid"], labelcolor=COLORS["text"])

        self._anim_ts_line, = ax_r.plot([], [], color=COLORS["accent1"],
                                         linewidth=1.2)
        self._anim_dot, = ax_r.plot([], [], "o", color=COLORS["accent2"],
                                     markersize=6)
        ax_r.set_xlim(0, 60)
        ax_r.set_ylim(-1.3, 1.3)
        ax_r.set_xlabel("n (итерация)", fontsize=11)
        ax_r.set_ylabel("x_n", fontsize=11)
        ax_r.set_title("Временной ряд", fontsize=12, pad=8)

        self._anim_iter_text = ax_r.text(
            0.98, 0.96, "n = 0",
            transform=ax_r.transAxes, fontsize=11,
            verticalalignment="top", horizontalalignment="right",
            color=COLORS["accent3"],
            bbox=dict(boxstyle="round,pad=0.3", facecolor=COLORS["bg"], alpha=0.8),
        )

    def _draw_dynamics(self):
        self._stop_animation()
        self._prepare_dyn_axes()
        self._anim_x = 0.1
        self._anim_xs = [0.1]
        self._anim_cobweb_lines = []

    def _anim_step(self, frame):
        ax_l = self.ax_dyn_left
        ax_r = self.ax_dyn_right
        mu = self.mu_val
        z = self.z_val

        x_old = self._anim_x
        y = 1.0 - mu * abs(x_old) ** z

        if abs(y) > 1e6:
            self._stop_animation()
            return

        n_lines = len(self._anim_cobweb_lines)
        alpha = max(0.15, 1.0 - n_lines * 0.003)

        l1, = ax_l.plot([x_old, x_old], [x_old, y],
                        color=COLORS["cobweb"], linewidth=0.9, alpha=alpha)
        l2, = ax_l.plot([x_old, y], [y, y],
                        color=COLORS["cobweb"], linewidth=0.9, alpha=alpha)
        self._anim_cobweb_lines.extend([l1, l2])

        if len(self._anim_cobweb_lines) > 400:
            for old_line in self._anim_cobweb_lines[:2]:
                old_line.remove()
            self._anim_cobweb_lines = self._anim_cobweb_lines[2:]

        self._anim_x = y
        self._anim_xs.append(y)

        ns = list(range(len(self._anim_xs)))
        self._anim_ts_line.set_data(ns, self._anim_xs)
        self._anim_dot.set_data([ns[-1]], [self._anim_xs[-1]])

        n_total = len(self._anim_xs) - 1
        xlim_max = max(60, n_total + 10)
        ax_r.set_xlim(0, xlim_max)

        y_min = min(self._anim_xs[-200:]) - 0.1 if len(self._anim_xs) > 1 else -1.3
        y_max = max(self._anim_xs[-200:]) + 0.1 if len(self._anim_xs) > 1 else 1.3
        margin = max(0.2, (y_max - y_min) * 0.15)
        ax_r.set_ylim(y_min - margin, y_max + margin)

        self._anim_iter_text.set_text(f"n = {n_total}")

    # ----------------------------------------- animation controls
    def _on_play(self, event):
        if self.current_tab != DYN_TAB or self._anim_running:
            return
        interval = max(10, 1000 // self.speed_val)
        if self._timer is None:
            self._timer = self.fig.canvas.new_timer(interval=interval)
            self._timer.add_callback(self._timer_tick)
        else:
            self._timer.interval = interval
        self._timer.start()
        self._anim_running = True

    def _on_pause(self, event):
        if self._anim_running:
            if self._timer is not None:
                self._timer.stop()
            self._anim_running = False

    def _on_reset(self, event):
        if self.current_tab == DYN_TAB:
            self._reset_dynamics()

    def _reset_dynamics(self):
        self._stop_animation()
        self._draw_dynamics()
        self.fig.canvas.draw_idle()

    def _stop_animation(self):
        if self._timer is not None:
            self._timer.stop()
        self._anim_running = False

    def _timer_tick(self):
        if not self._anim_running or self.current_tab != DYN_TAB:
            return
        self._anim_step(None)
        self.fig.canvas.draw_idle()

    # -------------------------------------------------------------------- run
    def run(self):
        plt.show()
