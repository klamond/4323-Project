import ast
import os
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure


STEPS_PER_FRAME = 10  # sim steps between redraws
MAX_RANGE = 1000  # m, stop the run if the drone gets this far from the origin

# Plotted on the EKF states tab: (name, unit, state index). The rate is always at index + 1.
STATES = [
    ("Roll", "rad", 0),
    ("Pitch", "rad", 2),
    ("Yaw", "rad", 4),
    ("Z", "m", 6),
    ("X", "m", 8),
    ("Y", "m", 10),
]


def load_work():
    """Run work.py up to (not including) its main sim loop, and return its namespace.
    Gives us the functions, classes and parameters without running the sim or showing plots."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "work.py")
    tree = ast.parse(open(path).read(), path)
    for i, node in enumerate(tree.body):
        if isinstance(node, ast.For):
            tree.body = tree.body[:i]
            break
    ns = {"__name__": "work"}
    exec(compile(tree, path, "exec"), ns)
    return ns


wk = load_work()


class SimGUI:

    def __init__(self, root):
        self.root = root
        root.title("Quadrotor Coverage Sim")
        self.running = False

        # --- Left: controls ---
        panel = ttk.Frame(root, padding=8)
        panel.pack(side=tk.LEFT, fill=tk.Y)

        self.entries = {}
        cols = ttk.Frame(panel)
        cols.pack(anchor="w")
        left = ttk.Frame(cols)
        left.pack(side=tk.LEFT, anchor="n")
        right = ttk.Frame(cols)
        right.pack(side=tk.LEFT, anchor="n", padx=(12, 0))

        ttk.Label(left, text="Simulation", font=("", 11, "bold")).pack(anchor="w")
        self._entry(left, "sim time (s)", wk["timeArr"][-1] + wk["dt"])

        ttk.Label(left, text="Initial conditions", font=("", 11, "bold")).pack(anchor="w", pady=(8, 0))
        for k, v in wk["initial_cnds"].items():
            self._entry(left, k, v)

        ttk.Label(right, text="Target", font=("", 11, "bold")).pack(anchor="w")
        self._entry(right, "xd (m)", wk["xd"])
        self._entry(right, "yd (m)", wk["yd"])
        self._entry(right, "zd (m)", wk["zd"])
        self._entry(right, "yawd (deg)", np.rad2deg(wk["yawd"]))

        ttk.Label(right, text="Gains", font=("", 11, "bold")).pack(anchor="w", pady=(8, 0))
        for k, v in vars(wk["GAINS"]).items():
            self._entry(right, k, v)

        btns = ttk.Frame(panel)
        btns.pack(anchor="w", pady=8)
        ttk.Button(btns, text="Start", command=self.start).pack(side=tk.LEFT)
        ttk.Button(btns, text="Pause", command=self.pause).pack(side=tk.LEFT)
        ttk.Button(btns, text="Reset", command=self.reset).pack(side=tk.LEFT)

        self.status = tk.StringVar(value="Ready")
        ttk.Label(panel, textvariable=self.status, justify=tk.LEFT).pack(anchor="w")

        # --- Right: plots ---
        self.tabs = ttk.Notebook(root)
        self.tabs.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        tab_overview = ttk.Frame(self.tabs)
        tab_states = ttk.Frame(self.tabs)
        self.tabs.add(tab_overview, text="Overview")
        self.tabs.add(tab_states, text="EKF states")

        self.fig = Figure(figsize=(11, 6))
        self.ax_map = self.fig.add_subplot(1, 2, 1)
        self.ax_x = self.fig.add_subplot(2, 2, 2)
        self.ax_err = self.fig.add_subplot(2, 2, 4)
        self.canvas = FigureCanvasTkAgg(self.fig, master=tab_overview)
        NavigationToolbar2Tk(self.canvas, tab_overview)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self._build_states_tab(tab_states)
        self.tabs.bind("<<NotebookTabChanged>>", lambda e: self._redraw())

        self.reset()

    def _build_states_tab(self, parent):
        """One column per variable: value, its error, rate, its error."""
        self.fig_states = Figure(figsize=(11, 6))
        axes = self.fig_states.subplots(4, len(STATES), sharex=True)
        self.axes_states = axes
        self.state_lines = []  # (state index, actual line, EKF line, error line)

        for col, (name, unit, idx) in enumerate(STATES):
            for row, i, title, u in ((0, idx, name, unit), (2, idx + 1, f"{name} rate", f"{unit}/s")):
                ax, ax_err = axes[row, col], axes[row + 1, col]
                (l_true,) = ax.plot([], [], lw=1, label="Actual")
                (l_est,) = ax.plot([], [], "--", lw=1, label="EKF Estimate")
                (l_err,) = ax_err.plot([], [], lw=1, color="C2")
                ax.set_title(f"{title} ({u})", fontsize=8)
                ax_err.set_title("error", fontsize=7)
                self.state_lines.append((i, l_true, l_est, l_err))

        for ax in axes.flat:
            ax.grid(True)
            ax.tick_params(labelsize=6)
            ax.yaxis.get_offset_text().set_fontsize(6)
        for ax in axes[-1]:
            ax.set_xlabel("Time (s)", fontsize=7)

        self.fig_states.legend(handles=[l_true, l_est], loc="upper center", ncol=2, fontsize=8)
        self.fig_states.subplots_adjust(left=0.04, right=0.99, top=0.91, bottom=0.07, wspace=0.35, hspace=0.4)

        self.canvas_states = FigureCanvasTkAgg(self.fig_states, master=parent)
        NavigationToolbar2Tk(self.canvas_states, parent)
        self.canvas_states.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _entry(self, parent, name, default):
        row = ttk.Frame(parent)
        row.pack(anchor="w", fill=tk.X)
        ttk.Label(row, text=name, width=12).pack(side=tk.LEFT)
        e = ttk.Entry(row, width=10)
        e.insert(0, f"{default:g}")
        e.pack(side=tk.LEFT)
        self.entries[name] = e

    # --- Sim state ---

    def reset(self):
        self.running = False
        try:
            T = float(self.entries["sim time (s)"].get())
            x0 = [float(self.entries[k].get()) for k in wk["initial_cnds"]]
            self.target = (float(self.entries["xd (m)"].get()), float(self.entries["yd (m)"].get()),
                           float(self.entries["zd (m)"].get()), np.deg2rad(float(self.entries["yawd (deg)"].get())))
            self.gains = wk["gains"](**{k: float(self.entries[k].get()) for k in vars(wk["GAINS"])})
        except ValueError as e:
            messagebox.showerror("Bad input", str(e))
            return

        W1 = wk["W1"]
        self.W = wk["workspace"](W1.x_dim, W1.y_dim, W1.x_func, W1.y_func, W1.res)
        self.timeArr = np.arange(0, T, wk["dt"])
        self.x_true = np.array(x0, dtype=float)
        self.x_est = self.x_true.copy()
        self.P = wk["P"].copy()
        self.u = wk["u"].copy()
        self.k = 0
        self.diverged = False
        self.true_log = np.zeros((len(self.timeArr), 12))
        self.est_log = np.zeros((len(self.timeArr), 12))

        self._setup_plots()
        self.status.set("Ready")

    def _setup_plots(self):
        for ax in (self.ax_map, self.ax_x, self.ax_err):
            ax.clear()

        self.img = self.ax_map.imshow(self.W.confidence.T, origin="lower", cmap="plasma", vmin=0.5, vmax=1,
                                      extent=[0, self.W.x_dim, 0, self.W.y_dim])
        if not hasattr(self, "cbar"):
            self.cbar = self.fig.colorbar(self.img, ax=self.ax_map, label="Confidence", shrink=0.7)
        (self.l_true,) = self.ax_map.plot([], [], "c-", lw=1, label="Actual")
        (self.l_est,) = self.ax_map.plot([], [], "w:", lw=1, label="EKF")
        (self.l_drone,) = self.ax_map.plot([], [], "ro", ms=5)
        self.ax_map.plot([self.target[0]], [self.target[1]], "gx", ms=8, label="Target", scalex=False, scaley=False)
        self.ax_map.set_title("Confidence map")
        self.ax_map.set_xlabel("X-Position")
        self.ax_map.set_ylabel("Y-Position")
        self.ax_map.legend(loc="upper right", fontsize=7)

        (self.l_x,) = self.ax_x.plot([], [], label="Actual X-position")
        (self.l_xe,) = self.ax_x.plot([], [], "--", label="EKF Estimate")
        self.ax_x.set_xlim(0, self.timeArr[-1])
        self.ax_x.set_ylabel("X-position (m)")
        self.ax_x.set_title("EKF Position Estimation")
        self.ax_x.grid(True)
        self.ax_x.legend(fontsize=7)

        (self.l_ex,) = self.ax_err.plot([], [])
        self.ax_err.set_xlim(0, self.timeArr[-1])
        self.ax_err.set_xlabel("Time (s)")
        self.ax_err.set_ylabel("X-position error (m)")
        self.ax_err.grid(True)

        self.fig.tight_layout()
        self.canvas.draw()

        for _, *lines in self.state_lines:
            for line in lines:
                line.set_data([], [])
        self.axes_states[0, 0].set_xlim(0, self.timeArr[-1])  # shared by every axis on the tab
        self.canvas_states.draw_idle()

    def start(self):
        if self.k >= len(self.timeArr) or self.diverged:
            self.reset()
        if not self.running:
            self.running = True
            self._loop()

    def pause(self):
        self.running = False

    def _step(self):
        """One pass of the work.py loop body. Returns False if the drone has flown away."""
        params, Q, R, H, dt, omegaR = (wk[n] for n in ("params", "Q", "R", "H", "dt", "omegaR"))

        self.u = wk["controller"](*self.target, self.x_est, params, self.gains)

        self.x_true = self.x_true + wk["update_A"](self.x_true, self.u, params, omegaR) * dt \
            + np.random.multivariate_normal(np.zeros(12), Q)
        measurement = H @ self.x_true + np.random.multivariate_normal(np.zeros(6), R)
        self.x_est, self.P = wk["EKF"](self.x_est, self.P, self.u, measurement, Q, R, H, params, omegaR, dt)

        # camera_ft's footprint grows with height, so a flyaway would otherwise stall or crash the GUI
        pos = np.concatenate((self.x_true[[6, 8, 10]], self.x_est[[6, 8, 10]]))
        if not np.all(np.abs(pos) < MAX_RANGE):  # also catches nan
            return False

        wk["camera_ft"](self.x_est, self.W, wk["C1"])

        self.true_log[self.k] = self.x_true
        self.est_log[self.k] = self.x_est
        self.k += 1
        return True

    def _loop(self):
        if not self.running:
            return
        for _ in range(STEPS_PER_FRAME):
            if self.k >= len(self.timeArr):
                self.running = False
                break
            if not self._step():
                self.running = False
                self.diverged = True
                break
        self._redraw()
        if self.running:
            self.root.after(1, self._loop)

    def _redraw(self):
        k = self.k
        if k == 0:
            return
        t = self.timeArr[:k]
        tr, es = self.true_log[:k], self.est_log[:k]

        # Only the visible tab is drawn; the states tab has too many axes to redraw every frame
        if self.tabs.index("current") == 1:
            self._redraw_states(t, tr, es)
        else:
            self._redraw_overview(t, tr, es)

        self.status.set(f"t = {t[-1]:.2f} s\n"
                        f"x, y, z = {tr[-1, 8]:.1f}, {tr[-1, 10]:.1f}, {tr[-1, 6]:.1f}\n"
                        f"avg confidence = {np.mean(self.W.confidence):.4f}"
                        + ("\nDIVERGED: tune gains, Reset" if self.diverged else ""))

    def _redraw_overview(self, t, tr, es):
        self.img.set_data(self.W.confidence.T)
        self.l_true.set_data(tr[:, 8], tr[:, 10])
        self.l_est.set_data(es[:, 8], es[:, 10])
        self.l_drone.set_data([tr[-1, 8]], [tr[-1, 10]])

        self.l_x.set_data(t, tr[:, 8])
        self.l_xe.set_data(t, es[:, 8])
        self.l_ex.set_data(t, tr[:, 8] - es[:, 8])
        for ax in (self.ax_x, self.ax_err):
            ax.relim()
            ax.autoscale_view(scalex=False)
        self.canvas.draw_idle()

    def _redraw_states(self, t, tr, es):
        for i, l_true, l_est, l_err in self.state_lines:
            l_true.set_data(t, tr[:, i])
            l_est.set_data(t, es[:, i])
            l_err.set_data(t, tr[:, i] - es[:, i])
        for ax in self.axes_states.flat:
            ax.relim()
            ax.autoscale_view(scalex=False)
        self.canvas_states.draw_idle()


if __name__ == "__main__":
    root = tk.Tk()
    SimGUI(root)
    root.mainloop()
