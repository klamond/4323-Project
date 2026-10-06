"""
ECE 4323 - Milestone 1: Discretization of the 2D ground workspace.

Frame convention (NED): x = North, y = East, ground plane at z = 0.
Altitude z_t is passed as a positive height above ground.

Run this file directly to produce the demo figure:
    python workspace.py
"""

import numpy as np
import matplotlib.pyplot as plt


def max_cell_size(sigma_cam, p_max=1.0, tol=0.05):
    """Largest cell size for which confidence is ~uniform over one cell.

    The camera model is p(r) = p_max * exp(-r^2 / (2 sigma^2)). Its steepest
    slope is at r = sigma, where |dp/dr| = p_max * exp(-1/2) / sigma.
    Requiring the change in p across one cell to stay below `tol` gives
        delta <= tol * sigma / (p_max * exp(-1/2)).
    """
    return tol * sigma_cam / (p_max * np.exp(-0.5))


class GridWorkspace:
    """Rectangular workspace W split into N = nx * ny uniform cells."""

    def __init__(self, x_min, x_max, y_min, y_max, cell_size):
        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max

        # Round the cell count up so cells tile W exactly (no partial cells).
        self.nx = int(np.ceil((x_max - x_min) / cell_size))
        self.ny = int(np.ceil((y_max - y_min) / cell_size))
        self.dx = (x_max - x_min) / self.nx
        self.dy = (y_max - y_min) / self.ny
        self.N = self.nx * self.ny

        # Cell centres c_i = [x_i, y_i]^T
        xc = x_min + (np.arange(self.nx) + 0.5) * self.dx
        yc = y_min + (np.arange(self.ny) + 0.5) * self.dy
        self.X, self.Y = np.meshgrid(xc, yc, indexing="ij")   # shape (nx, ny)
        self.centers = np.column_stack([self.X.ravel(), self.Y.ravel()])  # (N, 2)

    # ---- indexing -------------------------------------------------------
    def index(self, ix, iy):
        """Grid subscripts (ix, iy) -> flat cell index i."""
        return ix * self.ny + iy

    def subscripts(self, i):
        """Flat cell index i -> grid subscripts (ix, iy)."""
        return i // self.ny, i % self.ny

    def world_to_cell(self, x, y):
        """World position -> flat cell index, or -1 if outside W."""
        if not (self.x_min <= x < self.x_max and self.y_min <= y < self.y_max):
            return -1
        ix = int((x - self.x_min) // self.dx)
        iy = int((y - self.y_min) // self.dy)
        return self.index(ix, iy)

    def new_map(self, fill=0.0):
        """Per-cell array (e.g. the confidence map), shape (nx, ny)."""
        return np.full((self.nx, self.ny), fill, dtype=float)

    # ---- camera geometry (preview of Task 2) ----------------------------
    def relative_positions(self, x_t, y_t, psi=0.0):
        """Cell centres expressed in the camera frame (yaw-only rotation)."""
        d = self.centers - np.array([x_t, y_t])
        c, s = np.cos(psi), np.sin(psi)
        xr = c * d[:, 0] + s * d[:, 1]
        yr = -s * d[:, 0] + c * d[:, 1]
        return xr, yr

    def cells_in_footprint(self, x_t, y_t, z_t, fov_x, fov_y, psi=0.0):
        """Boolean mask (N,) of cells inside the rectangular footprint."""
        field_x = 2 * z_t * np.tan(fov_x / 2)
        field_y = 2 * z_t * np.tan(fov_y / 2)
        xr, yr = self.relative_positions(x_t, y_t, psi)
        return (np.abs(xr) <= field_x / 2) & (np.abs(yr) <= field_y / 2)

    # ---- plotting -------------------------------------------------------
    def plot(self, ax, values=None, show_lines=True, **kw):
        """Draw the grid. East on the horizontal axis, North on the vertical."""
        if values is not None:
            im = ax.pcolormesh(
                np.linspace(self.y_min, self.y_max, self.ny + 1),
                np.linspace(self.x_min, self.x_max, self.nx + 1),
                np.asarray(values).reshape(self.nx, self.ny),
                **kw,
            )
        else:
            im = None
        if show_lines:
            for x in np.linspace(self.x_min, self.x_max, self.nx + 1):
                ax.axhline(x, color="k", lw=0.3, alpha=0.4)
            for y in np.linspace(self.y_min, self.y_max, self.ny + 1):
                ax.axvline(y, color="k", lw=0.3, alpha=0.4)
        ax.set_xlim(self.y_min, self.y_max)
        ax.set_ylim(self.x_min, self.x_max)
        ax.set_aspect("equal")
        ax.set_xlabel("y, East (m)")
        ax.set_ylabel("x, North (m)")
        return im


if __name__ == "__main__":
    # ---- placeholder parameters: replace with your team's values --------
    X_MIN, X_MAX, Y_MIN, Y_MAX = 0.0, 20.0, 0.0, 20.0   # workspace (m)
    FOV_X = FOV_Y = np.deg2rad(60)                       # camera FOV
    Z_REF = 5.0                                          # reference altitude (m)
    P_MAX = 1.0
    SIGMA_CAM = 2.0                                      # confidence falloff (m)
    TOL = 0.05                                           # allowed change in p per cell

    limit = max_cell_size(SIGMA_CAM, P_MAX, TOL)
    cell = 0.15                                          # chosen size, must be <= limit
    ws = GridWorkspace(X_MIN, X_MAX, Y_MIN, Y_MAX, cell)

    print(f"Max cell size for {TOL:.0%} uniformity: {limit:.3f} m")
    print(f"Chosen cell size: {ws.dx:.3f} m x {ws.dy:.3f} m")
    print(f"Grid: {ws.nx} x {ws.ny} = {ws.N} cells")
    i = ws.world_to_cell(7.3, 12.9)
    print(f"Point (7.3, 12.9) -> cell {i}, subscripts {ws.subscripts(i)}, "
          f"centre {ws.centers[i]}")

    # Example pose and single-frame confidence, to show the grid in use
    x_t, y_t, z_t, psi = 8.0, 11.0, Z_REF, np.deg2rad(20)
    mask = ws.cells_in_footprint(x_t, y_t, z_t, FOV_X, FOV_Y, psi)
    xr, yr = ws.relative_positions(x_t, y_t, psi)
    p = P_MAX * np.exp(-(xr**2 + yr**2) / (2 * SIGMA_CAM**2)) \
        * Z_REF / (Z_REF + abs(z_t - Z_REF))
    p = np.where(mask, p, 0.0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    coarse = GridWorkspace(X_MIN, X_MAX, Y_MIN, Y_MAX, 1.0)  # readable lines
    coarse.plot(axes[0])
    axes[0].plot(coarse.centers[:, 1], coarse.centers[:, 0], ".", ms=3)
    axes[0].set_title("Discretization (shown at 1 m cells for legibility)")

    im = ws.plot(axes[1], p, show_lines=False, cmap="viridis", vmin=0, vmax=1)
    axes[1].plot(y_t, x_t, "r^", label="quadrotor")
    axes[1].legend(loc="upper right")
    axes[1].set_title(f"Single-frame confidence on {ws.nx}x{ws.ny} grid")
    fig.colorbar(im, ax=axes[1], label="p(c_i)")

    fig.tight_layout()
    fig.savefig("milestone1_grid.png", dpi=150)
    print("Saved milestone1_grid.png")
