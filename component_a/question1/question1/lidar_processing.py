"""
LiDAR processing and occupancy-grid mapping for Component A (Question 1).

Pipeline per scan:
    1. clean_scan  - mark invalid returns (optionally median-filter spikes)
    2. scan_to_points - convert ranges to world (x, y) using the robot pose
    3. OccupancyGrid.update - log-odds ray casting into a 2-D grid

Design decisions:
    * Validity check: returns outside [min_range, max_range) carry no
      surface information and are marked invalid (NaN).
    * Optional median filter (3 rays wide): removes isolated range spikes
      while keeping sharp edges better than a mean. It is OFF by default
      because testing on this dataset showed the simulated noise (5 mm) is
      already below the 2 cm cell size, and the filter rounds off corners;
      it only pays off once outlier spikes exceed about 4% of rays (see the
      robustness test in perception_pipeline.ipynb).
    * Log-odds occupancy grid: each scan only gives weak evidence about each
      cell. Adding log-odds lets repeated observations reinforce each other
      and lets later "free" observations correct earlier false hits, so the
      map rejects random noise as the robot revisits areas.
    * Ray casting marks cells *between* the robot and a hit as free, so the
      grid distinguishes "free" from "never observed" (unknown).
"""

import json
import math
import os
import warnings

import numpy as np
import pandas as pd


def load_dataset(data_dir):
    """Load scans, poses and ground truth logged by the Webots controller."""
    scans = pd.read_csv(os.path.join(data_dir, "lidar_scans.csv"))
    with open(os.path.join(data_dir, "ground_truth.json")) as f:
        truth = json.load(f)
    range_cols = [c for c in scans.columns if c.startswith("r")]
    ranges = scans[range_cols].replace("inf", np.inf).astype(float).values
    return scans, ranges, truth


def ray_angles(n_rays, fov):
    """Ray angles in the robot frame. Webots orders rays left to right,
    starting at +FOV/2, so ray i points at FOV/2 - i * FOV / n_rays."""
    return fov / 2.0 - np.arange(n_rays) * fov / n_rays


def clean_scan(ranges, max_range, min_range=0.05, use_median=False):
    """Mark invalid returns as NaN; optionally remove isolated spikes."""
    r = np.array(ranges, dtype=float)
    r[~np.isfinite(r) | (r < min_range) | (r >= max_range)] = np.nan
    if not use_median:
        return r

    # Circular 3-ray median (the scan wraps around through 360 degrees).
    stacked = np.vstack([np.roll(r, 1), r, np.roll(r, -1)])
    with warnings.catch_warnings():
        # Rays with no valid neighbours stay NaN; that is expected.
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(stacked, axis=0)


def scan_to_points(ranges, pose, angles):
    """Convert valid ranges to world-frame (x, y) points."""
    x, y, yaw = pose
    valid = np.isfinite(ranges)
    a = yaw + angles[valid]
    return np.column_stack([x + ranges[valid] * np.cos(a),
                            y + ranges[valid] * np.sin(a)])


def bresenham(x0, y0, x1, y1):
    """Integer grid cells on the line from (x0, y0) to (x1, y1)."""
    cells = []
    dx, dy = abs(x1 - x0), -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        cells.append((x0, y0))
        if x0 == x1 and y0 == y1:
            return cells
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


class OccupancyGrid:
    """2-D log-odds occupancy grid.

    Cell (row, col) covers world x in [x_min + col*res, ...) and
    y in [y_min + row*res, ...), so row 0 is the bottom (smallest y).
    """

    L_OCC = math.log(0.7 / 0.3)    # evidence added for a hit
    L_FREE = math.log(0.4 / 0.6)   # evidence added for a pass-through
    L_MIN, L_MAX = -4.0, 4.0       # clamp so cells can still change later

    def __init__(self, x_min, y_min, size_x, size_y, resolution):
        self.x_min, self.y_min = x_min, y_min
        self.res = resolution
        self.cols = int(round(size_x / resolution))
        self.rows = int(round(size_y / resolution))
        self.log_odds = np.zeros((self.rows, self.cols))
        self.observed = np.zeros((self.rows, self.cols), dtype=bool)

    def world_to_cell(self, x, y):
        col = int(math.floor((x - self.x_min) / self.res))
        row = int(math.floor((y - self.y_min) / self.res))
        return row, col

    def cell_to_world(self, row, col):
        return (self.x_min + (col + 0.5) * self.res,
                self.y_min + (row + 0.5) * self.res)

    def in_bounds(self, row, col):
        return 0 <= row < self.rows and 0 <= col < self.cols

    def update(self, pose, ranges, angles, max_range):
        """Integrate one cleaned scan."""
        x, y, yaw = pose
        r0, c0 = self.world_to_cell(x, y)

        for r, a in zip(ranges, angles):
            hit = np.isfinite(r)
            dist = r if hit else max_range   # no return: free up to max
            ex = x + dist * math.cos(yaw + a)
            ey = y + dist * math.sin(yaw + a)
            r1, c1 = self.world_to_cell(ex, ey)

            line = bresenham(c0, r0, c1, r1)
            for c, rr in line[:-1]:
                if self.in_bounds(rr, c):
                    self.log_odds[rr, c] += self.L_FREE
                    self.observed[rr, c] = True
            if hit and self.in_bounds(r1, c1):
                self.log_odds[r1, c1] += self.L_OCC
                self.observed[r1, c1] = True

        np.clip(self.log_odds, self.L_MIN, self.L_MAX, out=self.log_odds)

    def probability(self):
        return 1.0 - 1.0 / (1.0 + np.exp(self.log_odds))

    def classify(self, occupied_threshold=0.65, free_threshold=0.35):
        """Return grid of -1 unknown, 0 free, 1 occupied."""
        p = self.probability()
        out = np.full(p.shape, -1, dtype=int)
        out[self.observed & (p <= free_threshold)] = 0
        out[self.observed & (p >= occupied_threshold)] = 1
        return out

    def extent(self):
        """Matplotlib imshow extent (left, right, bottom, top)."""
        return (self.x_min, self.x_min + self.cols * self.res,
                self.y_min, self.y_min + self.rows * self.res)


def ground_truth_grid(grid, truth):
    """Rasterise the true walls, obstacles and landmarks onto the grid."""
    occ = np.zeros((grid.rows, grid.cols), dtype=bool)
    half_x = truth["arena"]["size_x"] / 2
    half_y = truth["arena"]["size_y"] / 2

    for row in range(grid.rows):
        for col in range(grid.cols):
            wx, wy = grid.cell_to_world(row, col)
            if abs(wx) >= half_x or abs(wy) >= half_y:
                occ[row, col] = True
                continue
            for o in truth["obstacles"]:
                if (abs(wx - o["x"]) <= o["size_x"] / 2
                        and abs(wy - o["y"]) <= o["size_y"] / 2):
                    occ[row, col] = True
            for lm in truth["landmarks"]:
                if math.hypot(wx - lm["x"], wy - lm["y"]) <= lm["radius"]:
                    occ[row, col] = True
    return occ
