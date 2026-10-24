"""
Motion planning for Component A (Question 2).

Plans on the fused occupancy grid produced by Question 1
(component_a/question1/outputs/fused_map.npz).

Contents:
    * build_planning_grid - turn the semantic map into a binary planning grid
      (occupied OR unknown = blocked), then inflate obstacles by the robot's
      radius so the robot can be treated as a single point
    * search - Dijkstra (no heuristic) or A* (Euclidean or Manhattan
      heuristic) on an 8-connected grid with diagonal cost sqrt(2)
    * simplify_path - line-of-sight shortcutting that turns the cell path
      into a few waypoints for the PID path follower (Questions 3 and 4)
"""

import heapq
import math
import time

import numpy as np
from scipy.ndimage import binary_dilation

FREE, BLOCKED = 0, 1

# 8-connected moves: (d_row, d_col, cost in cells)
MOVES = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
         (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)),
         (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2))]


def load_map(path):
    data = np.load(path)
    return {"fused": data["fused"], "x_min": float(data["x_min"]),
            "y_min": float(data["y_min"]),
            "resolution": float(data["resolution"])}


def build_planning_grid(fused, resolution, robot_radius, margin=0.01):
    """Binary grid: blocked where occupied, a landmark, or unknown.

    Unknown cells are blocked because the robot has no evidence they are
    safe. Obstacles are then inflated by robot_radius + margin, the
    configuration-space step that lets the planner treat the robot as a
    point.
    """
    blocked = fused != 0   # -1 unknown, 1 obstacle, 2-4 landmarks
    cells = int(math.ceil((robot_radius + margin) / resolution))
    yy, xx = np.mgrid[-cells:cells + 1, -cells:cells + 1]
    disk = xx ** 2 + yy ** 2 <= cells ** 2
    inflated = binary_dilation(blocked, structure=disk)
    return inflated.astype(int), cells


def world_to_cell(x, y, m):
    return (int(math.floor((y - m["y_min"]) / m["resolution"])),
            int(math.floor((x - m["x_min"]) / m["resolution"])))


def cell_to_world(row, col, m):
    return (m["x_min"] + (col + 0.5) * m["resolution"],
            m["y_min"] + (row + 0.5) * m["resolution"])


def euclidean(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def search(grid, start, goal, heuristic=None):
    """Dijkstra if heuristic is None, otherwise A*.

    Returns dict with path (list of cells), cost (cells), nodes_expanded,
    and the set of expanded cells (for visualisation).
    """
    rows, cols = grid.shape
    h = (lambda c: heuristic(c, goal)) if heuristic else (lambda c: 0.0)

    frontier = [(h(start), 0.0, start)]
    g_cost = {start: 0.0}
    parent = {}
    expanded = set()

    while frontier:
        _, cost, current = heapq.heappop(frontier)
        if current in expanded:
            continue        # stale queue entry
        expanded.add(current)
        if current == goal:
            break

        r, c = current
        for dr, dc, step in MOVES:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if grid[nr, nc] == BLOCKED:
                continue
            # No corner cutting: a diagonal move needs both side cells free.
            if dr and dc and (grid[r + dr, c] or grid[r, c + dc]):
                continue
            new_cost = cost + step
            if new_cost < g_cost.get((nr, nc), float("inf")):
                g_cost[(nr, nc)] = new_cost
                parent[(nr, nc)] = current
                heapq.heappush(frontier,
                               (new_cost + h((nr, nc)), new_cost, (nr, nc)))

    if goal not in expanded:
        return {"path": [], "cost": float("inf"),
                "nodes_expanded": len(expanded), "expanded": expanded}

    path = [goal]
    while path[-1] != start:
        path.append(parent[path[-1]])
    path.reverse()
    return {"path": path, "cost": g_cost[goal],
            "nodes_expanded": len(expanded), "expanded": expanded}


def time_search(grid, start, goal, heuristic=None, runs=20):
    """Average wall-clock time (ms) over several runs."""
    t0 = time.perf_counter()
    for _ in range(runs):
        search(grid, start, goal, heuristic)
    return 1000 * (time.perf_counter() - t0) / runs


def line_is_free(grid, a, b):
    """True if the straight segment between cells a and b is collision free
    (sampled at quarter-cell steps)."""
    n = int(max(abs(b[0] - a[0]), abs(b[1] - a[1])) * 4) + 1
    for t in np.linspace(0, 1, n):
        r = int(round(a[0] + t * (b[0] - a[0])))
        c = int(round(a[1] + t * (b[1] - a[1])))
        if grid[r, c] == BLOCKED:
            return False
    return True


def simplify_path(grid, path):
    """Greedy line-of-sight shortcutting: keep only the waypoints needed so
    that each straight segment stays in free space."""
    if len(path) < 3:
        return list(path)
    waypoints = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not line_is_free(grid, path[i], path[j]):
            j -= 1
        waypoints.append(path[j])
        i = j
    return waypoints


def path_length_m(cells, m):
    pts = [cell_to_world(r, c, m) for r, c in cells]
    return sum(math.dist(pts[k], pts[k + 1]) for k in range(len(pts) - 1))
