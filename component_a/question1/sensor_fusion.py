"""
LiDAR-camera sensor fusion for Component A (Question 1).

The two sensors are complementary:
    * The camera identifies WHAT an object is (its colour) and its bearing,
      but a single camera cannot measure distance.
    * The LiDAR measures WHERE surfaces are (range and bearing) very
      accurately, but cannot tell a landmark from a wall or an obstacle.

Fusion method (feature-level, bearing association):
    For each camera detection, take the LiDAR rays within a small angular
    window around the detection's bearing and use the nearest return as the
    landmark's range. Range from the LiDAR plus bearing and identity from
    the camera give a labelled landmark position in the world. Estimates
    from many frames are combined with a median, which is robust to the
    occasional wrong association (for example, when another object partly
    occludes the landmark).

The fused result is a single semantic occupancy grid:
    -1 unknown, 0 free, 1 obstacle/wall, 2 red, 3 green, 4 blue landmark
which is the unified environment representation passed to the planner.
"""

import math

import numpy as np

LANDMARK_CODES = {"red": 2, "green": 3, "blue": 4}
UNKNOWN, FREE, OCCUPIED = -1, 0, 1


def fuse_detection(detection, ranges, angles, pose, landmark_radius,
                   window_deg=2.0):
    """Turn one camera detection plus the matching LiDAR scan into a world
    position for the landmark centre. Returns (x, y) or None."""
    if detection["touches_border"]:
        return None   # centroid of a clipped blob gives a biased bearing

    bearing = detection["bearing"]
    # Angular difference wrapped to [-pi, pi].
    diff = np.angle(np.exp(1j * (angles - bearing)))
    in_window = np.abs(diff) <= math.radians(window_deg)
    candidates = ranges[in_window]
    candidates = candidates[np.isfinite(candidates)]
    if candidates.size == 0:
        return None

    # The LiDAR hits the near surface of the cylinder, so add the radius
    # to reach its centre.
    distance = float(np.min(candidates)) + landmark_radius
    x, y, yaw = pose
    return (x + distance * math.cos(yaw + bearing),
            y + distance * math.sin(yaw + bearing))


def aggregate_landmarks(estimates):
    """Combine per-frame estimates {colour: [(x, y), ...]} with a median."""
    result = {}
    for colour, points in estimates.items():
        if not points:
            continue
        pts = np.array(points)
        centre = np.median(pts, axis=0)
        spread = np.median(np.linalg.norm(pts - centre, axis=1))
        result[colour] = {"x": float(centre[0]), "y": float(centre[1]),
                          "n_frames": len(points),
                          "median_spread_m": float(spread)}
    return result


def build_fused_map(occupancy_grid, classified, landmarks, landmark_radius):
    """Label occupied cells near each fused landmark with its colour code."""
    fused = classified.copy()
    reach = landmark_radius + 1.5 * occupancy_grid.res
    for colour, lm in landmarks.items():
        for row in range(occupancy_grid.rows):
            for col in range(occupancy_grid.cols):
                if classified[row, col] != OCCUPIED:
                    continue
                wx, wy = occupancy_grid.cell_to_world(row, col)
                if math.hypot(wx - lm["x"], wy - lm["y"]) <= reach:
                    fused[row, col] = LANDMARK_CODES[colour]
    return fused
