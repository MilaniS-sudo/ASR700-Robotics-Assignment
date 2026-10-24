"""
ASR700 Component A - Webots data-collection controller (e-puck).

The robot explores the arena with simple LiDAR-based obstacle avoidance
and logs synchronised sensor data for the perception pipeline (Q1):

    data/lidar_scans.csv   one row per sample: time, true pose, 360 ranges
    data/images/frame_XXXX.png   camera image taken at the same moment
    data/ground_truth.json       true obstacle/landmark positions (for
                                 evaluating perception accuracy later)

The robot is a Supervisor only so it can read its true pose and the
true world layout. That ground truth is used for logging and evaluation,
NOT for steering, which uses the LiDAR alone.

Uses only the Python standard library, so no extra packages are needed
inside Webots.
"""

import csv
import json
import math
import os
import random
import shutil

from controller import Supervisor

# ---------------------------------------------------------------- settings
RUN_SECONDS = 90.0        # length of the data-collection run (sim time)
LOG_EVERY_SECONDS = 0.5   # sampling period for scans and images
MAX_SPEED = 6.28          # e-puck wheel limit (rad/s)
CRUISE = 0.6 * MAX_SPEED
SAFE_DISTANCE = 0.18      # metres: turn away below this in front sector
RANDOM_SEED = 42          # fixed seed so every run is reproducible

DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data")
)
IMAGE_DIR = os.path.join(DATA_DIR, "images")

# ------------------------------------------------------------------ setup
robot = Supervisor()
timestep = int(robot.getBasicTimeStep())
random.seed(RANDOM_SEED)

left_motor = robot.getDevice("left wheel motor")
right_motor = robot.getDevice("right wheel motor")
for motor in (left_motor, right_motor):
    motor.setPosition(float("inf"))   # velocity control mode
    motor.setVelocity(0.0)

camera = robot.getDevice("camera")
camera.enable(timestep)

lidar = robot.getDevice("lidar")
lidar.enable(timestep)

N_RAYS = lidar.getHorizontalResolution()
FOV = lidar.getFov()
MAX_RANGE = lidar.getMaxRange()


def ray_angle(i):
    """Angle of ray i in the robot frame (0 = straight ahead, +ve = left).

    Webots orders lidar rays from left to right, starting at +FOV/2.
    """
    return FOV / 2.0 - i * FOV / N_RAYS


def sector_min(ranges, start_deg, end_deg):
    """Minimum valid range for rays whose angle lies in [start, end] deg."""
    values = [
        r for i, r in enumerate(ranges)
        if start_deg <= math.degrees(ray_angle(i)) <= end_deg
        and math.isfinite(r)
    ]
    return min(values) if values else MAX_RANGE


def true_pose(node):
    """Return (x, y, yaw) of a node from the supervisor."""
    x, y, _ = node.getPosition()
    m = node.getOrientation()            # 3x3 rotation matrix, row-major
    yaw = math.atan2(m[3], m[0])
    return x, y, yaw


def export_ground_truth():
    """Write true obstacle and landmark geometry to ground_truth.json."""
    root = robot.getRoot().getField("children")
    obstacles, landmarks = [], []
    arena = None

    for k in range(root.getCount()):
        node = root.getMFNode(k)
        type_name = node.getTypeName()
        name_field = node.getField("name")
        name = name_field.getSFString() if name_field else ""

        if name.startswith("obstacle_"):
            x, y, _ = node.getField("translation").getSFVec3f()
            box = node.getField("boundingObject").getSFNode()
            sx, sy, sz = box.getField("size").getSFVec3f()
            obstacles.append({"name": name, "x": x, "y": y,
                              "size_x": sx, "size_y": sy})
        elif name.startswith("landmark_"):
            x, y, _ = node.getField("translation").getSFVec3f()
            landmarks.append({"name": name, "colour": name.split("_", 1)[1],
                              "x": x, "y": y, "radius": 0.04})
        elif type_name == "RectangleArena":
            fx, fy = node.getField("floorSize").getSFVec2f()
            arena = {"size_x": fx, "size_y": fy}

    with open(os.path.join(DATA_DIR, "ground_truth.json"), "w") as f:
        json.dump({"arena": arena, "obstacles": obstacles,
                   "landmarks": landmarks,
                   "lidar": {"rays": N_RAYS, "fov_rad": FOV,
                             "max_range_m": MAX_RANGE},
                   "camera": {"width": camera.getWidth(),
                              "height": camera.getHeight(),
                              "fov_rad": camera.getFov()}},
                  f, indent=2)


# --------------------------------------------- prepare a clean data folder
if os.path.isdir(IMAGE_DIR):
    shutil.rmtree(IMAGE_DIR)
os.makedirs(IMAGE_DIR, exist_ok=True)
export_ground_truth()

csv_file = open(os.path.join(DATA_DIR, "lidar_scans.csv"), "w", newline="")
writer = csv.writer(csv_file)
writer.writerow(["sample", "time_s", "x", "y", "yaw", "image"]
                + [f"r{i}" for i in range(N_RAYS)])

self_node = robot.getSelf()
steps_per_log = max(1, int(round(LOG_EVERY_SECONDS * 1000 / timestep)))
sample = 0
step = 0
turn_direction = 0      # 0 = not turning, +1 = left, -1 = right

print(f"[controller] logging to {DATA_DIR}")
print(f"[controller] lidar: {N_RAYS} rays, FOV {math.degrees(FOV):.0f} deg")

# -------------------------------------------------------------- main loop
while robot.step(timestep) != -1:
    t = robot.getTime()
    if t >= RUN_SECONDS:
        break

    ranges = lidar.getRangeImage()

    # --- reactive obstacle avoidance using LiDAR sectors
    front = sector_min(ranges, -25, 25)
    left = sector_min(ranges, 25, 80)
    right = sector_min(ranges, -80, -25)

    if front < SAFE_DISTANCE:
        # Commit to one turn direction until the way ahead is clear,
        # which avoids dithering in corners.
        if turn_direction == 0:
            turn_direction = 1 if left > right else -1
        left_speed = -turn_direction * 0.5 * MAX_SPEED
        right_speed = turn_direction * 0.5 * MAX_SPEED
    else:
        turn_direction = 0
        # Gentle steering away from the closer side, plus small random
        # wander so the robot does not repeat the same loop.
        bias = (left - right) * 2.0 + random.uniform(-0.3, 0.3)
        bias = max(-1.5, min(1.5, bias))
        left_speed = CRUISE - bias
        right_speed = CRUISE + bias

    left_motor.setVelocity(max(-MAX_SPEED, min(MAX_SPEED, left_speed)))
    right_motor.setVelocity(max(-MAX_SPEED, min(MAX_SPEED, right_speed)))

    # --- log a synchronised sample
    if step % steps_per_log == 0:
        x, y, yaw = true_pose(self_node)
        image_name = f"frame_{sample:04d}.png"
        camera.saveImage(os.path.join(IMAGE_DIR, image_name), 100)
        writer.writerow([sample, round(t, 3), round(x, 4), round(y, 4),
                         round(yaw, 4), image_name]
                        + [round(r, 4) if math.isfinite(r) else "inf"
                           for r in ranges])
        sample += 1
    step += 1

# ------------------------------------------------------------------ finish
left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)
csv_file.close()
print(f"[controller] done: {sample} samples saved to {DATA_DIR}")
robot.simulationSetMode(Supervisor.SIMULATION_MODE_PAUSE)
