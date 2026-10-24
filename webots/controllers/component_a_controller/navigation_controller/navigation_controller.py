"""
ASR700 Component A, Question 4: closed-loop autonomous navigation.

Integrates the three earlier stages in one control loop on the e-puck:

    perception (Q1)  ->  fused occupancy map built from Webots LiDAR + camera
    planning   (Q2)  ->  A* route on that map, simplified to waypoints
                         (component_a/question2/outputs/planned_path.json)
    control    (Q3)  ->  PID on heading error drives the robot through the
                         waypoints; the LiDAR runs live as a safety monitor

Each run tests three PID gain settings in sequence. Between trials the
supervisor puts the robot back at the start, so all three are compared on
the same route. The results are logged to webots/data/navigation/ for the
analysis notebook in component_a/question4.

Localisation: the controller uses the robot's pose from the Webots
supervisor (ideal localisation). Wheel-encoder odometry is computed and
logged alongside it, so the notebook can show how much drift a robot that
relied on odometry alone would have had.

Uses only the Python standard library.
"""

import csv
import json
import math
import os

from controller import Supervisor

# ------------------------------------------------------------- parameters
WHEEL_RADIUS = 0.02        # m (e-puck)
AXLE_LENGTH = 0.052        # m, distance between the wheels
MAX_WHEEL_SPEED = 6.28     # rad/s
V_MAX = 0.10               # m/s forward speed limit (~80% of the e-puck max)
OMEGA_MAX = 3.0            # rad/s turn-rate limit
SLOW_RADIUS = 0.10         # m: slow down within this distance of the goal
WAYPOINT_TOLERANCE = 0.04  # m: switch to the next waypoint inside this
GOAL_TOLERANCE = 0.02      # m: stop when this close to the final goal
EMERGENCY_DISTANCE = 0.055 # m: LiDAR safety stop (sensor min range = 0.05)
TRIAL_TIMEOUT = 60.0       # s of simulated time per trial

# Gain sets compared. "Tuned" is the setting chosen in Question 3.
GAIN_SETS = [
    {"name": "low_kp", "kp": 0.8, "ki": 0.0, "kd": 0.0},
    {"name": "tuned", "kp": 2.0, "ki": 0.0, "kd": 0.1},
    {"name": "high_kp", "kp": 6.0, "ki": 0.0, "kd": 0.0},
]

HERE = os.path.dirname(os.path.abspath(__file__))
PLAN_FILE = os.path.join(HERE, "..", "..", "..", "component_a", "question2",
                         "outputs", "planned_path.json")
OUT_DIR = os.path.join(HERE, "..", "..", "data", "navigation")


class PID:
    """Discrete PID controller (same structure as Question 3)."""

    def __init__(self, kp, ki, kd, integral_limit=1.0):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.integral_limit = integral_limit
        self.reset()

    def reset(self):
        self.integral = 0.0
        self.previous_error = None

    def update(self, error, dt):
        self.integral += error * dt
        # Anti-windup: bound the integral so a long turn cannot saturate it.
        self.integral = max(-self.integral_limit,
                            min(self.integral_limit, self.integral))
        derivative = (0.0 if self.previous_error is None
                      else (error - self.previous_error) / dt)
        self.previous_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative


def wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


# ------------------------------------------------------------------ setup
robot = Supervisor()
timestep = int(robot.getBasicTimeStep())
dt = timestep / 1000.0

left_motor = robot.getDevice("left wheel motor")
right_motor = robot.getDevice("right wheel motor")
for motor in (left_motor, right_motor):
    motor.setPosition(float("inf"))
    motor.setVelocity(0.0)

left_encoder = robot.getDevice("left wheel sensor")
right_encoder = robot.getDevice("right wheel sensor")
left_encoder.enable(timestep)
right_encoder.enable(timestep)

lidar = robot.getDevice("lidar")
lidar.enable(timestep)
N_RAYS = lidar.getHorizontalResolution()
FOV = lidar.getFov()
FRONT_RAYS = [i for i in range(N_RAYS)
              if abs(FOV / 2 - i * FOV / N_RAYS) <= math.radians(30)]

self_node = robot.getSelf()
translation_field = self_node.getField("translation")
rotation_field = self_node.getField("rotation")
start_translation = translation_field.getSFVec3f()
start_rotation = rotation_field.getSFRotation()

with open(PLAN_FILE) as f:
    plan = json.load(f)
waypoints = [tuple(p) for p in plan["waypoints"]][1:]   # skip the start
os.makedirs(OUT_DIR, exist_ok=True)
print(f"[nav] {len(waypoints)} waypoints loaded from planned_path.json")


def true_pose():
    x, y, _ = self_node.getPosition()
    m = self_node.getOrientation()
    return x, y, math.atan2(m[3], m[0])


def drive(v, omega):
    """Convert (v, omega) to wheel speeds, scaling both down together if
    either wheel would exceed its limit (keeps the curvature)."""
    left = (v - omega * AXLE_LENGTH / 2) / WHEEL_RADIUS
    right = (v + omega * AXLE_LENGTH / 2) / WHEEL_RADIUS
    scale = max(1.0, abs(left) / MAX_WHEEL_SPEED,
                abs(right) / MAX_WHEEL_SPEED)
    left_motor.setVelocity(left / scale)
    right_motor.setVelocity(right / scale)


def reset_robot():
    drive(0.0, 0.0)
    translation_field.setSFVec3f(start_translation)
    rotation_field.setSFRotation(start_rotation)
    self_node.resetPhysics()
    for _ in range(10):              # let the robot settle
        robot.step(timestep)


# ------------------------------------------------------------- the trials
summary = []
for gains in GAIN_SETS:
    reset_robot()
    pid = PID(gains["kp"], gains["ki"], gains["kd"])
    x, y, yaw = true_pose()
    odom = [x, y, yaw]                          # odometry starts aligned
    enc_prev = (left_encoder.getValue(), right_encoder.getValue())
    t0 = robot.getTime()
    target = 0
    emergency_stops = 0
    min_front = float("inf")
    reached = False

    log_path = os.path.join(OUT_DIR, f"trial_{gains['name']}.csv")
    log_file = open(log_path, "w", newline="")
    log = csv.writer(log_file)
    log.writerow(["t", "x", "y", "yaw", "odom_x", "odom_y", "odom_yaw",
                  "target", "heading_error", "distance_to_target", "v",
                  "omega", "front_min"])
    print(f"[nav] trial '{gains['name']}' Kp={gains['kp']} "
          f"Ki={gains['ki']} Kd={gains['kd']}")

    while robot.step(timestep) != -1:
        t = robot.getTime() - t0
        x, y, yaw = true_pose()

        # Wheel odometry (dead reckoning from encoder increments).
        enc = (left_encoder.getValue(), right_encoder.getValue())
        dl = (enc[0] - enc_prev[0]) * WHEEL_RADIUS
        dr = (enc[1] - enc_prev[1]) * WHEEL_RADIUS
        enc_prev = enc
        ds, dth = (dl + dr) / 2, (dr - dl) / AXLE_LENGTH
        odom[0] += ds * math.cos(odom[2] + dth / 2)
        odom[1] += ds * math.sin(odom[2] + dth / 2)
        odom[2] = wrap(odom[2] + dth)

        # Waypoint management.
        tx, ty = waypoints[target]
        distance = math.hypot(tx - x, ty - y)
        is_last = target == len(waypoints) - 1
        if is_last and distance < GOAL_TOLERANCE:
            reached = True
            break
        if not is_last and distance < WAYPOINT_TOLERANCE:
            target += 1
            pid.reset()             # avoid a derivative kick at the switch
            tx, ty = waypoints[target]
            distance = math.hypot(tx - x, ty - y)
        if t > TRIAL_TIMEOUT:
            break

        # Heading PID: the controlled variable is the robot's heading.
        heading_error = wrap(math.atan2(ty - y, tx - x) - yaw)
        omega = max(-OMEGA_MAX, min(OMEGA_MAX, pid.update(heading_error, dt)))

        # Forward speed: only drive when roughly facing the target, and
        # slow down on the approach to the final goal.
        v = V_MAX * max(0.0, math.cos(heading_error)) ** 2
        if is_last:
            v *= min(1.0, distance / SLOW_RADIUS)

        # LiDAR safety monitor (live perception in the loop).
        ranges = lidar.getRangeImage()
        front = min((ranges[i] for i in FRONT_RAYS if math.isfinite(ranges[i])),
                    default=float("inf"))
        min_front = min(min_front, front)
        if front < EMERGENCY_DISTANCE and v > 0:
            v = 0.0
            emergency_stops += 1

        drive(v, omega)
        log.writerow([round(t, 3), round(x, 5), round(y, 5), round(yaw, 5),
                      round(odom[0], 5), round(odom[1], 5), round(odom[2], 5),
                      target, round(heading_error, 5), round(distance, 5),
                      round(v, 5), round(omega, 5),
                      round(front, 4) if math.isfinite(front) else "inf"])

    drive(0.0, 0.0)
    log_file.close()
    result = {**gains, "reached_goal": reached,
              "time_to_goal_s": round(robot.getTime() - t0, 3),
              "final_error_m": round(math.hypot(waypoints[-1][0] - x,
                                                waypoints[-1][1] - y), 4),
              "odometry_drift_m": round(math.hypot(odom[0] - x,
                                                   odom[1] - y), 4),
              "emergency_stop_steps": emergency_stops,
              "min_front_clearance_m": round(min_front, 4)
              if math.isfinite(min_front) else None}
    summary.append(result)
    print(f"[nav]   reached={reached} time={result['time_to_goal_s']} s "
          f"final_error={result['final_error_m']} m")

with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
    json.dump({"plan": plan, "trials": summary}, f, indent=2)
print("[nav] done: results saved to webots/data/navigation")
robot.simulationSetMode(Supervisor.SIMULATION_MODE_PAUSE)
