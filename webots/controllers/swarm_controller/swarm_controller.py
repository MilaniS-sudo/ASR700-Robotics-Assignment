"""
ASR700 Component B: controller run by every e-puck in the swarm.

Cooperative task: four robots search an arena for a hidden signal source
(think of a gas leak). Each robot carries a simulated concentration sensor
and can only measure the signal where it currently is.

Each experiment has three phases (started by the supervisor):
    1. CONSENSUS  - the robots agree on a rendezvous point (the average of
                    their start positions) using average consensus.
    2. FORMATION  - each drives to its slot on a circle around that point.
    3. PSO SEARCH - Particle Swarm Optimisation: every robot is a particle
                    whose fitness is the signal it measures. Robots share
                    their best-found positions and move towards the best.

Coordination modes (set per experiment by the supervisor):
    distributed  - robots talk only to their two ring neighbours
                   (0-1-2-3-0). No robot has global information.
    centralised  - the supervisor acts as a coordinator: robots report to
                   it and it broadcasts the global answer back.

Low-level navigation picks one of three discrete actions each step
(FORWARD, LEFT, RIGHT), from either a hand-written expert rule or a neural
network trained to imitate it (Question 3).

Uses only the Python standard library.
"""

import csv
import json
import math
import os
import random

from controller import Robot

# ----------------------------------------------------------------- settings
N_ROBOTS = 4
NEIGHBOURS = {i: [(i - 1) % N_ROBOTS, (i + 1) % N_ROBOTS]
              for i in range(N_ROBOTS)}              # ring topology

CONSENSUS_ROUNDS = 25
ROUND_TIME = 0.25            # s per consensus round
EPSILON = 0.3                # consensus step size (< 1 / max degree = 0.5)
FORMATION_TIME = 20.0        # s allowed to reach formation slots
FORMATION_RADIUS = 0.25      # m
PSO_ITERATIONS = 20
PSO_ITER_TIME = 4.0          # s per PSO iteration
# Inertia weight falls linearly from W_START (explore) to W_END (settle), the
# standard remedy for particles oscillating around the best point instead
# of converging (Shi and Eberhart, 1998).
PSO_W_START, PSO_W_END = 0.9, 0.4
PSO_C1, PSO_C2 = 1.5, 1.5
PSO_VMAX = 0.4              # m: largest move per iteration
ARENA_HALF = 1.0
SAFE_MARGIN = 0.12           # keep PSO targets this far inside the walls

SOURCE = (0.55, 0.5)         # hidden from the swarm's logic: sensor only
SIGNAL_WIDTH = 0.5           # m (Gaussian plume width)
SENSOR_NOISE = 0.01

ACTIONS = ["FORWARD", "LEFT", "RIGHT"]
ESCAPE_STEPS = 31            # ~0.5 s forward after clearing an obstacle
# Turns are gentle arcs (some forward speed) rather than spins on the spot,
# so a robot slides around a blocker instead of oscillating in front of it.
ACTION_CMD = {"FORWARD": (0.09, 0.0), "LEFT": (0.03, 1.8),
              "RIGHT": (0.03, -1.8)}          # (v m/s, omega rad/s)
WHEEL_RADIUS, AXLE = 0.02, 0.052
MAX_WHEEL = 6.27

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "..", "data", "swarm")
NN_FILE = os.path.join(DATA_DIR, "nn_policy.json")

# ------------------------------------------------------------------- setup
robot = Robot()
timestep = int(robot.getBasicTimeStep())
NAME = robot.getName()
ID = int(NAME.split("_")[1])

left_motor = robot.getDevice("left wheel motor")
right_motor = robot.getDevice("right wheel motor")
for m in (left_motor, right_motor):
    m.setPosition(float("inf"))
    m.setVelocity(0.0)
gps = robot.getDevice("gps")
gps.enable(timestep)
imu = robot.getDevice("imu")
imu.enable(timestep)
lidar = robot.getDevice("lidar")
lidar.enable(timestep)
emitter = robot.getDevice("emitter")
receiver = robot.getDevice("receiver")
receiver.enable(timestep)

N_RAYS = lidar.getHorizontalResolution()
FOV = lidar.getFov()
os.makedirs(DATA_DIR, exist_ok=True)


# ---------------------------------------------------------------- helpers
def pose():
    x, y, _ = gps.getValues()
    return x, y, imu.getRollPitchYaw()[2]


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def measure_signal(x, y, rng):
    """Simulated concentration sensor: Gaussian plume plus noise."""
    d2 = (x - SOURCE[0]) ** 2 + (y - SOURCE[1]) ** 2
    return math.exp(-d2 / (2 * SIGNAL_WIDTH ** 2)) + rng.gauss(0, SENSOR_NOISE)


def lidar_sectors():
    """Minimum range in 8 sectors of 45 deg (0 = ahead, then
    anticlockwise: front-left, left, ..., front-right), capped at 1 m."""
    ranges = lidar.getRangeImage()
    sectors = [1.0] * 8
    for i, r in enumerate(ranges):
        angle = math.degrees(FOV / 2 - i * FOV / N_RAYS) % 360
        s = int(((angle + 22.5) % 360) // 45)
        if math.isfinite(r):
            sectors[s] = min(sectors[s], r)
    return sectors


def expert_action(sectors, bearing):
    """Hand-written navigation rule (the 'teacher' for the network).

    Obstacle avoidance has priority over goal seeking: if the way ahead is
    blocked, turn towards the clearly more open side. If neither side is
    clearly more open, always turn LEFT: a fixed convention (like a traffic
    rule) that stops two robots facing each other from mirroring each other's
    choice and deadlocking. Otherwise turn towards the goal until roughly
    facing it, then drive forward."""
    if is_blocked(sectors):
        right_space = sectors[7] + sectors[6]
        left_space = sectors[1] + sectors[2]
        return "RIGHT" if right_space > left_space + 0.1 else "LEFT"
    if bearing > 0.35:
        return "LEFT"
    if bearing < -0.35:
        return "RIGHT"
    return "FORWARD"


class NeuralPolicy:
    """Forward pass of the MLP trained in Question 3 (weights from JSON),
    written in plain Python so the controller needs no extra packages."""

    def __init__(self, path):
        with open(path) as f:
            d = json.load(f)
        self.mean, self.scale = d["mean"], d["scale"]
        self.layers = list(zip(d["weights"], d["biases"]))
        self.classes = d["classes"]

    def __call__(self, features):
        h = [(v - m) / s for v, m, s in zip(features, self.mean, self.scale)]
        for k, (W, b) in enumerate(self.layers):
            h = [sum(h[i] * W[i][j] for i in range(len(h))) + b[j]
                 for j in range(len(b))]
            if k < len(self.layers) - 1:
                h = [max(0.0, v) for v in h]          # ReLU
        return self.classes[max(range(len(h)), key=h.__getitem__)]


def is_blocked(sectors):
    return sectors[0] < 0.12 or min(sectors[1], sectors[7]) < 0.08


class Actuation:
    """Turns the policy's decision into motion, adding commitment.

    A purely reactive policy (rule or network) decides afresh every step,
    so near another robot it can flip between LEFT and RIGHT forever: turn
    away, see a clear path, turn back to the goal, be blocked again. This
    layer, shared by the rule and the network so the comparison is fair:
      * while blocked: spins on the spot, keeping the first chosen
        direction until the way ahead is clear;
      * after clearing: drives forward for ESCAPE_STEPS before handing
        control back to the policy, so the robot leaves the trap.
    """

    def __init__(self):
        self.avoid_dir = None
        self.escape = 0

    def __call__(self, policy_action, sectors):
        if policy_action is None:
            self.avoid_dir, self.escape = None, 0
            return None, (0.0, 0.0)
        if is_blocked(sectors):
            if self.avoid_dir is None:
                self.avoid_dir = policy_action if policy_action != "FORWARD" \
                    else "LEFT"
            self.escape = ESCAPE_STEPS
            w = ACTION_CMD[self.avoid_dir][1]
            return self.avoid_dir, (0.0, w)               # spin in place
        self.avoid_dir = None
        if self.escape > 0:
            self.escape -= 1
            return "FORWARD", ACTION_CMD["FORWARD"]
        return policy_action, ACTION_CMD[policy_action]


def features(sectors, bearing):
    return sectors + [math.sin(bearing), math.cos(bearing)]


def drive(command):
    v, w = command
    left = (v - w * AXLE / 2) / WHEEL_RADIUS
    right = (v + w * AXLE / 2) / WHEEL_RADIUS
    scale = max(1.0, abs(left) / MAX_WHEEL, abs(right) / MAX_WHEEL)
    left_motor.setVelocity(left / scale)
    right_motor.setVelocity(right / scale)


def send(msg):
    msg["src"] = ID
    emitter.send(json.dumps(msg).encode("utf-8"))
    return 1


def receive_all():
    msgs = []
    while receiver.getQueueLength() > 0:
        try:
            data = receiver.getString()
        except AttributeError:                     # older Webots versions
            data = receiver.getData().decode("utf-8")
        receiver.nextPacket()
        try:
            msgs.append(json.loads(data))
        except ValueError:
            pass
    return msgs


def clamp_to_arena(x, y):
    lim = ARENA_HALF - SAFE_MARGIN
    return max(-lim, min(lim, x)), max(-lim, min(lim, y))


# ------------------------------------------------------------- experiment
class Experiment:
    def __init__(self, cfg, t0):
        self.cfg = cfg
        self.t0 = t0
        self.mode = cfg["mode"]
        self.policy_name = cfg["policy"]
        self.policy = NeuralPolicy(NN_FILE) if cfg["policy"] == "nn" \
            else None
        self.rng = random.Random(1000 * cfg["seed"] + ID)
        self.failed = False
        self.actuation = Actuation()
        self.messages_sent = 0
        x, y, _ = pose()
        # Consensus state: the robot's estimate of the rendezvous point.
        self.state = [x, y]
        self.neighbour_state = {}
        self.round = -1
        self.centre = None
        # PSO state.
        self.velocity = [0.0, 0.0]
        self.pbest, self.pbest_fit = None, -1.0
        self.gbest, self.gbest_fit = None, -1.0
        self.reports = {}                  # latest pbest heard per robot
        self.iteration = -1
        self.updated_this_iter = False
        self.target = None

        path = os.path.join(DATA_DIR, f"{cfg['name']}_robot_{ID}.csv")
        self.log_file = open(path, "w", newline="")
        self.log = csv.writer(self.log_file)
        self.log.writerow(
            ["t", "phase", "x", "y", "yaw", "state_x", "state_y",
             "consensus_round", "pso_iter", "fitness", "pbest_fit",
             "gbest_x", "gbest_y", "gbest_fit", "target_x", "target_y"]
            + [f"s{k}" for k in range(8)]
            + ["bearing", "expert_action", "action", "applied", "failed",
               "messages_sent"])

    # -- phase 1 -------------------------------------------------------
    def consensus_step(self, t, msgs):
        for m in msgs:
            if m.get("type") == "state" and m["src"] in NEIGHBOURS[ID]:
                self.neighbour_state[m["src"]] = m["value"]
            if m.get("type") == "centre":                 # centralised
                self.state = list(m["value"])
                self.centre = list(m["value"])
        r = int(t // ROUND_TIME)
        if r != self.round and r < CONSENSUS_ROUNDS:
            # Close the previous round: x_i += eps * sum_j (x_j - x_i)
            if self.mode == "distributed" and self.round >= 0:
                for k in (0, 1):
                    self.state[k] += EPSILON * sum(
                        v[k] - self.state[k]
                        for v in self.neighbour_state.values())
            self.round = r
            if self.mode == "distributed":
                self.messages_sent += send({"type": "state",
                                            "value": self.state})
            elif r == 0:
                self.messages_sent += send({"type": "position",
                                            "value": self.state})

    # -- phase 3 -------------------------------------------------------
    def pso_step(self, t_pso, msgs, x, y):
        for m in msgs:
            t = m.get("type")
            if t == "pbest" and (self.mode == "distributed"
                                 and m["src"] in NEIGHBOURS[ID]):
                self.reports[m["src"]] = (m["pos"], m["fit"])
            if t == "gbest" and self.mode == "centralised":
                if m["fit"] > self.gbest_fit:
                    self.gbest, self.gbest_fit = m["pos"], m["fit"]

        it = int(t_pso // PSO_ITER_TIME)
        if it >= PSO_ITERATIONS:
            self.target = None
            return
        if it != self.iteration:
            # Start of an iteration: sense, update personal best, share it.
            self.iteration = it
            self.updated_this_iter = False
            fit = measure_signal(x, y, self.rng)
            self.last_fitness = fit
            if fit > self.pbest_fit:
                self.pbest, self.pbest_fit = [x, y], fit
            self.messages_sent += send({"type": "pbest", "pos": self.pbest,
                                        "fit": self.pbest_fit, "iter": it})
        if not self.updated_this_iter and \
                t_pso - it * PSO_ITER_TIME > 0.2:
            # 0.2 s later: neighbours' (or the coordinator's) replies are in.
            if self.mode == "distributed":
                candidates = [(self.pbest, self.pbest_fit)] + \
                    list(self.reports.values())
                best = max(candidates, key=lambda c: c[1])
                if best[1] > self.gbest_fit:
                    self.gbest, self.gbest_fit = best
            elif self.gbest is None or self.pbest_fit > self.gbest_fit:
                self.gbest, self.gbest_fit = self.pbest, self.pbest_fit
            # Standard PSO velocity update.
            r1, r2 = self.rng.random(), self.rng.random()
            w = PSO_W_START - (PSO_W_START - PSO_W_END) * it / \
                max(1, PSO_ITERATIONS - 1)
            for k, pos in ((0, x), (1, y)):
                self.velocity[k] = (w * self.velocity[k]
                                    + PSO_C1 * r1 * (self.pbest[k] - pos)
                                    + PSO_C2 * r2 * (self.gbest[k] - pos))
            speed = math.hypot(*self.velocity)
            if speed > PSO_VMAX:
                self.velocity = [v * PSO_VMAX / speed for v in self.velocity]
            self.target = list(clamp_to_arena(x + self.velocity[0],
                                              y + self.velocity[1]))
            self.updated_this_iter = True

    def close(self):
        self.log_file.close()


exp = None
print(f"[{NAME}] ready")

while robot.step(timestep) != -1:
    msgs = receive_all()
    for m in msgs:
        if m.get("type") == "start" and m.get("src") == -1:
            if exp:
                exp.close()
            exp = Experiment(m["config"], robot.getTime())
        elif m.get("type") == "stop" and exp:
            exp.close()
            exp = None
            drive((0.0, 0.0))
        elif m.get("type") == "fail" and exp and m.get("robot") == ID:
            exp.failed = True
            print(f"[{NAME}] simulated failure")
    if exp is None:
        continue
    if exp.failed:
        drive((0.0, 0.0))   # failed robot: motors off, radio silent
        continue

    t = robot.getTime() - exp.t0
    x, y, yaw = pose()
    t_formation = CONSENSUS_ROUNDS * ROUND_TIME
    t_pso_start = t_formation + FORMATION_TIME
    phase, target = "consensus", None

    if t < t_formation:
        exp.consensus_step(t, msgs)
    elif t < t_pso_start:
        phase = "formation"
        if exp.centre is None:
            exp.centre = list(exp.state)       # agreed rendezvous point
        angle = math.pi / 2 * ID
        target = clamp_to_arena(exp.centre[0] + FORMATION_RADIUS * math.cos(angle),
                                exp.centre[1] + FORMATION_RADIUS * math.sin(angle))
    else:
        phase = "pso"
        exp.pso_step(t - t_pso_start, msgs, x, y)
        target = exp.target

    sectors = lidar_sectors()
    bearing, expert, decision = 0.0, "", None
    if target is not None and math.hypot(target[0] - x,
                                         target[1] - y) > 0.05:
        bearing = wrap(math.atan2(target[1] - y, target[0] - x) - yaw)
        expert = expert_action(sectors, bearing)
        decision = exp.policy(features(sectors, bearing)) if exp.policy \
            else expert
    applied, command = exp.actuation(decision, sectors)
    drive(command)

    exp.log.writerow(
        [round(t, 3), phase, round(x, 4), round(y, 4), round(yaw, 4),
         round(exp.state[0], 5), round(exp.state[1], 5), exp.round,
         exp.iteration, round(getattr(exp, "last_fitness", 0.0), 5),
         round(exp.pbest_fit, 5),
         *(round(v, 4) for v in (exp.gbest or [math.nan, math.nan])),
         round(exp.gbest_fit, 5),
         *(round(v, 4) for v in (target or [math.nan, math.nan])),
         *(round(s, 4) for s in sectors), round(bearing, 4), expert,
         decision or "", applied or "", int(exp.failed),
         exp.messages_sent])
