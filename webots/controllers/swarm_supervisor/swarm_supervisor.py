"""
ASR700 Component B: supervisor for the swarm experiments.

Roles:
    * Experiment manager: resets the four robots to their start poses and
      runs each experiment in turn, three times with different random seeds
      (one press of Play runs them all, about 21 minutes of simulated time).
    * Central coordinator (centralised experiments only): collects reports
      from the robots and broadcasts the global answer back (the rendezvous
      point, then the global best position for PSO).
    * Fault injector: makes robot_2 fail, or makes the coordinator itself
      fail, partway through the PSO search.
    * Observer: logs every robot's true position and its clearance to
      walls, obstacles and other robots, so collisions can be counted.

If webots/data/swarm/nn_policy.json exists (written by the Question 3
notebook), only the neural-network experiment is run, so it can be compared
with the earlier rule-based run.

Uses only the Python standard library.
"""

import csv
import json
import math
import os

from controller import Supervisor

N_ROBOTS = 4
ROBOT_RADIUS = 0.037
CONSENSUS_TIME = 25 * 0.25
FORMATION_TIME = 20.0
PSO_ITER_TIME = 4.0
PSO_ITERATIONS = 20
EXPERIMENT_TIME = CONSENSUS_TIME + FORMATION_TIME + \
    PSO_ITERATIONS * PSO_ITER_TIME + 1.0
FAULT_ITERATION = 5               # PSO iteration at which faults happen
PSO_START = CONSENSUS_TIME + FORMATION_TIME

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "..", "data", "swarm")
NN_FILE = os.path.join(DATA_DIR, "nn_policy.json")

SEEDS = [1, 2, 3]          # each experiment is repeated with 3 seeds
KINDS = [
    {"kind": "distributed", "mode": "distributed", "policy": "expert",
     "fault": None},
    {"kind": "centralised", "mode": "centralised", "policy": "expert",
     "fault": None},
    {"kind": "distributed_robot_failure", "mode": "distributed",
     "policy": "expert", "fault": "robot"},
    {"kind": "centralised_coordinator_failure", "mode": "centralised",
     "policy": "expert", "fault": "coordinator"},
]
NN_KINDS = [
    {"kind": "distributed_nn", "mode": "distributed", "policy": "nn",
     "fault": None},
]


def expand(kinds):
    return [{**k, "seed": seed, "name": f"{k['kind']}_s{seed}"}
            for seed in SEEDS for k in kinds]


BASE_EXPERIMENTS = expand(KINDS)
NN_EXPERIMENTS = expand(NN_KINDS)

sup = Supervisor()
timestep = int(sup.getBasicTimeStep())
emitter = sup.getDevice("emitter")
receiver = sup.getDevice("receiver")
receiver.enable(timestep)
os.makedirs(DATA_DIR, exist_ok=True)

robots = [sup.getFromDef(f"ROBOT_{i}") for i in range(N_ROBOTS)]
starts = [r.getField("translation").getSFVec3f() for r in robots]


def load_obstacles():
    root = sup.getRoot().getField("children")
    boxes = []
    for k in range(root.getCount()):
        node = root.getMFNode(k)
        f = node.getField("name")
        if f and f.getSFString().startswith("obstacle_"):
            x, y, _ = node.getField("translation").getSFVec3f()
            sx, sy, _ = node.getField("boundingObject").getSFNode() \
                .getField("size").getSFVec3f()
            boxes.append((x, y, sx, sy))
    return boxes


OBSTACLES = load_obstacles()


def clearance(i, positions):
    """Gap between robot i's body and the nearest wall, obstacle or robot."""
    x, y = positions[i]
    gaps = [1.0 - abs(x), 1.0 - abs(y)]
    for ox, oy, sx, sy in OBSTACLES:
        gaps.append(math.hypot(max(abs(x - ox) - sx / 2, 0),
                               max(abs(y - oy) - sy / 2, 0)))
    for j, (xj, yj) in enumerate(positions):
        if j != i:
            gaps.append(math.hypot(x - xj, y - yj) - ROBOT_RADIUS)
    return min(gaps) - ROBOT_RADIUS


def send(msg):
    msg["src"] = -1
    emitter.send(json.dumps(msg).encode("utf-8"))


def receive_all():
    msgs = []
    while receiver.getQueueLength() > 0:
        try:
            data = receiver.getString()
        except AttributeError:
            data = receiver.getData().decode("utf-8")
        receiver.nextPacket()
        try:
            msgs.append(json.loads(data))
        except ValueError:
            pass
    return msgs


def reset_robots():
    send({"type": "stop"})
    for r, s in zip(robots, starts):
        r.getField("translation").setSFVec3f(s)
        r.getField("rotation").setSFRotation([0, 0, 1, 0])
        r.resetPhysics()
    for _ in range(20):
        sup.step(timestep)
    receive_all()                       # drop stale messages


experiments = NN_EXPERIMENTS if os.path.exists(NN_FILE) else BASE_EXPERIMENTS
summary_path = os.path.join(DATA_DIR, "experiments.json")
summary = {}
if os.path.exists(summary_path):
    with open(summary_path) as f:
        summary = json.load(f)
print(f"[supervisor] running: {[e['name'] for e in experiments]}")

for cfg in experiments:
    reset_robots()
    send({"type": "start", "config": cfg})
    t0 = sup.getTime()
    coordinator_alive = True
    fault_sent = False
    positions_reported = {}
    global_best = (None, -1.0)
    coordinator_messages = 0
    in_contact = [False] * N_ROBOTS
    collisions = [0] * N_ROBOTS
    min_clear = [float("inf")] * N_ROBOTS

    log_file = open(os.path.join(DATA_DIR, f"{cfg['name']}_supervisor.csv"),
                    "w", newline="")
    log = csv.writer(log_file)
    log.writerow(["t"] + [f"{a}{i}" for i in range(N_ROBOTS)
                          for a in ("x", "y", "clear")])
    print(f"[supervisor] experiment '{cfg['name']}'")
    step = 0

    while sup.step(timestep) != -1:
        t = sup.getTime() - t0
        if t > EXPERIMENT_TIME:
            break
        pso_iter = int((t - PSO_START) // PSO_ITER_TIME) if t >= PSO_START \
            else -1

        # Fault injection.
        if cfg["fault"] and not fault_sent and pso_iter >= FAULT_ITERATION:
            fault_sent = True
            if cfg["fault"] == "robot":
                send({"type": "fail", "robot": 2})
                print("[supervisor]   robot_2 failed")
            else:
                coordinator_alive = False
                print("[supervisor]   coordinator failed")

        # Central coordinator.
        msgs = receive_all()
        if cfg["mode"] == "centralised" and coordinator_alive:
            for m in msgs:
                if m.get("type") == "position":
                    positions_reported[m["src"]] = m["value"]
                    if len(positions_reported) == N_ROBOTS:
                        centre = [sum(p[k] for p in positions_reported.values())
                                  / N_ROBOTS for k in (0, 1)]
                        send({"type": "centre", "value": centre})
                        coordinator_messages += 1
                elif m.get("type") == "pbest" and m["fit"] > global_best[1]:
                    global_best = (m["pos"], m["fit"])
                    send({"type": "gbest", "pos": m["pos"], "fit": m["fit"]})
                    coordinator_messages += 1

        # Ground-truth observation.
        positions = [r.getPosition()[:2] for r in robots]
        row = [round(t, 3)]
        for i in range(N_ROBOTS):
            c = clearance(i, positions)
            min_clear[i] = min(min_clear[i], c)
            if c < 0 and not in_contact[i]:
                collisions[i] += 1
            in_contact[i] = c < 0
            row += [round(positions[i][0], 4), round(positions[i][1], 4),
                    round(c, 4)]
        if step % 4 == 0:
            log.writerow(row)
        step += 1

    log_file.close()
    summary[cfg["name"]] = {**cfg, "coordinator_messages": coordinator_messages,
                            "collisions": collisions,
                            "min_clearance_m": [round(c, 4) for c in min_clear]}
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[supervisor]   collisions={collisions}")

send({"type": "stop"})
for _ in range(5):
    sup.step(timestep)
print("[supervisor] done: results saved to webots/data/swarm")
sup.simulationSetMode(Supervisor.SIMULATION_MODE_PAUSE)
sup.step(timestep)
