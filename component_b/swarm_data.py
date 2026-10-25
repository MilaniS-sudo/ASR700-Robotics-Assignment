"""Shared loading helpers for the Component B notebooks."""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "webots" / "data" / "swarm"

# GitHub's web uploader rejects large uploads, so the repository stores the
# swarm logs as webots/data/swarm.zip. Extract it on first use.
_ZIP = DATA_DIR.parent / "swarm.zip"
if not (DATA_DIR / "experiments.json").exists() and _ZIP.exists():
    import zipfile
    with zipfile.ZipFile(_ZIP) as archive:
        archive.extractall(DATA_DIR.parent,
                           [m for m in archive.namelist()
                            if m.startswith("swarm/")])
N_ROBOTS = 4
SOURCE = (0.55, 0.5)
STARTS = [(-0.8, -0.8), (0.8, -0.8), (0.8, 0.0), (-0.8, 0.0)]
OBSTACLES = [(-0.35, 0.35, 0.2, 0.2), (0.35, 0.1, 0.15, 0.3),
             (-0.55, -0.35, 0.15, 0.15)]
ROBOT_COLOURS = ["#e63333", "#3380e6", "#f2bf1a", "#9a4dcc"]
KINDS = ["distributed", "centralised", "distributed_robot_failure",
         "centralised_coordinator_failure"]
SEEDS = [1, 2, 3]
BASE_EXPERIMENTS = [f"{k}_s{s}" for s in SEEDS for k in KINDS]
FAILED_ROBOT = 2           # robot that fails in the robot-failure runs
SENSOR_COLS = [f"s{k}" for k in range(8)]


def robot_log(experiment, robot):
    return pd.read_csv(DATA_DIR / f"{experiment}_robot_{robot}.csv")


def supervisor_log(experiment):
    return pd.read_csv(DATA_DIR / f"{experiment}_supervisor.csv")


def summary():
    with open(DATA_DIR / "experiments.json") as f:
        return json.load(f)


def has(experiment):
    return (DATA_DIR / f"{experiment}_supervisor.csv").exists()


def draw_arena(ax, source=True):
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((-1, -1), 2, 2, fill=False, lw=1.5))
    for x, y, sx, sy in OBSTACLES:
        ax.add_patch(Rectangle((x - sx / 2, y - sy / 2), sx, sy,
                               color="grey", alpha=0.6))
    if source:
        ax.plot(*SOURCE, marker="*", ms=16, color="orange", mec="k",
                label="hidden source", ls="")
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")


def distance_to_source(x, y):
    return math.hypot(x - SOURCE[0], y - SOURCE[1])


def pso_progress(experiment):
    """Per PSO iteration: each robot's best-known distance to the source
    (its gbest) and actual distance; NaN after a robot has failed."""
    rows = []
    for i in range(N_ROBOTS):
        d = robot_log(experiment, i)
        p = d[d.phase == "pso"]
        for it, g in p.groupby("pso_iter"):
            last = g.iloc[-1]
            rows.append({"robot": i, "iteration": int(it),
                         "gbest_error": distance_to_source(last.gbest_x,
                                                           last.gbest_y),
                         "robot_distance": distance_to_source(last.x, last.y),
                         "gbest_fit": last.gbest_fit})
    return pd.DataFrame(rows)


def runs(kind):
    """Names of the runs of one experiment kind that exist on disk."""
    return [f"{kind}_s{s}" for s in SEEDS if has(f"{kind}_s{s}")]


def first_within(series, threshold):
    hit = series[series < threshold]
    return int(hit.index[0]) if len(hit) else np.nan


def run_metrics(name):
    """Headline metrics for one run."""
    prog = pso_progress(name)
    if "robot_failure" in name:
        prog = prog[prog.robot != FAILED_ROBOT]     # score the survivors
    final = prog[prog.iteration == prog.iteration.max()]
    info = summary()[name]
    sup = supervisor_log(name)
    messages = sum(int(robot_log(name, i).messages_sent.iloc[-1])
                   for i in range(N_ROBOTS)) + info["coordinator_messages"]
    return {
        "gbest error (cm)": 100 * final.gbest_error.mean(),
        "robot distance (cm)": 100 * final.robot_distance.mean(),
        "first within 10 cm (iter)": first_within(
            prog.groupby("iteration").gbest_error.min(), 0.10),
        "all within 10 cm (iter)": first_within(
            prog.groupby("iteration").gbest_error.max(), 0.10),
        "contact events": sum(info["collisions"]),
        "closest gap (mm)": 1000 * min(sup[f"clear{i}"].min()
                                       for i in range(N_ROBOTS)),
        "messages": messages,
    }


def kind_table(kinds, columns=None):
    """Mean +/- standard deviation over seeds for each experiment kind."""
    rows = {}
    for kind in kinds:
        m = pd.DataFrame([run_metrics(r) for r in runs(kind)])
        cols = columns or list(m.columns)
        rows[kind] = {c: (f"{m[c].mean():.1f} ± {m[c].std():.1f}"
                          if m[c].notna().all() else
                          f"{m[c].mean():.1f} ({m[c].notna().sum()}/{len(m)} runs)")
                      for c in cols}
        rows[kind]["runs"] = len(m)
    return pd.DataFrame(rows).T
