# Component A — Question 2: Motion Planning and Pathfinding

Plans a route across the arena on the **fused occupancy grid from Question 1**
(`../question1/outputs/fused_map.npz`), which was built from Webots sensor data.

## Files
| File | Purpose |
|---|---|
| `motion_planning.py` | Planning grid (unknown = blocked, obstacles inflated by robot radius), Dijkstra and A* on an 8-connected grid, line-of-sight path simplification |
| `motion_planning.ipynb` | Comparison of Dijkstra, A* (Euclidean) and A* (Manhattan) on one route and on 300 random routes, plus waypoint generation |
| `outputs/planned_path.json` | Start, goal and waypoints for the PID controller (Questions 3 and 4) |
| `outputs/*.png` | Figures |

## Key results (300 random routes)
| Algorithm | Nodes vs Dijkstra | Mean time | Optimal paths |
|---|---|---|---|
| Dijkstra | 100% | 23.2 ms | 300/300 |
| A* (Euclidean) | 27% | 7.9 ms | 300/300 |
| A* (Manhattan) | 14% | 6.1 ms | 284/300 (worst +11.5%) |

A* with the admissible Euclidean heuristic is used for the robot.

## How to run
Run Question 1 first (it produces `fused_map.npz`), then run all cells of
`motion_planning.ipynb` from this folder.
Dependencies: numpy, pandas, matplotlib, scipy.
