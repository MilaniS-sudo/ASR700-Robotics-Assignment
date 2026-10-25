# Autonomous Systems and Robotics 700: Intelligent Autonomous Robotic Systems

Two simulated robotic systems built in **Webots** and analysed in **Python**.
Full setup, versions, run order and runtimes are in [`README.txt`](README.txt).

## Component A: autonomous mobile robot (e-puck)
| Question | Folder | Summary |
|---|---|---|
| A1 Perception | `component_a/question1` | LiDAR occupancy grid + HSV camera landmark detection, fused into a semantic map (landmarks within about 1 cm) |
| A2 Planning | `component_a/question2` | Dijkstra vs A* on the real map, 300-route benchmark, waypoints |
| A3 Control | `component_a/question3` | Differential-drive kinematics, PID tuning and analysis |
| A4 Integration | `component_a/question4` | Closed-loop navigation in Webots, 3 PID gain sets, odometry drift, trade-offs and ethics |

## Component B: multi-agent swarm (4 × e-puck)
| Question | Folder | Summary |
|---|---|---|
| B1 Setup | `component_b/question1` | Cooperative source search, ring and star topologies |
| B2 Consensus & swarm | `component_b/question2` | Average consensus, formation, PSO search, emergent behaviour |
| B3 Learning & decisions | `component_b/question3` | Neural-network navigation (imitation learning), centralised vs distributed with fault injection |
| B4 Ethics & risk | `component_b/question4` | Bias, explainability, safety, fault tolerance, South African context |

## Simulation
| Folder | Contents |
|---|---|
| `webots/worlds` | `component_a.wbt`, `component_a_navigation.wbt`, `component_b_swarm.wbt` |
| `webots/controllers` | `component_a_controller`, `navigation_controller`, `swarm_controller`, `swarm_supervisor` |
| `webots/data` | All logged simulation data used by the notebooks |

All reported results come from simulations that were actually run. Random
seeds are fixed, and Component B repeats every experiment with 3 seeds.
