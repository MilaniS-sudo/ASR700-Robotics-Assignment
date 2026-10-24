# Component A — Question 4: System Integration and Evaluation

Closed-loop autonomous navigation in Webots that chains perception (Q1),
planning (Q2) and PID control (Q3), with live LiDAR as a safety monitor.

## How it runs
1. `webots/worlds/component_a_navigation.wbt` uses the controller
   `webots/controllers/navigation_controller/navigation_controller.py`.
2. The controller loads the A* waypoints from
   `component_a/question2/outputs/planned_path.json` and drives the e-puck
   through them with a heading PID. It runs three gain settings in turn
   (low Kp, tuned, high Kp), resetting the robot to the start each time.
3. Logs are written to `webots/data/navigation/` (one CSV per trial plus
   `summary.json`).
4. `integration.ipynb` analyses the logs: planned vs actual trajectory,
   tracking errors, performance table, odometry drift, trade-offs and ethics.

## Files
| File | Purpose |
|---|---|
| `integration.ipynb` | Evaluation and discussion |
| `outputs/q4_performance.csv` | Performance evaluation table |
| `outputs/*.png` | Trajectory, error, and odometry figures |

Run the Webots navigation world first, then run all cells of the notebook.
Dependencies: numpy, pandas, matplotlib.
