ASR700 - Intelligent Autonomous Robotic Systems Project
=======================================================
Student: 402412405

Two simulated robotic systems built in Webots and analysed with Python
(Matplotlib, pandas, scikit-learn). No ROS or Gazebo is used.

  Component A  Single e-puck: LiDAR + camera perception, sensor fusion,
               Dijkstra / A* planning, PID control, closed-loop navigation.
  Component B  Four-robot swarm: average consensus, formation, Particle
               Swarm Optimisation search, neural-network navigation,
               centralised vs distributed coordination, ethics.


1. SOFTWARE VERSIONS
--------------------
  Webots   R2025a (world files are written for R2023b or newer)
  Python   3.9 or newer (developed with 3.9 on macOS and 3.13 on Linux)

  Webots controllers use only the Python standard library.

  Python dependencies for the notebooks:
      numpy >= 1.24      pandas >= 2.0      matplotlib >= 3.7
      scipy >= 1.10      scikit-learn >= 1.3
      opencv-python >= 4.8   jupyter (notebook or lab)

  Install with:
      pip install numpy pandas matplotlib scipy scikit-learn opencv-python jupyter


2. HARDWARE
-----------
  Any laptop that runs Webots (tested on a MacBook Air). No GPU needed.
  About 200 MB free disk space for the simulation logs.


3. FOLDER STRUCTURE
-------------------
  webots/
    worlds/component_a.wbt              A: sensor data collection world
    worlds/component_a_navigation.wbt   A: closed-loop navigation world (Q4)
    worlds/component_b_swarm.wbt        B: four-robot swarm world
    controllers/component_a_controller  A: explores and logs LiDAR + camera
    controllers/navigation_controller   A: follows the A* route with PID
    controllers/swarm_controller        B: one controller for every robot
    controllers/swarm_supervisor        B: runs experiments, faults, logging
    data/                               all simulation outputs (CSV/JSON/PNG)
  component_a/question1 ... question4   A: notebooks, modules, figures
  component_b/question1 ... question4   B: notebooks and figures
  component_b/swarm_data.py             B: shared data-loading helpers


4. HOW TO RUN (in this order)
-----------------------------
  First, set Webots to use your Python: Webots > Preferences > General >
  Python command = the output of `which python3` in a terminal.

  Every simulation pauses itself when finished and prints "done" in the
  Webots console. Press the fast-forward button to run faster than real time.

  Component A
    1. Open webots/worlds/component_a.wbt and press Play.
       -> webots/data/lidar_scans.csv, images/, ground_truth.json
    2. Run component_a/question1/perception_pipeline.ipynb
       -> question1/outputs/fused_map.npz (camera frames are unzipped
          automatically from images.zip if needed)
    3. Run component_a/question2/motion_planning.ipynb
       -> question2/outputs/planned_path.json
    4. Run component_a/question3/pid_controller.ipynb
    5. Open webots/worlds/component_a_navigation.wbt and press Play.
       -> webots/data/navigation/
    6. Run component_a/question4/integration.ipynb

  Component B
    7. Open webots/worlds/component_b_swarm.wbt and press Play.
       Runs 12 experiments (4 types x 3 seeds) -> webots/data/swarm/
    8. Run component_b/question3/learning_and_decisions.ipynb
       Trains the network and writes webots/data/swarm/nn_policy.json
    9. Open component_b_swarm.wbt again and press Play. Because
       nn_policy.json now exists, it runs the 3 neural-network runs.
   10. Run all four Component B notebooks (question1 ... question4).
       Re-run question3 so it includes the neural-network results.

  Run each notebook from its own folder (Kernel > Restart & Run All).


5. EXPECTED RUNTIME
-------------------
  Simulated time per world (real time is shorter with fast-forward):
    component_a.wbt              90 s
    component_a_navigation.wbt   about 70 s (3 trials)
    component_b_swarm.wbt        about 21 min (12 runs of 107 s), then
                                 about 5.5 min for the 3 neural-network runs
  Notebooks: each runs in under 1 minute, except
    A question2 (300-route benchmark) about 1-2 min
    B question3 (network training) about 1-2 min


6. NOTES ON REPRODUCIBILITY
---------------------------
  * Worlds use plain colours and built-in lighting, so nothing needs to be
    downloaded at runtime except the e-puck and arena models (first load
    needs internet).
  * Random seeds are fixed in all controllers and notebooks. Webots adds
    fresh LiDAR noise each run, so repeated runs differ slightly. That is
    why Component B repeats each experiment with 3 seeds and reports
    mean +/- standard deviation.
  * The Webots supervisor provides each robot's true pose. Component A
    compares this with wheel odometry to show the localisation gap.
