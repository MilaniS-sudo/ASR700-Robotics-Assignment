ASR700 Component A - Webots project
===================================

Contents
  worlds/component_a.wbt          Arena (1.5 m x 1.5 m), 5 box obstacles,
                                  red/green/blue cylinder landmarks, e-puck
                                  with a 360-degree LiDAR and a 160x120 camera
  controllers/component_a_controller/component_a_controller.py
                                  Explores with LiDAR obstacle avoidance and
                                  logs synchronised sensor data
  data/                           Output of the controller (regenerated on
                                  every run)
    lidar_scans.csv               time, true pose (x, y, yaw), 360 ranges
    images/frame_XXXX.png         camera frame for each CSV row
    ground_truth.json             true obstacle/landmark layout

How to run
  1. Webots > Preferences > General > Python command: set it to your python3
     (e.g. /usr/local/bin/python3 or the output of `which python3`).
  2. File > Open World... > worlds/component_a.wbt
  3. Press Play. The run lasts 90 s of simulated time, then pauses itself.
     The console prints "[controller] done: N samples saved".

Notes
  - Tested with Webots R2025a; the world file targets R2023b or newer.
  - First load needs internet (Webots downloads the robot/arena PROTOs).
  - The controller uses only the Python standard library.
  - The robot runs as a Supervisor only to log its true pose and the true
    world layout for evaluation; steering uses the LiDAR alone.
  - Fixed random seed (42) makes runs reproducible.
