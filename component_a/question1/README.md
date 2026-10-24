# Component A — Question 1: Sensor Simulation and Perception

## Objective
Build a perception pipeline that processes LiDAR and camera data from the
Webots simulation, detects obstacles and landmarks, and fuses both sensors
into a single environment representation for path planning.

## Data source
Sensor data is logged by the Webots controller
(`webots/controllers/component_a_controller`) while an e-puck explores the
arena in `webots/worlds/component_a.wbt`:

- 360-ray LiDAR scans (1.5 m range) and the robot's pose, every 0.5 s → `webots/data/lidar_scans.csv`
- 160×120 camera frames at the same moments → `webots/data/images.zip`
- the true obstacle and landmark layout, for evaluation → `webots/data/ground_truth.json`

## Files
| File | Purpose |
|---|---|
| `perception.py` | Camera: HSV colour segmentation, morphological cleaning, landmark detection, pixel-to-bearing conversion |
| `lidar_processing.py` | LiDAR: validity checking, optional median filter, world projection, log-odds occupancy grid with Bresenham ray casting |
| `sensor_fusion.py` | Fusion: camera bearing and identity + LiDAR range → labelled landmark positions → semantic occupancy grid |
| `perception_pipeline.ipynb` | Runs the full pipeline with visualisations and evaluation against ground truth |
| `outputs/` | Saved figures and `fused_map.npz` (input to Question 2) |

## Method summary
1. **Camera.** Convert to HSV and threshold saturation to reject grey and pale surfaces. Hue identifies red, green or blue. Opening and closing remove noise, and a pinhole model converts the blob centroid to a bearing.
2. **LiDAR.** Invalid returns are removed. A median filter was tested and rejected, because it reduced edge coverage on this data and only helps above about 5% outlier spikes. Scans are integrated into a 2 cm log-odds occupancy grid.
3. **Fusion.** Each camera detection is matched to the LiDAR return at the same bearing (±2°). Range plus radius gives the landmark centre. The median across frames gives the final estimate, which then labels the occupancy grid.

## How to run
From this folder: `jupyter notebook perception_pipeline.ipynb` and run all
cells. Camera frames are extracted from `images.zip` automatically on the
first run.

Dependencies: numpy, pandas, matplotlib, opencv-python, scipy.
