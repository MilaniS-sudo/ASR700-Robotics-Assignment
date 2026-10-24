
import numpy as np


def fuse_sensor_data(lidar_distances, camera_obstacles):
    """
    Combine LiDAR distance readings with camera obstacle detections.

    lidar_distances:
        List of distances measured by LiDAR, in metres.

    camera_obstacles:
        List of dictionaries containing camera detections.
        Each detection can include a label and confidence score.
    """

    lidar = np.asarray(lidar_distances, dtype=float)

    # Keep only valid, positive distance readings.
    valid_lidar = lidar[np.isfinite(lidar) & (lidar > 0)]

    fused_data = {
        "lidar_min_distance_m": (
            float(np.min(valid_lidar))
            if valid_lidar.size > 0 else None
        ),
        "camera_obstacles": camera_obstacles,
    }

    # Give a simple warning if LiDAR detects a nearby obstacle.
    if fused_data["lidar_min_distance_m"] is not None:
        fused_data["nearby_obstacle"] = (
            fused_data["lidar_min_distance_m"] < 1.0
        )
    else:
        fused_data["nearby_obstacle"] = None

    return fused_data


if __name__ == "__main__":
    example_lidar = [2.5, 1.8, 0.7, float("inf")]
    example_camera = [
        {"label": "obstacle", "confidence": 0.85}
    ]

    result = fuse_sensor_data(example_lidar, example_camera)
    print(result)
