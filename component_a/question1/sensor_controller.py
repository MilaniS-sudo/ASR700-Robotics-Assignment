
import numpy as np


def simulate_lidar_readings(
    number_of_rays=9,
    max_range=5.0,
    obstacle_distance=2.0
):
    """
    Generate example LiDAR readings for initial testing.

    These are synthetic readings, not real Webots measurements.
    """

    if number_of_rays < 1:
        raise ValueError("number_of_rays must be at least 1.")

    if max_range <= 0:
        raise ValueError("max_range must be positive.")

    if not 0 <= obstacle_distance <= max_range:
        raise ValueError(
            "obstacle_distance must be between 0 and max_range."
        )

    readings = np.full(number_of_rays, max_range, dtype=float)

    # Simulate an obstacle in the centre of the sensor's view.
    centre_ray = number_of_rays // 2
    readings[centre_ray] = obstacle_distance

    return readings


if __name__ == "__main__":
    distances = simulate_lidar_readings()

    print("Simulated LiDAR distances (metres):")
    print(distances)

    print("Closest obstacle (metres):", np.min(distances))
