import math


def update_robot_position(x, y, theta, linear_velocity,
                         angular_velocity, dt):
    """
    Update a differential-drive robot's position.

    x, y: current position in metres
    theta: heading in radians
    linear_velocity: forward speed in m/s
    angular_velocity: turning speed in rad/s
    dt: time step in seconds
    """

    if dt <= 0:
        raise ValueError("dt must be greater than zero")

    new_x = x + linear_velocity * math.cos(theta) * dt
    new_y = y + linear_velocity * math.sin(theta) * dt
    new_theta = theta + angular_velocity * dt

    # Keep heading within -pi to pi
    new_theta = (new_theta + math.pi) % (2 * math.pi) - math.pi

    return new_x, new_y, new_theta


if __name__ == "__main__":
    x, y, theta = update_robot_position(
        x=0.0,
        y=0.0,
        theta=0.0,
        linear_velocity=1.0,
        angular_velocity=0.5,
        dt=1.0
    )

    print("Robot x-position:", round(x, 3))
    print("Robot y-position:", round(y, 3))
    print("Robot heading (radians):", round(theta, 3))
