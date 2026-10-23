{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyMnFYI0fzSZduwFiyVQu6ys",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/MilaniS-sudo/ASR700-Robotics-Assignment/blob/main/component_a/question3/robot_kinematics.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 2,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "3Y7qP3KM1hO2",
        "outputId": "f2a896ac-ff0f-4f94-c2a6-7938fb29e464"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Robot x-position: 1.0\n",
            "Robot y-position: 0.0\n",
            "Robot heading (radians): 0.5\n"
          ]
        }
      ],
      "source": [
        "import math\n",
        "\n",
        "\n",
        "def update_robot_position(x, y, theta, linear_velocity,\n",
        "                         angular_velocity, dt):\n",
        "    \"\"\"\n",
        "    Update a differential-drive robot's position.\n",
        "\n",
        "    x, y: current position in metres\n",
        "    theta: heading in radians\n",
        "    linear_velocity: forward speed in m/s\n",
        "    angular_velocity: turning speed in rad/s\n",
        "    dt: time step in seconds\n",
        "    \"\"\"\n",
        "\n",
        "    if dt <= 0:\n",
        "        raise ValueError(\"dt must be greater than zero\")\n",
        "\n",
        "    new_x = x + linear_velocity * math.cos(theta) * dt\n",
        "    new_y = y + linear_velocity * math.sin(theta) * dt\n",
        "    new_theta = theta + angular_velocity * dt\n",
        "\n",
        "    # Keep heading within -pi to pi\n",
        "    new_theta = (new_theta + math.pi) % (2 * math.pi) - math.pi\n",
        "\n",
        "    return new_x, new_y, new_theta\n",
        "\n",
        "\n",
        "if __name__ == \"__main__\":\n",
        "    x, y, theta = update_robot_position(\n",
        "        x=0.0,\n",
        "        y=0.0,\n",
        "        theta=0.0,\n",
        "        linear_velocity=1.0,\n",
        "        angular_velocity=0.5,\n",
        "        dt=1.0\n",
        "    )\n",
        "\n",
        "    print(\"Robot x-position:\", round(x, 3))\n",
        "    print(\"Robot y-position:\", round(y, 3))\n",
        "    print(\"Robot heading (radians):\", round(theta, 3))"
      ]
    }
  ]
}