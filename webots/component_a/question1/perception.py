
import cv2
import numpy as np


def detect_obstacles(image, threshold=100):
    """
    Detect possible obstacles in a camera image.

    Args:
        image: Camera image as a NumPy array.
        threshold: Brightness threshold used for segmentation.

    Returns:
        Processed image and number of detected regions.
    """

    if image is None:
        raise ValueError("No camera image was provided.")

    # Convert the image to grayscale.
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Reduce noise before detecting regions.
    blurred = cv2.GaussianBlur(grayscale, (5, 5), 0)

    # Separate darker regions from brighter regions.
    _, binary = cv2.threshold(
        blurred, threshold, 255, cv2.THRESH_BINARY_INV
    )

    # Find separate regions in the processed image.
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Ignore very small regions that may be noise.
    obstacles = [
        contour for contour in contours
        if cv2.contourArea(contour) > 100
    ]

    # Draw boxes around the detected regions.
    result = image.copy()

    for contour in obstacles:
        x, y, width, height = cv2.boundingRect(contour)
        cv2.rectangle(
            result, (x, y), (x + width, y + height),
            (0, 255, 0), 2
        )

    return result, len(obstacles)


if __name__ == "__main__":
    print("Perception module loaded successfully.")
    print("Waiting for a camera image to process.")
