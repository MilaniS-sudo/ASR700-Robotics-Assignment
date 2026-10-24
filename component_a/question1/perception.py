"""
Camera perception for Component A (Question 1).

Detects the coloured landmark cylinders in e-puck camera frames using HSV
colour thresholding, and converts each detection into a bearing (angle
relative to the robot's heading) using a pinhole camera model.

Why HSV rather than grayscale or RGB thresholds:
    The landmarks differ from the grey obstacles, grey walls, light floor and
    pale-blue sky mainly in *saturation*, not brightness. HSV separates hue
    and saturation from brightness (value), so a single threshold on
    saturation removes every unsaturated surface regardless of how brightly
    it is lit, and hue then identifies which landmark it is.
"""

import math

import cv2
import numpy as np

# Hue ranges in OpenCV units (0-179). Red wraps around 0, so it has two.
COLOUR_RANGES = {
    "red": [(0, 10), (170, 179)],
    "green": [(45, 85)],
    "blue": [(100, 130)],
}

MIN_SATURATION = 150   # rejects floor, walls, grey boxes and the pale sky
MIN_VALUE = 50         # rejects very dark pixels where hue is unreliable
MIN_AREA_PX = 40       # rejects specks (e.g. a landmark edge at the border)

KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))


def colour_mask(hsv, colour):
    """Binary mask of pixels belonging to one landmark colour."""
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for low, high in COLOUR_RANGES[colour]:
        mask |= cv2.inRange(
            hsv,
            (low, MIN_SATURATION, MIN_VALUE),
            (high, 255, 255),
        )
    # Opening (erode then dilate) removes isolated noise pixels without
    # shrinking real blobs; closing fills small holes inside a blob.
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL)
    return mask


def pixel_to_bearing(cx, image_width, horizontal_fov):
    """Bearing of image column cx in radians (0 = ahead, positive = left).

    Pinhole model: focal length f = (W/2) / tan(FOV/2), so a pixel offset
    u from the image centre lies at angle atan(u / f).
    """
    focal = (image_width / 2.0) / math.tan(horizontal_fov / 2.0)
    return math.atan((image_width / 2.0 - cx) / focal)


def detect_landmarks(image_bgr, horizontal_fov):
    """Detect coloured landmarks in one BGR image.

    Returns a list of dicts with colour, centroid, bounding box, pixel area
    and bearing (radians). Only the largest blob per colour is kept, because
    each colour appears exactly once in the world.
    """
    if image_bgr is None:
        raise ValueError("No camera image was provided.")

    # A small blur suppresses aliasing on object edges before thresholding.
    blurred = cv2.GaussianBlur(image_bgr, (3, 3), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    width = image_bgr.shape[1]

    detections = []
    for colour in COLOUR_RANGES:
        contours, _ = cv2.findContours(
            colour_mask(hsv, colour), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        contours = [c for c in contours if cv2.contourArea(c) >= MIN_AREA_PX]
        if not contours:
            continue

        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        moments = cv2.moments(largest)
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]

        detections.append({
            "colour": colour,
            "cx": cx,
            "cy": cy,
            "bbox": (x, y, w, h),
            "area": cv2.contourArea(largest),
            # A blob cut off by the image border has an unreliable centre.
            "touches_border": x <= 0 or x + w >= width,
            "bearing": pixel_to_bearing(cx, width, horizontal_fov),
        })
    return detections


def draw_detections(image_bgr, detections):
    """Return a copy of the image with labelled bounding boxes."""
    result = image_bgr.copy()
    for d in detections:
        x, y, w, h = d["bbox"]
        cv2.rectangle(result, (x, y), (x + w, y + h), (255, 255, 255), 1)
        label = f"{d['colour']} {math.degrees(d['bearing']):+.0f}deg"
        cv2.putText(result, label, (x, max(8, y - 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    return result
