Component A — Question 1: Sensor Simulation and Perception

1. Objective

The objective is to develop a basic perception pipeline that processes camera images, simulates LiDAR distance readings, and combines information from both sensors to support obstacle awareness.

2. LiDAR Simulation

A Python function generates example distance readings in metres. The closest valid reading is calculated after removing invalid values, including non-finite and non-positive readings.

This provides sample data for initial testing. These readings are synthetic and must eventually be replaced with measurements from a LiDAR sensor in Webots.

3. Camera Image Processing

The perception module uses the following preprocessing steps:

1. Grayscale conversion: simplifies the image by reducing it to intensity information.
2. Gaussian blur: reduces small variations and image noise.
3. Thresholding: separates darker pixels from brighter pixels.
4. Contour detection: identifies connected regions in the processed image.
5. Area filtering: ignores regions that are too small to be useful.
6. Bounding boxes: marks detected regions for visual inspection.

The initial test uses a synthetic image containing two dark rectangles. Detecting these regions verifies the basic processing logic, but dark regions are not necessarily real obstacles.

4. Sensor Fusion

The initial fusion function combines the closest valid LiDAR distance with the camera’s list of detected objects. A warning is raised when the closest valid LiDAR reading is below one metre.

This approach was selected because it is simple to implement and provides a common output containing distance and visual detection information. However, it does not yet match camera detections to specific LiDAR measurements or confirm that both sensors observed the same object.

5. Initial Testing

The following initial tests were performed in Google Colab:

* Generated synthetic LiDAR distance readings.
* Processed a synthetic image and checked for two dark regions.
* Combined sample LiDAR and camera data using the fusion function.

These tests check the initial Python logic only. They do not demonstrate successful Webots sensor integration.

6. Further Work

The next stage is to connect the pipeline to actual Webots robot sensors, test it in a simulated environment, evaluate its performance, and record the results. Further improvements could include camera-based object classification, sensor calibration, and more robust matching between camera detections and LiDAR measurements.
