# Whoopnet

Whoopnet transforms standard FPV drones into an autonomous vehicle by harnessing off-board neural networks that process raw visual and inertial data and output RC flight commands.

While compatible with any unmodified FPV multirotor that supports ELRS, HDZero and Betaflight, the project focuses on targeting sub-25-gram vehicles and advancing offboard compute capabilities to enable AI-driven autonomous software.

This flexibility allows your 20-gram robot to leverage a 20-kilogram brain, depending on your edge compute requirements, enabling scalable computational power for advanced tasks and experimentation.

Read the [docs](https://github.com/nfreq/whoopnet/wiki).
## Goal
* Reproduce [Deep Drone Acrobatics](https://arxiv.org/pdf/2006.05768) with a **tiny whoop**.

## Challenges
* Un-syncronized Camera and IMU feeds
* Low Frequency IMU (~100hz)
* Rolling Shutter Camera
* ELRS Telemetry Bandwidth
* Noise
* Murphy

## Strategy
* Betaflight FC firmware modifications:
  1. Telemetry: Disable Existing Telemetry. Send raw IMU CRSF telemetry packets @ ~100hz (include FC RTC timestamp)
  2. OSD: Increase OSD Refresh Rate to 30hz and send FC RTC timestamp (data intake will extract via OCR)
* ELRS v3 F1000HZ to support the new telemetry bandwidth requirements
* ROS2 timestamps in IMU and Camera topics will use FC RTC timestamp to keep the data as synchronized as possible
