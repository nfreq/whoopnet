# Whoopnet
Whoopnet enables tiny FPV drones to operate autonomously by leveraging offboard neural networks. It facilitates the processing of raw visual and inertial data while providing the interface to deliver RC flight commands for classical and ai driven control.

While compatible with any unmodified FPV multirotor that supports ELRS, HDZero and Betaflight, the project focuses on targeting sub-25-gram vehicles and advancing capabilities via offboard compute.

This allows a 20g robot to have a 20kg brain.

Read the [docs](https://github.com/nfreq/whoopnet/wiki).
___
### Goals
* Stable State Estimation with VINS (Visual-Inertial Navigation)
* Reproduce [Deep Drone Acrobatics](https://arxiv.org/pdf/2006.05768) with a **tiny whoop** and offboard compute.
___
<div align="center">
  <img src="https://github.com/user-attachments/assets/414608c7-fce0-4ee1-b526-d78d7cb91b39" alt="whoopnet_hw" width="30%" style="display:inline-block;">
  <img src="https://github.com/user-attachments/assets/0b8ead3e-dcb9-4141-bffe-6c3b835faf4a" alt="whoopnet_hw" width="45%" style="display:inline-block;">
</div>
<div align="center">
  <img src="https://github.com/user-attachments/assets/7331890b-623d-4bba-9e12-3f2614ad0907" alt="whoopnet_hw" width="80%" style="display:inline-block;"> 
</div>

___

### Challenges
* Un-syncronized Camera and IMU feeds
* Low Frequency IMU (~100hz)
* Rolling Shutter Camera
* ELRS Telemetry Bandwidth
* Noise
* Murphy

### Strategy
* Betaflight FC firmware modifications:
  1. Disable Existing Telemetry. Send raw IMU CRSF telemetry packets @ ~100hz (include FC RTC timestamp)
  2. Increase OSD Refresh Rate to 30hz and send FC RTC timestamp (recieve and extract via OCR)
* ELRS v3 F1000HZ to support the new telemetry bandwidth requirements
* ROS2 timestamps in IMU and Camera topics will use FC RTC timestamp to keep the data as synchronized as possible
