# Whoopnet

Whoopnet transforms standard FPV drones into an autonomous vehicle by harnessing off-board neural networks that process raw visual and inertial data and output RC flight commands.

While compatible with any unmodified FPV multirotor that supports ELRS and Betaflight, the project focuses on targeting sub-25-gram vehicles and advancing offboard compute capabilities to enable AI-driven models and autonomous software.  

This flexibility allows your 20-gram robot to leverage a 20-kilogram brain, depending on your edge compute requirements, enabling scalable computational power for advanced tasks and experimentation.

Challenges:
- Unsyncronized Camera+IMU 
- Low Frequency IMU (~100hz)
- Rolling Shutter Camera
- Saturation of Telemetry Bandwidth via ELRS
- Noise

---

## Major Components of Whoopnet

### ROS Integration
- **Video Feedback:** Captures and streams real-time visual data for processing by AI models.  
- **IMU Feedback:** Provides orientation and motion data, fused with vision to enable accurate state estimation for visual-inertial odometry, navigation and autonomous decision-making.  
- **Handset Input:** Enables manual control inputs from a pilot, serving as a fallback or supplement to AI operations.  
- **RC Output:** Transmits AI-processed or pilot commands to the multirotor’s control system for execution.  
- **Real-Time Control and Data Collection:** Facilitates real-time interactions and logs data critical for training AI models, enabling iterative improvements to system performance.

---

### IMU-Camera Calibration Process Documentation
- A detailed guide for aligning inertial measurement units (IMUs) with cameras on FPV-based multirotors, ensuring accurate sensor fusion for precise navigation and control.  
- Includes step-by-step instructions for calibration, troubleshooting tips, and recommendations for maintaining alignment during operations.

---

### ELRS Repeater/RC Path Mixer
- **Human Manual Control:** Allows pilots to fully control the multirotor in scenarios requiring human oversight.  
- **Handoff:** Facilitates seamless transitions between human control and autonomous AI-driven operations.  
- **Mixed Autonomous Operation:** Combines human input with AI adjustments for enhanced control and safety, ideal for complex flight scenarios.

---

### Betaflight Modifications and ExpressLRS 
- **Betaflight Modifications:** Custom changes to the flight control software to enable high-speed raw IMU telemetry, including the creation of a new IMU extended CRSF packet and dedicating the entire telemetry channel for this purpose. We also highjack the craftname field of OSD to push FC timestamp so we can attempt to syncronize the video and imu data.
- **ExpressLRS:** Utilizes ExpressLRS's existing F1000Hz mode to provide the necessary telemetry bandwidth for real-time data transmission.

---
## Hardware
### FPV Multirotors
- Any multirotor with ELRS and a VRX that has HDMI OUT
- 65mm class (sub-25-gram) multirotors:  
  - **Mobula6 HDZero ECO 2024**  
  - **Mobula6 Freestyle HD**
- 3.5" class (sub-250-gram) multirotors:  
  - **Crux35 HDZero**

### ELRS Transmitter(s)
- Used for remote communication and control with "high-speed" telemetry.  
- Example: **BETAFPV 2.4GHZ 1W Micro RF Module (Supports CRSF over USB)**

### VRX with HDMI Output
- A video receiver capable of streaming real-time FPV video feed through an HDMI interface.  
- Example: **HDZero VRX (HDZero Monitor) /w HDMI Output**

### Video Capture Device
- Captures the video feed from the VRX and integrates it into the system for processing and display.  
- **Recommendation:** 60Hz for calibration; 30Hz should be fine for real-time control.

### ELRS Receiver to USB Gamepad
- Converts ELRS signals to a USB gamepad for utilization with the RC handset input and mixing capabilities.  
- Examples: **SquidStick, RC Handset USB**, etc.

### RC Handset
- For Manual Flight Control and Expert Pilot Training
- Examples: RadioMaster Boxer Radio Transmitter

### FPV Goggles (Optional)
- For Manual Flight Feedback and Expert Pilot Training
- Examples: HDZero Goggles

## Instructions

### Build Betaflight
- Modified Betaflight 4.5.1 Firmware 
- Approx 92hz Telemetry (Includes, AccX,AccY,AccZ,VelX,VelY,VelZ,FC-Timestamp)
- Must run ELRS F1000HZ (1:2) for this to work. Otherwise telemetry bandwidth will saturate which results in much lower update rate.

git clone https://github.com/nfreq/betaflight

make arm_sdk_install

make configs

#### HDZero Mobula6 ECO 2024
make CRAZYBEEF4DX EXTRA_FLAGS="-D'RELEASE_NAME=4.5.1-imu-osd-mod' -DCLOUD_BUILD -DUSE_DSHOT -DUSE_OSD_HD -DUSE_PINIO -DUSE_SERIALRX -DUSE_SERIALRX_CRSF -DUSE_TELEMETRY -DUSE_TELEMETRY_CRSF -DUSE_VTX"
