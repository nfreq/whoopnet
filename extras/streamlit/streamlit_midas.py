import cv2
import streamlit as st
import time
import numpy as np
from transformers import pipeline
from PIL import Image
import json

image_width = 1280
image_height = 720
target_aspect_ratio = 4 / 3
frame_skip = 2  # Process every nth frame

# Initialize MiDaS pipeline
depth_pipe = pipeline(task="depth-estimation", model="Intel/dpt-hybrid-midas", device="cuda:0")

#Load Camera Calibration File
def load_calibration_data(file_path):
    with open(file_path, "r") as f:
        calibration_data = json.load(f)
    K = np.array(calibration_data["K"])  # Convert list back to numpy array
    D = np.array(calibration_data["D"])
    image_size = tuple(calibration_data["image_size"])
    print(f"Calibration data loaded from {file_path}")
    return K, D, image_size

# Function to crop the frame dynamically based on image_width and image_height
def crop(frame):
    h, w, _ = frame.shape
    target_width = int(w / target_aspect_ratio)
    if w > target_width:
        x_offset = (w - target_width) // 2
        cropped_frame = frame[:, x_offset:x_offset + target_width]
    else:
        cropped_frame = frame
    return cropped_frame

st.title("Video Stream with MiDaS Depth Estimation")

stframe1 = st.empty()  # Placeholder for original frames
stframe2 = st.empty()  # Placeholder for depth images
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, image_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, image_height)

fps_display = st.sidebar.empty()

if not cap.isOpened():
    st.error("Error: Could not open video device.")
    st.stop()

K,D,image_size = load_calibration_data("hdzero_eco_960x720.json")
map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, image_size, cv2.CV_16SC2)

frame_count = 0
last_fps_update = time.time()
while True:
    ret, frame = cap.read()
    if not ret:
        st.error("Error: Failed to capture image.")
        break

    frame_count += 1
    # Skip frames based on frame_skip value
    if frame_count % frame_skip != 0:
        continue

    cropped_frame = crop(frame)
    undistorted_img = cv2.remap(cropped_frame, map1, map2, interpolation=cv2.INTER_LINEAR)
    rgb_frame = cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2RGB)
    input_image = Image.fromarray(rgb_frame)

    # Use pipeline to get depth estimation
    result = depth_pipe(input_image)
    depth_image = result["depth"]

    depth_array = np.array(depth_image, dtype=np.float32)
    # Normalize depth values to a range between 0 and 1
    depth_normalized = cv2.normalize(depth_array, None, 0, 1, cv2.NORM_MINMAX)

    # Convert depth to uint8 for colormap application
    depth_uint8 = (depth_normalized * 255).astype(np.uint8)

    # Apply a colormap to the depth image
    depth_colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)
    stframe2.image(depth_colored, caption="MiDaS Depth Frame", channels="BGR")

    if time.time() > last_fps_update + 1:
        fps = frame_count / (time.time() - last_fps_update)
        last_fps_update = time.time()
        frame_count = 0
        fps_display.metric("FPS", f"{fps:.2f}")


cap.release()
cv2.destroyAllWindows()
