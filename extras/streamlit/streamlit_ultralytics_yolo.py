import numpy as np
from PIL import Image
import time
from collections import defaultdict, deque
import cv2
from ultralytics import YOLO
import streamlit as st
import json

frame_height = 720
frame_width = 1280 
frame_rate = 25

#Load Camera Calibration File
def load_calibration_data(file_path):
    with open(file_path, "r") as f:
        calibration_data = json.load(f)
    K = np.array(calibration_data["K"])  # Convert list back to numpy array
    D = np.array(calibration_data["D"])
    image_size = tuple(calibration_data["image_size"])
    print(f"Calibration data loaded from {file_path}")
    return K, D, image_size

model = YOLO("yolo11n-seg.pt")  # load an official model (n nano, x largest)
model.to('cuda:1')

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FPS, frame_rate)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)


st.title("YOLOv11 Real-Time Segmentation")
stframe = st.empty()    
fps_display = st.sidebar.empty()
stop_button = st.sidebar.button("Stop Stream")

K,D,image_size = load_calibration_data("hdzero_eco_960x720.json")
map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, image_size, cv2.CV_16SC2)

last_fps_update = time.time()
frame_count = 0
frame_skip = 0
while True:
    ret, frame = cap.read()
    if frame_skip > 0 and frame_count % frame_skip != 0:
        frame_count += 1
        continue
    if not ret:
        st.error("Error: Failed to capture image.")
        break
    
    start = time.time()
    framecrop = frame[:, 160:1120] # crop to 960x720
    undistorted_frame = cv2.remap(framecrop, map1, map2, interpolation=cv2.INTER_LINEAR)
    #downscaled_frame = cv2.resize(undistorted_frame, (480, 360), interpolation=cv2.INTER_AREA)
    image = Image.fromarray(cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2RGB))
    preprocess_time = time.time()

    results = model.track(image, device='cuda:1', verbose=False, persist=True)
    annotated_frame = results[0].plot()
    inference_time = time.time()
    
    # Display the annotated frame in Streamlit
    if frame_count % 3 == 0:
        stframe.image(annotated_frame, channels="BGR")

    frame_count += 1
    if time.time() > last_fps_update + 1:
        fps = frame_count / (time.time() - last_fps_update)
        last_fps_update = time.time()
        frame_count = 0
        #print(f"{fps} hz")
        #print(f"Preprocessing time: {preprocess_time - start:.4f} seconds")
        #print(f"Inference time: {inference_time - preprocess_time:.4f} seconds")
        #print(f"Total time per loop: {inference_time - start:.4f} seconds")
        fps_display.metric("Loop", f"{fps:.2f} hz")

# Release the camera and video writer
print("Video File Released")
cv2.destroyAllWindows()
