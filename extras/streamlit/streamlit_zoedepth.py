import cv2
import streamlit as st
import time
from transformers import AutoImageProcessor, ZoeDepthForDepthEstimation
from PIL import Image
import numpy as np
import torch
import json

image_width = 1280
image_height = 720
target_aspect_ratio = 4 / 3
frame_rate = 15

# Calculate target width based on the crop
target_width = int(image_width / target_aspect_ratio)


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


image_processor = AutoImageProcessor.from_pretrained("Intel/zoedepth-nyu-kitti")
#model = ZoeDepthForDepthEstimation.from_pretrained("Intel/zoedepth-nyu-kitti").half().to("cuda:0")
model = ZoeDepthForDepthEstimation.from_pretrained("Intel/zoedepth-nyu-kitti").to("cuda:0")

# Open the video device (e.g., webcam or external camera). Replace 0 with the appropriate device ID if needed.
# v4l2-ctl --device=/dev/video0 --list-formats-ext   MJPG mode offers higher framerates up to 60hz, but manual says 30hz max up to 4k
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FPS, frame_rate)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, image_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, image_height)
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Video Capture Device FPS: {fps}")
#cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Check if the video device is opened successfully
if not cap.isOpened():
    st.error("Error: Could not open video device.")
    st.stop()

#UI Elements
st.title("Video Stream")
stframe1 = st.empty()  # Placeholder for original frames
stframe2 = st.empty()  # Placeholder for flow images
fps_display = st.sidebar.empty()

K,D,image_size = load_calibration_data("hdzero_eco_960x720.json")
map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, image_size, cv2.CV_16SC2)

last_update = time.time()
last_fps_update = time.time()
frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        st.error("Error: Failed to capture image.")
        break

    start_time = time.time()
    current_frame = crop(frame)
    undistorted_img = cv2.remap(current_frame, map1, map2, interpolation=cv2.INTER_LINEAR)
    downscaled_frame = cv2.resize(current_frame, (256, 256), interpolation=cv2.INTER_AREA)
    current_frame_rgb = cv2.cvtColor(downscaled_frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    current_frame_pil = Image.fromarray(current_frame_rgb) 
    #result = pipe(current_frame_pil)
    #depth = result["depth"]
    preproc_time = time.time()

    inputs = image_processor(images=current_frame_pil, return_tensors="pt").to("cuda:0")

    # Move tensors to GPU and convert to half precision
    #inputs = {k: v.to("cuda:0").half() for k, v in inputs.items()}
    #inputs = {k: v.to("cuda:0") for k, v in inputs.items()}
    loading_time = time.time()

    with torch.no_grad():
        outputs = model(**inputs)
    inference_time = time.time()

    post_processed_output = image_processor.post_process_depth_estimation(
        outputs,
        source_sizes=[(current_frame_pil.height, current_frame_pil.width)]
    )
    predicted_depth = post_processed_output[0]["predicted_depth"]

    depth = (predicted_depth - predicted_depth.min()) / (predicted_depth.max() - predicted_depth.min())
    depth_image_np = (depth.detach().cpu().numpy() * 255).astype(np.uint8)

    # Apply a colormap to the depth image
    depth_colored = cv2.applyColorMap(depth_image_np, cv2.COLORMAP_BONE)

    #stframe2.image(downscaled_frame, caption="Optical Flow Frame", channels="BGR")
    stframe2.image(depth_colored, caption="ZoeDepth Frame")

    fps_enable = False
    if fps_enable:
        frame_count += 1
        if time.time() > last_fps_update + 1:
            fps = frame_count / (time.time() - last_fps_update)
            last_fps_update = time.time()
            frame_count = 0
            fps_display.metric("FPS", f"{fps:.2f}")
            print(f"Pre Process Time: {preproc_time - start_time}")
            print(f"Loading Time: {loading_time - preproc_time}")
            print(f"Inference Time: {inference_time - loading_time}")
            print(f"Total Time: {inference_time - start_time}")
            print(f"Depth min: {predicted_depth.min().item()} meters")
            print(f"Depth max: {predicted_depth.max().item()} meters")
            height, width = predicted_depth.shape
            x, y = width // 2, height // 2  # Example: Center pixel
            depth_center = predicted_depth[y, x].item()
            print(f"Depth at center ({x}, {y}): {depth_center:.2f} meters")

cap.release()
cv2.destroyAllWindows()
