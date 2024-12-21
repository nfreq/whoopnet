import torch
import cv2
import numpy as np
import os
import streamlit as st
import pandas as pd
from NeuFlow.neuflow import NeuFlow
from NeuFlow.backbone_v7 import ConvBlock
from data_utils import flow_viz
import asciichartpy
import random
import time
from collections import deque
import torch.nn.functional as F
import altair as alt
import json

# Set resolution to 720p (1280x720)
CAPTURE_FPS = 30
image_width = 1280
image_height = 720

roi_w = 100
roi_h = 100
target_aspect_ratio = 4 / 3
actual_width = int(image_width / target_aspect_ratio)

# Streamlit settings
st.title("neuflow v2")
stframe1 = st.empty()  # Placeholder for original frames
stframe2 = st.empty()  # Placeholder for flow images


#Load Camera Calibration File
def load_calibration_data(file_path):
    with open(file_path, "r") as f:
        calibration_data = json.load(f)
    K = np.array(calibration_data["K"])  # Convert list back to numpy array
    D = np.array(calibration_data["D"])
    image_size = tuple(calibration_data["image_size"])
    print(f"Calibration data loaded from {file_path}")
    return K, D, image_size

def get_cuda_image_from_frame(frame):
    frame = cv2.resize(frame, (roi_w, roi_h))
    image = torch.from_numpy(frame).permute(2, 0, 1).half()
    return image[None].cuda()

def smooth_flow(flow, kernel_size=3): #gaussian blur
    flow = flow.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    smoothed_flow = F.avg_pool2d(flow, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
    return smoothed_flow.squeeze(0).squeeze(0)  # Remove batch and channel dimensions

def apply_temporal_smoothing(current, previous, alpha=0.2):
    return alpha * current + (1 - alpha) * previous

def crop(frame):
    h, w, _ = frame.shape
    target_width = int(w / target_aspect_ratio)
    if w > target_width:
        x_offset = (w - target_width) // 2
        cropped_frame = frame[:, x_offset:x_offset + target_width]
    else:
        cropped_frame = frame
    return cropped_frame

def get_center_roi(frame, roi_width, roi_height):
    """
    Extracts a rectangular ROI centered in the frame.

    Parameters:
        frame (numpy.ndarray): Input image (H, W, C).
        roi_width (int): Width of the ROI.
        roi_height (int): Height of the ROI.

    Returns:
        numpy.ndarray: Cropped ROI from the frame.
    """
    h, w, _ = frame.shape  # Get frame dimensions
    
    # Calculate the top-left corner of the ROI
    x_start = max((w - roi_width) // 2, 0)
    y_start = max((h - roi_height) // 2, 0)

    # Calculate the bottom-right corner of the ROI
    x_end = min(x_start + roi_width, w)
    y_end = min(y_start + roi_height, h)

    # Extract the ROI
    roi = frame[y_start:y_end, x_start:x_end]

    return roi

def fuse_conv_and_bn(conv, bn):
    fusedconv = torch.nn.Conv2d(
        conv.in_channels, conv.out_channels, kernel_size=conv.kernel_size,
        stride=conv.stride, padding=conv.padding, dilation=conv.dilation,
        groups=conv.groups, bias=True).requires_grad_(False).to(conv.weight.device)
    
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    b_conv = torch.zeros(conv.weight.shape[0], device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)
    return fusedconv


previous_motion_x = None
previous_motion_y = None
previous_motion_mag = None
previous_motion_dir = None

def extract_motion(flow):
    global previous_motion_x, previous_motion_y, previous_motion_mag, previous_motion_dir
    """
    Extracts the X and Y components from the optical flow and computes motion magnitude and direction.
    Applies spatial and temporal smoothing to all outputs.

    Returns:
        smoothed_motion_x (torch.Tensor): Smoothed horizontal (X) flow motion.
        smoothed_motion_y (torch.Tensor): Smoothed vertical (Y) flow motion.
        smoothed_motion_magnitude (torch.Tensor): Smoothed magnitude of motion.
        smoothed_motion_direction (torch.Tensor): Smoothed direction of motion (in radians).
    """

    # Extract X and Y flow components directly from a 3D flow tensor (H, W, 2)
    motion_x = flow[0, :, :]  # Horizontal flow (X direction)
    motion_y = flow[1, :, :]  # Vertical flow (Y direction)

    # Apply spatial smoothing
    motion_x = smooth_flow(motion_x)
    motion_y = smooth_flow(motion_y)

    # Calculate motion magnitude and direction
    motion_magnitude = torch.sqrt(motion_x ** 2 + motion_y ** 2)  # Magnitude of motion
    motion_direction = torch.atan2(motion_y, motion_x)  # Direction in radians

    # Initialize previous motion tensors if None
    if previous_motion_x is None:
        previous_motion_x = motion_x.clone()
        previous_motion_y = motion_y.clone()
        previous_motion_mag = motion_magnitude.clone()
        previous_motion_dir = motion_direction.clone()

    # Apply temporal smoothing to motion components
    smoothed_motion_x = apply_temporal_smoothing(motion_x, previous_motion_x)
    smoothed_motion_y = apply_temporal_smoothing(motion_y, previous_motion_y)

    # Apply temporal smoothing to magnitude and direction
    smoothed_motion_magnitude = apply_temporal_smoothing(motion_magnitude, previous_motion_mag)
    smoothed_motion_direction = apply_temporal_smoothing(motion_direction, previous_motion_dir)

    # Update previous motion tensors
    previous_motion_x = smoothed_motion_x.clone()
    previous_motion_y = smoothed_motion_y.clone()
    previous_motion_mag = smoothed_motion_magnitude.clone()
    previous_motion_dir = smoothed_motion_direction.clone()

    return smoothed_motion_x, smoothed_motion_y, smoothed_motion_magnitude, smoothed_motion_direction



# Initialize model and load checkpoint
device = torch.device('cuda')
model = NeuFlow().to(device)
checkpoint = torch.load('neuflow_mixed.pth', map_location='cuda')
model.load_state_dict(checkpoint['model'], strict=True)

for m in model.modules():
    if isinstance(m, ConvBlock):
        m.conv1 = fuse_conv_and_bn(m.conv1, m.norm1)
        m.conv2 = fuse_conv_and_bn(m.conv2, m.norm2)
        delattr(m, "norm1")
        delattr(m, "norm2")
        m.forward = m.forward_fuse

model.eval()
model.half()

#model.init_bhwd(1, image_height, actual_width, 'cuda')
model.init_bhwd(1, roi_h, roi_w, 'cuda')

previous_frame = None

# Open the video device (e.g., webcam or external camera). Replace 0 with the appropriate device ID if needed.
# v4l2-ctl --device=/dev/video0 --list-formats-ext   MJPG mode offers higher framerates up to 60hz, but manual says 30hz max up to 4k
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FPS, CAPTURE_FPS)
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
#chart = st.line_chart()  # Initialize an empty line chart
chart_alt = st.empty() 
fps_display = st.sidebar.empty()
x_motion_display = st.sidebar.empty()
y_motion_display = st.sidebar.empty()
mag_motion_display = st.sidebar.empty()
rot_motion_display = st.sidebar.empty()

K,D,image_size = load_calibration_data("../hdzero_eco_960x720.json")
map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, image_size, cv2.CV_16SC2)

motion_data_buffer = deque(maxlen=50)
last_update = time.time()
last_fps_update = time.time()
frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        st.error("Error: Failed to capture image.")
        break
    
    undistorted_frame = cv2.remap(crop(frame), map1, map2, interpolation=cv2.INTER_LINEAR)
    current_frame = get_center_roi(undistorted_frame, roi_width=roi_w, roi_height=roi_h)
    
    #current_frame = crop(frame)
    #print(current_frame.shape)

    if previous_frame is not None:
        image_0 = get_cuda_image_from_frame(previous_frame)
        image_1 = get_cuda_image_from_frame(current_frame)
        #print(f"image_0 shape: {image_0.shape}")
        #print(f"image_1 shape: {image_1.shape}")

        with torch.no_grad():
            flow = model(image_0, image_1)[-1][0]
            flow2 = flow.permute(1,2,0)

            flow_image = flow_viz.flow_to_image_gpu(flow2)
            stframe2.image(flow_image, caption="Optical Flow Frame")

            ##stframe1.image(previous_frame, channels="BGR", caption="og frame")
            motion_x, motion_y, motion_magnitude, motion_rotation = extract_motion(flow)

        frame_count += 1
        if time.time() > last_fps_update + 1:
            fps = frame_count / (time.time() - last_fps_update)
            last_fps_update = time.time()
            frame_count = 0
            fps_display.metric("FPS", f"{fps:.2f}")

        if time.time() > last_update + 0.01:
            last_update = time.time()
            x = torch.mean(motion_x).item()
            y = torch.mean(motion_y).item()
            mag = torch.mean(motion_magnitude).item()
            rot = torch.mean(motion_rotation).item()
            #x_motion_display.metric("X Flow", f"{x:.2f}")
            #y_motion_display.metric("Y Flow", f"{y:.2f}")
            #mag_motion_display.metric("Magitude", f"{mag:.2f}")
            #rot_motion_display.metric("Rotation", f"{rot:.2f}")

            motion_data_buffer.append({'motion_x': x, 'motion_y': y, 'mag': mag})
            #motion_data_buffer.append({'motion_x': x, 'motion_y': y})
            motion_df = pd.DataFrame(motion_data_buffer)
            
            fixed_domain = [-1, 1]
            motion_df['mag'] = motion_df['mag'].clip(*fixed_domain)
            #motion_df['rot'] = motion_df['rot'].clip(*fixed_domain)
            # Ensure the DataFrame has a "time" column for the x-axis
            motion_df['time'] = range(len(motion_df))
            chart = alt.Chart(motion_df).transform_fold(
                ['motion_x', 'motion_y', 'mag'], as_=['motion_type', 'value']
            ).mark_line().encode(
                x=alt.X('time:Q', title='Time'),
                y=alt.Y('value:Q', title='Motion Value', scale=alt.Scale(domain=[-1, 1])),  # Fixed y-axis
                color='motion_type:N'
            ).properties(
                width=800,
                height=400
            )
            chart_alt.altair_chart(chart, use_container_width=True)
        
    previous_frame = current_frame

cap.release()
cv2.destroyAllWindows()
