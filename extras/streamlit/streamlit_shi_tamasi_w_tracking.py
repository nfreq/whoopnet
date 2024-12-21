import torch
import cv2
import numpy as np
import os
import streamlit as st
import time
import copy as cp
import json
from scipy.spatial import cKDTree


# Set resolution to 720p (1280x720)
image_width = 1280
image_height = 720

# Streamlit settings
st.title("Corner Detection Experimentation")

stframe1 = st.empty()  # Placeholder for Harris corners
stframe2 = st.empty()  # Placeholder for shi-tamasi corners
stframe3 = st.empty()  # sillouette

max_corners = st.number_input("Max Corners", value=100, min_value=0, max_value=2000, step=10)
quality_level = st.number_input("Quality Level", value=0.1, min_value=0.0, max_value=1.0, step=0.01)
min_distance = st.number_input("Min Distance", value=10, min_value=0, max_value=100, step=1)
tracking_thr = st.number_input("Tracking Threshold", value=10, min_value=0, max_value=100, step=1)
block_size = st.number_input("Block Size", value=10, min_value=0, max_value=100, step=1)

prev_frame = None
tracked_features = {}

#Load Camera Calibration File
def load_calibration_data(file_path):
    with open(file_path, "r") as f:
        calibration_data = json.load(f)
    K = np.array(calibration_data["K"])  # Convert list back to numpy array
    D = np.array(calibration_data["D"])
    image_size = tuple(calibration_data["image_size"])
    print(f"Calibration data loaded from {file_path}")
    return K, D, image_size


def crop(frame, target_width=960, target_height=720):
    # Get the original dimensions of the frame
    h, w, _ = frame.shape

    # Ensure the original frame has the expected height (720) and width (1280)
    if w == 1280 and h == 720:
        # Calculate the crop for the left and right sides (80px each)
        x_offset = (w - target_width) // 2
        cropped_frame = frame[:, x_offset:x_offset + target_width]  # Crop 80px from each side
    else:
        raise ValueError(f"Unexpected frame dimensions: {w}x{h}. Expected 1280x720.")

    # Return the cropped frame (960x720), no need to resize as target dimensions match the cropped size
    return cropped_frame


def shi_tomasi_with_optimized_tracking(
    image, tracked_features, prev_frame=None, distance_threshold=15, max_displacement=50, tracking_threshold=5, max_tracked_features=500
):
    """
    Detect corners and track their lifespan with better handling of dynamic movement.

    :param image: Input frame (BGR).
    :param tracked_features: Dictionary of tracked features with their lifespan.
    :param prev_frame: Previous frame for optical flow tracking.
    :param distance_threshold: Maximum distance for feature matching.
    :param max_displacement: Maximum allowed movement for valid features.
    :param tracking_threshold: Number of frames a feature must be tracked to change color.
    :param max_tracked_features: Maximum number of features to track at any given time.
    :return: Processed image, updated tracking dictionary, current gray frame.
    """
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    new_tracked_features = {}

    # Optical flow prediction if previous frame exists
    if prev_frame is not None and tracked_features:
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        prev_points = np.array(list(tracked_features.keys()), dtype=np.float32).reshape(-1, 1, 2)
        next_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray_img, prev_points, None)

        valid_features = []
        for i, (new, old) in enumerate(zip(next_points, prev_points)):
            if status[i]:  # Successfully tracked
                displacement = np.linalg.norm(new - old)
                if displacement < max_displacement:  # Movement threshold
                    valid_features.append((new.ravel(), old.ravel()))

        print(f"Valid features after filtering: {len(valid_features)} of {len(next_points)}")

        # Update lifespans for valid features
        for new, old in valid_features:
            x, y = new
            old_x, old_y = old
            lifespan = tracked_features[(old_x, old_y)] + 1
            new_tracked_features[(x, y)] = lifespan

    # Shi-Tomasi corner detection
    corners_img = cv2.goodFeaturesToTrack(
        gray_img, maxCorners=100, qualityLevel=0.1, minDistance=10, blockSize=3
    )
    print(f"Detected corners: {len(corners_img) if corners_img is not None else 0}")

    # Add new features
    if corners_img is not None:
        for corner in corners_img:
            x, y = corner.ravel()
            if (x, y) not in new_tracked_features:
                new_tracked_features[(x, y)] = 1  # New feature

    # Remove stale features
    for (x, y), lifespan in tracked_features.items():
        if (x, y) not in new_tracked_features:
            # Decay lifespan for unmatched features
            new_tracked_features[(x, y)] = max(lifespan - 1, 0)

    # Limit the number of tracked features
    if len(new_tracked_features) > max_tracked_features:
        # Sort by lifespan and keep the top N
        new_tracked_features = dict(
            sorted(new_tracked_features.items(), key=lambda item: item[1], reverse=True)[:max_tracked_features]
        )

    # Reset tracking if no features remain
    if len(new_tracked_features) == 0:
        print("Resetting tracking: No features left.")
        new_tracked_features = {}

    # Draw features with lifespan-based colors
    for (x, y), lifespan in new_tracked_features.items():
        color = (0, 255, 0) if lifespan >= tracking_threshold else (255, 255, 0)
        cv2.circle(image, (int(x), int(y)), 3, color, -1)

    print(f"Tracked features after cleanup: {len(new_tracked_features)}")

    return image, new_tracked_features




def shi_tomasi_with_optimized_tracking_old(image, tracked_features, prev_frame, distance_threshold=20, max_displacement=10,tracking_threshold=5):
    """
    Detect corners and track their lifespan with robust matching and debugging.

    :param image: Input frame (BGR).
    :param tracked_features: Dictionary of tracked features with their lifespan.
    :param prev_frame: Previous frame for optical flow tracking.
    :param distance_threshold: Maximum distance for feature matching.
    :param tracking_threshold: Number of frames a feature must be tracked to change color.
    :return: Processed image, updated tracking dictionary, current gray frame.
    """
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    new_tracked_features = {}

    # Optical flow prediction if previous frame exists
    if prev_frame is not None and tracked_features:
        print("process optical flow")
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        prev_points = np.array(list(tracked_features.keys()), dtype=np.float32).reshape(-1, 1, 2)
        next_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray_img, prev_points, None)

        print(f"Optical flow status: {status.sum()} of {len(status)} features matched.")

        # Filter features based on displacement
        valid_features = []
        for i, (new, old) in enumerate(zip(next_points, prev_points)):
            if status[i]:  # Successfully tracked
                displacement = np.linalg.norm(new - old)
                if displacement < max_displacement:  # Movement threshold
                    valid_features.append((new, old))
        print(f"Valid features after filtering: {len(valid_features)} of {len(next_points)}")

         # Update lifespans for valid features
        for new, old in valid_features:
            x, y = new.ravel()
            old_x, old_y = old.ravel()
            lifespan = tracked_features[(old_x, old_y)] + 1
            new_tracked_features[(x, y)] = lifespan

    # Shi-Tomasi corner detection
    corners_img = cv2.goodFeaturesToTrack(gray_img, max_corners, quality_level, min_distance, blockSize=block_size)
    #print(f"Detected corners: {len(corners_img) if corners_img is not None else 0}")

    # KD-tree for spatial matching
    if new_tracked_features:
        kd_tree = cKDTree(list(new_tracked_features.keys()))
    else:
        kd_tree = None

    for corner in corners_img:
        x, y = corner.ravel().astype(int)
        if kd_tree is not None:
            dist, idx = kd_tree.query((x, y), distance_upper_bound=distance_threshold)
            if dist != np.inf:  # Match found
                matched_x, matched_y = list(new_tracked_features.keys())[idx]
                new_tracked_features[(x, y)] = new_tracked_features.pop((matched_x, matched_y)) + 1
            else:
                new_tracked_features[(x, y)] = 1  # New feature
        else:
            new_tracked_features[(x, y)] = 1  # New feature

    
    # Draw features with lifespan-based colors
    for (x, y), lifespan in new_tracked_features.items():
        color = (0, 255, 0) if lifespan >= tracking_threshold else (255, 255, 0)
        cv2.circle(image, (int(x), int(y)), 3, color, -1)
        #if (lifespan > 1):
        #    print(f"Feature at ({x}, {y}) has lifespan: {lifespan}")
    return image, new_tracked_features


# Open the video device (e.g., webcam or external camera). Replace 0 with the appropriate device ID if needed.
# v4l2-ctl --device=/dev/video0 --list-formats-ext   MJPG mode offers higher framerates up to 60hz, but manual says 30hz max up to 4k

# Initialize the camera (only if it's not already initialized)
if 'cap' not in st.session_state:
    st.session_state.cap = cv2.VideoCapture(0)
    st.session_state.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    st.session_state.cap.set(cv2.CAP_PROP_FPS, 20)
    st.session_state.cap.set(cv2.CAP_PROP_FRAME_WIDTH, image_width)
    st.session_state.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, image_height)
    fps = st.session_state.cap.get(cv2.CAP_PROP_FPS)
    st.write(f"Video Capture Device FPS: {fps}")

# Check if the video device is opened successfully
if not st.session_state.cap.isOpened():
    st.error("Error: Could not open video device.")
    st.stop()

K,D,image_size = load_calibration_data("hdzero_eco_960x720.json")
map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, image_size, cv2.CV_16SC2)



last_chart_update = time.time()
last_fps_update = time.time()
frame_count = 0

while True:
    #ret, frame = cap.read()
    ret, frame = st.session_state.cap.read()
    if not ret:
        st.error("Error: Failed to capture image.")
        break

    
    current_frame = crop(frame)
    undistorted_frame = cv2.remap(current_frame, map1, map2, interpolation=cv2.INTER_LINEAR)

    # Detect and track corners
    shitomasi_image, tracked_features = shi_tomasi_with_optimized_tracking(
        undistorted_frame,
        tracked_features,
        prev_frame,
        distance_threshold=min_distance,
        tracking_threshold=tracking_thr
    )

    prev_frame = undistorted_frame
    #stframe1.image(previous_frame, channels="BGR", caption="Original Frame")
    #stframe1.image(silhouette, caption="silhouette")
    stframe2.image(shitomasi_image, channels="BGR",caption="shitomasi with tracking")


    frame_count += 1
    if time.time() > last_chart_update + 1:
        last_chart_update = time.time()
        fps = frame_count / (last_chart_update - last_fps_update)  # Calculate FPS
        last_fps_update = last_chart_update # Update the last FPS measurement time
        frame_count = 0  # Reset frame count
        
        print("\033[2J\033[H", end='')  # Clear the screen and move cursor to top-left
        print(f"FPS: {fps:.2f}")

st.session_state.cap.release()
cv2.destroyAllWindows()
