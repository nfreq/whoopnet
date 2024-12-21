import torch
import cv2
import numpy as np
import os
import streamlit as st
import time
import copy as cp
import json

# Set resolution to 720p (1280x720)
image_width = 1280
image_height = 720

# Streamlit settings
st.title("Corner Detection Experimentation")

stframe1 = st.empty()  # Placeholder for Harris corners
stframe2 = st.empty()  # Placeholder for shi-tamasi corners
stframe3 = st.empty()  # sillouette

max_corners = st.number_input("Max Corners", value=400, min_value=0, max_value=2000, step=10)
quality_level = st.number_input("Quality Level", value=0.05, min_value=0.0, max_value=1.0, step=0.01)
min_distance = st.number_input("Min Distance", value=10, min_value=0, max_value=100, step=1)
block_size = st.number_input("Block Size", value=10, min_value=0, max_value=100, step=1)

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



#Finds corners : harris and shitomasi
def find_corners(img):
    img_dup = cp.copy(img)
    shitomasi,silhouette = shi_tomasi(img_dup)
    return shitomasi,silhouette




#Function: cv2.goodFeaturesToTrack(image,maxCorners, qualityLevel, minDistance[, corners[, mask[, blockSize[, useHarrisDetector[, k]]]]])
#image – Input 8-bit or floating-point 32-bit (grayscale image).
#maxCorners – You can specify the maximum no. of corners to be detected. (Strongest ones are returned if detected more than max.)
#qualityLevel – Minimum accepted quality of image corners.
#minDistance – Minimum possible Euclidean distance between the returned corners.
#corners – Output vector of detected corners.
#mask – Optional region of interest. 
#blockSize – Size of an average block for computing a derivative covariation matrix over each pixel neighborhood. 
#useHarrisDetector – Set this to True if you want to use Harris Detector with this function.
#k – Free parameter of the Harris detector (used in computing R)

def shi_tomasi(image):

    gray_img = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    
    #max_corners = 400
    #quality_level = 0.05
    #min_distance = 10

    corners_img = cv2.goodFeaturesToTrack(gray_img,max_corners,quality_level,min_distance,block_size)

    blank_img = np.zeros((image.shape[0],image.shape[1],3),np.uint8)

    for corners in corners_img:
        x, y = corners.ravel().astype(int)
        cv2.circle(image,(x,y),3,[255,255,0],-1)
        cv2.circle(blank_img,(x,y),2,[255,255,0],-1)

    return image,blank_img


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
    shitomasi,silhouette= find_corners(undistorted_frame)

    #stframe1.image(previous_frame, channels="BGR", caption="Original Frame")
    #stframe1.image(silhouette, caption="silhouette")
    stframe2.image(shitomasi, channels="BGR",caption="shitomasi")


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
