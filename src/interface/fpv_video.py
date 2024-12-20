from whoopnet.src.fpv_node import FpvNode
import rclpy
import cv2
import numpy as np
import json
import signal

runtime_exec = True

image_width = 1280
image_height = 720
target_aspect_ratio = 4 / 3
frame_rate = 30
target_width = int(image_width / target_aspect_ratio)


def load_calibration_data(file_path):
    with open(file_path, "r") as f:
        calibration_data = json.load(f)
    K = np.array(calibration_data["K"])
    D = np.array(calibration_data["D"])
    image_size = tuple(calibration_data["image_size"])
    print(f"Calibration data loaded from {file_path}")
    return K, D, image_size

def compute_crop_params(image_width, image_height, target_aspect_ratio):
    target_width = int(image_height * target_aspect_ratio)
    x_offset = max((image_width - target_width) // 2, 0)
    return target_width, x_offset

def crop(frame, target_width, x_offset):
    return frame[:, x_offset:x_offset + target_width]

def signal_handler(sig, frame):
    global runtime_exec
    print("\nSIGINT received. Exiting gracefully...")
    runtime_exec = False
signal.signal(signal.SIGINT, signal_handler)


# v4l2-ctl --device=/dev/video0 --list-formats-ext
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FPS, frame_rate)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, image_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, image_height)
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Video Capture FPS: {fps}")

K,D,image_size = load_calibration_data("hdzero_eco_960x720.json")
map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, image_size, cv2.CV_16SC2)

rclpy.init()
ros_node = FpvNode()

target_width, x_offset = compute_crop_params(image_width, image_height, target_aspect_ratio)

while runtime_exec:
    while rclpy.ok() and runtime_exec:
        ret, frame = cap.read()

        #cropped_frame = crop(frame, target_width, x_offset)
        #undistorted_img = cv2.remap(cropped_frame, map1, map2, interpolation=cv2.INTER_LINEAR)
        #downscaled_frame = cv2.resize(undistorted_img, (256, 256), interpolation=cv2.INTER_AREA)

        ros_node.publish_camera_raw_feed(frame)
        #ros_node.publish_camera_feed(undistorted_img)
        rclpy.spin_once(ros_node, timeout_sec=0.0)        

cap.release() 
cv2.destroyAllWindows()