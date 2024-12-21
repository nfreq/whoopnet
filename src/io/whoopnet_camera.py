from whoopnet_node import WhoopnetNode
import rclpy
import cv2
import numpy as np
import json
import signal
import easyocr
import math

runtime_exec = True

image_width = 1280
image_height = 720
target_aspect_ratio = 4 / 3
frame_rate = 30
target_width = int(image_width / target_aspect_ratio)

reader = easyocr.Reader(['en'])

def extract_timestamp(frame):
    # OSD Timestamp field crop size should be 200x20px

    # Crop the bottom-left timestamp region
    height, width = frame.shape[:2]
    cropped = frame[height - 20:height, 230:430]  # Adjust coordinates as needed

    # Convert to grayscale for better OCR
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    resized = cv2.resize(binary, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # Perform OCR on the grayscale image
    results = reader.readtext(resized, detail=0, allowlist='0123456789')

    # Extract and return the first detected numeric string
    timestamp = float(results[0]) if results else 0
    #cv2.imwrite('ocr.jpg', frame)
    cv2.imwrite('ocr_cropped.jpg', resized)
    return timestamp

def within_percentage(a, b, threshold=0.1):  # threshold is 10% by default
    return abs(a - b) / max(a, b) <= threshold

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

#K,D,image_size = load_calibration_data("hdzero_eco_960x720.json")
#map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, image_size, cv2.CV_16SC2)

rclpy.init()
ros_node = WhoopnetNode()

target_width, x_offset = compute_crop_params(image_width, image_height, target_aspect_ratio)

prev_timestamp = 0.0
pred_timestamp = 0.0
dt_conseq_count= 0
while runtime_exec:
    while rclpy.ok() and runtime_exec:
        ret, frame = cap.read()

        timestamp = extract_timestamp(frame)

        if timestamp != 0:
            if pred_timestamp == 0:
                pred_timestamp = timestamp

            dt = timestamp - prev_timestamp
            if dt == 33 or dt == 34:        # 30hz  1/30 = 0.033-->
                dt_conseq_count += 1
            else:
                dt_conseq_count = 0

            if dt_conseq_count >= 5:
                dt_conseq_count = 0
                if within_percentage(pred_timestamp, timestamp, threshold=0.1):
                    pred_timestamp = timestamp
            else:
                pred_timestamp += 33.333333

            #print(f"Extracted tstamp: {timestamp} - dT {timestamp - prev_timestamp} - predicted: {pred_timestamp}")

            prev_timestamp = timestamp

        #cropped_frame = crop(frame, target_width, x_offset)
        #undistorted_img = cv2.remap(cropped_frame, map1, map2, interpolation=cv2.INTER_LINEAR)
        #downscaled_frame = cv2.resize(undistorted_img, (256, 256), interpolation=cv2.INTER_AREA)

        ros_node.publish_camera_raw_feed(frame, math.ceil(timestamp))
        #ros_node.publish_camera_feed(undistorted_img)
        rclpy.spin_once(ros_node, timeout_sec=0.0)        

cap.release() 
cv2.destroyAllWindows()