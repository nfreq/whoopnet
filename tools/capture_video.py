import cv2
import numpy as np
import json
import signal

image_width = 1280
image_height = 720
target_aspect_ratio = 4 / 3
frame_rate = 30

runtime_exec = True
def signal_handler(sig, frame):
    global runtime_exec
    print("\nExiting gracefully.")
    runtime_exec = False
signal.signal(signal.SIGINT, signal_handler)
    


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

K,D,image_size = load_calibration_data("hdzero_eco_960x720.json")
map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, image_size, cv2.CV_16SC2)

# Initialize the VideoWriter
output_filename = "fpvai_camera_record_video.avi"  # Change to your desired output path
fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # Use XVID or other suitable codec
output_fps = frame_rate
output_size = (image_size[0], image_size[1])  # Ensure it matches undistorted frame size
#out = cv2.VideoWriter(output_filename, fourcc, output_fps, output_size)
out = cv2.VideoWriter(output_filename, fourcc, output_fps, (1280, 720))
assert out.isOpened()


print(f"Recording video to {output_filename}. Ctrl-C to stop.")

while runtime_exec:
    try:
        ret, frame = cap.read()
    except:
        print("capture exception")

    #current_frame = crop(frame)
    #undistorted_img = cv2.remap(current_frame, map1, map2, interpolation=cv2.INTER_LINEAR)
    #downscaled_frame = cv2.resize(undistorted_img, (256, 256), interpolation=cv2.INTER_AREA)
    out.write(frame)
   

cap.release()
cv2.destroyAllWindows()

