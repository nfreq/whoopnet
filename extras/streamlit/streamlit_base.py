import cv2
import streamlit as st
import time

def crop(frame, target_width=960, target_height=720):
    # Get the original dimensions of the frame
    h, w, _ = frame.shape

    # Ensure the original frame has the expected height (720) and width (1280)
    if w == image_width and h == image_height:
        # Calculate the crop for the left and right sides (80px each)
        x_offset = (w - target_width) // 2
        cropped_frame = frame[:, x_offset:x_offset + target_width]  # Crop 80px from each side
    else:
        raise ValueError(f"Unexpected frame dimensions: {w}x{h}. Expected 1280x720.")

    # Return the cropped frame (960x720), no need to resize as target dimensions match the cropped size
    return cropped_frame

image_width = 1280
image_height = 720

st.title("Video Stream")

stframe1 = st.empty()  # Placeholder for original frames
stframe2 = st.empty()  # Placeholder for flow images
# Open the video device (e.g., webcam or external camera). Replace 0 with the appropriate device ID if needed.
# v4l2-ctl --device=/dev/video0 --list-formats-ext   MJPG mode offers higher framerates up to 60hz, but manual says 30hz max up to 4k
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FPS, 30)
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
chart = st.line_chart()  # Initialize an empty line chart
fps_display = st.sidebar.empty()

last_update = time.time()
last_fps_update = time.time()
frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        st.error("Error: Failed to capture image.")
        break

    current_frame = crop(frame)
    downscaled_frame = cv2.resize(current_frame, (400, 400), interpolation=cv2.INTER_AREA)
    stframe2.image(downscaled_frame, caption="Optical Flow Frame", channels="BGR")

    frame_count += 1
    if time.time() > last_fps_update + 1:
        fps = frame_count / (time.time() - last_fps_update)
        last_fps_update = time.time()
        frame_count = 0
        fps_display.metric("FPS", f"{fps:.2f}")

    if time.time() > last_update + 0.1:
        last_update = time.time()
        pass
    time.sleep(0.01)

cap.release()
cv2.destroyAllWindows()
