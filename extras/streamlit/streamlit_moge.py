import cv2
import streamlit as st
import time
from PIL import Image
import numpy as np
import torch
from moge.model import MoGeModel

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

def normalize_and_convert_to_image(data):
    if isinstance(data, torch.Tensor):
        data = data.numpy()  # Convert PyTorch tensor to NumPy array
    elif not isinstance(data, np.ndarray):
        data = np.array(data)  # Convert to NumPy array if needed

    # Normalize to the range [0, 255]
    normalized_data = (255 * (data - np.min(data)) / (np.max(data) - np.min(data))).astype(np.uint8)
    return cv2.applyColorMap(normalized_data, cv2.COLORMAP_JET)

device = torch.device("cuda")

# Load the model from huggingface hub (or load from local).
model = MoGeModel.from_pretrained("Ruicheng/moge-vitl").to(device)                             


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
    #downscaled_frame = cv2.resize(current_frame, (128, 128), interpolation=cv2.INTER_AREA)

    # Read the input image and convert to tensor (3, H, W) and normalize to [0, 1]
    #input_image = cv2.cvtColor(cv2.imread("PATH_TO_IMAGE.jpg"), cv2.COLOR_BGR2RGB)                       
    input_image = torch.tensor(current_frame / 255, dtype=torch.float32, device=device).permute(2, 0, 1)    
    output = model.infer(input_image)

    depth_image = normalize_and_convert_to_image(output["depth"])
    mask_image = (output["mask"] * 255).astype(np.uint8)

    # Visualize point map as an image
    # For visualization, let's extract the x, y, z channels separately and normalize them
    x_map = normalize_and_convert_to_image(output["points"][..., 0])
    y_map = normalize_and_convert_to_image(output["points"][..., 1])
    z_map = normalize_and_convert_to_image(output["points"][..., 2])

    point_map_image = cv2.merge([x_map, y_map, z_map])


    stframe2.image(depth_image, caption="Depth Map", use_column_width=True)
    #stframe2.image(mask_image, caption="Mask", use_column_width=True)
    #stframe2.image(point_map_image, caption="Point Map", use_column_width=True)
    

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
