import sys
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import streamlit as st
import time

# Add the path to the `DeepLabV3Plus-Pytorch` directory to the Python path
sys.path.append('./DeepLabV3Plus-Pytorch')  # Replace with the actual path to the cloned repository

# Import the custom network module
from network import modeling
from datasets import VOCSegmentation, Cityscapes

# Define model parameters
MODEL_NAME = "deeplabv3plus_resnet101"  # Replace with the actual model name in `modeling.py`
NUM_CLASSES = 19  # Replace this with the number of classes used (e.g., 19 for Cityscapes)
OUTPUT_STRIDE = 16  # Replace this with the output stride used during training

# Create the model using the custom implementation
model = modeling.__dict__[MODEL_NAME](num_classes=NUM_CLASSES, output_stride=OUTPUT_STRIDE)

# Load the model weights from the `.pth` file
model_weights_path = "./deeplabv3-models/best_deeplabv3plus_resnet101_cityscapes_os16.pth.tar"  # Replace with the path to your `.pth` file
state_dict = torch.load(model_weights_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))['model_state']

# Load the state dictionary into the model
model.load_state_dict(state_dict)

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Streamlit settings
st.title("DeepLabV3+ Real-Time Segmentation")
stframe = st.empty()  # Placeholder for video frames

# Define Cityscapes categories
CITYSCAPES_CLASSES = [
    ('unlabeled', 0),
    ('ego vehicle', 1),
    ('rectification border', 2),
    ('out of roi', 3),
    ('static', 4),
    ('dynamic', 5),
    ('ground', 6),
    ('road', 7),
    ('sidewalk', 8),
    ('parking', 9),
    ('rail track', 10),
    ('building', 11),
    ('wall', 12),
    ('fence', 13),
    ('guard rail', 14),
    ('bridge', 15),
    ('tunnel', 16),
    ('pole', 17),
    ('polegroup', 18),
    ('traffic light', 19),
    ('traffic sign', 20),
    ('vegetation', 21),
    ('terrain', 22),
    ('sky', 23),
    ('person', 24),
    ('rider', 25),
    ('car', 26),
    ('truck', 27),
    ('bus', 28),
    ('caravan', 29),
    ('trailer', 30),
    ('train', 31),
    ('motorcycle', 32),
    ('bicycle', 33),
    ('license plate', -1)
]

CITYSCAPES_CATEGORIES = {idx: name for name, idx in CITYSCAPES_CLASSES}

# Allow user to select categories to display
displayed_categories = st.multiselect(
    'Select categories to display:',
    options=[cls[1] for cls in CITYSCAPES_CLASSES if cls[1] >= 0],
    format_func=lambda x: CITYSCAPES_CATEGORIES[x],
    default=[7, 23]  # Default to 'road' and 'sky'
)

# If no categories are selected, show all categories
if not displayed_categories:
    displayed_categories = [cls[1] for cls in CITYSCAPES_CLASSES if cls[1] >= 0]

# Initialize the camera
if 'cap' not in st.session_state:
    st.session_state.cap = cv2.VideoCapture(0)
    st.session_state.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    st.session_state.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not st.session_state.cap.isOpened():
    st.error("Error: Could not open video device.")
    st.stop()

# Preprocessing transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define decode function for Cityscapes
decode_fn = Cityscapes.decode_target

fps_display = st.empty()  # Placeholder for FPS display
last_fps_update = time.time()
frame_count = 0
while True:
    ret, frame = st.session_state.cap.read()
    if not ret:
        st.error("Error: Failed to capture image.")
        break

    # Convert frame to PIL Image
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Preprocess the image
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Perform inference
    with torch.no_grad():
        output = model(input_tensor)

    # Get the predicted class for each pixel
    preds = output.max(1)[1].detach().cpu().numpy()

    # Filter the predictions to show only certain categories
    filtered_preds = np.zeros_like(preds[0])
    for category in displayed_categories:
        filtered_preds[preds[0] == category] = category

    # Colorize the filtered segmentation maps
    colorized_preds = decode_fn(filtered_preds).astype('uint8')
    colorized_image = Image.fromarray(colorized_preds)  # to PIL Image

    # Annotate the segments
    draw = ImageDraw.Draw(colorized_image)
    font = ImageFont.truetype("DejaVuSans-Bold.ttf", 24)
    unique_categories = np.unique(filtered_preds)
    for category in unique_categories:
        if category in CITYSCAPES_CATEGORIES:
            mask = (filtered_preds == category).astype(np.uint8)
            ys, xs = np.where(mask)
            if len(xs) > 0 and len(ys) > 0:
                x, y = int(np.mean(xs)), int(np.mean(ys))
                draw.text((x, y), CITYSCAPES_CATEGORIES[category], fill=(255, 255, 255), font=font)

    # Resize the colorized image back to the original size
    colorized_image_upscaled = colorized_image.resize((frame.shape[1], frame.shape[0]), Image.NEAREST)

    # Convert back to OpenCV format and display
    colorized_frame = cv2.cvtColor(np.array(colorized_image_upscaled), cv2.COLOR_RGB2BGR)


    # Display the segmented frame in Streamlit
    stframe.image(colorized_frame, channels="BGR")

    frame_count += 1
    if time.time() > last_fps_update + 1:
        fps = frame_count / (time.time() - last_fps_update)
        last_fps_update = time.time()
        frame_count = 0
        fps_display.metric("FPS", f"{fps:.2f}")

# Release the camera
st.session_state.cap.release()
cv2.destroyAllWindows()
