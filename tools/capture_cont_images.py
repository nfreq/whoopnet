import cv2
import os
import time

def capture_images(camera_device, output_dir, capture_interval=10):
    os.makedirs(output_dir, exist_ok=True)

    # Determine the starting index by checking existing images
    existing_files = [f for f in os.listdir(output_dir) if f.startswith("image_") and f.endswith(".jpg")]
    if existing_files:
        existing_indices = [int(f.split("_")[1].split(".")[0]) for f in existing_files]
        image_index = max(existing_indices) + 1
    else:
        image_index = 0

    # Open the camera device
    cap = cv2.VideoCapture(camera_device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    if not cap.isOpened():
        print(f"Error: Unable to open camera device {camera_device}.")
        return

    print(f"Starting capture from {camera_device}. Press Ctrl+C to stop.")

    try:
        last_save_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame. Exiting.")
                break

            current_time = time.time()
            if current_time - last_save_time >= capture_interval:
                filename = os.path.join(output_dir, f"image_{image_index:04d}.jpg")
                cv2.imwrite(filename, frame)
                print(f"saved {filename}")
                last_save_time = current_time
                image_index += 1

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    camera_device = 0  # /dev/video0
    output_dir = "images"
    capture_interval = 1
    capture_images(camera_device, output_dir, capture_interval)
