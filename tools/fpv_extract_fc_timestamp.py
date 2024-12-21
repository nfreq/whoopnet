import cv2
import signal
import easyocr

image_width = 1280
image_height = 720
target_aspect_ratio = 4 / 3
frame_rate = 30

target_width = int(image_width / target_aspect_ratio)

runtime_exec = True
def signal_handler(sig, frame):
    global runtime_exec
    print("\nExiting gracefully.")
    runtime_exec = False
signal.signal(signal.SIGINT, signal_handler)
    
reader = easyocr.Reader(['en'])

def extract_timestamp(frame):
    # OSD Timestamp "craftname" field crop size should be ~200x20px
    height, width = frame.shape[:2]
    cropped = frame[height - 20:height, 230:430]  # Adjust coordinates as needed
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    resized = cv2.resize(binary, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    results = reader.readtext(resized, detail=0, allowlist='0123456789')
    timestamp = float(results[0]) if results else 0
    #cv2.imwrite('ocr.jpg', frame)
    cv2.imwrite('ocr_cropped.jpg', resized)
    return timestamp

def within_percentage(a, b, threshold=0.1):  # 10% relative
    return abs(a - b) / max(a, b) <= threshold

# v4l2-ctl --device=/dev/video0 --list-formats-ext
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FPS, frame_rate)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, image_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, image_height)
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Video Capture FPS: {fps}")

prev_timestamp = 0.0
pred_timestamp = 0.0
dt_conseq_count= 0
while runtime_exec:
    try:
        ret, frame = cap.read()
    except:
        print("capture exception")

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

        print(f"Extracted tstamp: {timestamp} - dT {timestamp - prev_timestamp} - predicted: {pred_timestamp}")
        prev_timestamp = timestamp

cap.release()
cv2.destroyAllWindows()