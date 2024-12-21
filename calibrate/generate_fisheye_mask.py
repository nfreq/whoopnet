import cv2
import numpy as np

image = np.zeros((720, 960), dtype=np.uint8)
cv2.circle(image, (480, 360), 480, 255, -1)
cv2.imwrite("mask.jpg", image)