import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import numpy as np
import cv2
import torch
from PIL import Image as PILImage
from ultralytics import YOLO

class YoloV11(Node):
    def __init__(self):
        super().__init__('yolo_node')
        self.image_subscriber = self.create_subscription(
            Image,
            '/whoopnet/io/camera',
            self.image_callback,
            10
        )
        self.yolo_publisher = self.create_publisher(
            Image,
            '/whoopnet/perception/yolo',
            10
        )
        self.model = YOLO("yolo11x-seg.pt")  # load an official model (n nano, x largest)
        self.model.to('cuda:1')
        
        self.frame_skip = 1
        self.frame_count = 0

        self.get_logger().info("Yolo v11 Node initialized.")

    def image_callback(self, msg):
        try:
            self.frame_count += 1
            if self.frame_count % self.frame_skip != 0:
                return
            
            frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
            image = PILImage.fromarray(frame)
            results = self.model.track(image, device='cuda:1', verbose=False, persist=True)
            annotated_frame = results[0].plot()

            yolo_msg = Image()
            yolo_msg.header = msg.header  # Preserve the original message header
            yolo_msg.height = annotated_frame.shape[0]
            yolo_msg.width = annotated_frame.shape[1]
            yolo_msg.encoding = "rgb8"  # Assuming the annotated frame is in RGB format
            yolo_msg.step = annotated_frame.shape[1] * 3  # Width * 3 (bytes per pixel for RGB)
            yolo_msg.data = annotated_frame.tobytes()  # Convert NumPy array to byte data

            self.yolo_publisher.publish(yolo_msg)

        except Exception as e:
            self.get_logger().error(f"Failed to process image: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = YoloV11()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()