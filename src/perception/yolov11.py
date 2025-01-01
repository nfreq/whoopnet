import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
import numpy as np
import cv2
import torch
from PIL import Image as PILImage
from ultralytics import YOLO

class YoloV11(Node):
    def __init__(self):
        super().__init__('yolo_node')
        self.image_subscriber = self.create_subscription(
            CompressedImage,  # Use CompressedImage type
            '/whoopnet/io/camera_compressed',
            self.image_callback,
            10
        )
        self.yolo_publisher = self.create_publisher(
            CompressedImage,
            '/whoopnet/perception/yolo',
            10
        )
        self.model = YOLO("yolo11x-seg.pt")  # load an official model (n nano, x largest)
        self.model.to('cuda:1')
        
        self.frame_skip = 1
        self.frame_count = 0

        self.get_logger().info("Yolo v11 Node initialized.")


    def image_callback(self, msg: CompressedImage):
        try:
            # Skip frames based on frame_skip parameter
            self.frame_count += 1
            if self.frame_count % self.frame_skip != 0:
                return

            # Decode the compressed image from the message
            frame = np.frombuffer(msg.data, np.uint8)  # Convert raw bytes to numpy array
            frame_rgb = cv2.imdecode(frame, cv2.IMREAD_COLOR)  # Decode the image
            if frame_rgb is None:
                self.get_logger().error("Failed to decode compressed image")
                return

            # Convert to PIL Image for YOLO model processing
            image = PILImage.fromarray(cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB
            results = self.model.track(image, device='cuda:1', verbose=False, persist=True)

            # Annotate the frame with YOLO results
            annotated_frame = results[0].plot()

            # Encode the annotated frame as JPEG for publishing
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, 80]
            success, encoded_image = cv2.imencode('.jpg', annotated_frame, encode_params)
            if not success:
                self.get_logger().error("Failed to encode image for compressed feed")
                return

            # Create a new CompressedImage message
            yolo_msg = CompressedImage()
            yolo_msg.header.stamp = self.get_clock().now().to_msg()  # Add current timestamp
            yolo_msg.header.frame_id = "yolov11"  # Optional, update as needed
            yolo_msg.format = "jpeg"
            yolo_msg.data = encoded_image.tobytes()  # Convert numpy array to bytes

            # Publish the annotated compressed image
            self.yolo_publisher.publish(yolo_msg)

        except Exception as e:
            self.get_logger().error(f"Error in image_callback: {e}")



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