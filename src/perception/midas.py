import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
import numpy as np
import cv2
from transformers import pipeline
from PIL import Image as PILImage

class DepthEstimationNode(Node):
    def __init__(self):
        super().__init__('depth_estimation_node')
        self.image_subscriber = self.create_subscription(
            CompressedImage,  # Use CompressedImage type
            '/whoopnet/io/camera_compressed',
            self.image_callback,
            10
        )
        self.depth_publisher = self.create_publisher(
            CompressedImage,
            '/whoopnet/perception/midas',
            10
        )
        self.depth_pipe = pipeline(task="depth-estimation", model="Intel/dpt-hybrid-midas", device="cuda:0")

        self.frame_skip = 6
        self.frame_count = 0
        
        self.get_logger().info("midas initialized.")

    def image_callback(self, msg):
        try:
            self.frame_count += 1
            if self.frame_count % self.frame_skip != 0:
                return
            
            frame = np.frombuffer(msg.data, np.uint8)  # Convert raw bytes to numpy array
            frame_rgb = cv2.imdecode(frame, cv2.IMREAD_COLOR)  # Decode the image
            #downscaled_frame = cv2.resize(frame_rgb, (128, 128), interpolation=cv2.INTER_AREA)            
            
            input_image = PILImage.fromarray(frame_rgb)
            result = self.depth_pipe(input_image)
            depth_image = result["depth"]

            depth_array = np.array(depth_image, dtype=np.float32)
            depth_normalized = cv2.normalize(depth_array, None, 0, 1, cv2.NORM_MINMAX)
            depth_uint8 = (depth_normalized * 255).astype(np.uint8)


            encode_params = [cv2.IMWRITE_JPEG_QUALITY, 80]
            success, encoded_image = cv2.imencode('.jpg', depth_uint8, encode_params)
            if not success:
                self.get_logger().error("Failed to encode image for compressed feed")
                return
            
            depth_msg = CompressedImage()
            depth_msg.header.stamp = self.get_clock().now().to_msg()  # Add current timestamp
            depth_msg.header.frame_id = "midas"  # Optional, update as needed
            depth_msg.format = "jpeg"
            depth_msg.data = encoded_image.tobytes()  # Convert numpy array to bytes
            self.depth_publisher.publish(depth_msg)

        except Exception as e:
            self.get_logger().error(f"Failed to process image: {e}")


def main(args=None):
   

    rclpy.init(args=args)
    node = DepthEstimationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()