import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
import numpy as np
import cv2
from transformers import AutoImageProcessor, ZoeDepthForDepthEstimation
import torch

class DepthEstimationNode(Node):
    def __init__(self):
        super().__init__('depth_estimation_node')
        #self.image_subscriber = self.create_subscription(
        #    Image,
        #    '/whoopnet/io/camera_compressed',
        #    self.image_callback,
        #    10
        #)
        self.image_subscriber = self.create_subscription(
            CompressedImage,  # Use CompressedImage type
            '/whoopnet/io/camera_compressed',
            self.image_callback,
            10
        )
        self.depth_publisher = self.create_publisher(
            CompressedImage,
            '/whoopnet/perception/depth_zoedepth',
            10
        )
        self.image_processor = AutoImageProcessor.from_pretrained("Intel/zoedepth-nyu-kitti")
        self.model = ZoeDepthForDepthEstimation.from_pretrained("Intel/zoedepth-nyu-kitti").to("cuda:0")
        
        self.frame_skip = 6
        self.frame_count = 0

        self.get_logger().info("zoedepth initialized")

    def image_callback(self, msg: CompressedImage):
        try:
            self.frame_count += 1
            if self.frame_count % self.frame_skip != 0:
                return
            frame = np.frombuffer(msg.data, np.uint8)  # Convert the data back into a numpy array
            frame_rgb = cv2.imdecode(frame, cv2.IMREAD_COLOR)  # Decode the image
            #downscaled_frame = cv2.resize(frame_rgb, (128, 128), interpolation=cv2.INTER_AREA)            
            
            inputs = self.image_processor(images=frame_rgb, return_tensors="pt").to("cuda:0")
            with torch.no_grad():
                outputs = self.model(**inputs)

            post_processed_output = self.image_processor.post_process_depth_estimation(
                outputs,
                source_sizes=[frame_rgb.shape[:2]]
            )
            predicted_depth = post_processed_output[0]["predicted_depth"]

            depth = (predicted_depth - predicted_depth.min()) / (predicted_depth.max() - predicted_depth.min())
            depth_image_np = (depth.detach().cpu().numpy() * 255).astype(np.uint8)

            encode_params = [cv2.IMWRITE_JPEG_QUALITY, 80]
            success, encoded_image = cv2.imencode('.jpg', depth_image_np, encode_params)
            if not success:
                self.get_logger().error("Failed to encode image for compressed feed")
                return
            
            depth_msg = CompressedImage()
            depth_msg.header.stamp = self.get_clock().now().to_msg()  # Add current timestamp
            depth_msg.header.frame_id = "zoedepth"  # Optional, update as needed
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