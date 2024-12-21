import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import numpy as np
import cv2
from transformers import AutoImageProcessor, ZoeDepthForDepthEstimation
import torch

class DepthEstimationNode(Node):
    def __init__(self):
        super().__init__('depth_estimation_node')
        self.image_subscriber = self.create_subscription(
            Image,
            '/drone/camera_raw',
            self.image_callback,
            10
        )
        self.depth_publisher = self.create_publisher(
            Image,
            '/drone/camera_depth',
            10
        )
        self.image_processor = AutoImageProcessor.from_pretrained("Intel/zoedepth-nyu-kitti")
        self.model = ZoeDepthForDepthEstimation.from_pretrained("Intel/zoedepth-nyu-kitti").to("cuda:0")

        self.get_logger().info("Depth Estimation Node initialized.")

    def image_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV (NumPy) array
            frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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

            depth_msg = Image()
            depth_msg.header = msg.header
            depth_msg.height = depth_image_np.shape[0]
            depth_msg.width = depth_image_np.shape[1]
            depth_msg.encoding = "mono8"
            depth_msg.step = depth_image_np.shape[1]
            depth_msg.data = depth_image_np.tobytes()

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