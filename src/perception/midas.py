import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import numpy as np
import cv2
from transformers import pipeline
from PIL import Image as PILImage

class DepthEstimationNode(Node):
    def __init__(self):
        super().__init__('depth_estimation_node')
        self.image_subscriber = self.create_subscription(
            Image,
            '/whoopnet/io/camera',
            self.image_callback,
            10
        )
        self.depth_publisher = self.create_publisher(
            Image,
            '/whoopnet/perception/depth_midas',
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
            # Convert ROS Image message to OpenCV (NumPy) array
            frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            #downscaled_frame = cv2.resize(frame_rgb, (128, 128), interpolation=cv2.INTER_AREA)            
            
            input_image = PILImage.fromarray(frame_rgb)
            result = self.depth_pipe(input_image)
            depth_image = result["depth"]

            depth_array = np.array(depth_image, dtype=np.float32)
            depth_normalized = cv2.normalize(depth_array, None, 0, 1, cv2.NORM_MINMAX)
            depth_uint8 = (depth_normalized * 255).astype(np.uint8)

            depth_msg = Image()
            depth_msg.header = msg.header
            depth_msg.height = depth_uint8.shape[0]
            depth_msg.width = depth_uint8.shape[1]
            depth_msg.encoding = "mono8"
            depth_msg.step = depth_uint8.shape[1]
            depth_msg.data = depth_uint8.tobytes()

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