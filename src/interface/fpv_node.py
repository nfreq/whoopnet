from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.node import Node
from sensor_msgs.msg import Imu, BatteryState, Image
from geometry_msgs.msg import Vector3Stamped
from cv_bridge import CvBridge
import builtin_interfaces.msg

class FpvNode(Node):
    def __init__(self):
        super().__init__('flight_interface_node')
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE, 
            history=HistoryPolicy.KEEP_LAST,
            depth=10 
        )
        self.imu_publisher = self.create_publisher(Imu, 'drone/imu', qos_profile=qos_profile)
        self.camera_raw_publisher = self.create_publisher(Image, 'drone/camera_raw', qos_profile=qos_profile)
        self.camera_publisher = self.create_publisher(Image, 'drone/camera', qos_profile=qos_profile)
        self.attitude_publisher = self.create_publisher(Vector3Stamped, 'attitude', qos_profile=qos_profile)
        self.battery_publisher = self.create_publisher(BatteryState, 'battery', qos_profile=qos_profile)
        self.bridge = CvBridge()

    def publish_imu(self, imu_data):
        timestamp_ms = imu_data[6]
        secs = int(timestamp_ms / 1000)
        nsecs = int((timestamp_ms % 1000) * 1_000_000)
        stamp_msg = builtin_interfaces.msg.Time()
        stamp_msg.sec = secs
        stamp_msg.nanosec = nsecs

        msg = Imu()
        msg.header.stamp =  stamp_msg
        msg.header.frame_id = "imu_frame"
        msg.linear_acceleration.x = imu_data[0] * 9.81  # Convert g to m/s^2
        msg.linear_acceleration.y = imu_data[1] * 9.81
        msg.linear_acceleration.z = imu_data[2] * 9.81
        msg.angular_velocity.x = imu_data[3]
        msg.angular_velocity.y = imu_data[4]
        msg.angular_velocity.z = imu_data[5]
        self.imu_publisher.publish(msg)

    def publish_camera_feed(self, undistorted_img):
        try:
            ros_image = self.bridge.cv2_to_imgmsg(undistorted_img, encoding="bgr8")
            ros_image.header.stamp = self.get_clock().now().to_msg() 
            ros_image.header.frame_id = "camera"
            self.camera_publisher.publish(ros_image)
        except Exception as e:
            self.get_logger().error(f"Failed to publish camera feed: {e}")

    def publish_camera_raw_feed(self, raw_img):
        try:
            ros_image = self.bridge.cv2_to_imgmsg(raw_img, encoding="bgr8")
            ros_image.header.stamp = self.get_clock().now().to_msg() 
            ros_image.header.frame_id = "camera_raw"
            self.camera_raw_publisher.publish(ros_image)
        except Exception as e:
            self.get_logger().error(f"Failed to publish camera raw feed: {e}")

    def publish_attitude(self, attitude_data):
        msg = Vector3Stamped()
        msg.vector.x = attitude_data[0]
        msg.vector.y = attitude_data[1]
        msg.vector.z = attitude_data[2]
        self.attitude_publisher.publish(msg)

    def publish_battery(self, battery_data):
        msg = BatteryState()
        msg.voltage = battery_data[0]
        msg.current = battery_data[1]
        msg.charge = battery_data[2] / 1000.0  # Convert mAh to Ah
        msg.percentage = battery_data[3] / 100.0
        self.battery_publisher.publish(msg)