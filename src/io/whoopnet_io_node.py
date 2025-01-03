import signal
import argparse
import rclpy
import signal
from whoopnet_io import WhoopnetIO
from whoopnet_rc_mixer import RCMixer
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.node import Node
from sensor_msgs.msg import Imu, BatteryState
from geometry_msgs.msg import Vector3Stamped
from sensor_msgs.msg import Joy
from cv_bridge import CvBridge
import builtin_interfaces.msg

"""
Published ROS Topics
/whoopnet/io/attitude
/whoopnet/io/battery
/whoopnet/io/camera
/whoopnet/io/camera_compressed
/whoopnet/io/camera_corrected
/whoopnet/io/command
/whoopnet/io/imu
/whoopnet/perception/depth_zoedepth
/whoopnet/perception/midas
/whoopnet/perception/yolo
"""


class WhoopnetIONode(Node): 
    def __init__(self):
        super().__init__('whoopnet_io_node')
        qos_besteffort = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT, 
            history=HistoryPolicy.KEEP_LAST,
            depth=100 
        )
        self.imu_publisher = self.create_publisher(Imu, 'whoopnet/io/imu', qos_profile=qos_besteffort)
        self.attitude_publisher = self.create_publisher(Vector3Stamped, 'whoopnet/io/attitude', qos_profile=qos_besteffort)
        self.battery_publisher = self.create_publisher(BatteryState, 'whoopnet/io/battery', qos_profile=qos_besteffort)
        self.rc_publisher = self.create_publisher(Joy, 'whoopnet/io/command', qos_profile=qos_besteffort)
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

    def publish_rc_values(self, channel_data):
        msg = Joy()
        msg.header.stamp = self.get_clock().now().to_msg()  # Add timestamp

        # Normalize 16-bit integer values to the range [-1.0, 1.0]
        msg.axes = [
            2 * (channel - 1000) / 1000.0 - 1 if 1000 <= channel <= 2000 else 0.0
            for channel in [
                channel_data[0],  # chT
                channel_data[1],  # chA
                channel_data[2],  # chE
                channel_data[3],  # chR
                channel_data[4],  # aux1
                channel_data[6],  # aux3
                channel_data[11], # aux8
            ]
        ]
        self.rc_publisher.publish(msg)



runtime_exec = True
def signal_handler(sig, frame):
    global runtime_exec
    print("\nSIGINT received and exiting")
    runtime_exec = False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="whoopnet-io fpv i/o interface (control and telemetry)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="/dev/ttyUSB0",
        help="video capture device (default: /dev/ttyUSB0)"
    )
    args = parser.parse_args()

    rclpy.init()
    ros2_node = WhoopnetIONode()
       
    print("whoopnet-io-ros started")
    print(f"device: {args.device}")

    def device_info_event_handler(device_infpo):
        pass

    def command_event_handler(command_data):
        ros2_node.publish_rc_values(command_data)

    def imu_event_handler(imu_data):
        ros2_node.publish_imu(imu_data)

    def attitude_event_handler(attitude_data):
        ros2_node.publish_attitude(attitude_data)

    def battery_event_handler(battery_data):
        ros2_node.publish_battery(battery_data)

    signal.signal(signal.SIGINT, signal_handler)

    whoopnet_io = WhoopnetIO(args.device)
    whoopnet_io.set_command_callback(command_event_handler)
    whoopnet_io.set_imu_callback(imu_event_handler)
    #whoopnet_io.set_device_info_callback(device_info_event_handler)
    #whoopnet_io.set_attitude_callback(attitude_event_handler)
    #whoopnet_io.set_battery_callback(battery_event_handler)
    whoopnet_io.start()
    whoopnet_io.init_rc_channels()

    mixer = RCMixer(whoopnet_io)
    mixer.start_mixer()

    while runtime_exec:
        rclpy.spin_once(ros2_node, timeout_sec=0.2)
       
    mixer.stop_mixer()
    whoopnet_io.stop()