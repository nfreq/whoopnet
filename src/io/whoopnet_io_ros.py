import time
import signal
import argparse
import rclpy
from whoopnet_io import WhoopnetIO
from whoopnet_node import WhoopnetNode
import signal

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
    ros2_node = WhoopnetNode()
       
    print("whoopnet-io-ros started")
    print(f"device: {args.device}")

    def device_info_event_handler(device_infpo):
        pass

    def imu_event_handler(imu_data):
        ros2_node.publish_imu(imu_data)

    def attitude_event_handler(attitude_data):
        ros2_node.publish_attitude(attitude_data)

    def battery_event_handler(battery_data):
        ros2_node.publish_battery(battery_data)

    signal.signal(signal.SIGINT, signal_handler)

    whoopnet_io = WhoopnetIO(args.device)
    whoopnet_io.set_imu_callback(imu_event_handler)
    #whoopnet_io.set_device_info_callback(device_info_event_handler)
    #whoopnet_io.set_attitude_callback(attitude_event_handler)
    #whoopnet_io.set_battery_callback(battery_event_handler)
    whoopnet_io.start()
    whoopnet_io.set_channel_init()

    while runtime_exec:
        if args.use_ros:
            rclpy.spin_once(ros2_node, timeout_sec=0.1)
        time.sleep(0.001)

    whoopnet_io.stop()