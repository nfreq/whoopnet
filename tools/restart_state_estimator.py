import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
import sys
import termios
import tty

class RestartPublisher(Node):
    def __init__(self):
        super().__init__('restart_publisher')
        # Create a publisher for the restart topic
        self.publisher_ = self.create_publisher(Bool, 'whoopnet/perception/vins_mono/estimator/restart_manual', rclpy.qos.QoSProfile(depth=10, reliability=rclpy.qos.ReliabilityPolicy.RELIABLE))
        self.get_logger().info("Press the spacebar to send the restart signal.")

    def publish_restart(self):
        # Create and publish the Bool message
        msg = Bool()
        msg.data = True  # Signal restart
        self.publisher_.publish(msg)
        self.get_logger().info(f'Published restart message: {msg.data}')

def wait_for_spacebar():
    """Wait for the spacebar press."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        while True:
            char = sys.stdin.read(1)
            if char == ' ':
                break
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

def main(args=None):
    rclpy.init(args=args)  # Initialize the ROS 2 Python client library
    node = RestartPublisher()  # Create the RestartPublisher node

    while True:
        try:
            wait_for_spacebar()  # Wait for spacebar press
            node.publish_restart()  # Send the restart signal
        except KeyboardInterrupt:
            node.get_logger().info("Interrupted by user.")
            break

if __name__ == '__main__':
    main()
