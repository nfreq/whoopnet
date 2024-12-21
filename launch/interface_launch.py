from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='whoopnet',
            executable='fpv_interface',
            name='fpv_interface_node',
            output='screen'
        ),
        Node(
            package='whoopnet',
            executable='fpv_video',
            name='fpv_video_node',
            output='screen'
        )
    ])
