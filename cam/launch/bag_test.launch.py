from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='cam',
            executable='lane_detector',
            name='lane_detector_node',
        ),
        Node(
            package='cam',
            executable='integrated_stanley_controller',
            name='stanley_node',
        ),
        Node(
            package='cam',
            executable='yolo_node',
            name='yolo_node',
        )
    ])