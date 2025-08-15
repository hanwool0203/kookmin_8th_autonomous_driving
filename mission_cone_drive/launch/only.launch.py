from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='mission_cone_drive',
            executable='preprocessing_node',
            name='preprocessing_node'
        ),
        Node(
            package='mission_cone_drive',
            executable='path_planning_node',
            name='path_planning_node'
        ),
        Node(
            package='mission_cone_drive',
            executable='pure_pursuit_node',
            name='pure_pursuit_node'
        ),
        Node(
            package='mission_cone_drive',
            executable='visualization_node',
            name='visualization_node',
            output='screen'
        ),
    ])
