import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import LifecycleNode, Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():

    lidar2cam_projector_node = Node(
        package='lidar2cam_projector',
        executable='lidar2cam_projector',
        name='lidar2cam_projector_node'
    )

    yolo_lidar_fusion_node = Node(
        package='lidar2cam_projector',
        executable='yolo_lidar_fusion_node',
        name='yolo_lidar_fusion_node'
    )

    objacc_controller = Node(
        package='objacc_controller',
        executable='objacc_controller',
        name='objacc_controller'
    )
    
    return LaunchDescription([
        objacc_controller,
        yolo_lidar_fusion_node,
        lidar2cam_projector_node,

    ])

