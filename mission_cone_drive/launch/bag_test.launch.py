import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    mission_cone_drive_share_dir = get_package_share_directory('mission_cone_drive')
    rviz_config_file = os.path.join(mission_cone_drive_share_dir, 'rviz', 'cone_drive.rviz')
    
    base_to_laser_tf = Node(package='tf2_ros',
                    executable='static_transform_publisher',
                    name='static_tf_pub_laser',
                    arguments=['0', '0', '0.02', '0', '0', '0', '1', 'base_link', 'laser_frame'])
    
    lidar_to_rear_axle_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_pub_rear_axle',
        arguments=['-0.42', '0', '0', '0', '0', '0', 'laser_frame', 'rear_axle']
    )
    
    rviz2_node = Node(package='rviz2',
                      executable='rviz2',
                      name='rviz2',
                      arguments=['-d', rviz_config_file])
    
    sign_detector_node = Node(
        package='cam',
        executable='sign_detector',
        name='SignDetector'
    )
    yolo_node = Node(
        package='cam',
        executable='yolo_node',
        name='yolo_node'
    )

    scan_rotator_node = Node(
        package='mission_cone_drive',
        executable='scan_rotator',
        name='scan_rotator_node'
    )

    preprocessing_node = Node(
        package='mission_cone_drive',
        executable='preprocessing_node',
        name='preprocessing_node'
    )

    path_planning_node = Node(
        package='mission_cone_drive',
        executable='path_planning_node',
        name='path_planning_node'
    )

    pure_pursuit_node = Node(
        package='mission_cone_drive',
        executable='pure_pursuit_node',
        name='pure_pursuit_node'
    )

    rviz_visualizer_node = Node(
        package='mission_cone_drive',
        executable='rviz_visualizer_node',
        name='rviz_visualizer_node'
    )
    checkerboard_node = Node(
        package='mission_cone_drive',
        executable='checkerboard_detector',
        name='checkerboard_node'
    )
    stanley_controller_node = Node(
        package='cam',
        executable='integrated_stanley_controller',
        name='stanley_controller_node'
    )

    lane_detector_node = Node(
        package='cam',
        executable='lane_detector',
        name='lane_detector_node'
    )

    centerlane_tracer_node = Node(
        package='cam',
        executable='centerlane_tracer',
        name='centerlane_tracer_node'
    )

    target_lane_planner_node = Node(
        package='cam',
        executable='target_lane_planner',
        name='target_lane_planner_node'
    )

    ultra_node = Node(
        package='cam',
        executable='ultra_node',
        name='ultra_node'
    )
    
    return LaunchDescription([
        lidar_to_rear_axle_tf,
        base_to_laser_tf,
        rviz2_node,

        scan_rotator_node,
        preprocessing_node,
        path_planning_node,
        pure_pursuit_node,
        rviz_visualizer_node, 

        sign_detector_node,
        checkerboard_node,
        yolo_node,
        centerlane_tracer_node,
        target_lane_planner_node,
        ultra_node,
        stanley_controller_node,
        lane_detector_node,
        
    ])
