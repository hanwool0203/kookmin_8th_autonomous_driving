import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import LifecycleNode, Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():

    # xycar_lidar 패키지의 공유 디렉토리 경로를 찾습니다.
    share_dir = get_package_share_directory('xycar_lidar')
    # mission_cone_drive_share_dir = get_package_share_directory('mission_cone_drive')
    # rviz_config_file = os.path.join(mission_cone_drive_share_dir, 'rviz', 'cone_drive.rviz')
   
    # Lidar 파라미터 파일 경로를 설정합니다.
    parameter_file = LaunchConfiguration('params_file')
    params_declare = DeclareLaunchArgument('params_file',
                                           default_value=os.path.join(share_dir, 'params', 'ydlidar.yaml'),
                                           description='Path to the ROS2 parameters file to use.')

    # Lidar 드라이버 노드 (LifecycleNode로 관리)
    driver_node = LifecycleNode(package='xycar_lidar',
                                executable='xycar_lidar_node',
                                name='xycar_lidar_node',
                                output='screen',
                                emulate_tty=True,
                                parameters=[parameter_file],
                                namespace='/')

    # 정적 TF 발행 노드 (base_link -> laser_frame)
    base_to_laser_tf = Node(package='tf2_ros',
                    executable='static_transform_publisher',
                    name='static_tf_pub_laser',
                    arguments=['0', '0', '0.02', '0', '0', '0', '1','base_link', 'laser_frame'])
    
    lidar_to_rear_axle_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_pub_rear_axle',
        arguments=['-0.42', '0', '0', '0', '0', '0', 'laser_frame', 'rear_axle']
    )
    
    lidar2cam_projector_node = Node(
        package='lidar2cam_projector',
        executable='lidar2cam_projector',
        name='lidar2cam_projector_node'
    )


    objacc_controller = Node(
        package='objacc_controller',
        executable='objacc_controller',
        name='objacc_controller'
    )
    
    # RViz2 실행 노드
    # rviz2_node = Node(package='rviz2',
    #                   executable='rviz2',
    #                   name='rviz2',
    #                   arguments=['-d', rviz_config_file])
    

    
    return LaunchDescription([
        # params_declare,
        # driver_node,
        # lidar_to_rear_axle_tf,
        # base_to_laser_tf,
        #rviz2_node,  # RViz2를 사용하지 않을 경우 주석 처리
        objacc_controller,
        yolo_lidar_fusion_node,
        lidar2cam_projector_node,

    ])

