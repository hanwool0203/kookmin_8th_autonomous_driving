import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import LifecycleNode, Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
        # xycar_lidar 패키지의 공유 디렉토리 경로를 찾습니다.
    share_dir = get_package_share_directory('xycar_lidar')
   
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

    camera_node = Node(
            package='cam',
            executable='camera_publisher',
            name='camera_node'
        )
    lane_detector_node = Node(
            package='cam',
            executable='lane_detector',
            name='lane_detector_node',
        )
    integrated_stanley_controller = Node(
            package='cam',
            executable='integrated_stanley_controller',
            name='integrated_stanley_controller',
        )
    yolo_node = Node(
            package='cam',
            executable='yolo_node',
            name='yolo_node',
        )
    yolo_lidar_fusion_node = Node(
            package='lidar2cam_projector',
            executable='yolo_lidar_fusion_node',
            name='yolo_lidar_fusion_node'
        )

    ultrasonic_node = Node(
        package='xycar_ultrasonic',  # <-- 이 부분을 실제 초음파 센서 패키지 이름으로 바꿔주세요.
        executable='xycar_ultrasonic', # <-- CMakeLists.txt에 정의된 실행 파일 이름
        name='xycar_ultrasonic_node',  # 런치 파일에서 노드를 구별하기 위한 이름 (자유롭게 지정)
        output='screen',
        emulate_tty=True
    )


    return LaunchDescription([
        params_declare,
        driver_node,


        camera_node,
        lane_detector_node,
        integrated_stanley_controller,
        #yolo_node,
        #yolo_lidar_fusion_node,

        ultrasonic_node,

    ])