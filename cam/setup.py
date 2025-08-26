from setuptools import find_packages, setup

package_name = 'cam'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/common.launch.py']),
        ('share/' + package_name + '/launch', ['launch/bag_test.launch.py']),
        ('share/' + package_name + '/launch', ['launch/obstacle.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='xytron',
    maintainer_email='xytron@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'camera_publisher = cam.Camera_Publisher:main',
            'lane_detector = cam.Lane_Detector:main',
            'sign_detector = cam.SignDetector:main',
            'integrated_stanley_controller = cam.Integrated_Stanley_Controller:main',
            'yolo_node = cam.yolo_node:main',
            'centerlane_tracer = cam.centerlane_tracer:main',
            'curve_subscriber = cam.curve_subscriber:main',
            'topic_stamper = cam.topic_stamper:main',
            'target_lane_planner = cam.target_lane_planner:main',
            'ultra_node = cam.ultra_node:main',
        ],
    },
)
