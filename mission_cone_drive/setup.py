from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'mission_cone_drive'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*.launch.py'))),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='hanwool0203',
    maintainer_email='hanwool0203@naver.com',
    description='Only 2D LiDAR Detection',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'preprocessing_node = mission_cone_drive.preprocessing_node:main',
            'path_planning_node = mission_cone_drive.path_planning_node:main',
            'pure_pursuit_node = mission_cone_drive.pure_pursuit_node:main',
            'visualization_node = mission_cone_drive.visualization_node:main',
            'rviz_visualizer_node = mission_cone_drive.rviz_visualizer_node:main',
            'checkerboard_detector = mission_cone_drive.checkerboard_detector:main',
            'scan_rotator = mission_cone_drive.scan_rotator:main',
        ],
    },
)
