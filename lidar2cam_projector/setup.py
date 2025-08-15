from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'lidar2cam_projector'

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
    maintainer_email='hanwool0203@todo.todo',
    description='TODO: Package description',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'lidar2cam_projector = lidar2cam_projector.lidar2cam_projector:main',
            'yolo_lidar_fusion_node = lidar2cam_projector.yolo_lidar_fusion_node:main',
        ],
    },
)
