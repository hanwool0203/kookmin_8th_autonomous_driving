from setuptools import setup

package_name = 'lidar_point_clicker'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],  # rclpy, numpy 등은 따로 pip install
    zip_safe=True,
    maintainer='your_name',            # 본인 이름
    maintainer_email='your_email@example.com',
    description='LiDAR 클릭 특이점 추출용 예제 패키지',
    license='Apache License 2.0',
    entry_points={
        'console_scripts': [
            'click_saver = lidar_point_clicker.click_saver:main',
            
        ],
    },
)
