from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'dqn_stanley_test'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=[
        'setuptools',
        # ROS2 패키지 의존성은 package.xml에 명시되므로 여기에는 포함하지 않습니다.
        # 아래에 Non-ROS Python 라이브러리 의존성을 명시합니다.
        'torch',          # PyTorch (CPU 버전)
        'numpy',          # 수치 연산 라이브러리
        'matplotlib',     # 학습 결과 시각화 라이브러리
        # termios는 표준 라이브러리이므로 보통 명시할 필요가 없습니다.
    ],
    zip_safe=True,
    maintainer='xytron',
    maintainer_email='kdh9981@konkuk.ac.kr',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'train = dqn_stanley_test.train:main',
            'test = dqn_stanley_test.test:main',
        ],
    },
)
