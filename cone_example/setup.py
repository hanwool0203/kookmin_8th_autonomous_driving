from setuptools import find_packages, setup

package_name = 'cone_example'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='hanwool0203',
    maintainer_email='hanwool0203@todo.todo',
    description='TODO: Package description',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'move2rviz = cone_example.move2rviz:main',
            'pure_pursuit_visualizer_node_v003 = cone_example.pure_pursuit_visualizer_node_v003:main',
            'pure_pursuit_visualizer_node_v004 = cone_example.pure_pursuit_visualizer_node_v004:main',
            'pure_pursuit_visualizer_node_v005 = cone_example.pure_pursuit_visualizer_node_v005:main',
            'pure_pursuit_visualizer_node_v007 = cone_example.pure_pursuit_visualizer_node_v007:main',
            'pure_pursuit_visualizer_node = cone_example.pure_pursuit_visualizer_node:main',
            'pure_pursuit_node = cone_example.pure_pursuit_node:main'
        ],
    },
)
