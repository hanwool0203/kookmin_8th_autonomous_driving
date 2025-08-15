from setuptools import find_packages, setup

package_name = 'objacc_controller'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
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
            'objacc_controller = objacc_controller.objacc_controller:main',
            'simple = objacc_controller.simple:main'
        ],
    },
)
