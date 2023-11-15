from setuptools import setup, find_packages

package_name = 'driving'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name, 'driving/packages'],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jetbot',
    maintainer_email='marino.dominik.dm@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'driving = driving.driving:main'
        ],
    },
)
