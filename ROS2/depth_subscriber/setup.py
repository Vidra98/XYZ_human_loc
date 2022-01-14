from setuptools import setup

package_name = 'depth_subscriber'

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
    maintainer='drame',
    maintainer_email='victor.drame@gmail.com',
    description='suscribers for data acquisition processing',
    license='OpenSource',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'listener = depth_subscriber.D430i_subscriber:main',
        ],
    },
)
