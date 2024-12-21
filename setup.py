from setuptools import setup

package_name = 'whoopnet'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
     install_requires=[
        'pyserial',
        'numpy',
        'opencv-python',
        'easyocr',
        'inputs',
        'transformers',
        'ultralytics'
    ],
    zip_safe=True,
    maintainer='nfreq',
    maintainer_email='nfreq@whoopnet.ai',
    description='autonomous fpv interface',
    license='GPL-3.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'fpv_interface = interface.fpv_interface:main',
            'fpv_video = interface.fpv_video:main',
        ],
    },
)