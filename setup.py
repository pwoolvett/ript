from setuptools import setup
from setuptools import find_packages

setup(
    name='realtimeipt',
    version='0.1.0',
    url='https://github.com/pwoolvett/realtimeipt.git',
    author='Pablo Woolvett',
    author_email='realtimeipt@devx.pw',
    description='Realtime Inverse Perspective Transform',
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        'opencv-python >= 4.5.3 ',
        'requests>=2.26.0',
        'uvicorn>=0.14.0',
        'bentley-ottmann>=6.0.0',
        'python-dotenv>=0.19.0',
    ],
        entry_points = {
        'console_scripts': [
            'ript-serve=realtimeipt.api:serve',
            'ript-gui=realtimeipt.roi:main',
            'ript=realtimeipt.full:main',
        ],
    }
)
