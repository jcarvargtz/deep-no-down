from setuptools import find_packages
from setuptools import setup


REQUIRED_PACKAGES = ['docopt', 'pandas', 'keras', 'tensorflow','opencv-python','tqdm','pathlib','torch','torchvision']


setup(
    name='pantunfla',
    version='1.666667',
    author = 'Juan Carlos Vargas',
    author_email = 'jcarvargtz@hotmail.com',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    package_data={'pantunfla':[r'pantunfla/*.pth',r'pantunfla/*.npy',r'pantunfla/*.sh']},
    description="training setup, data download, and procesing for the kaggle's Deepfake detection challenge")

