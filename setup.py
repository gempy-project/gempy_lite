from setuptools import setup, find_packages
version = '0.1'

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='gempy_lite',
    version=version,
    packages=find_packages(exclude=('test', 'docs', 'examples')),
    include_package_data=True,
    install_requires=[
        'pandas',
        'numpy',
        'pytest',
        'subsurface'
        'scikit-image>=0.17', # Probably this fits more in the engine but I leave it so far

    ],
    license='LGPL v3',
    author='Miguel de la Varga',
    author_email='varga@aices.rwth-aachen.de',
    description='An Open-source, Python-based 3-D structural geological modeling software.',
    keywords=['geology', '3-D modeling', 'structural geology', 'uncertainty']
)
