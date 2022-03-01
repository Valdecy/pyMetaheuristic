from setuptools import setup, find_packages

setup(
    name='pyMetaheuristic',
    version='1.0.1',
    license='GNU',
    author='Valdecy pereira',
    author_email='valdecy.pereira@gmail.com',
    url='https://github.com/Valdecy/pyMetaheuristic',
    packages=find_packages(),
    install_requires=[
        'matplotlib',
        'numpy',
        'scipy'
    ],
)
