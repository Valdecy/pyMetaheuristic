from pathlib import Path

from setuptools import find_packages, setup

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="pymetaheuristic",
    version="5.5.9",
    license="GNU",
    author="Valdecy Pereira",
    author_email="valdecy.pereira@gmail.com",
    url="https://github.com/Valdecy/pyMetaheuristic",
    packages=find_packages(include=["pyMetaheuristic", "pyMetaheuristic.*", "pymetaheuristic", "pymetaheuristic.*"]),
    python_requires=">=3.9",
    install_requires=[
        "kaleido",
        "matplotlib",
        "numpy",
        "pandas",
        "plotly",
        "scipy",
        "tabulate",
    ],
    description="pymetaheuristic: A Python Library for Metaheuristic Optimization and Collaborative Search",
    long_description=long_description,
    long_description_content_type="text/markdown",
    include_package_data=True,
    package_data={
        "pymetaheuristic.src": ["cec2022_input_data/*.txt"],
    },
)
