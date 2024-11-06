from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    name="VERSUS",  # Name of the project
    ext_modules=cythonize("VERSUS/sphericalvoids.pyx"),  # Cythonize the .pyx file
    include_dirs=[numpy.get_include()],
    install_requires=[
        "numpy"
        "pyrecon"
    ],
    author="Nathan Findlay",
    description="Implementation of Pylians3 spherical void finder adapted for survey geometries.",
)
