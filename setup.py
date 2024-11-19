from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy
from platform import machine, system

is_mac = system() == "Darwin"
is_arm = machine() == "arm64"
is_m1 = is_mac and is_arm
omp_flag = "-Xpreprocessor -fopenmp" if is_m1 else "-fopenmp"
extra_compile_args = ["-O3", "-ffast-math", omp_flag]

extensions = [
        # Extension(
            # "sphericalvoids",
            # ["VERSUS/sphericalvoids.pyx"],
            # extra_compile_args=extra_compile_args,
            # extra_link_args=[omp_flag],
            # libraries=["m"]
            # ),
        # Extension(
            # "void_openmp_library",
            # ["VERSUS/void_openmp_library.c"],
            # extra_compile_args=extra_compile_args,
            # extra_link_args=[omp_flag],
            # libraries=["m"]
            # )
        Extension(
            "VERSUS.sphericalvoids",
            [
                "VERSUS/sphericalvoids.pyx",
                "VERSUS/void_openmp_library.c"
            ],
            extra_compile_args=extra_compile_args,
            extra_link_args=[omp_flag],
            libraries=["m"]
            )
        ]

setup(
    name="VERSUS",  # Name of the project
    # ext_modules=cythonize("VERSUS/sphericalvoids.pyx",
    ext_modules=cythonize(
        extensions,
        compiler_directives={'language_level': "3"},
        include_path=["VERSUS/"]),
    include_dirs=[numpy.get_include()],
    install_requires=[
        "numpy",
        "pyrecon"
    ],
    author="Nathan Findlay",
    description="Implementation of Pylians3 spherical void finder adapted for survey geometries.",
)
