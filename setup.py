from setuptools import setup, Extension
from pathlib import Path

import pybind11

eigen_include = Path(__file__).resolve().parent / "third_party" / "eigen-3.4.0"

ext_modules = [
    Extension(
        "MBFVAR.cholcov.cholcov_module",
        sources=[
            "MBFVAR/cholcov/cholcov.cpp",
            "MBFVAR/cholcov/cholcov_bindings.cpp",
        ],
        include_dirs=[pybind11.get_include(), str(eigen_include)],
        language="c++",
    ),
    Extension(
        "MBFVAR.inverse.matrix_inversion",
        sources=["MBFVAR/inverse/matrix_inversion.cpp"],
        include_dirs=[pybind11.get_include(), str(eigen_include)],
        language="c++",
    ),
    Extension(
        "MBFVAR.pseudo_inverse.pseudo_inverse",
        sources=[
            "MBFVAR/pseudo_inverse/pseudo_inverse.cpp",
            "MBFVAR/pseudo_inverse/pseudo_inverse_bindings.cpp",
        ],
        include_dirs=[pybind11.get_include(), str(eigen_include)],
        language="c++",
    ),
    Extension(
        "MBFVAR.solve.solve",
        sources=["MBFVAR/solve/solve.cpp"],
        include_dirs=[pybind11.get_include(), str(eigen_include)],
        language="c++",
    ),
]

setup(ext_modules=ext_modules)
