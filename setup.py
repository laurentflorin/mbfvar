from setuptools import setup, Extension
from pathlib import Path

import pybind11

eigen_include = Path(__file__).resolve().parent / "third_party" / "eigen-3.4.0"

ext_modules = [
    Extension(
        "MUFBVAR.cholcov.cholcov_module",
        sources=[
            "MUFBVAR/cholcov/cholcov.cpp",
            "MUFBVAR/cholcov/cholcov_bindings.cpp",
        ],
        include_dirs=[pybind11.get_include(), str(eigen_include)],
        language="c++",
    ),
    Extension(
        "MUFBVAR.inverse.matrix_inversion",
        sources=["MUFBVAR/inverse/matrix_inversion.cpp"],
        include_dirs=[pybind11.get_include(), str(eigen_include)],
        language="c++",
    ),
    Extension(
        "MUFBVAR.pseudo_inverse.pseudo_inverse",
        sources=[
            "MUFBVAR/pseudo_inverse/pseudo_inverse.cpp",
            "MUFBVAR/pseudo_inverse/pseudo_inverse_bindings.cpp",
        ],
        include_dirs=[pybind11.get_include(), str(eigen_include)],
        language="c++",
    ),
    Extension(
        "MUFBVAR.solve.solve",
        sources=["MUFBVAR/solve/solve.cpp"],
        include_dirs=[pybind11.get_include(), str(eigen_include)],
        language="c++",
    ),
]

setup(ext_modules=ext_modules)
