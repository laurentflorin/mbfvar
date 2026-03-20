from setuptools import setup, Extension
from pathlib import Path
import pybind11

eigen_include = Path(__file__).resolve().parents[2] / "third_party" / "eigen-3.4.0"

ext_modules = [
    Extension(
        "solve",
        sources=["solve.cpp"],
        include_dirs=[pybind11.get_include(), eigen_include],
        language="c++",
    ),
]

setup(
    name="solve",
    version="0.1",
    ext_modules=ext_modules,
)