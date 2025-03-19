# Available at setup time due to pyproject.toml
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup, find_packages

ext_modules = [
    Pybind11Extension(
        "noFFT",
        ["src/resonate.cpp", "src/ResonatorBank.cpp"],
    ),
]

setup(
    name="noFFT",
    version="0.0.1",
    author="Alexandre R.J. Francois",
    author_email="alexandrefrancois@gmail.com",
    url="https://github.com/alexandrefrancois/noFFT",
    description="A reference implementation of the Resonate algorithm in C++ for Python",
    long_description="",
    packages=find_packages(),
    ext_modules=ext_modules,
    extras_require={"test": "pytest"},
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.8",
)
