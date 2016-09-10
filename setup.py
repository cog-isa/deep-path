from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules = cythonize("dlpf/gym_environ/utils_compiled.pyx"),
    include_dirs = [numpy.get_include()]
)
