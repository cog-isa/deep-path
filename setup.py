from distutils.core import setup, Extension

import numpy
from Cython.Build import cythonize

extensions = [Extension("dlpf.gym_environ.utils_compiled",
                        ["dlpf/gym_environ/utils_compiled.pyx"]),
              Extension("dlpf.agents.training_data_gen",
                        ["dlpf/agents/training_data_gen.pyx"]), ]

setup(
    ext_modules=cythonize(extensions),
    include_dirs=[numpy.get_include()]
)
