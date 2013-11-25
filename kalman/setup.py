import numpy
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
from numpy.distutils.system_info import get_info

#blas_include = get_info('blas_opt')['extra_compile_args'][1][2:]
#includes = [blas_include,numpy.get_include()]

#ext_modules = cythonize([
#    Extension("cylapack", ["cylapack.pyx"],
#              include_dirs = includes,
#              libraries=['blas','lapack']),
#    Extension("kalman_filter", ["kalman_filter.pyx"])
#])
ext_modules = [
    Extension("kalman_filter", ["kalman_filter.pyx"]),
    Extension("dkalman_filter", ["dkalman_filter.pyx"]),
]

setup(
  name = 'Markov Switching Model Tools',
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules,
  include_dirs=[numpy.get_include()]
)
