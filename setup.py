from setuptools import setup, find_packages, Extension
# from distutils.core import setup
# from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
from numpy.distutils.system_info import get_info

import numpy

ext_modules = [
    Extension("kalman_filter", sources=[
              "kalman/blas_lapack.pxd", "kalman/kalman_filter.pyx"],
              include_dirs=[numpy.get_include()]),
]

setup(
    name='pykalman_filter',
    version='0.1',
    packages=find_packages(),

    author='Chad Fulton',
    description='Multivariate Kalman Filter',
    license='BSD Simplified',
    url='https://github.com/ChadFulton/pykalman_filter',

    install_requires=['numpy', 'scipy >= 0.13', 'Cython'],
    setup_requires=['nose>=1.0'],
    test_suite='nose.collector',

    cmdclass={'build_ext': build_ext},
    ext_modules=ext_modules
)
