#!/usr/bin/env python
"""pykf

Kalman filter and state-space models.
"""
from __future__ import division, print_function

DISTNAME            = 'pykf'
DESCRIPTION         = 'Kalman filter and state-space models'
AUTHOR              = 'Chad Fulton',
URL                 = 'https://github.com/ChadFulton/pykalman_filter/'
LICENSE             = 'BSD Simplified'
VERSION             = '0.1'

import os
import sys

# may need to work around setuptools bug by providing a fake Pyrex
# sourced from scikits-sparce
project_path = os.path.split(__file__)[0]
sys.path.append(os.path.join(project_path, 'fake_pyrex'))
from setuptools import setup, find_packages, Extension
from Cython.Distutils import build_ext

from numpy.distutils.misc_util import get_info

ext_data = {
    'sources': [
        "pykf/blas_lapack.pxd",
        "pykf/kalman_filter.pyx"
    ]
}
ext_data.update(get_info("npymath"))

ext_modules = [
    Extension("pykf.kalman_filter", **ext_data),
]


def setup_package():
    setup(
        name=DISTNAME,
        version=VERSION,
        packages=find_packages(),

        author=AUTHOR,
        description=DESCRIPTION,
        license=LICENSE,
        url=URL,

        install_requires=['numpy', 'scipy'],
        setup_requires=['nose>=1.0'],
        test_suite='nose.collector',
        zip_safe=False,

        cmdclass={'build_ext': build_ext},
        ext_modules=ext_modules
    )

if __name__ == '__main__':
    setup_package()