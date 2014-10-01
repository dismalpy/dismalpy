from __future__ import division, print_function, absolute_import

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('src', parent_package, top_path)

    config.add_data_files('capsule.h')
    config.add_data_files('*.pxd')

    config.add_extension('blas',
                         sources=['blas.c'],
                         include_dirs=['.'])
    config.add_extension('lapack',
                         sources=['lapack.c'],
                         include_dirs=['.'])

    config.make_config_py()
    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())