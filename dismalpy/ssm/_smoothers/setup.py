from __future__ import division, print_function, absolute_import

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration, get_info

    config = Configuration('_smoothers', parent_package, top_path)

    config.add_extension('_conventional',
                         include_dirs=['dismalpy/src'],
                         sources=['_conventional.c'])
    config.add_extension('_univariate',
                         include_dirs=['dismalpy/src'],
                         sources=['_univariate.c'])
    config.make_config_py()
    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())