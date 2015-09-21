#!/usr/bin/env python
""" sphinx_notebooks

Run IPython example notebooks in [source_dir] and copy them to [output_dir].

Usage: sphinxify_notebooks [source_dir] [output_dir]

Default [source_dir] is 'doc/notebooks'.
Default [ouptut_dir] is'doc/source/user/notebooks'.

Much of this was cribbe from @hadim
(https://gistgithub.com/hadim/16e29b5848672e2e497c).
"""

from __future__ import division, print_function, absolute_import

import os
import sys
import shutil
import subprocess
import re

from IPython.nbconvert import RSTExporter
from IPython.nbconvert.writers import FilesWriter
from IPython import nbformat

DEFAULT_SOURCE = 'doc/notebooks'
DEFAULT_OUTPUT = 'doc/source/user'
DEFAULT_GEN_DIR = '_notebooks'

# WindowsError is not defined on unix systems
try:
    WindowsError
except NameError:
    WindowsError = None

def convert_notebook(fromfile, tofile, output_subdir):
    # ipython nbconvert --to rst --execute fromfile --output tofile
    r = subprocess.call(['ipython nbconvert --to rst --execute %s --output %s' %(fromfile, tofile)], shell=True)
    if r != 0:
        raise Exception('Sphinxify failed')

    # Replace absolute (image) file paths with relative paths
    with open(tofile + '.rst', 'r') as f:
        string = f.read()
        string = re.sub(output_subdir, '', string)

    with open(tofile + '.rst', 'w') as f:
        f.write(string)

def find_process_files(source_dir, output_dir):
    # Figure out what processors we have available
    try:
        import IPython
        if not int(IPython.__version__.split('.')[0]) >= 3:
            raise ImportError
    except ImportError:
        raise Exception('IPython >= 3.0.0 required to build documentation.')

    # Walk through the notebooks
    this_dir = os.path.abspath(os.getcwd())
    for cur_dir, dirs, files in os.walk(source_dir):
        if cur_dir.endswith(os.sep + '.ipynb_checkpoints'):
            continue
        rel_cur_dir = os.path.relpath(cur_dir, source_dir)
        if rel_cur_dir == '.':
            rel_cur_dir = ''
        for filename in files:
            if filename.endswith('.ipynb'):
                # Need to change the working directory to the one in which
                # the notebook is in, in case it e.g. opens files using
                # relative paths
                os.chdir(cur_dir)

                # Create the output subdir if it does not exist
                output_subdir = os.path.join(output_dir, rel_cur_dir)
                if not os.path.exists(output_subdir):
                    os.makedirs(output_subdir)

                # Get the file paths
                fromfile = os.path.join(source_dir, rel_cur_dir, filename)
                tofile = os.path.join(output_subdir, filename)

                # Run and convert the notebook
                convert_notebook(fromfile, tofile, output_subdir)
        os.chdir(this_dir)


def main():
    # Arguments
    try:
        source_dir = sys.argv[1]
    except IndexError:
        source_dir = os.path.abspath(DEFAULT_SOURCE)
    try:
        output_loc = sys.argv[2]
    except IndexError:
        output_loc = os.path.abspath(DEFAULT_OUTPUT)

    # Double check source directory actually exists
    if not os.path.exists(source_dir):
        raise Exception('Source path %s does not exist' % source_dir)

    # Clean the output directory
    output_dir = os.path.join(output_loc, DEFAULT_GEN_DIR)
    delete = raw_input('Are you sure you want to delete %s? [y/n]: ' % output_dir).lower()
    if delete in ['y', 'yes']:
        try:
            shutil.rmtree(output_dir)
        except Exception as e:
            pass
        os.makedirs(output_dir)

    # Get absolute paths
    source_dir = os.path.abspath(source_dir)
    output_dir = os.path.abspath(output_dir)

    # Do processing
    find_process_files(source_dir, output_dir)

if __name__ == '__main__':
    main()