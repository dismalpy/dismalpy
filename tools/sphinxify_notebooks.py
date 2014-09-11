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


def _prepare_execute(file, _nbformat=None):
    if _nbformat is None:
        _nbformat = nbformat.current
    return _nbformat.reads_json(open(file).read())


def _execute_runipy(file):
    notebook = _prepare_execute(file)

    from runipy.notebook_runner import NotebookRunner

    runner = NotebookRunner(notebook)
    runner.run_notebook(skip_exceptions=True)

    return runner.nb, {}


def _execute_ipython(file, resources=None):
    notebook = _prepare_execute(file)

    from IPython.nbconvert.preprocessors import ExecutePreprocessor
    if resources is None:
        resources = {}
    notebook, resources = ExecutePreprocessor().preprocess(notebook, resources)

    return notebook, resources


def convert_notebook(notebook, tofile, resources=None):
    todir = os.path.dirname(tofile)

    exporter = RSTExporter()
    if resources is None:
        resources = {}
    output, resources = exporter.from_notebook_node(notebook,
                                                    resources=resources)

    writer = FilesWriter()
    writer.build_directory = todir
    writer.write(output, resources, notebook_name=tofile)


def find_process_files(source_dir, output_dir):
    # Figure out what processors we have available
    try:
        import runipy
        execute_runipy = True
        execute_notebook = _execute_runipy
    except ImportError:
        execute_runipy = False
    try:
        import IPython
        execute_ipython = int(IPython.__version__.split('.')[0]) >= 3
        if execute_ipython:
            execute_notebook = _execute_ipython
    except ImportError:
        raise Exception('IPython required to build documentation.')

    # Make sure one of them is available
    if not execute_ipython and not execute_runipy:
        raise Exception('Either IPython >= 3.0.0 or runipy required to'
                        ' build documentation.')

    # Walk through the notebooks
    this_dir = os.path.abspath(os.getcwd())
    for cur_dir, dirs, files in os.walk(source_dir):
        if cur_dir.endswith(os.sep + '.ipynb_checkpoints'):
            continue
        rel_cur_dir = os.path.relpath(cur_dir, source_dir)
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
                notebook, resources = execute_notebook(fromfile)
                convert_notebook(notebook, tofile, resources)
        os.chdir(this_dir)


def main():
    # Arguments
    try:
        source_dir = sys.argv[1]
    except IndexError:
        source_dir = DEFAULT_SOURCE
    try:
        output_loc = sys.argv[2]
    except IndexError:
        output_loc = DEFAULT_OUTPUT

    # Double check source directory actually exists
    if not os.path.exists(source_dir):
        raise Exception('Source path %s does not exist' % source_dir)

    # Clean the output directory
    output_dir = os.path.join(output_loc, DEFAULT_GEN_DIR)
    try:
        shutil.rmtree(output_dir)
    except:
        pass
    os.makedirs(output_dir)

    # Get absolute paths
    source_dir = os.path.abspath(source_dir)
    output_dir = os.path.abspath(output_dir)

    # Do processing
    find_process_files(source_dir, output_dir)

if __name__ == '__main__':
    main()