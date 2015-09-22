************
Installation
************

Dependencies
------------

- NumPy
- SciPy >= 0.14.0
- Pandas >= 0.16.0
- Cython >= 0.20.0
- Statsmodels >= 0.8.0; note that this has not yet been released, so for the
  time being the development version must be installed prior to installing
  DismalPy.
- Git (this is required to install the development version of Statsmodels)

There are a few optional dependencies:

- Matplotlib; this is required for plotting functionality
- Nose; this is required for running the test suite
- IPython; this is required for running the examples or building the
  documentation

Procedure
---------

The most straightforward way to install DismalPy is using pip. The following
steps should be followed.

1. Install git. Instructions are available many places, for example at
   https://git-scm.com/book/en/v2/Getting-Started-Installing-Git
2. Install the development version of Statsmodels using the following command:
   
   ::
       
       pip install git+git://github.com/statsmodels/statsmodels.git

3. Install DismalPy
   
   ::

       pip install dismalpy

At this point, the package should installed. If you have the Nose package
installed, you can test for a successful installation by running the following
command (this may take a few minutes):

::

    python -c "import dismalpy as dp; dp.test();"

There should be no failures (although a number of Warnings are to be expected).

Installing from source
----------------------

Here we assume that the dependencies have already been installed (see above).

The source code can be obtained from the
`Github repository <http://github.com/dismalpy/dismalpy>`_. If you have git
installed (see above), you can use the following command:

::

    git clone git://github.com/statsmodels/statsmodels.git

Then assuming you have the appropriate compiler support (for example XCode on
Mac OS X and mingw32 or the Microsoft SDK on Windows), you can build the source
using the following command, from inside the base dismalpy directory:

::

    python setup.py install
