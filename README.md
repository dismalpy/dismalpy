DismalPy
========

**Note**: this package is largely obsolete, because most of its content has
been integrated into [Statsmodels](http://statsmodels.github.io/). Please see
my working paper [Estimating time series models by state space methods in Python: Statsmodels](http://www.chadfulton.com/research.html#est-ssm-py) for more
information on using Statsmodels to estimate state space models.

- [Working paper (PDF)](https://github.com/ChadFulton/fulton_statsmodels_2017/raw/master/fulton_statsmodels_2017_v1.pdf)
- [Working paper (HTML)](https://chadfulton.github.io/fulton_statsmodels_2017/)
- [Github Repository](https://github.com/ChadFulton/fulton_statsmodels_2017)

------------

A collection of resources for quantitative economics in Python. Includes:

- a Python wrapper for state space models along with a fast (compiled) Kalman
  filter, Kalman smoother, and simulation smoother.
- integration with the [Statsmodels](http://statsmodels.github.io/) module to
  allow maximum likelihood estimation of parameters in state space models,
  summary tables, diagnostic tests and plots, and post-estimation results.
- parameter estimation using Bayesian posterior simulation methods (either
  Metropolis-Hastings or Gibbs sampling - see the user guide for details,
  below), including integration with the
  [PyMC](https://pymc-devs.github.io/pymc/) module.
- built-in time series models: SARIMAX, unobserved components, VARMAX, and
  dynamic factor models.
- high test coverage

Documentation
-------------

- [User guide](http://dismalpy.github.io/user/index.html)
- [Reference guide](http://dismalpy.github.io/reference/index.html).

Installation
------------

See http://dismalpy.github.io/installation.html for details on installation.

- The up-to-date source code is available on GitHub: http://github.com/dismalpy/dismalpy
- Source distributions and some wheels are available on PyPi: https://pypi.python.org/pypi/dismalpy/

This package has the following dependencies:

- NumPy
- SciPy >= 0.14.0
- Pandas >= 0.16.0
- Cython >= 0.20.0
- Statsmodels >= 0.8.0; note that this has not yet been released, so for the
  time being the development version must be installed prior to installing
  DismalPy.

License
-------

Simplified-BSD License

Bug reports
-----------

Please submit bug reports to http://github.com/dismalpy/dismalpy/issues