DismalPy
========

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

- The up-to-date source code is available on GitHub: http://github.com/dismalpy/dismalpy
- Source distributions and some wheels are available on PyPi: https://pypi.python.org/pypi/dismalpy/

License
-------

Simplified-BSD License

Bug reports
-----------

Please submit bug reports to http://github.com/dismalpy/dismalpy/issues