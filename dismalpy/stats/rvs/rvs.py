"""
Random Variates

TODO Use __call__ for setting posterior parameters
TODO Use slice notation for old __call__ functionality
"""
from __future__ import division

import numpy as np
from scipy import stats

# Shim for old Scipy versions
try:
    raise ImportError
    from scipy.stats import wishart, invwishart
except ImportError:
    from _wishart import wishart, invwishart

def _process_size(size):
    """
    Validate and standardize array shape
    """
    size = np.array(size, dtype=float)

    # Exapand numbers (zero-dimensional sizes) to 1-dim
    if size.ndim == 0:
        size = size[np.newaxis]
    # Cannot have 2-dimensional size
    elif size.ndim > 1:
        raise ValueError('Size must be an integer or tuple of integers;'
                         ' thus must have dimension <= 1.'
                         ' Got size.ndim = %s' % str(tuple(size)))

    n = size.prod()
    shape = tuple(size)

    return n, shape

_numpy_distribution_map = {
    'beta': np.random.beta,
    'binomial': np.random.binomial,
    'chisquare': np.random.chisquare,
    'dirichlet': np.random.dirichlet,
    'exponential': np.random.exponential,
    'f': np.random.f,
    'gamma': np.random.gamma,
    'geometric': np.random.geometric,
    'gumbel': np.random.gumbel,
    'hypergeometric': np.random.hypergeometric,
    'laplace': np.random.laplace,
    'logistic': np.random.logistic,
    'lognormal': np.random.lognormal,
    'logseries': np.random.logseries,
    'multinomial': np.random.multinomial,
    'multivariate_normal': np.random.multivariate_normal,
    'negative_binomial': np.random.negative_binomial,
    'noncentral_chisquare': np.random.noncentral_chisquare,
    'noncentral_f': np.random.noncentral_f,
    'normal': np.random.normal,
    'pareto': np.random.pareto,
    'poisson': np.random.poisson,
    'power': np.random.power,
    'rayleigh': np.random.rayleigh,
    'standard_cauchy': np.random.standard_cauchy,
    'standard_exponential': np.random.standard_exponential,
    'standard_gamma': np.random.standard_gamma,
    'standard_normal': np.random.standard_normal,
    'standard_t': np.random.standard_t,
    'triangular': np.random.triangular,
    'uniform': np.random.uniform,
    'vonmises': np.random.vonmises,
    'wald': np.random.wald,
    'weibull': np.random.weibull,
    'zipf': np.random.zipf,
}

_scipy_distribution_map = {
    # Continuous
    'alpha': stats.alpha,
    'anglit': stats.anglit,
    'arcsine': stats.arcsine,
    'beta': stats.beta,
    'betaprime': stats.betaprime,
    'bradford': stats.bradford,
    'burr': stats.burr,
    'cauchy': stats.cauchy,
    'chi': stats.chi,
    'chi2': stats.chi2,
    'cosine': stats.cosine,
    'dgamma': stats.dgamma,
    'dweibull': stats.dweibull,
    'erlang': stats.erlang,
    'expon': stats.expon,
    'exponweib': stats.exponweib,
    'exponpow': stats.exponpow,
    'f': stats.f,
    'fatiguelife': stats.fatiguelife,
    'fisk': stats.fisk,
    'foldcauchy': stats.foldcauchy,
    'foldnorm': stats.foldnorm,
    'frechet_r': stats.frechet_r,
    'frechet_l': stats.frechet_l,
    'genlogistic': stats.genlogistic,
    'genpareto': stats.genpareto,
    'genexpon': stats.genexpon,
    'genextreme': stats.genextreme,
    'gausshyper': stats.gausshyper,
    'gamma': stats.gamma,
    'gengamma': stats.gengamma,
    'genhalflogistic': stats.genhalflogistic,
    'gilbrat': stats.gilbrat,
    'gompertz': stats.gompertz,
    'gumbel_r': stats.gumbel_r,
    'gumbel_l': stats.gumbel_l,
    'halfcauchy': stats.halfcauchy,
    'halflogistic': stats.halflogistic,
    'halfnorm': stats.halfnorm,
    'hypsecant': stats.hypsecant,
    'invgamma': stats.invgamma,
    'invgauss': stats.invgauss,
    'invweibull': stats.invweibull,
    'invwishart': invwishart,
    'johnsonsb': stats.johnsonsb,
    'johnsonsu': stats.johnsonsu,
    'ksone': stats.ksone,
    'kstwobign': stats.kstwobign,
    'laplace': stats.laplace,
    'logistic': stats.logistic,
    'loggamma': stats.loggamma,
    'loglaplace': stats.loglaplace,
    'lognorm': stats.lognorm,
    'lomax': stats.lomax,
    'maxwell': stats.maxwell,
    'mielke': stats.mielke,
    'nakagami': stats.nakagami,
    'ncx2': stats.ncx2,
    'ncf': stats.ncf,
    'nct': stats.nct,
    'norm': stats.norm,
    'pareto': stats.pareto,
    'pearson3': stats.pearson3,
    'powerlaw': stats.powerlaw,
    'powerlognorm': stats.powerlognorm,
    'powernorm': stats.powernorm,
    'rdist': stats.rdist,
    'reciprocal': stats.reciprocal,
    'rayleigh': stats.rayleigh,
    'rice': stats.rice,
    'recipinvgauss': stats.recipinvgauss,
    'semicircular': stats.semicircular,
    't': stats.t,
    'triang': stats.triang,
    'truncexpon': stats.truncexpon,
    'truncnorm': stats.truncnorm,
    'tukeylambda': stats.tukeylambda,
    'uniform': stats.uniform,
    'vonmises': stats.vonmises,
    'wald': stats.wald,
    'weibull_min': stats.weibull_min,
    'weibull_max': stats.weibull_max,
    'wishart': wishart,
    'wrapcauchy': stats.wrapcauchy,

    # Multivariate
    'multivariate_normal': stats.multivariate_normal,

    # Discrete
    'bernoulli': stats.bernoulli,
    'binom': stats.binom,
    'boltzmann': stats.boltzmann,
    'dlaplace': stats.dlaplace,
    'geom': stats.geom,
    'hypergeom': stats.hypergeom,
    'logser': stats.logser,
    'nbinom': stats.nbinom,
    'planck': stats.planck,
    'poisson': stats.poisson,
    'randint': stats.randint,
    'skellam': stats.skellam,
    'zipf': stats.zipf,
}

def _process_distribution(distribution):
    if distribution is None:
        distribution = np.zeros
    elif isinstance(distribution, str):
        distribution = distribution.lower()
        # If the random variable is in numpy, use the callable function
        if distribution in _numpy_distribution_map:
            distribution = _numpy_distribution_map[distribution]
        # Otherwise if it is in scipy, use the rvs function
        elif distribution in _scipy_distribution_map:
            distribution = _scipy_distribution_map[distribution].rvs
        else:
            raise ValueError('Invalid distribution name: %s' % distribution)
    elif not callable(distribution):
        raise ValueError('Invalid distribution object. Must be the name of'
                         ' a numpy.random or scipy.stats distribution or must'
                         ' be callable.')

    return distribution

class RandomVariable(object):
    
    def __init__(self, distribution=None, distribution_args=None,
                 distribution_kwargs=None, size=1, preload=1,
                 *args, **kwargs):
        # Iteration number
        self.i = -1

        # Save the distribution (if any)
        self.distribution_rvs = _process_distribution(distribution)
        if distribution_args is None:
            distribution_args = ()
        if distribution_kwargs is None:
            distribution_kwargs = {}
        self.distribution_args = distribution_args
        self.distribution_kwargs = distribution_kwargs

        # Process size of output random variates
        self.n, self.size = _process_size(size)

        # Setup parameters for limited iterator runs created via __call__
        self._limited_n = None
        self._limited_i = None

        # Process size of preloading
        self.preload_n = int(preload)
        self.preload_size = (self.preload_n,)

        # Setup the cache dimensions
        self._cache_n = self.n * self.preload_n
        self._cache_size = self.preload_size + self.size

        # Initialize the caching variables
        self._cache = None
        self._cache_index = None

    def __iter__(self):
        return self

    def __call__(self, n):
        self._limited_n = int(n)
        # Set the counter to -1 because it will be advanced at the start of
        # each next() call rather than the end
        self._limited_i = -1
        return self

    def recache(self):
        # Re-create the cache
        del self._cache
        self._cache = self.distribution_rvs(size=self._cache_size,
                                            *self.distribution_args,
                                            **self.distribution_kwargs)

        # Re-initialize the index
        self._cache_index = np.ndindex(self.preload_size)

        # Return the first index element
        return next(self._cache_index)

    def next(self):
        # Advance the iteration number
        self.i += 1

        # See if we are in a limited run; if so advance or raise StopIteration
        if self._limited_n is not None:
            if self._limited_i >= self._limited_n:
                self._limited_n = None
                raise StopIteration
            else:
                self._limited_i += 1

        # Check the cache
        if self._cache_index is None:
            index = self.recache()
        else:
            try:
                index = next(self._cache_index)
            except StopIteration:
                index = self.recache()
            

        # Get the next element in the cache
        rvs = self._cache[index]

        return rvs
