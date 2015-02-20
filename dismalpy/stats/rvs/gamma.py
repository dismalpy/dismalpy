"""
Gamma Random Variates
"""
from __future__ import division

import numpy as np
from rvs import RandomVariable

class Gamma(RandomVariable):
    """
    Notes
    -----

    This function uses the shape and scale parameter definition.

    In calling this function, typically we use

    - shape
    - rate = inverse scale = 1 / scale

    One complication is that the PDF of the Gamma distribution according to
    the numpy definition of the shape and scale does not include the division
    by 2. Thus the ultimate signature of the function call will use:

    - shape = *shape / 2
    - rate = *rate / 2
    - [or] scale = 1 / rate = 2 / *rate

    Then the posteriors are given by:

    - posterior_shape = (*shape + T) / 2
    - posterior_rate / 2 = (*rate + e'e) / 2
    - [or] posterior_scale = 1 / posterior_rate = 2 / (*rate + e'e)
    """

    def __init__(self, shape, scale=1.0, size=1, preload=1, *args, **kwargs):
        # Initialize parameters
        self.shape = float(shape)
        self.scale = float(scale)
        self.prior_shape = self.shape
        self.prior_scale = self.scale

        # Calculate some quantities
        self._prior_scaled_rate = 2. / self.prior_scale

        # Setup holder variables for posterior-related quantities
        self._beta = None
        self._exog = None
        self._endog = None

        # Setup holder variables for calculated quantities
        self._xbeta = None
        self._resid = None
        self._posterior_shape = None
        self._posterior_scale = None

        # Set the flag to use the prior
        self._use_posterior = False

        # Initialize the distribution
        distribution = 'gamma'
        super(Gamma, self).__init__(distribution,
                                    distribution_args=(self.shape, 1.0),
                                    distribution_kwargs={},
                                    size=size, preload=preload,
                                    *args, **kwargs)

    @property
    def beta(self):
        return self._beta
    @beta.setter
    def beta(self, value):
        # Save the value
        value = np.array(value)

        # Check that dimensions match
        if value.ndim == 0:
            value = value[np.newaxis]
        elif not value.ndim == 1:
            raise ValueError('Invalid beta array dimensions. Required '
                             ' 1-dim, got %d' % value.ndim)

        if self._exog is not None:
            if not value.shape[0] == self._exog.shape[1]:
                raise ValueError('Invalid beta array dimensions. Required'
                                 ' (%d), got (%s)'
                                 % (self._exog.shape[1], value.shape[0]))

        # Set the underlying value
        self._beta = value

        # Clear calculated quantities
        self._xbeta = None
        self._resid = None
        self._posterior_scale = None

        # Set the posterior flag
        self._use_posterior = True
    @beta.deleter
    def beta(self):
        # Clear beta
        self._beta = None

        # Clear calculated quantities
        self._xbeta = None
        self._resid = None
        self._posterior_scale = None

        # Recalculate posterior flag
        self._use_posterior = not (
            self._exog is None and
            self._endog is None and
            self._beta is None
        )

    @property
    def exog(self):
        return self._exog
    @exog.setter
    def exog(self, value):
        # Save the exogenous dataset
        value = np.array(value)

        # Check that dimensions match
        if value.ndim == 1:
            value = value[:, np.newaxis]
        elif not value.ndim == 2:
            raise ValueError('Invalid exogenous array dimensions. Required '
                             ' (nobs, ncoef), got %s' % value.shape)

        if self._beta is not None:
            if not value.shape[1] == self._beta.shape[0]:
                raise ValueError('Invalid exogenous array dimensions. Required'
                                 ' (nobs, %d), got %s'
                                 % (self._beta.shape[0], value.shape))
        if self._endog is not None:
            if not value.shape[0] == self._endog.shape[0]:
                raise ValueError('Invalid exogenous array dimensions. Required'
                                 ' (%d, ncoef), got %s'
                                 % (self._endog.shape[0], value.shape))

        # Set the underlying value
        self._exog = value

        # Clear calculated quantities
        self._xbeta = None
        self._resid = None
        self._posterior_scale = None

        # Set the posterior flag
        self._use_posterior = True
    @exog.deleter
    def exog(self):
        # Clear exog
        self._exog = None

        # Clear calculated quantities
        self._xbeta = None
        self._resid = None
        self._posterior_scale = None

        # Recalculate posterior flag
        self._use_posterior = not (
            self._exog is None and
            self._endog is None and
            self._beta is None
        )

    @property
    def endog(self):
        return self._endog
    @endog.setter
    def endog(self, value):
        # Save the exogenous dataset
        value = np.array(value)

        # Record the old nobs (so that we avoid re-caching if we don't need to)
        nobs = None
        if self._endog is not None:
            nobs = self._endog.shape[0]

        # Check that dimensions match
        if value.ndim == 1:
            value = value[:, np.newaxis]
        elif not value.ndim == 2:
            raise ValueError('Invalid endogenous array dimension.'
                             ' Required (%d, 1), got %s'
                             % (self._endog.shape[0], value.shape))

        if self._exog is not None:
            if not value.shape[0] == self._exog.shape[0]:
                raise ValueError('Invalid endogenous array dimensions.'
                                 ' Required (%d, 1), got %s'
                                 % (self._endog.shape[0], value.shape))

        self._endog = value

        # Clear the cache (if the scale changed)
        if not self._endog.shape[0] == nobs:
            self._cache = None
            self._cache_index = None

        # Clear calculated quantities
        self._resid = None
        self._posterior_shape
        self._posterior_scale = None

        # Set the posterior flag
        self._use_posterior = True
    @endog.deleter
    def endog(self):
        # Clear endog
        self._endog = None

        # Clear calculated quantities
        self._resid = None
        self._posterior_shape
        self._posterior_scale = None

        # Clear the cache (because scale will change)
        self._cache = None
        self._cache_index = None

        # Recalculate posterior flag
        self._use_posterior = not (
            self._exog is None and
            self._endog is None and
            self._beta is None
        )
    
    @property
    def posterior_shape(self):
        if self._posterior_shape is None:
            # Get intermediate calculated quantity
            if self._endog is None:
                raise RuntimeError('Endogenous array is not set; cannot'
                                   ' calculate posterior scale.')
            self._posterior_shape = self.prior_shape + self._endog.shape[0]/2
        return self._posterior_shape

    @property
    def posterior_scale(self):
        if self._posterior_scale is None:
            # Get intermediate calculated quantity
            if self._resid is None:
                # Make sure we have required quantities
                if self._endog is None:
                    raise RuntimeError('Endogenous array is not set; cannot'
                                       ' calculate posterior scale.')
                if self._exog is None:
                    raise RuntimeError('Exogenous array is not set; cannot'
                                       ' calculate posterior scale.')
                if self._beta is None:
                    raise RuntimeError('Beta array is not set; cannot'
                                       ' calculate posterior scale.')
                if self._xbeta is None:
                    self._xbeta = np.dot(self._exog, self._beta)
                self._resid = self._endog[:, 0] - self._xbeta
            # Calculate the posterior scale
            self._posterior_scale = 2 / (
                self._prior_scaled_rate + np.sum(self._resid**2)
            )
        return self._posterior_scale

    def recache(self):
        # Always draw with scale parameter 1.0 (since we can change the scale
        # later via multiplication), but we need the shape parameter to be set
        if self._use_posterior:
            self.distribution_args = (self.posterior_shape, 1.0)
        else:
            self.distribution_args = (self.prior_shape, 1.0)
        
        return super(Gamma, self).recache()

    def next(self):
        # Get the Gamma(scale, 1.0) variate
        rvs = super(Gamma, self).next()

        # Get shape to transform to Gamma(scale, shape)
        return self.posterior_scale * rvs
