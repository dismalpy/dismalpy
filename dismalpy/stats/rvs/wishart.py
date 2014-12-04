"""
Wishary Random Variates

TODO want to move the specific implementation of posteriors outside of the
     general `stats.rvs` libraries, since e.g. the below Wishart implementation
     is only valid for vector autoregressions.
"""
from __future__ import division

import numpy as np
from rvs import RandomVariable
# Shim for old Scipy versions
try:
    raise ImportError
    from scipy.stats import wishart, invwishart
except ImportError:
    from _wishart import wishart, invwishart
from scipy.linalg import lapack

class Wishart(RandomVariable):

    """
    Posterior calculation is for a VAR(1) process, where the endogenous
    variable is shaped (M, T) where M is the number of endogenous variables
    (and so also the number of equations) and T is the number of observations.

    The posterior degrees of freedom calculation is simple:

    .. math::

        posterior\_df = prior\_df + T

    The posterior scale matrix is calculated as follows:

    Since each of the M equations has the same right-hand-side variables, the
    provided exogenous array is also shaped (M, T) and is just the first lag of
    the endogenous variable.

    The M equations can be written as:

    .. math::

        y_i = L y' \hat \phi_i + \varepsilon_i

    where :math:`y_i` is shaped :math:`T \times 1`, L is the lag operator,
    :math:`y` is shaped :math:`M \times T` and :math:`\hat \phi_i` is a vector
    holding the coefficients in the ith equation, shaped :math:`M \times 1`.

    M residual vectors are created as :math:`e_i = y_i - L y' \hat \phi_i`
    (each with shape :math:`T \times 1`) and stacked as columns to create an
    :math:`M \times T` matrix :math:`e`.

    Alternatively, given the :math:`M \times M` matrix :math:`\phi`, the
    residual vector can be calculated as:

    .. math::

        e = y - \phi L y

    Then the posterior scale matrix is constructed in the following way:

    .. math::

        posterior_scale = [ prior\_scale^{-1} + e e']^{-1}

    The result is the following:

    Presuming the prior for the precision matrix of the VAR was given by
    W(prior_df, prior_scale), the posterior for the precision matrix of the
    VAR is given by W(posterior_df, posterior_scale).

    **Inverse Wishart**

    Drawing from the Wishart distribution provides draws of the precision
    matrix, but what is usually required in other steps of, for example, a
    Gibbs-sampling procedure is a covariance matrix (i.e. the inverse of the
    precision matrix). Instead of drawing a precision matrix and then inverting
    it, it is instead possible, and more computationally convenient, to draw
    covariance matrices directly from the inverse Wishart distribution.

    Deriving the posterior degrees of freedom and posterior scale for the case
    of an inverse Wishart is not difficult due to the following result:

    If :math:`S \sim W(df, scale)`, then :math:`S^{-1} \sim iW(df, scale^{-1})`

    If we had specified :math:`prior\_df_W` and `prior\_scale_W` for a Wishart
    distribution, then we could alternatively draw from an inverse Wishart
    with parameters :math:`prior\_df_{iW} = prior\_df_{W}` and
    :math:`prior\_scale_{iW} = prior\_scale_{W}^{-1}`.

    The posterior degrees of freedom is the same as for the Wishart
    distribution.

    .. math::

        posterior\_df_{*} = prior\_df_{*} + T

    The posterior scale is simply the inverse of the Wishart posterior scale:

    .. math::

       posterior\_scale_{iW} & = posterior\_scale_W^{-1} \\
       & = \left \{[ prior\_scale_W^{-1} + e e']^{-1} \right \}^{-1} \\
       & = prior\_scale_{iW} + e e' \\

    This saves us two matrix inversions: one in calculating the posterior
    scale, and one in retrieving a draw of a covariance matrix from the draw
    of a precision matrix.

    **Sampling from Wishart vs. inverse Wishart**

    The actual situation is more complicated than it might appear above,
    because the process of sampling from an inverse Wishart distribution is
    the following:

    Suppose a draw :math:`T \sim iW(df_{iW}, scale_{iw})` is required.

    1. :math:`scale_W = scale_{iw}^{-1}` is calculated.
    2. :math:`S \sim W(df, scale_W)` is drawn
    3. :math:`T \sim iW(df, scale_{iW})` is calculated as :math:`T = S^{-1}`

    Thus it appears that by drawing from an inverse Wishart, we have simply
    traded the point at which we perform the two inversions (that we had
    thought we could avoid by drawing from an inverse Wishart).

    However, we can further break down the steps of drawing from an inverse
    Wishart as follows:

    Suppose a draw :math:`T \sim iW(df_{iW}, scale_{iw})` is required.

    1. :math:`scale_W = scale_{iw}^{-1}` is calculated.
    2. :math:`S \sim W(df, scale_W)` is drawn
        a. The lower triangular matrix :math:`A` is created by filling the
           diagonal with the square roots of :math:`\chi^2` random variables,
           and the the lower triangle as independent N(0,1) draws.
        b. The Cholesky factorization :math:`DD' = scale_W` is calculated.
        c. By the Bartlett (1933) decomposition, :math:`S = D A A' D'`
    3. :math:`T \sim iW(df, scale_{iW})` is calculated as :math:`T = S^{-1}`
        a. However, instead of calculating :math:`S` and then separately
           calculating :math:`T = S^{-1}`, notice that
           :math:`T = (DA)^{-1'} (DA)^{-1}`, where :math:`DA` is lower
           triangular.
        b. Thus, :math:`T` is constructed directly, which only requires
           inverting a single lower triangular matrix.

    Steps 3a-3b show the performance improvement that can be achieved by
    drawing directly from the inverse Wishart rather than from the Wishart:
    we have substituted the inverse of dense precision matrix for the
    easier inverse of a lower triangular matrix.
    """

    def __init__(self, df, scale, size=1, preload=1, *args, **kwargs):
        # Initialize parameters
        self._frozen = wishart(df, scale)
        self._rvs = self._frozen._wishart
        self.df = self.prior_df = self._frozen.df
        self.scale = self.prior_scale = self._frozen.scale

        # Calculated quantities
        self._inv_prior_scale = np.linalg.inv(self.prior_scale)

        # Setup holder variables for posterior-related quantities
        self._phi = None     # (M x M)
        self._lagged = None  # (M x T)
        self._endog = None   # (M x T)

        # Setup holder variables for calculated quantities
        self._philagged = None
        self._posterior_df = None
        self._posterior_scale = None
        self._posterior_cholesky = None

        # Set the flag to use the prior
        self._use_posterior = False

        # Initialize the distribution
        super(Wishart, self).__init__(None, size=size, preload=preload,
                                      *args, **kwargs)

    @property
    def phi(self):
        return self._phi
    @phi.setter
    def phi(self, value):
        # Save the value
        value = np.array(value)

        # Check that dimensions match
        if not value.ndim == 2:
            raise ValueError('Invalid phi array dimensions. Required '
                             ' 2-dim, got %d-dim.' % value.ndim)

        if self._lagged is not None:
            if not value.shape[1] == self._lagged.shape[0]:
                raise ValueError('Invalid phi array dimension. Required'
                                 ' (n, %d), got %s'
                                 % (self._lagged.shape[0], str(value.shape)))
        elif self._endog is not None:
            if not value.shape[0] == self._endog.shape[0]:
                raise ValueError('Invalid phi array dimension. Required'
                                 ' (%d, k), got %s'
                                 % (self._endog.shape[0], str(value.shape)))

        # Set the underlying value
        self._phi = value

        # Clear calculated quantities
        self._philagged = None
        self._posterior_scale = None
        self._posterior_cholesky = None

        # Set the posterior flag
        self._use_posterior = True
    @phi.deleter
    def phi(self):
        # Clear phi
        self._phi = None

        # Clear calculated quantities
        self._philagged = None
        self._posterior_scale = None
        self._posterior_cholesky = None

        # Recalculate posterior flag
        self._use_posterior = not (
            self._lagged is None and
            self._endog is None and
            self._phi is None
        )

    @property
    def lagged(self):
        return self._lagged
    @lagged.setter
    def lagged(self, value):
        # Save the laggedenous dataset
        value = np.array(value)

        # Check that dimensions match
        if not value.ndim == 2:
            raise ValueError('Invalid lagged array dimensions. Required '
                             ' (k, nobs), got %s' % str(value.shape))
        if self._phi is not None:
            if not value.shape[0] == self._phi.shape[1]:
                raise ValueError('Invalid lagged array dimensions. Required'
                                 ' (%d, nobs), got %s'
                                 % (self._phi.shape[1], str(value.shape)))
        if self._endog is not None:
            if not value.shape[1] == self._endog.shape[1]:
                raise ValueError('Invalid lagged array dimensions.'
                                 ' Required (k, %d), got %s'
                                 % (self._endog.shape[1], str(value.shape)))

        # Set the underlying value
        self._lagged = value

        # Clear calculated quantities
        self._philagged = None
        self._posterior_scale = None

        # Set the posterior flag
        self._use_posterior = True
    @lagged.deleter
    def lagged(self):
        # Clear lagged
        self._lagged = None

        # Clear calculated quantities
        self._philagged = None
        self._posterior_scale = None
        self._posterior_cholesky = None

        # Recalculate posterior flag
        self._use_posterior = not (
            self._lagged is None and
            self._endog is None and
            self._phi is None
        )

    @property
    def endog(self):
        return self._endog
    @endog.setter
    def endog(self, value):
        # Save the endogenous dataset
        value = np.array(value)

        # Record the old nobs (so that we avoid re-caching if we don't need to)
        nobs = None
        if self._endog is not None:
            nobs = self._endog.shape[1]

        # Check that dimensions match
        if not value.ndim == 2:
            raise ValueError('Invalid endogenous array dimension.'
                             ' Required (k, nobs), got %s' % str(value.shape))
        if self._phi is not None:
            if not value.shape[0] == self._phi.shape[0]:
                raise ValueError('Invalid endogenous array dimensions.'
                                 ' Required (%d, nobs), got %s'
                                 % (self._phi.shape[0], str(value.shape)))
        if self._lagged is not None:
            if not value.shape[1] == self._lagged.shape[1]:
                raise ValueError('Invalid endogenous array dimensions.'
                                 ' Required (n, %d), got %s'
                                 % (str(self._lagged.shape[1]),
                                    str(value.shape)))

        # Set the underlying value
        self._endog = value

        # Clear calculated quantities
        self._posterior_df = None
        self._posterior_scale = None
        self._posterior_cholesky = None

        # Clear the cache (if the scale changed)
        if not self._endog.shape[1] == nobs:
            self._cache = None
            self._cache_index = None

        # Set the posterior flag
        self._use_posterior = True
    @endog.deleter
    def endog(self):
        # Clear endog
        self._endog = None

        # Clear calculated quantities
        self._posterior_df = None
        self._posterior_scale = None
        self._posterior_cholesky = None

        # Clear the cache (because scale will change)
        self._cache = None
        self._cache_index = None

        # Recalculate posterior flag
        self._use_posterior = not (
            self._lagged is None and
            self._endog is None and
            self._phi is None
        )
    
    @property
    def posterior_df(self):
        if self._posterior_df is None:
            # Get intermediate calculated quantity
            if self._endog is None:
                raise RuntimeError('Endogenous array is not set; cannot'
                                   ' calculate posterior degrees of freedom.')
            self._posterior_df = self.prior_df + self._endog.shape[1]
        return self._posterior_df

    @property
    def _resid(self):
        # Note: does no caching, should not be called twice

        # Make sure we have required quantities
        if self._endog is None:
            raise RuntimeError('Endogenous array is not set; cannot'
                               ' calculate posterior scale.')
        if self._lagged is None:
            raise RuntimeError('Lagged array is not set; cannot'
                               ' calculate posterior scale.')
        if self._phi is None:
            raise RuntimeError('Phi array is not set; cannot'
                               ' calculate posterior scale.')
        # This corresponds to a SUR model, where the residuals will
        # be shaped (k x T)
        if self._philagged is None:
            self._philagged = np.dot(self._phi, self._lagged)
        return self._endog - self._philagged

    @property
    def posterior_scale(self):
        if self._posterior_scale is None:
            resid = self._resid
            # Calculate the posterior scale
            # TODO inverse via Cholesky (?)
            self._posterior_scale = np.linalg.inv(
                self._inv_prior_scale + np.inner(resid, resid)
            )
        return self._posterior_scale

    @property
    def posterior_cholesky(self):
        if self._posterior_cholesky is None:
            self._posterior_cholesky = (
                np.linalg.cholesky(self.posterior_scale)
            )
        return self._posterior_cholesky

    def recache(self):
        # Set the appropriate degrees of freedom parameter
        if self._use_posterior:
            self._frozen.df = self.posterior_df
        else:
            self._frozen.df = self.prior_df
        
        # All of the cached draws are from a "standard" Wishart - meaning with
        # an identity scale matrix, but with the degrees of freedom set above.
        # In the `next` function, the cached variables are transformed to the
        # appropriate Wishart random variable

        # Re-create the cache
        del self._cache
        self._cache = self._rvs._standard_rvs(
            # n, shape, dim, df
            self._cache_n, self._cache_size, self._frozen.dim,
            self._frozen.df
        )

        # Re-initialize the index
        self._cache_index = np.ndindex(self.preload_size)

        # Return the first index element
        return next(self._cache_index)

    def next(self):
        rvs = super(Wishart, self).next()

        # Transformation
        if self._use_posterior:
            D = self.posterior_cholesky
        else:
            D = self._frozen.C

        if self.size == (1,):
            DA = np.dot(D, rvs[0])
            rvs = np.dot(DA, DA.T)
        else:
            for index in np.ndindex(rvs.shape[:-2]):
                DA = np.dot(D, rvs[index])
                rvs[index] = np.dot(DA, DA.T)

        return rvs

class InverseWishart(Wishart):
    # TODO: this probably should subclass RandomVariable directly, with the
    #       common functions separated into helpers or a second common
    #       superclass
    def __init__(self, df, scale, size=1, preload=1, *args, **kwargs):
        # Initialize the Wishart
        super(InverseWishart, self).__init__(df, scale, size, preload,
                                             *args, **kwargs)
        # Replace the wishart _rvs with an invwishart
        self._frozen = invwishart(self.df, self.scale)
        self._rvs = self._frozen._invwishart
        # df, scale are the same

        # Helpers for the triangular matrix inversion
        self._trtri = lapack.get_lapack_funcs(('trtri'), (self.scale,))

    @property
    def posterior_scale(self):
        if self._posterior_scale is None:
            resid = self._resid
            # Calculate the posterior scale
            self._posterior_scale = (
                self._inv_prior_scale + np.inner(resid, resid)
            )
        return self._posterior_scale

    def next(self):
        # Don't want to call the Wishart next, want to call the RandomVariable
        # next independently to get the standard Wishart distributed rvs
        rvs = RandomVariable.next(self)

        # Transformation
        if self._use_posterior:
            D = self.posterior_cholesky
        else:
            D = self._frozen.C

        if self.size == (1,):
            DA = np.dot(D, rvs[0])
            DA, info = self._trtri(DA, lower=True)
            if info > 0:
                raise np.linalg.LinAlgError("Singular matrix.")
            if info < 0:
                raise ValueError('Illegal value in %d-th argument of'
                                 ' internal trtrs' % -info)
            rvs = np.dot(DA.T, DA)
        else:
            for index in np.ndindex(rvs.shape[:-2]):
                # Calculate CA
                DA = np.dot(D, rvs[index])
                DA, info = self._trtrs(DA, self._eye, lower=True)
                if info > 0:
                    raise np.linalg.LinAlgError("Singular matrix.")
                if info < 0:
                    raise ValueError('Illegal value in %d-th argument of'
                                     ' internal trtrs' % -info)
                # Get SA
                rvs[index] = np.dot(DA.T, DA)

        return rvs