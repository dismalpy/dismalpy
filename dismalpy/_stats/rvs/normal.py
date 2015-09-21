"""
Normal Random Variates

TODO add possibility of specifying resid directly
"""
from __future__ import division

import numpy as np
from rvs import RandomVariable
from _normal import _rvs_quantities

class Normal(RandomVariable):
    """
    Posterior calculation is slightly different depending on the shape of the
    provided arrays. In general, their shapes are:

    `endog` :math:`Y (TM \times 1)`
    `exog` :math:`Z (TM \times k)`
    `beta` :math:`\beta (k \times 1)`
    `epsilon` :math:`\varepsilon (TM \times 1)`

    where :math:`k = \sum_{i=1}^M` k_i` and
    :math:`\varepsilon \sim N(0, \Omega)`.

    1. A typical linear regression model would have :math:`M = 1`, :math:`k`
       equal to the number of regression coefficients, and
       :math:`\Omega = h^{-1} I_T`.
    2. A SUR would have :math:`\Omega = I_T \otimes \Sigma` where
       :math:`\varepsilon_t \sim N(0, \Sigma)`
    3. A VAR(1) with no intercept is a SUR with :math:`k = M`.

    In the latter two cases, the matrix :math:`\Sigma` captures the covariance
    between equations *at a given time* :math:`t`; this covariance is assumed
    to be time-invariant.

    Practically, this class offers posterior estimation of either #1 or #2
    (where #3 is a special case of #2). It selects the appropriate model
    depending on the shape of the provided `precision`, where ultimately
    precision is shaped `M \times M`.

    1. If `precision` has shape :math:`1 \times 1`, then it is assumed to be
       a linear regression model. Then `endog` is assumed to be
       :math:`T \times 1` and `exog` is assumed to be :math:`T \times k`.
    2. Otherwise it is a SUR model (so `precision` is :math:`M \times M`), and
       there are multiple possibilities:
       - `endog` can be :math:`TM \times 1` or :math:`T \times M`
       - `exog` can be :math:`TM \times k` or :math:`T \times M \times k`

    **Independent Normal-Gamma Priors**

    If :math:`\Sigma = h^{-1} I_M`, then the entire model collapses
    to a ordinary linear regression with unknown variance. In this case, the
    prior on the precision parameter :math:`h^{-1}` is Gamma(v_0/2, d_0/2)
    and the prior on the coefficients is N(b_0, V_0).

    Posteriors are then Gamma(v_1/2, d_1/2) and N(b_1, V_1) where the posterior
    parameters are defined as usual.

    Note that a version of the  Minnesota prior falls into this category, in
    which the :math:`\Sigma` matrix is no longer considered to be fixed, but is
    required to be diagonal. In this case, there is no covariance between
    equations, and each equation can be estimated separately. Allowing each
    variance to be Gamma distributed allows the usual Gibbs-Sampling procedure
    to sample from the joint posterior.

    **Independent Normal-Wishart Priors**

    If :math:`\Sigma` is a general covariance matrix, then the model is a
    seemingly unrelated regression (SUR) - i.e. it consists of a set of
    apparently unrelated linear regressions for which efficiency in estimation
    can be improved by taking into account the structure of correlation in the
    error term (captured by :math:`\Sigma`) across equations.

    **Minnesota Prior**

    The Minnesota prior simply replaces the :math:`\Sigma` matrix with an
    estimate (i.e. not in a Gibbs step, where the estimate will be updated - it
    is assumed to be *fixed* at the estimated value). Often times it is further
    assumed to be diagonal (in which case the VAR is not only a seemingly
    unrelated regression, but a truly unrelated regression).

    The Minnesota prior attempts to enforce variable shrinkage in the easy-to-
    overparameterize VAR models. The prior itself models each equation
    as either as separate random noise processes (in the case of growth-rate
    data)

    .. math::

        y_{m,t} = \varepsilon_{m,t}

    or as separate random walks (in the case of levels data).

    .. math::

        y_{m,t} = y_{m,t-1} + \varepsilon_{m,t}

    Since the covariance matrix :math:`\Sigma` is fixed, the only prior
    parameters are on the distribution of the coefficients :math:`\beta`.

    The prior mean is usually specifyied to be zero for all coefficients except
    for on the first-own-lag of levels data in each equation (this expresses
    the prior of a random walk for such series). However this prior is flexible
    and could instead be set to arbitrary values as required.

    The prior covariance matrix is usually diagonal, and is set so as to
    produce variable shrinkage; in particular, the prior variance on
    own-lag coefficients is the highest (allowing them the most
    flexibility to move). For all variables, the prior variance is higher on
    lower lags and decreases as the lag length increases. Finally, the
    prior variances are scaled by a ratio of (usually estimated) variances so
    as to make the prior variances consistent with the scale of the variable
    in question.

    The posterior hyperparameters for the coefficients then applies standard
    results for the linear regression case, using the fixed :math:`\Sigma`
    matrix.

    This prior structure can also be implemented using the independent
    Normal-Wishart prior, which means the drawback of requiring a fixed
    covariance matrix is eliminated.
    """

    def __init__(self, loc=0.0, scale=1.0, size=1, preload=1, *args, **kwargs):
        # Initialize parameters
        # Note: dim refers to k in the description above. T and M are only
        #       determined by the `endog`, `exog`, and `precision` params.
        self.dim, self.loc, self.scale = self._process_parameters(loc, scale)
        self.prior_loc = self.loc
        self.prior_scale = self.scale

        # Cache some calculated quantities
        self.cholesky = np.linalg.cholesky(self.prior_scale)
        # TODO use cholesky in inversion
        self._prior_inv_scale = np.linalg.inv(self.prior_scale)
        self._prior_inv_scale_loc = np.dot(self._prior_inv_scale,
                                           self.prior_loc)[:,np.newaxis]

        # Setup holder variables for posterior-related quantities
        self._precision = None
        self._exog = None
        self._endog = None

        # Setup holder variables for calculated quantities
        self._ZH = None
        self._ZHZ = np.zeros((self.dim, self.dim), order='F')
        self._ZHy = np.zeros((self.dim,), order='F')
        self._posterior_loc = None
        self._posterior_scale = None
        self._posterior_cholesky = None

        # Set the flag to use the prior
        self._use_posterior = False

        # Initialize the distribution
        distribution = 'normal'
        super(Normal, self).__init__(distribution,
                                     distribution_args=(),
                                     distribution_kwargs={},
                                     size=size, preload=preload,
                                     *args, **kwargs)

        # Alter the size (so that we can use normal rather than multivariate
        # normal)
        self.size = (self.dim,) + self.size
        self._cache_n *= self.dim
        self._cache_size = self.preload_size + self.size

    def _process_parameters(self, loc, scale):
        loc = np.array(loc)
        scale = np.array(scale)
        # The mean parameter is a vector
        if loc.ndim == 0:
            loc = loc[np.newaxis]
        elif not loc.ndim == 1:
            raise ValueError('Location vector must be 1-dimensional;'
                             ' Got loc.ndim = %d' % loc.ndim)
        dim = loc.shape[0]

        # Expand scalars and vectors to a scale matrix
        if scale.ndim == 0:
            scale = np.eye(dim) * scale
        elif scale.ndim == 1:
            scale = np.diag(scale)

        # Make sure scale matrix is square, dimension matches loc
        if not scale.shape == (dim, dim):
            raise ValueError('Scale matrix must be a square matrix whose'
                             ' dimension matches the provided location vector.'
                             ' Required %s, got %s'
                             % ((dim, dim), scale.shape))

        return dim, loc, scale

    @property
    def precision(self):
        return self._precision
    @precision.setter
    def precision(self, value):
        # Set the value
        self._precision = np.array(value, dtype=float)

        # Clear calculated quantities
        self._ZH = None
        self._ZHZ = None
        self._ZHy = None
        self._posterior_loc = None
        self._posterior_scale = None
        self._posterior_cholesky = None

        # Set the posterior flag
        self._use_posterior = True
    @precision.deleter
    def precision(self):
        # Clear precision
        self._precision = None

        # Clear calculated quantities
        self._ZH = None
        self._ZHZ = None
        self._ZHy = None
        self._posterior_loc = None
        self._posterior_scale = None
        self._posterior_cholesky = None

        # Recalculate posterior flag
        self._use_posterior = not (
            self._exog is None and
            self._endog is None and
            self._posterior is None
        )

    @property
    def exog(self):
        return self._exog
    @exog.setter
    def exog(self, value):
        # Save the exogenous dataset
        value = np.array(value)

        # Check that dimensions match

        # Convert everything to a 2-dim array
        if value.ndim == 1:
            # In this case, assume just 1 equation and exogenous variable, so
            # the shape should be 1 x 1 x T
            value = value[np.newaxis, np.newaxis, :]
        elif value.ndim == 2:
            # In this case, assume that we have k x T, so the shape should be
            # 1 x k x T
            value = value[np.newaxis, :, :]
        elif not value.ndim == 3:
            raise ValueError('Invalid exogenous array dimensions. Required '
                             ' (neqs, ncoef, nobs), got %s' % str(value.shape))

        # Make sure the coefficients (k) is correct
        if not value.shape[1] == self.dim:
            raise ValueError('Invalid exogenous array dimensions. Required '
                             ' (neqs, %d, nobs), got %s'
                             % (self.dim, str(value.shape)))

        # If `endog` was provided, we have an independent assessment of T and M
        if self._endog is not None:
            if not value.shape[0:3:2] == self._endog.shape:
                raise ValueError('Invalid exogenous array dimensions. Required'
                                 ' (%d, %d, %d), got %s'
                                 % (self._endog.shape[0], self.dim,
                                    self._endog.shape[1], str(value.shape)))

        # If `precision` was provided, we have an independent assessment of M,
        if self._precision is not None:
            if self._precision.shape == ():
                if not value.shape[0] == 1:
                    raise ValueError('Invalid exogenous array dimensions.'
                                     ' Required shape (1, neqs, nobs), got %s'
                                     % str(value.shape[0]))
            elif not value.shape[0] == self._precision.shape[0]:
                raise ValueError('Invalid exogenous array dimensions. Required'
                                 ' shape (%d, neqs, nobs), got %s'
                                 % (self._precision.shape[0],
                                    str(value.shape[0])))

        # Set the underlying value
        self._exog = np.asfortranarray(value)

        # Clear calculated quantities
        self._ZH = None
        self._ZHZ = None
        self._ZHy = None
        self._posterior_loc = None
        self._posterior_scale = None
        self._posterior_cholesky = None

        # Set the posterior flag
        self._use_posterior = True
    @exog.deleter
    def exog(self):
        # Clear exog
        self._exog = None

        # Clear calculated quantities
        self._ZH = None
        self._ZHZ = None
        self._ZHy = None
        self._posterior_loc = None
        self._posterior_scale = None
        self._posterior_cholesky = None

        # Recalculate posterior flag
        self._use_posterior = not (
            self._exog is None and
            self._endog is None and
            self._posterior is None
        )

    @property
    def endog(self):
        return self._endog
    @endog.setter
    def endog(self, value):
        # Save the exogenous dataset
        value = np.array(value)

        # Check that dimensions match

        # Convert everything to a 2-dim array
        if value.ndim == 1:
            # Assume we have a (T,) matrix, so expand to (1,T)
            value = value[np.newaxis, :]
        elif not value.ndim == 2:
            raise ValueError('Invalid endogenous array dimension.'
                             ' Required (neqs, nobs), got %s'
                             % str(value.shape))

        # If `exog` was provided, we have an independent assessment of
        # T and M (the first dimensions of `exog`), so make sure it matches
        if self._exog is not None:
            if not value.shape[0:2] == self._exog.shape[0:3:2]:
                raise ValueError('Invalid endogenous array dimensions.'
                                 ' Required (%d, %d), got %s'
                                 % (self._exog.shape[0], self._endog.shape[-1],
                                    str(value.shape)))

        # If `precision` was provided, we have an independent assessment of M,
        if self._precision is not None:
            if self._precision.shape == ():
                if not value.shape[0] == 1:
                    raise ValueError('Invalid endogenous array dimensions.'
                                     ' Required shape (1, nobs), got %s'
                                     % str(value.shape[0]))
            elif not value.shape[0] == self._precision.shape[0]:
                raise ValueError('Invalid endogenous array dimensions.'
                                 ' Required shape (%d, nobs), got %s'
                                 % (self._precision.shape[0],
                                    str(value.shape)))

        # Set the underlying value
        self._endog = np.asfortranarray(value)

        # Clear calculated quantities
        self._ZHy = None
        self._posterior_loc = None

        # Set the posterior flag
        self._use_posterior = True
    @endog.deleter
    def endog(self):
        # Clear endog
        self._endog = None

        # Clear calculated quantities
        self._ZHy = None
        self._posterior_loc = None

        # Recalculate posterior flag
        self._use_posterior = not (
            self._exog is None and
            self._endog is None and
            self._posterior is None
        )
    
    @property
    def posterior_loc(self):
        if self._posterior_loc is None:
            # Getting this also implicitly calculates xTHy
            posterior_scale = self.posterior_scale

            # Calculate the posterior loc
            self._posterior_loc = np.dot(
                posterior_scale,
                self._prior_inv_scale_loc + self._ZHy[:, None]
            )
        return self._posterior_loc[:,0]

    @property
    def posterior_scale(self):
        if self._posterior_scale is None:
            # Get intermediate calculated quantity
            if self._ZHZ is None or self._ZHy is None:
                # Make sure we have required quantities
                if self._precision is None:
                    raise RuntimeError('Precision is not set; cannot calculate'
                                       ' posterior scale matrix.')
                if self._exog is None:
                    raise RuntimeError('Exogenous array is not set; cannot'
                                       ' calculate posterior scale matrix.')
                if self._endog is None:
                    raise RuntimeError('Endogenous array is not set; cannot'
                                       ' calculate posterior location vector.')

                # Iterate
                # self._ZHZ = np.zeros((self.dim, self.dim))
                # self._ZHy = np.zeros((self.dim,))
                # for t in range(self._exog.shape[-1]):
                #     self._ZH = np.dot(self._exog[:,:,t].T, self._precision)
                #     self._ZHZ += np.dot(self._ZH, self._exog[:,:,t])
                #     self._ZHy += np.dot(self._ZH, self._endog[:,t])


                M, k, T = self._exog.shape
                if M == 1:
                    self._ZH = self._precision * self._exog[0]
                    self._ZHZ = np.dot(self._ZH, self._exog[0].T)
                    self._ZHy = np.dot(self._ZH, self._endog[0])
                else:
                    if self._ZH is None:
                        self._ZH = np.zeros((k, M), order='F')
                    if self._ZHZ is None:
                        self._ZHZ = np.zeros((k, k), order='F')
                    if self._ZHy is None:
                        self._ZHy = np.zeros((k,), order='F')

                    _rvs_quantities(self._ZH, self._ZHZ, self._ZHy,
                                    self._endog, self._exog, self._precision.T,
                                    T, M, k)

            # Calculate the posterior scale
            self._posterior_scale = np.linalg.inv(
                self._prior_inv_scale + self._ZHZ
            )
        return self._posterior_scale

    @property
    def posterior_cholesky(self):
        if self._posterior_cholesky is None:
            self._posterior_cholesky = np.linalg.cholesky(self.posterior_scale)
        return self._posterior_cholesky

    def next(self):
        # Get the N(0,1) variate
        rvs = super(Normal, self).next()

        # Get loc and cholesky to transform to N(\beta, V) random variate
        if not self._use_posterior:
            loc = self.loc
            cholesky = self.cholesky
        else:
            loc = self.posterior_loc
            cholesky = self.posterior_cholesky

        # Perform the transformation
        if self.dim == 1:
            rvs = loc + cholesky * rvs[:,0]
        else:
            rvs = loc + np.dot(cholesky, rvs)[:,0]
        return rvs
