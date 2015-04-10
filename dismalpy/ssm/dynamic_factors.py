"""
Dynamic Factors model

Author: Chad Fulton
License: Simplified-BSD
"""
from __future__ import division, absolute_import, print_function

from warnings import warn

import numpy as np
from .model import Model
from .tools import (
    concat
)

class FAVAR(Model):
    r"""
    Factor-Augmented Vector Autoregression

    Parameters
    ----------
    observed : array_like
        The observed fundamental time-series process :math:`y`
    informational : array_like, optional
        The observed informational time-series :math:`x`
    k_factors : int
        The number of unobserved factors
    order : int
        The order of the vector autoregression

    Notes
    -----

    **Model**

    The FAVAR model can be written as (see [1]_ for all notation and details):

    .. math::

        \begin{bmatrix} F_t \\ Y_t \end{bmatrix} & = \Phi(L)
            \begin{bmatrix} F_{t-1} \\ Y_{t-1} \end{bmatrix} + v_t \\
        X_t & = \Lambda^f F_t + \Lambda^y Y_t + e_t \\
        e_t \sim N(0, R) \\
        v_t \sim N(0, Q)

    with :math:`R` diagonal and :math:`Q` unrestricted. Let the dimensions of
    the model be denoted:

    - :math:`X_t` is :math:`N \times 1`
    - :math:`F_t` is :math:`K \times 1`
    - :math:`Y_t` is :math:`M \times 1`

    **State-space Representation**

    The mapping to state-space form is fairly straightforward, with:

    .. math::

        Z_t & = \begin{bmatrix} \Lambda^f & \Lambda^y \\ 0 & I \end{bmatrix} \\
        T_t & = T_{\Phi(L)}^c\\

    where :T_{\Phi(L)}^c: represents the companion matrix associated with the
    VAR(1) representation of :math:`\Phi(L)`.

    **Identification** (see [1]_, section II.D)

    Since the combination of the factors and the factor loading matrix are
    (jointly) fundamentally indeterminate, the first level of identifying
    restrictions require:

    .. math::

        \Lambda^f & = \begin{bmatrix} I_{K \times K} \\ \tilde \Lambda^f \end{bmatrix} \\
        \Lambda^y & = \begin{bmatrix} 0_{K \times M} \\ \tilde \Lambda^y \end{bmatrix} \\

    where:

    - :math:`\tilde \Lambda^f` is :math:`N-K \times K`
    - :math:`\tilde \Lambda^y` is :math:`N-K \times M`

    Additional identifying restrictions (e.g. for the identification of
    structural shocks) can be placed on the lag polynomial :math:`\Phi(L)`. In
    particular, we assume (as in [1]_) a recursive ordering of the factors and
    observables, such that:

    - :math:`F_{1,t}` can depend only on lagged values
    - :math:`F_{2,t}` can depend only on :math:`F_{1,t}` and lagged values
    - ...
    - :math:`F_{i,t}` can depend only on :math:`F_{1,t}, \dots, F_{i-1, t}` and lagged values
    - ...
    - :math:`Y_{1,t}` can depend only on `F_{t-1}`
    - ...
    - :math:`Y_{M,t}` can depend on all other variables contemporaneously

    **Parameters**

    There are parameters to be estimated in the following matrices:

    - :math:`\tilde \Lambda^f`: :math:`(N-K) \times K` parameters (due to fundamental identifying restriction)
    - :math:`\tilde \Lambda^y`: :math:`(N-K) \times M` parameters (due to fundamental identifying restriction)
    - :math:`R`: :math:`N` parameters (restriction to uncorrelated series - diagonal matrix)
    - :math:`T_{\Phi(L)}^c`: :math:`d*(K+M)^2: (where d is the lag order)
    - :math:`Q`: :math:`(K+M)**2` (unrestricted state covariance matrix)

    In total, there are:

    .. math::
        (N-K) * (K+M) + N + d*(K+M)^2 + (K+M)^2 = [(N-K) + (d+1)*(K+M)] (K+M) + N

    For example, if as in [1]_, :math:`N=109, d=1, K=5, M=1`, then the number
    of parameters is 805.

    References
    ----------
    .. [1] Bernanke, Ben S., Jean Boivin, and Piotr Eliasz. 2005.
       "Measuring the Effects of Monetary Policy: A Factor-Augmented Vector
       Autoregressive (FAVAR) Approach."
       The Quarterly Journal of Economics 120 (1): 387-422.
    .. [2] Durbin, James, and Siem Jan Koopman. 2012.
       Time Series Analysis by State Space Methods: Second Edition.
       Oxford University Press.
    """

    def __init__(self, observed, informational, k_factors, order=1, *args, **kwargs):
        # Model orders
        self.k_obs = observed.shape[1] if observed.ndim > 1 else 1
        self.k_info = informational.shape[1] if informational.ndim > 1 else 1
        self.k_factors = k_factors
        self.order = order

        k_posdef = k_factors + self.k_obs
        k_states = self.order * k_posdef
        

        # Parameter dimensions
        self.k_loadings = (self.k_info - self.k_factors) * k_posdef
        self.k_ar = self.order * k_posdef
        self.k_var = k_posdef * self.k_ar
        self.k_params = (
            self.k_loadings +   # design
            self.k_info +       # (diagonal) obs cov
            self.k_var +        # lag polynomial
            k_posdef**2         # state cov
        )
        
        # Construct the endogenous vector from the background and observed
        endog = concat((informational, observed), axis=1)
    
        # Initialize the statespace
        super(FAVAR, self).__init__(
            endog, k_states=k_states, k_posdef=k_posdef, *args, **kwargs
        )

        # Initialize known elements of the state space matrices
        # Note: no need to zet zeroing restrictions, because the matrices are
        # initialized to zeros. The commented lines are left for documenation
        # purposes.

        # The design matrix is the matrix of factor loadings, Lambda
        # From the fundamental identification issue, we can immediately set:
        self['design', :self.k_factors, :self.k_factors] = np.eye(self.k_factors)
        # self['design', :self.k_info, self.k_info:] = np.zeros(self.k_info, self.k_info)
        self['design', -self.k_obs:, self.k_factors:self.k_posdef] = np.eye(self.k_obs)

        # The observation covariance matrix has zeros in the last M rows and
        # columns, due to the way the observed series are integrated into the
        # measurement equation. But for now, we cannot accomodate a positive
        # semi-definite matrix, so make those diagonal elements very small
        self['obs_cov', -self.k_obs:, -self.k_obs:] = np.eye(self.k_obs)*1e-6

        # Initialize the transition matrix for a VAR structure
        if self.order > 1:
            self['transition', self.k_posdef:, :-self.k_posdef, 0] = np.eye(self.k_ar - self.k_posdef)

        # Identity for the selection matrix
        self['selection'] = np.zeros((self.k_states, self.k_posdef))
        self['selection', :self.k_posdef, :, 0] = np.eye(self.k_posdef)

        # Cache some slices
        start = 0; end = self.k_loadings;
        self._params_loadings = np.s_[start:end]
        start += self.k_loadings; end += self.k_info;
        self._params_obs_cov = np.s_[start:end]
        start += self.k_info; end += self.k_var;
        self._params_transition = np.s_[start:end]
        start += self.k_var; end += self.k_posdef**2;
        self._params_state_cov = np.s_[start:end]
        # Cache some indices
        self._design_idx = np.s_['design', self.k_factors:-self.k_obs, :self.k_posdef, 0]
        self._obs_cov_idx = ('obs_cov',) + np.diag_indices(self.k_info) + (0,)
        self._transition_idx = np.s_['transition', :self.k_posdef, :, 0]

        # Initialize as stationary
        self.initialize_stationary()

        # Set to use the univariate filter with observation collapsing
        self.filter_collapsed = True
        self.filter_univariate = True
        self._initialize_representation()
        self._statespace.subset_design = True
        self._statespace.companion_transition = True

    def _get_model_names(self, latex=False):
        return np.arange(self.k_params)

    @property
    def start_params(self):
        # Regress each X on Y, save OLS estimates and variances
        betas = []
        variances = [1] * self.k_factors # the covariances for the zeroes factor loadings
        exog = self.endog[-1][:,None]
        exog_pinv = np.linalg.pinv(exog)
        for i in range(self.k_factors, self.k_info - self.k_obs + 1):
            endog = self.endog[i]
            ols = exog_pinv.dot(endog)
            resid = endog - np.dot(exog, ols)

            betas.append(ols[0])
            variances.append(np.dot(resid.T, resid) / (self.nobs - 1))

        # Construct the final start parameters
        start_loadings = np.zeros((self.k_info - self.k_factors, self.k_posdef))
        start_loadings[:,-1] = betas
        return np.r_[
            start_loadings.reshape(start_loadings.size,),
            variances, # diagonals of obs cov
            [0] * self.k_var, # lag polynomial
            np.eye(self.k_posdef).reshape(self.k_posdef**2,), # state cov
        ]

    def update(self, params, *args, **kwargs):
        params = super(FAVAR, self).update(params, *args, **kwargs)

        # Update factor loadings (design matrix)
        self[self._design_idx] = np.reshape(
            params[self._params_loadings],
            (self.k_info - self.k_factors, self.k_posdef)
        )

        # Update observation covariance matrix
        self[self._obs_cov_idx] = params[self._params_obs_cov]

        # Update VAR lag polynomial (transition matrix)
        self[self._transition_idx] = np.reshape(
            params[self._params_transition], (self.k_posdef, self.k_states)
        )

        # Update state covariance matrix
        self['state_cov', :] = np.reshape(
            params[self._params_state_cov], (self.k_posdef, self.k_posdef, 1)
        )
