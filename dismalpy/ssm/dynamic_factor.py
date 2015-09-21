"""
Dynamic Factor Model

Author: Chad Fulton
License: Simplified-BSD
"""
from __future__ import division, absolute_import, print_function

import numpy as np
from .mlemodel import MLEMixin, MLEResultsMixin
try:
    raise ImportError
    from statsmodels.tsa.statespace import mlemodel, dynamic_factor
except ImportError:
    from .compat import mlemodel, dynamic_factor
import statsmodels.base.wrapper as wrap

class DynamicFactor(MLEMixin, dynamic_factor.DynamicFactor):
    r"""
    Dynamic factor model

    Parameters
    ----------
    endog : array_like
        The observed time-series process :math:`y`
    exog : array_like, optional
        Array of exogenous regressors for the observation equation, shaped
        nobs x k_exog.
    k_factors : int
        The number of unobserved factors.
    factor_order : int
        The order of the vector autoregression followed by the factors.
    error_cov_type : {'scalar', 'diagonal', 'unstructured'}, optional
        The structure of the covariance matrix of the observation error term,
        where "unstructured" puts no restrictions on the matrix, "diagonal"
        requires it to be any diagonal matrix (uncorrelated errors), and
        "scalar" requires it to be a scalar times the identity matrix. Default
        is "diagonal".
    error_order : int, optional
        The order of the vector autoregression followed by the observation
        error component. Default is None, corresponding to white noise errors.
    error_var : boolean, optional
        Whether or not to model the errors jointly via a vector autoregression,
        rather than as individual autoregressions. Has no effect unless
        `error_order` is set. Default is False.
    enforce_stationarity : boolean, optional
        Whether or not to transform the AR parameters to enforce stationarity
        in the autoregressive component of the model. Default is True.
    **kwargs
        Keyword arguments may be used to provide default values for state space
        matrices or for Kalman filtering options. See `Representation`, and
        `KalmanFilter` for more details.

    Attributes
    ----------
    exog : array_like, optional
        Array of exogenous regressors for the observation equation, shaped
        nobs x k_exog.
    k_factors : int
        The number of unobserved factors.
    factor_order : int
        The order of the vector autoregression followed by the factors.
    error_cov_type : {'diagonal', 'unstructured'}
        The structure of the covariance matrix of the error term, where
        "unstructured" puts no restrictions on the matrix and "diagonal"
        requires it to be a diagonal matrix (uncorrelated errors).
    error_order : int
        The order of the vector autoregression followed by the observation
        error component.
    error_var : boolean
        Whether or not to model the errors jointly via a vector autoregression,
        rather than as individual autoregressions. Has no effect unless
        `error_order` is set.
    enforce_stationarity : boolean, optional
        Whether or not to transform the AR parameters to enforce stationarity
        in the autoregressive component of the model. Default is True.

    Notes
    -----
    The dynamic factor model considered here is in the so-called static form,
    and is specified:

    .. math::

        y_t & = \Lambda f_t + B x_t + u_t \\
        f_t & = A_1 f_{t-1} + \dots + A_p f_{t-p} + \eta_t \\
        u_t & = C_1 u_{t-1} + \dots + C_1 f_{t-q} + \varepsilon_t

    where there are `k_endog` observed series and `k_factors` unobserved
    factors. Thus :math:`y_t` is a `k_endog` x 1 vector and :math:`f_t` is a
    `k_factors` x 1 vector.

    :math:`x_t` are optional exogenous vectors, shaped `k_exog` x 1.

    :math:`\eta_t` and :math:`\varepsilon_t` are white noise error terms. In
    order to identify the factors, :math:`Var(\eta_t) = I`. Denote
    :math:`Var(\varepsilon_t) \equiv \Sigma`.

    Options related to the unobserved factors:

    - `k_factors`: this is the dimension of the vector :math:`f_t`, above.
      To exclude factors completely, set `k_factors = 0`.
    - `factor_order`: this is the number of lags to include in the factor
      evolution equation, and corresponds to :math:`p`, above. To have static
      factors, set `factor_order = 0`.

    Options related to the observation error term :math:`u_t`:

    - `error_order`: the number of lags to include in the error evolution
      equation; corresponds to :math:`q`, above. To have white noise errors,
      set `error_order = 0` (this is the default).
    - `error_cov_type`: this controls the form of the covariance matrix
      :math:`\Sigma`. If it is "dscalar", then :math:`\Sigma = \sigma^2 I`. If
      it is "diagonal", then
      :math:`\Sigma = \text{diag}(\sigma_1^2, \dots, \sigma_n^2)`. If it is
      "unstructured", then :math:`\Sigma` is any valid variance / covariance
      matrix (i.e. symmetric and positive definite).
    - `error_var`: this controls whether or not the errors evolve jointly
      according to a VAR(q), or individually according to separate AR(q)
      processes. In terms of the formulation above, if `error_var = False`,
      then the matrices :math:C_i` are diagonal, otherwise they are general
      VAR matrices.

    See Also
    --------
    dismalpy.ssm.mlemodel.MLEModel
    dismalpy.ssm.kalman_smoother.KalmanSmoother
    dismalpy.ssm.kalman_filter.KalmanFilter
    dismalpy.ssm.representation.Representation

    References
    ----------
    .. [1] Lutkepohl, Helmut. 2007.
       New Introduction to Multiple Time Series Analysis.
       Berlin: Springer.

    """

    def filter(self, params, transformed=True, cov_type=None, return_ssm=False,
               **kwargs):
        params = np.array(params, ndmin=1)

        # Transform parameters if necessary
        if not transformed:
            params = self.transform_params(params)
            transformed = True

        # Get the state space output
        results = super(DynamicFactor, self).filter(
            params, transformed, cov_type, return_ssm=True, **kwargs)

        # Wrap in a results object
        if not return_ssm:
            result_kwargs = {}
            if cov_type is not None:
                result_kwargs['cov_type'] = cov_type
            results = DynamicFactorResultsWrapper(
                DynamicFactorResults(self, params, results, **result_kwargs)
            )

        return results
    filter.__doc__ = MLEMixin.filter.__doc__

    def smooth(self, params, transformed=True, cov_type=None, return_ssm=False,
               **kwargs):
        params = np.array(params, ndmin=1)

        if not transformed:
            params = self.transform_params(params)
            transformed = True

        # Get the state space output
        results = super(DynamicFactor, self).smooth(
            params, transformed, cov_type, return_ssm=True, **kwargs)

        # Wrap in a results object
        if not return_ssm:
            result_kwargs = {}
            if cov_type is not None:
                result_kwargs['cov_type'] = cov_type
            results = DynamicFactorResultsWrapper(
                DynamicFactorResults(self, params, results, **result_kwargs)
            )

        return results
    smooth.__doc__ = MLEMixin.smooth.__doc__


class DynamicFactorResults(MLEResultsMixin,
                           dynamic_factor.DynamicFactorResults):
    """
    Class to hold results from fitting an DynamicFactor model.

    Parameters
    ----------
    model : DynamicFactor instance
        The fitted model instance

    Attributes
    ----------
    specification : dictionary
        Dictionary including all attributes from the DynamicFactor model
        instance.
    coefficient_matrices_var : array
        Array containing autoregressive lag polynomial coefficient matrices,
        ordered from lowest degree to highest.

    See Also
    --------
    dismalpy.ssm.mlemodel.MLEResults
    dismalpy.ssm.kalman_smoother.SmootherResults
    dismalpy.ssm.kalman_filter.FilterResults
    dismalpy.ssm.representation.FrozenRepresentation
    """
    pass


class DynamicFactorResultsWrapper(mlemodel.MLEResultsWrapper):
    _attrs = {}
    _wrap_attrs = wrap.union_dicts(
        mlemodel.MLEResultsWrapper._wrap_attrs, _attrs)
    _methods = {}
    _wrap_methods = wrap.union_dicts(
        mlemodel.MLEResultsWrapper._wrap_methods, _methods)
wrap.populate_wrapper(DynamicFactorResultsWrapper, DynamicFactorResults)
