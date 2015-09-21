"""
VARMAX Model

Author: Chad Fulton
License: Simplified-BSD
"""
from __future__ import division, absolute_import, print_function

import numpy as np
from .mlemodel import MLEMixin, MLEResultsMixin
try:
    from statsmodels.tsa.statespace import varmax
    from statsmodels.tsa.statespace import mlemodel
except ImportError:
    from .compat import mlemodel, varmax
import statsmodels.base.wrapper as wrap

class VARMAX(MLEMixin, varmax.VARMAX):
    r"""
    Vector Autoregressive Moving Average with eXogenous regressors model

    Parameters
    ----------
    endog : array_like
        The observed time-series process :math:`y`, , shaped nobs x k_endog.
    exog : array_like, optional
        Array of exogenous regressors, shaped nobs x k.
    order : iterable
        The (p,q) order of the model for the number of AR and MA parameters to
        use.
    trend : {'nc', 'c'}, optional
        Parameter controlling the deterministic trend polynomial.
        Can be specified as a string where 'c' indicates a constant intercept
        and 'nc' indicates no intercept term.
    error_cov_type : {'diagonal', 'unstructured'}, optional
        The structure of the covariance matrix of the error term, where
        "unstructured" puts no restrictions on the matrix and "diagonal"
        requires it to be a diagonal matrix (uncorrelated errors). Default is
        "unstructured".
    measurement_error : boolean, optional
        Whether or not to assume the endogenous observations `endog` were
        measured with error. Default is False.
    enforce_stationarity : boolean, optional
        Whether or not to transform the AR parameters to enforce stationarity
        in the autoregressive component of the model. Default is True.
    enforce_invertibility : boolean, optional
        Whether or not to transform the MA parameters to enforce invertibility
        in the moving average component of the model. Default is True.
    **kwargs
        Keyword arguments may be used to provide default values for state space
        matrices or for Kalman filtering options. See `Representation`, and
        `KalmanFilter` for more details.

    Attributes
    ----------
    order : iterable
        The (p,q) order of the model for the number of AR and MA parameters to
        use.
    trend : {'nc', 'c'}, optional
        Parameter controlling the deterministic trend polynomial.
        Can be specified as a string where 'c' indicates a constant intercept
        and 'nc' indicates no intercept term.
    error_cov_type : {'diagonal', 'unstructured'}, optional
        The structure of the covariance matrix of the error term, where
        "unstructured" puts no restrictions on the matrix and "diagonal"
        requires it to be a diagonal matrix (uncorrelated errors). Default is
        "unstructured".
    measurement_error : boolean, optional
        Whether or not to assume the endogenous observations `endog` were
        measured with error. Default is False.
    enforce_stationarity : boolean, optional
        Whether or not to transform the AR parameters to enforce stationarity
        in the autoregressive component of the model. Default is True.
    enforce_invertibility : boolean, optional
        Whether or not to transform the MA parameters to enforce invertibility
        in the moving average component of the model. Default is True.

    Notes
    -----
    Generically, the VARMAX model is specified (see for example chapter 18 of
    [1]_):

    .. math::

        y_t = \nu + A_1 y_{t-1} + \dots + A_p y_{t-p} + B x_t + \epsilon_t +
        M_1 \epsilon_{t-1} + \dots M_q \epsilon_{t-q}

    where :math:`\epsilon_t \sim N(0, \Omega)`, and where :math:`y_t` is a
    `k_endog x 1` vector. Additionally, this model allows considering the case
    where the variables are measured with error.

    Note that in the full VARMA(p,q) case there is a fundamental identification
    problem in that the coefficient matrices :math:`\{A_i, M_j\}` are not
    generally unique, meaning that for a given time series process there may
    be multiple sets of matrices that equivalently represent it. See Chapter 12
    of [1]_ for more informationl. Although this class can be used to estimate
    VARMA(p,q) models, a warning is issued to remind users that no steps have
    been taken to ensure identification in this case.

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
        results = super(VARMAX, self).filter(params, transformed, cov_type,
                                             return_ssm=True, **kwargs)

        # Wrap in a results object
        if not return_ssm:
            result_kwargs = {}
            if cov_type is not None:
                result_kwargs['cov_type'] = cov_type
            results = VARMAXResultsWrapper(
                VARMAXResults(self, params, results, **result_kwargs)
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
        results = super(VARMAX, self).smooth(params, transformed, cov_type,
                                             return_ssm=True, **kwargs)

        # Wrap in a results object
        if not return_ssm:
            result_kwargs = {}
            if cov_type is not None:
                result_kwargs['cov_type'] = cov_type
            results = VARMAXResultsWrapper(
                VARMAXResults(self, params, results, **result_kwargs)
            )

        return results
    smooth.__doc__ = MLEMixin.smooth.__doc__


class VARMAXResults(MLEResultsMixin, varmax.VARMAXResults):
    """
    Class to hold results from fitting an VARMAX model.

    Parameters
    ----------
    model : VARMAX instance
        The fitted model instance

    Attributes
    ----------
    specification : dictionary
        Dictionary including all attributes from the VARMAX model instance.
    coefficient_matrices_var : array
        Array containing autoregressive lag polynomial coefficient matrices,
        ordered from lowest degree to highest.
    coefficient_matrices_vma : array
        Array containing moving average lag polynomial coefficients,
        ordered from lowest degree to highest.

    See Also
    --------
    dismalpy.ssm.mlemodel.MLEResults
    dismalpy.ssm.kalman_smoother.SmootherResults
    dismalpy.ssm.kalman_filter.FilterResults
    dismalpy.ssm.representation.FrozenRepresentation
    """
    pass


class VARMAXResultsWrapper(mlemodel.MLEResultsWrapper):
    _attrs = {}
    _wrap_attrs = wrap.union_dicts(
        mlemodel.MLEResultsWrapper._wrap_attrs, _attrs)
    _methods = {}
    _wrap_methods = wrap.union_dicts(
        mlemodel.MLEResultsWrapper._wrap_methods, _methods)
wrap.populate_wrapper(VARMAXResultsWrapper, VARMAXResults)
