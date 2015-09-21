"""
SARIMAX Model

Author: Chad Fulton
License: Simplified-BSD
"""
from __future__ import division, absolute_import, print_function

import numpy as np
from .mlemodel import MLEMixin, MLEResultsMixin
try:
    from statsmodels.tsa.statespace import varmax
    from statsmodels.tsa.statespace import mlemodel, structural
except ImportError:
    from .compat import mlemodel, structural

import statsmodels.base.wrapper as wrap

class UnobservedComponents(MLEMixin, structural.UnobservedComponents):
    r"""
    Univariate unobserved components time series model

    These are also known as structural time series models, and decompose a
    (univariate) time series into trend, seasonal, cyclical, and irregular
    components.

    Parameters
    ----------

    level : bool or string, optional
        Whether or not to include a level component. Default is False. Can also
        be a string specification of the level / trend component; see Notes
        for available model specification strings.
    trend : bool, optional
        Whether or not to include a trend component. Default is False. If True,
        `level` must also be True.
    seasonal_period : int or None, optional
        The period of the seasonal component. Default is None.
    cycle : bool, optional
        Whether or not to include a cycle component. Default is False.
    ar : int or None, optional
        The order of the autoregressive component. Default is None.
    exog : array_like or None, optional
        Exoenous variables.
    irregular : bool, optional
        Whether or not to include an irregular component. Default is False.
    stochastic_level : bool, optional
        Whether or not any level component is stochastic. Default is False.
    stochastic_trend : bool, optional
        Whether or not any trend component is stochastic. Default is False.
    stochastic_seasonal : bool, optional
        Whether or not any seasonal component is stochastic. Default is False.
    stochastic_cycle : bool, optional
        Whether or not any cycle component is stochastic. Default is False.
    damped_cycle : bool, optional
        Whether or not the cycle component is damped. Default is False.
    cycle_period_bounds : tuple, optional
        A tuple with lower and upper allowed bounds for the period of the
        cycle. If not provided, the following default bounds are used:
        (1) if no date / time information is provided, the frequency is
        constrained to be between zero and :math:`\pi`, so the period is
        constrained to be in [0.5, infinity].
        (2) If the date / time information is provided, the default bounds
        allow the cyclical component to be between 1.5 and 12 years; depending
        on the frequency of the endogenous variable, this will imply different
        specific bounds.

    Notes
    -----

    Thse models take the general form (see [1]_ Chapter 3.2 for all details)

    .. math::

        y_t = \mu_t + \gamma_t + c_t + \varepsilon_t

    where :math:`y_t` refers to the observation vector at time :math:`t`,
    :math:`\mu_t` refers to the trend component, :math:`\gamma_t` refers to the
    seasonal component, :math:`c_t` refers to the cycle, and
    :math:`\varepsilon_t` is the irregular. The modeling details of these
    components are given below.

    **Trend**

    The trend component is a dynamic extension of a regression model that
    includes an intercept and linear time-trend. It can be written:

    .. math::

        \mu_t = \mu_{t-1} + \beta_{t-1} + \eta_{t-1} \\
        \beta_t = \beta_{t-1} + \zeta_{t-1}

    where the level is a generalization of the intercept term that can
    dynamically vary across time, and the trend is a generalization of the
    time-trend such that the slope can dynamically vary across time.

    Here :math:`\eta_t \sim N(0, \sigma_\eta^2)` and
    :math:`\zeta_t \sim N(0, \sigma_\zeta^2)`.

    For both elements (level and trend), we can consider models in which:

    - The element is included vs excluded (if the trend is included, there must
      also be a level included).
    - The element is deterministic vs stochastic (i.e. whether or not the
      variance on the error term is confined to be zero or not)
    
    The only additional parameters to be estimated via MLE are the variances of
    any included stochastic components.

    The level/trend components can be specified using the boolean keyword
    arguments `level`, `stochastic_level`, `trend`, etc., or all at once as a
    string argument to `level`. The following table shows the available
    model specifications:

    +----------------------------------+--------------------------------------+--------------------+--------------------------------------------------+
    | Model name                       | Full string syntax                   | Abbreviated syntax | Model                                            |
    +==================================+======================================+====================+==================================================+
    | No trend                         | `'irregular'`                        | `'ntrend'`         | .. math:: y_t &= \varepsilon_t                   |
    +----------------------------------+--------------------------------------+--------------------+--------------------------------------------------+
    | Fixed intercept                  | `'fixed intercept'`                  |                    | .. math:: y_t &= \mu                             |
    +----------------------------------+--------------------------------------+--------------------+--------------------------------------------------+
    | Deterministic constant           | `'deterministic constant'`           | `'dconstant'`      | .. math:: y_t &= \mu + \varepsilon_t             |
    +----------------------------------+--------------------------------------+--------------------+--------------------------------------------------+
    | Local level                      | `'local level'`                      | `'llevel'`         | .. math:: y_t &= \mu_t + \varepsilon_t \\        |
    |                                  |                                      |                    |     \mu_t &= \mu_{t-1} + \eta_t                  |
    +----------------------------------+--------------------------------------+--------------------+--------------------------------------------------+
    | Random walk                      | `'random walk'`                      | `'rwalk'`          | .. math:: y_t &= \mu_t \\                        |
    |                                  |                                      |                    |     \mu_t &= \mu_{t-1} + \eta_t                  |
    +----------------------------------+--------------------------------------+--------------------+--------------------------------------------------+
    | Fixed slope                      | `'fixed slope'`                      |                    | .. math:: y_t &= \mu_t \\                        |
    |                                  |                                      |                    |     \mu_t &= \mu_{t-1} + \beta                   |
    +----------------------------------+--------------------------------------+--------------------+--------------------------------------------------+
    | Deterministic trend              | `'deterministic trend'`              | `'dtrend'`         | .. math:: y_t &= \mu_t + \varepsilon_t \\        |
    |                                  |                                      |                    |     \mu_t &= \mu_{t-1} + \beta                   |
    +----------------------------------+--------------------------------------+--------------------+--------------------------------------------------+
    | Local linear deterministic trend | `'local linear deterministic trend'` | `'lldtrend'`       | .. math:: y_t &= \mu_t + \varepsilon_t \\        |
    |                                  |                                      |                    |     \mu_t &= \mu_{t-1} + \beta + \eta_t          |
    +----------------------------------+--------------------------------------+--------------------+--------------------------------------------------+
    | Random walk with drift           | `'random walk with drift'`           | `'rwdrift'`        | .. math:: y_t &= \mu_t \\                        |
    |                                  |                                      |                    |     \mu_t &= \mu_{t-1} + \beta + \eta_t          |
    +----------------------------------+--------------------------------------+--------------------+--------------------------------------------------+
    | Local linear trend               | `'local linear trend'`               | `'lltrend'`        | .. math:: y_t &= \mu_t + \varepsilon_t \\        |
    |                                  |                                      |                    |     \mu_t &= \mu_{t-1} + \beta_{t-1} + \eta_t \\ |
    |                                  |                                      |                    |     \beta_t &= \beta_{t-1} + \zeta_t             |
    +----------------------------------+--------------------------------------+--------------------+--------------------------------------------------+
    | Smooth trend                     | `'smooth trend'`                     | `'strend'`         | .. math:: y_t &= \mu_t + \varepsilon_t \\        |
    |                                  |                                      |                    |     \mu_t &= \mu_{t-1} + \beta_{t-1} \\          |
    |                                  |                                      |                    |     \beta_t &= \beta_{t-1} + \zeta_t             |
    +----------------------------------+--------------------------------------+--------------------+--------------------------------------------------+
    | Random trend                     | `'random trend'`                     | `'rtrend'`         | .. math:: y_t &= \mu_t \\                        |
    |                                  |                                      |                    |     \mu_t &= \mu_{t-1} + \beta_{t-1} \\          |
    |                                  |                                      |                    |     \beta_t &= \beta_{t-1} + \zeta_t             |
    +----------------------------------+--------------------------------------+--------------------+--------------------------------------------------+

    Following the fitting of the model, the unobserved level and trend
    component time series are available in the results class in the
    `level` and `trend` attributes, respectively.

    **Seasonal**

    The seasonal component is modeled as:

    .. math::

        \gamma_t = - \sum_{j=1}^{s-1} \gamma_{t+1-j} + \omega_t \\
        \omega_t \sim N(0, \sigma_\omega^2)

    The periodicity (number of seasons) is s, and the defining character is
    that (without the error term), the seasonal components sum to zero across
    one complete cycle. The inclusion of an error term allows the seasonal
    effects to vary over time (if this is not desired, :math:`\sigma_\omega^2`
    can be set to zero using the `stochastic_seasonal=False` keyword argument).

    This component results in one parameter to be selected via maximum
    likelihood: :math:`\sigma_\omega^2`, and one parameter to be chosen, the
    number of seasons `s`.

    Following the fitting of the model, the unobserved seasonal component
    time series is available in the results class in the `seasonal`
    attribute.

    **Cycle**

    The cyclical component is intended to capture cyclical effects at time
    frames much longer than captured by the seasonal component. For example,
    in economics the cyclical term is often intended to capture the business
    cycle, and is then expected to have a period between "1.5 and 12 years"
    (see Durbin and Koopman).

    .. math::

        c_{t+1} & = \rho_c (\tilde c_t \cos \lambda_c t
                + \tilde c_t^* \sin \lambda_c) +
                \tilde \omega_t \\
        c_{t+1}^* & = \rho_c (- \tilde c_t \sin \lambda_c  t +
                \tilde c_t^* \cos \lambda_c) +
                \tilde \omega_t^* \\

    where :math:`\omega_t, \tilde \omega_t iid N(0, \sigma_{\tilde \omega}^2)`

    The parameter :math:`\lambda_c` (the frequency of the cycle) is an
    additional parameter to be estimated by MLE.

    If the cyclical effect is stochastic (`stochastic_cycle=True`), then there
    is another parameter to estimate (the variance of the error term - note
    that both of the error terms here share the same variance, but are assumed
    to have independent draws).

    If the cycle is damped (`damped_cycle=True`), then there is a third
    parameter to estimate, :math:`\rho_c`.

    In order to achieve cycles with the appropriate frequencies, bounds are
    imposed on the parameter :math:`\lambda_c` in estimation. These can be
    controlled via the keyword argument `cycle_period_bounds`, which, if
    specified, must be a tuple of bounds on the **period** `(lower, upper)`.
    The bounds on the frequency are then calculated from those bounds.

    The default bounds, if none are provided, are selected in the following
    way:

    1. If no date / time information is provided, the frequency is
       constrained to be between zero and :math:`\pi`, so the period is
       constrained to be in :math:`[0.5, \infty]`.
    2. If the date / time information is provided, the default bounds
       allow the cyclical component to be between 1.5 and 12 years; depending
       on the frequency of the endogenous variable, this will imply different
       specific bounds.

    Following the fitting of the model, the unobserved cyclical component
    time series is available in the results class in the `cycle`
    attribute.

    **Irregular**

    The irregular components are independent and identically distributed (iid):

    .. math::

        \varepsilon_t \sim N(0, \sigma_\varepsilon^2)

    **Autoregressive Irregular**

    An autoregressive component (often used as a replacement for the white
    noise irregular term) can be specified as:

    .. math::

        \varepsilon_t = \rho(L) \varepsilon_{t-1} + \epsilon_t \\
        \epsilon_t \sim N(0, \sigma_\epsilon^2)

    In this case, the AR order is specified via the `autoregressive` keyword,
    and the autoregressive coefficients are estimated.

    Following the fitting of the model, the unobserved autoregressive component
    time series is available in the results class in the `autoregressive`
    attribute.

    **Regression effects**

    Exogenous regressors can be pass to the `exog` argument. The regression
    coefficients will be estimated by maximum likelihood unless
    `mle_regression=False`, in which case the regression coefficients will be
    included in the state vector where they are essentially estimated via
    recursive OLS.

    If the regression_coefficients are included in the state vector, the
    recursive estimates are available in the results class in the
    `regression_coefficients` attribute.

    See Also
    --------
    dismalpy.ssm.mlemodel.MLEModel
    dismalpy.ssm.kalman_smoother.KalmanSmoother
    dismalpy.ssm.kalman_filter.KalmanFilter
    dismalpy.ssm.representation.Representation

    References
    ----------

    .. [1] Durbin, James, and Siem Jan Koopman. 2012.
       Time Series Analysis by State Space Methods: Second Edition.
       Oxford University Press.
    """

    def filter(self, params, transformed=True, cov_type=None, return_ssm=False,
               **kwargs):
        params = np.array(params, ndmin=1)

        # Transform parameters if necessary
        if not transformed:
            params = self.transform_params(params)
            transformed = True

        # Get the state space output
        result = super(UnobservedComponents, self).filter(
            params, transformed, cov_type, return_ssm=True, **kwargs)

        # Wrap in a results object
        if not return_ssm:
            result_kwargs = {}
            if cov_type is not None:
                result_kwargs['cov_type'] = cov_type
            result = UnobservedComponentsResultsWrapper(
                UnobservedComponentsResults(self, params, result,
                                            **result_kwargs)
            )

        return result
    filter.__doc__ = MLEMixin.filter.__doc__

    def smooth(self, params, transformed=True, cov_type=None, return_ssm=False,
               **kwargs):
        params = np.array(params, ndmin=1)

        if not transformed:
            params = self.transform_params(params)
            transformed = True

        # Get the state space output
        result = super(UnobservedComponents, self).smooth(
            params, transformed, cov_type, return_ssm=True, **kwargs)

        # Wrap in a results object
        if not return_ssm:
            result_kwargs = {}
            if cov_type is not None:
                result_kwargs['cov_type'] = cov_type
            result = UnobservedComponentsResultsWrapper(
                UnobservedComponentsResults(self, params, result,
                                            **result_kwargs)
            )

        return result
    smooth.__doc__ = MLEMixin.smooth.__doc__


class UnobservedComponentsResults(MLEResultsMixin,
                                  structural.UnobservedComponentsResults):
    """
    Class to hold results from fitting an unobserved components model.

    Parameters
    ----------
    model : UnobservedComponents instance
        The fitted model instance

    Attributes
    ----------
    specification : dictionary
        Dictionary including all attributes from the unobserved components
        model instance.

    See Also
    --------
    dismalpy.ssm.mlemodel.MLEResults
    dismalpy.ssm.kalman_smoother.SmootherResults
    dismalpy.ssm.kalman_filter.FilterResults
    dismalpy.ssm.representation.FrozenRepresentation
    """
    pass


class UnobservedComponentsResultsWrapper(
        mlemodel.MLEResultsWrapper):
    _attrs = {}
    _wrap_attrs = wrap.union_dicts(
        mlemodel.MLEResultsWrapper._wrap_attrs, _attrs)
    _methods = {}
    _wrap_methods = wrap.union_dicts(
        mlemodel.MLEResultsWrapper._wrap_methods, _methods)
wrap.populate_wrapper(UnobservedComponentsResultsWrapper,
                      UnobservedComponentsResults)
