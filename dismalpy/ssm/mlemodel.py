"""
State Space Model

Author: Chad Fulton
License: Simplified-BSD
"""
from __future__ import division, absolute_import, print_function

import numpy as np
from .simulation_smoother import SimulationSmoother, SimulationSmoothResults
try:
    from statsmodels.tsa.statespace import mlemodel, varmax
    from statsmodels.tsa.statespace.mlemodel import PredictionResultsWrapper
except ImportError:
    from .compat import mlemodel
    from .compat.mlemodel import PredictionResultsWrapper
import statsmodels.base.wrapper as wrap

class MLEMixin(object):

    def initialize_statespace(self, **kwargs):
        """
        Initialize the state space representation

        Parameters
        ----------
        **kwargs
            Additional keyword arguments to pass to the state space class
            constructor.

        """
        # (Now self.endog is C-ordered and in long format (nobs x k_endog). To
        # get F-ordered and in wide format just need to transpose)
        endog = self.endog.T

        # Instantiate the state space object
        self.ssm = SimulationSmoother(endog.shape[0], self.k_states, **kwargs)
        # Bind the data to the model
        self.ssm.bind(endog)

        # Other dimensions, now that `ssm` is available
        self.k_endog = self.ssm.k_endog

    def fit(self, *args, **kwargs):
        """
        Fits the model by maximum likelihood via Kalman filter.

        Parameters
        ----------
        start_params : array_like, optional
            Initial guess of the solution for the loglikelihood maximization.
            If None, the default is given by Model.start_params.
        transformed : boolean, optional
            Whether or not `start_params` is already transformed. Default is
            True.
        method : str, optional
            The `method` determines which solver from `scipy.optimize`
            is used, and it can be chosen from among the following strings:

            - 'newton' for Newton-Raphson, 'nm' for Nelder-Mead
            - 'bfgs' for Broyden-Fletcher-Goldfarb-Shanno (BFGS)
            - 'lbfgs' for limited-memory BFGS with optional box constraints
            - 'powell' for modified Powell's method
            - 'cg' for conjugate gradient
            - 'ncg' for Newton-conjugate gradient
            - 'basinhopping' for global basin-hopping solver

            The explicit arguments in `fit` are passed to the solver,
            with the exception of the basin-hopping solver. Each
            solver has several optional arguments that are not the same across
            solvers. See the notes section below (or scipy.optimize) for the
            available arguments and for the list of explicit arguments that the
            basin-hopping solver supports.
        cov_type : str, optional
            The `cov_type` keyword governs the method for calculating the
            covariance matrix of parameter estimates. Can be one of:

            - 'opg' for the outer product of gradient estimator
            - 'oim' for the observed information matrix estimator, calculated
              using the method of Harvey (1989)
            - 'cs' for the observed information matrix estimator, calculated
              using a numerical (complex step) approximation of the Hessian
              matrix.
            - 'delta' for the observed information matrix estimator, calculated
              using a numerical (complex step) approximation of the Hessian
              along with the delta method (method of propagation of errors)
              applied to the parameter transformation function
              `transform_params`.
            - 'robust' for an approximate (quasi-maximum likelihood) covariance
              matrix that may be valid even in the presense of some
              misspecifications. Intermediate calculations use the 'oim'
              method.
            - 'robust_cs' is the same as 'robust' except that the intermediate
              calculations use the 'cs' method.
        cov_kwds : dict or None, optional
            See `MLEResults.get_robustcov_results` for a description required
            keywords for alternative covariance estimators
        maxiter : int, optional
            The maximum number of iterations to perform.
        full_output : boolean, optional
            Set to True to have all available output in the Results object's
            mle_retvals attribute. The output is dependent on the solver.
            See LikelihoodModelResults notes section for more information.
        disp : boolean, optional
            Set to True to print convergence messages.
        callback : callable callback(xk), optional
            Called after each iteration, as callback(xk), where xk is the
            current parameter vector.
        return_params : boolean, optional
            Whether or not to return only the array of maximizing parameters.
            Default is False.
        optim_hessian : {'opg','oim','cs'}, optional
            The method by which the Hessian is numerically approximated. 'opg'
            uses outer product of gradients, 'oim' uses the information
            matrix formula from Harvey (1989), and 'cs' uses second-order
            complex step differentiation. This keyword is only relevant if the
            optimization method uses the Hessian matrix.
        **kwargs
            Additional keyword arguments to pass to the optimizer.

        Returns
        -------
        MLEResults

        See also
        --------
        statsmodels.base.model.LikelihoodModel.fit
        MLEResults
        """
        # Save the return_params argument
        return_params = kwargs.get('return_params', False)
        kwargs['return_params'] = False
        results = super(MLEMixin, self).fit(*args, **kwargs)

        # Construct the results class if desired
        if return_params:
            results = results.params
        else:
            result_kwargs = {}
            if 'cov_type' in kwargs:
                result_kwargs['cov_type'] = kwargs['cov_type']
            if 'cov_kwds' in kwargs:
                result_kwargs['cov_kwds'] = kwargs['cov_kwds']
            mlefit = results.mlefit
            results = self.smooth(results.params, **result_kwargs)
            results.mlefit = mlefit
            results.mle_retvals = mlefit.mle_retvals
            results.mle_settings = mlefit.mle_settings

        return results

    def filter(self, params, transformed=True, cov_type=None, cov_kwds=None,
               return_ssm=False, **kwargs):
        """
        Kalman filtering

        Parameters
        ----------
        params : array_like
            Array of parameters at which to evaluate the loglikelihood
            function.
        transformed : boolean, optional
            Whether or not `params` is already transformed. Default is True.
        return_ssm : boolean,optional
            Whether or not to return only the state space output or a full
            results object. Default is to return a full results object.
        cov_type : str, optional
            See `MLEResults.fit` for a description of covariance matrix types
            for results object.
        cov_kwds : dict or None, optional
            See `MLEResults.get_robustcov_results` for a description required
            keywords for alternative covariance estimators
        **kwargs
            Additional keyword arguments to pass to the Kalman filter. See
            `KalmanFilter.filter` for more details.
        """
        params = np.array(params, ndmin=1)

        # Transform parameters if necessary
        if not transformed:
            params = self.transform_params(params)
            transformed = True

        # Get the state space output
        results = super(MLEMixin, self).filter(params, transformed,
                                               return_ssm=True, **kwargs)

        # Wrap in a results object
        if not return_ssm:
            result_kwargs = {}
            if cov_type is not None:
                result_kwargs['cov_type'] = cov_type
            if cov_kwds is not None:
                result_kwargs['cov_kwds'] = cov_kwds
            results = MLEResultsWrapper(
                MLEResults(self, params, results, **result_kwargs)
            )

        return results

    def smooth(self, params, transformed=True, cov_type=None, cov_kwds=None,
               return_ssm=False, **kwargs):
        """
        Kalman smoothing

        Parameters
        ----------
        params : array_like
            Array of parameters at which to evaluate the loglikelihood
            function.
        transformed : boolean, optional
            Whether or not `params` is already transformed. Default is True.
        return_ssm : boolean,optional
            Whether or not to return only the state space output or a full
            results object. Default is to return a full results object.
        cov_type : str, optional
            See `MLEResults.fit` for a description of covariance matrix types
            for results object.
        cov_kwds : dict or None, optional
            See `MLEResults.get_robustcov_results` for a description required
            keywords for alternative covariance estimators
        **kwargs
            Additional keyword arguments to pass to the Kalman filter. See
            `KalmanFilter.filter` for more details.
        """
        params = np.array(params, ndmin=1)

        if not transformed:
            params = self.transform_params(params)
        self.update(params, transformed=True)

        # Save the parameter names
        self.data.param_names = self.param_names

        # Get the state space output
        results = self.ssm.smooth(**kwargs)

        # Wrap in a results object
        if not return_ssm:
            result_kwargs = {}
            if cov_type is not None:
                result_kwargs['cov_type'] = cov_type
            if cov_kwds is not None:
                result_kwargs['cov_kwds'] = cov_kwds
            results = MLEResultsWrapper(
                MLEResults(self, params, results, **result_kwargs)
            )

        return results

    def simulation_smoother(self, **kwargs):
        """
        Retrieve a simulation smoother for the statespace model.

        Parameters
        ----------
        simulation_output : int, optional
            Determines which simulation smoother output is calculated.
            Default is all (including state and disturbances).
        simulation_smooth_results_class : class, optional
            Default results class to use to save output of simulation
            smoothing. Default is `SimulationSmoothResults`. If specified,
            class must extend from `SimulationSmoothResults`.
        prefix : string
            The prefix of the datatype. Usually only used internally.
        **kwargs
            Additional keyword arguments, used to set the simulation output.
            See `set_simulation_output` for more details.

        Returns
        -------
        SimulationSmoothResults
        """
        return self.ssm.simulation_smoother(**kwargs)


class MLEModel(MLEMixin, mlemodel.MLEModel):
    r"""
    State space model for maximum likelihood estimation

    Parameters
    ----------
    endog : array_like
        The observed time-series process :math:`y`
    k_states : int
        The dimension of the unobserved state process.
    exog : array_like, optional
        Array of exogenous regressors, shaped nobs x k. Default is no
        exogenous regressors.
    dates : array-like of datetime, optional
        An array-like object of datetime objects. If a Pandas object is given
        for endog, it is assumed to have a DateIndex.
    freq : str, optional
        The frequency of the time-series. A Pandas offset or 'B', 'D', 'W',
        'M', 'A', or 'Q'. This is optional if dates are given.
    **kwargs
        Keyword arguments may be used to provide default values for state space
        matrices or for Kalman filtering options. See `Representation`, and
        `KalmanFilter` for more details.

    Attributes
    ----------
    ssm : KalmanFilter
        Underlying state space representation.

    Notes
    -----
    This class wraps the state space model with Kalman filtering to add in
    functionality for maximum likelihood estimation. In particular, it adds
    the concept of updating the state space representation based on a defined
    set of parameters, through the `update` method or `updater` attribute (see
    below for more details on which to use when), and it adds a `fit` method
    which uses a numerical optimizer to select the parameters that maximize
    the likelihood of the model.

    The `start_params` `update` method must be overridden in the
    child class (and the `transform` and `untransform` methods, if needed).

    See Also
    --------
    MLEResults
    dismalpy.ssm.simulation_smoother.SimulationSmoother
    dismalpy.ssm.kalman_smoother.KalmanSmoother
    dismalpy.ssm.kalman_filter.KalmanFilter
    dismalpy.ssm.representation.Representation
    """
    pass


class MLEResultsMixin(object):
    def __init__(self, model, params, smoother_results, cov_type='opg',
                 cov_kwds=None, **kwargs):
        super(MLEResultsMixin, self).__init__(
            model, params, smoother_results,
            cov_type=cov_type, cov_kwds=cov_kwds, **kwargs
        )

        # Add the smoother results
        self.smoother_results = smoother_results

    @property
    def kalman_gain(self):
        """
        Kalman gain matrices
        """
        return self._kalman_gain
    @kalman_gain.setter
    def kalman_gain(self, value):
        self._kalman_gain = value


class MLEResults(MLEResultsMixin, mlemodel.MLEResults):
    r"""
    Class to hold results from fitting a state space model.

    Parameters
    ----------
    model : MLEModel instance
        The fitted model instance
    params : array
        Fitted parameters
    filter_results : KalmanFilter instance
        The underlying state space model and Kalman filter output

    Attributes
    ----------
    model : Model instance
        A reference to the model that was fit.
    filter_results : KalmanFilter instance
        The underlying state space model and Kalman filter output
    nobs : float
        The number of observations used to fit the model.
    params : array
        The parameters of the model.
    scale : float
        This is currently set to 1.0 and not used by the model or its results.

    See Also
    --------
    MLEModel
    dismalpy.ssm.kalman_smoother.SmootherResults
    dismalpy.ssm.kalman_filter.FilterResults
    dismalpy.ssm.representation.FrozenRepresentation
    """
    pass


class MLEResultsWrapper(mlemodel.MLEResultsWrapper):
    _attrs = {}
    _wrap_attrs = wrap.union_dicts(mlemodel.MLEResultsWrapper._wrap_attrs,
                                   _attrs)
    _methods = {}
    _wrap_methods = wrap.union_dicts(mlemodel.MLEResultsWrapper._wrap_methods,
                                     _methods)
wrap.populate_wrapper(MLEResultsWrapper, MLEResults)
