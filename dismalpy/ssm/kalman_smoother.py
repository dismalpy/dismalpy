"""
State Space Representation and Kalman Filter, Smoother

Author: Chad Fulton
License: Simplified-BSD
"""
from __future__ import division, absolute_import, print_function

import numpy as np
from .kalman_filter import KalmanFilter, FilterResults
from .tools import prefix_kalman_smoother_map

SMOOTHER_STATE = 0x01          # Durbin and Koopman (2012), Chapter 4.4.2
SMOOTHER_STATE_COV = 0x02      # ibid., Chapter 4.4.3
SMOOTHER_DISTURBANCE = 0x04    # ibid., Chapter 4.5
SMOOTHER_DISTURBANCE_COV = 0x08    # ibid., Chapter 4.5
SMOOTHER_ALL = (
    SMOOTHER_STATE | SMOOTHER_STATE_COV | SMOOTHER_DISTURBANCE |
    SMOOTHER_DISTURBANCE_COV
)

class KalmanSmoother(KalmanFilter):
    r"""
    State space representation of a time series process, with Kalman filter
    and smoother.
    """

    def __init__(self, *args, **kwargs):
        super(KalmanSmoother, self).__init__(*args, **kwargs)

        # Reset the results class
        self.results_class = kwargs.get('results_class', SmootherResults)

        # Setup the underlying Kalman smoother storage
        self._kalman_smoothers = {}

        self.smoother_output = kwargs.get(
            'smoother_output', SMOOTHER_ALL
        )

    @property
    def _kalman_smoother(self):
        prefix = self.prefix
        if prefix in self._kalman_smoothers:
            return self._kalman_smoothers[prefix]
        return None

    def _initialize_smoother(self, smoother_output=None, *args, **kwargs):
        if smoother_output is None:
            smoother_output = self.smoother_output

        # Make sure we have the required Kalman filter
        prefix, dtype, create_filter, create_statespace = (
            self._initialize_filter(*args, **kwargs)
        )

        # Determine if we need to (re-)create the smoother
        # (definitely need to recreate if we recreated the filter)
        create_smoother = create_filter or prefix not in self._kalman_smoothers
        if not create_smoother:
            kalman_smoother = self._kalman_smoothers[prefix]

            create_smoother = (
                not kalman_smoother.kfilter is self._kalman_filters[prefix]
            )

        # If the dtype-specific _kalman_smoother does not exist (or if we need
        # to re-create it), create it
        if create_smoother:
            # Setup the smoother
            cls = prefix_kalman_smoother_map[prefix]
            self._kalman_smoothers[prefix] = cls(
                self._statespaces[prefix], self._kalman_filters[prefix],
                smoother_output
            )
        # Otherwise, update the smoother parameters
        else:
            self._kalman_smoothers[prefix].set_smoother_output(smoother_output, False)

        return prefix, dtype, create_smoother, create_filter, create_statespace

    def smooth(self, smoother_output=None, results=None,
               *args, **kwargs):
        """
        Apply the Kalman smoother to the statespace model.

        Parameters
        ----------
        smoother_output : int, optional
            Determines which Kalman smoother output calculate. Default is all
            (including state, disturbances, and all covariances).
        results : class or object, optional
            If a class, then that class is instantiated and returned with the
            result of both filtering and smoothing.
            If an object, then that object is updated with the smoothing data.
            If None, then a FilterResults object is returned with both
            filtering and smoothing results.
        Returns
        -------
        FilterResults object
        """
        new_results = not isinstance(results, SmootherResults)

        # Set the class to be the default results class, if None provided
        if results is None:
            results = self.results_class

        # Initialize the smoother
        prefix, dtype, create_smoother, create_filter, create_statespace = (
        self._initialize_smoother(
            smoother_output, *args, **kwargs
        ))

        # Instantiate a new results object, if required
        if isinstance(results, type):
            if not issubclass(results, SmootherResults):
                raise ValueError('Invalid results class provided.')
            results = results(self)

        # Run the filter if necessary
        kfilter = self._kalman_filters[prefix]
        if not kfilter.t == self.nobs:
            self._initialize_state()
            kfilter()

        # Run the smoother
        smoother = self._kalman_smoothers[prefix]
        smoother()

        # Update the results object
        # Update the model features if we had to recreate the statespace
        if create_statespace:
            results.update_representation(self)
        if new_results or create_filter:
            results.update_filter(kfilter)
        results.update_smoother(smoother)

        return results


class SmootherResults(FilterResults):
    """
    Results from applying the Kalman smoother and/or filter to a state space
    model.

    Parameters
    ----------
    model : Representation
        A Statespace representation

    Attributes
    ----------
    nobs : int
        Number of observations.
    k_endog : int
        The dimension of the observation series.
    k_states : int
        The dimension of the unobserved state process.
    k_posdef : int
        The dimension of a guaranteed positive definite covariance matrix
        describing the shocks in the measurement equation.
    dtype : dtype
        Datatype of representation matrices
    prefix : str
        BLAS prefix of representation matrices
    shapes : dictionary of name:tuple
        A dictionary recording the shapes of each of the representation
        matrices as tuples.
    endog : array
        The observation vector.
    design : array
        The design matrix, :math:`Z`.
    obs_intercept : array
        The intercept for the observation equation, :math:`d`.
    obs_cov : array
        The covariance matrix for the observation equation :math:`H`.
    transition : array
        The transition matrix, :math:`T`.
    state_intercept : array
        The intercept for the transition equation, :math:`c`.
    selection : array
        The selection matrix, :math:`R`.
    state_cov : array
        The covariance matrix for the state equation :math:`Q`.
    missing : array of bool
        An array of the same size as `endog`, filled with boolean values that
        are True if the corresponding entry in `endog` is NaN and False
        otherwise.
    nmissing : array of int
        An array of size `nobs`, where the ith entry is the number (between 0
        and k_endog) of NaNs in the ith row of the `endog` array.
    time_invariant : bool
        Whether or not the representation matrices are time-invariant
    initialization : str
        Kalman filter initialization method.
    initial_state : array_like
        The state vector used to initialize the Kalamn filter.
    initial_state_cov : array_like
        The state covariance matrix used to initialize the Kalamn filter.
    filter_method : int
        Bitmask representing the Kalman filtering method
    inversion_method : int
        Bitmask representing the method used to invert the forecast error
        covariance matrix.
    stability_method : int
        Bitmask representing the methods used to promote numerical stability in
        the Kalman filter recursions.
    conserve_memory : int
        Bitmask representing the selected memory conservation method.
    tolerance : float
        The tolerance at which the Kalman filter determines convergence to
        steady-state.
    loglikelihood_burn : int
        The number of initial periods during which the loglikelihood is not
        recorded.
    converged : bool
        Whether or not the Kalman filter converged.
    period_converged : int
        The time period in which the Kalman filter converged.
    filtered_state : array
        The filtered state vector at each time period.
    filtered_state_cov : array
        The filtered state covariance matrix at each time period.
    predicted_state : array
        The predicted state vector at each time period.
    predicted_state_cov : array
        The predicted state covariance matrix at each time period.
    kalman_gain : array
        The Kalman gain at each time period.
    forecasts : array
        The one-step-ahead forecasts of observations at each time period.
    forecasts_error : array
        The forecast errors at each time period.
    forecasts_error_cov : array
        The forecast error covariance matrices at each time period.
    loglikelihood : array
        The loglikelihood values at each time period.
    collapsed_forecasts : array
        If filtering using collapsed observations, stores the one-step-ahead
        forecasts of collapsed observations at each time period.
    collapsed_forecasts_error : array
        If filtering using collapsed observations, stores the one-step-ahead
        forecast errors of collapsed observations at each time period.
    collapsed_forecasts_error_cov : array
        If filtering using collapsed observations, stores the one-step-ahead
        forecast error covariance matrices of collapsed observations at each
        time period.
    standardized_forecast_error : array
        The standardized forecast errors
    smoother_output : int
        Bitmask representing the generated Kalman smoothing output
    scaled_smoothed_estimator : array
        The scaled smoothed estimator at each time period.
    scaled_smoothed_estimator_cov : array
        The scaled smoothed estimator covariance matrices at each time period.
    smoothing_error : array
        The smoothing error covariance matrices at each time period.
    smoothed_state : array
        The smoothed state at each time period.
    smoothed_state_cov : array
        The smoothed state covariance matrices at each time period.
    smoothed_measurement_disturbance : array
        The smoothed measurement at each time period.
    smoothed_state_disturbance : array
        The smoothed state at each time period.
    smoothed_measurement_disturbance_cov : array
        The smoothed measurement disturbance covariance matrices at each time
        period.
    smoothed_state_disturbance_cov : array
        The smoothed state disturbance covariance matrices at each time period.
    """

    _smoother_attributes = [
        'smoother_output', 'scaled_smoothed_estimator',
        'scaled_smoothed_estimator_cov', 'smoothing_error',
        'smoothed_state', 'smoothed_state_cov',
        'smoothed_measurement_disturbance', 'smoothed_state_disturbance',
        'smoothed_measurement_disturbance_cov',
        'smoothed_state_disturbance_cov'
    ]

    _attributes = FilterResults._model_attributes + _smoother_attributes

    def update_smoother(self, smoother):
        # Copy the appropriate output
        attributes = []

        if smoother.smoother_output & (SMOOTHER_STATE | SMOOTHER_DISTURBANCE):
            attributes.append('scaled_smoothed_estimator')
        if smoother.smoother_output & (SMOOTHER_STATE_COV | SMOOTHER_DISTURBANCE_COV):
            attributes.append('scaled_smoothed_estimator_cov')
        if smoother.smoother_output & SMOOTHER_STATE:
            attributes.append('smoothed_state')
        if smoother.smoother_output & SMOOTHER_STATE_COV:
            attributes.append('smoothed_state_cov')
        if smoother.smoother_output & SMOOTHER_DISTURBANCE:
            attributes += [
                'smoothing_error',
                'smoothed_measurement_disturbance',
                'smoothed_state_disturbance'
            ]
        if smoother.smoother_output & SMOOTHER_DISTURBANCE_COV:
            attributes += [
                'smoothed_measurement_disturbance_cov',
                'smoothed_state_disturbance_cov'
            ]

        for name in self._smoother_attributes:
            if name in attributes:
                setattr(
                    self, name,
                    np.array(getattr(smoother, name, None), copy=True)
                )
            else:
                setattr(self, name, None)

        # Smoother output (note that this was unset just above)
        self.smoother_output = smoother.smoother_output

        # Adjustments

        # For r_t (and similarly for N_t), what was calculated was
        # r_T, ..., r_{-1}, and stored such that
        # scaled_smoothed_estimator[0] == r_{-1}. We only want r_0, ..., r_T
        # so exclude the zeroth element so that the time index is consistent
        # with the other returned output
        if 'scaled_smoothed_estimator' in attributes:
            self.scaled_smoothed_estimator = self.scaled_smoothed_estimator[:,1:]
        if 'scaled_smoothed_estimator_cov' in attributes:
            self.scaled_smoothed_estimator_cov = self.scaled_smoothed_estimator_cov[:,1:]
