"""
Multivariate Kalman Filter (Python)

Author: Chad Fulton
License: Simplified-BSD

"""
from __future__ import division, absolute_import, print_function

from collections import namedtuple
from warnings import warn
import numpy as np
from scipy.linalg import solve_discrete_lyapunov, blas, lapack
from .representation import (
    FILTER_CONVENTIONAL, INVERT_UNIVARIATE, SOLVE_CHOLESKY, INVERT_NUMPY,
    STABILITY_FORCE_SYMMETRY,
    MEMORY_STORE_ALL, FilterResults
)

# Create a named tuple to simulate the statespace and kalman_filter classes
_statespace = namedtuple('statespace', 'initial_state initial_state_cov')
_properties = [
    'model',
    'filter_method', 'inversion_method', 'stability_method', 'conserve_memory',
    'tolerance', 'loglikelihood_burn', 'converged', 'period_converged',
    'filtered_state', 'filtered_state_cov', 'predicted_state',
    'predicted_state_cov', 'forecast', 'forecast_error', 'forecast_error_cov',
    'loglikelihood'
]
_kalman_filter = namedtuple('kalman_filter', ' '.join(_properties))


def kalman_filter(model, return_loglike=False):
    # Parameters
    dtype = model.dtype

    # Kalman filter properties
    filter_method = model.filter_method
    inversion_method = model.inversion_method
    stability_method = model.stability_method
    conserve_memory = model.conserve_memory
    tolerance = model.tolerance
    loglikelihood_burn = model.loglikelihood_burn

    # Check for acceptable values
    if not filter_method == FILTER_CONVENTIONAL:
        warn('The pure Python version of the kalman filter only supports the'
             ' conventional Kalman filter')
    implemented_inv_methods = INVERT_NUMPY | INVERT_UNIVARIATE | SOLVE_CHOLESKY
    if not inversion_method & implemented_inv_methods:
        warn('The pure Python version of the kalman filter only performs'
             ' inversion using `numpy.linalg.inv`.')
    if not tolerance == 0:
        warn('The pure Python version of the kalman filter does not check'
             ' for convergence.')

    # Convergence (this implementation does not consider convergence)
    converged = False
    period_converged = 0

    # Dimensions
    nobs = model.nobs
    k_endog = model.k_endog
    k_states = model.k_states
    k_posdef = model.k_posdef

    # Allocate memory for variables
    filtered_state = np.zeros((k_states, nobs), dtype=dtype)
    filtered_state_cov = np.zeros((k_states, k_states, nobs), dtype=dtype)
    predicted_state = np.zeros((k_states, nobs+1), dtype=dtype)
    predicted_state_cov = np.zeros((k_states, k_states, nobs+1), dtype=dtype)
    forecast = np.zeros((k_endog, nobs), dtype=dtype)
    forecast_error = np.zeros((k_endog, nobs), dtype=dtype)
    forecast_error_cov = np.zeros((k_endog, k_endog, nobs), dtype=dtype)
    loglikelihood = np.zeros((nobs+1,), dtype=dtype)

    # Selected state covariance matrix
    selected_state_cov = (
        np.dot(
            np.dot(model.selection[:, :, 0],
                   model.state_cov[:, :, 0]),
            model.selection[:, :, 0].T
        )
    )

    # Initial values
    if model.initialization == 'known':
        initial_state = model._initial_state.astype(dtype)
        initial_state_cov = model._initial_state_cov.astype(dtype)
    elif model.initialization == 'approximate_diffuse':
        initial_state = np.zeros((k_states,), dtype=dtype)
        initial_state_cov = (
            np.eye(k_states).astype(dtype) * model._initial_variance
        )
    elif model.initialization == 'stationary':
        initial_state = np.zeros((k_states,), dtype=dtype)
        initial_state_cov = solve_discrete_lyapunov(
            np.array(model.transition[:, :, 0], dtype=dtype),
            np.array(selected_state_cov[:, :], dtype=dtype),
        )
    else:
        raise RuntimeError('Statespace model not initialized.')

    # Copy initial values to predicted
    predicted_state[:, 0] = initial_state
    predicted_state_cov[:, :, 0] = initial_state_cov
    # print(predicted_state_cov[:, :, 0])

    # Setup indices for possibly time-varying matrices
    design_t = 0
    obs_intercept_t = 0
    obs_cov_t = 0
    transition_t = 0
    state_intercept_t = 0
    selection_t = 0
    state_cov_t = 0

    # Iterate forwards
    time_invariant = model.time_invariant
    for t in range(nobs):
        # Get indices for possibly time-varying arrays
        if not time_invariant:
            if model.design.shape[2] > 1:             design_t = t
            if model.obs_intercept.shape[1] > 1:      obs_intercept_t = t
            if model.obs_cov.shape[2] > 1:            obs_cov_t = t
            if model.transition.shape[2] > 1:         transition_t = t
            if model.state_intercept.shape[1] > 1:    state_intercept_t = t
            if model.selection.shape[2] > 1:          selection_t = t
            if model.state_cov.shape[2] > 1:          state_cov_t = t

        # Selected state covariance matrix
        if model.selection.shape[2] > 1 or model.state_cov.shape[2] > 1:
            selected_state_cov = (
                np.dot(
                    np.dot(model.selection[:, :, selection_t],
                           model.state_cov[:, :, state_cov_t]),
                    model.selection[:, :, selection_t].T
                )
            )

        # #### Forecast for time t
        # `forecast` $= Z_t a_t + d_t$
        #
        # *Note*: $a_t$ is given from the initialization (for $t = 0$) or
        # from the previous iteration of the filter (for $t > 0$).
        forecast[:, t] = (
            np.dot(model.design[:, :, design_t], predicted_state[:, t]) +
            model.obs_intercept[:, obs_intercept_t]
        )

        # *Intermediate calculation* (used just below and then once more)  
        # `tmp1` array used here, dimension $(m \times p)$  
        # $\\#_1 = P_t Z_t'$  
        # $(m \times p) = (m \times m) (p \times m)'$
        tmp1 = np.dot(predicted_state_cov[:, :, t],
                      model.design[:, :, design_t].T)

        # #### Forecast error for time t
        # `forecast_error` $\equiv v_t = y_t -$ `forecast`
        forecast_error[:, t] = model.obs[:, t] - forecast[:, t]

        # #### Forecast error covariance matrix for time t
        # $F_t \equiv Z_t P_t Z_t' + H_t$
        forecast_error_cov[:, :, t] = (
            np.dot(model.design[:, :, design_t], tmp1) +
            model.obs_cov[:, :, obs_cov_t]
        )
        # Store the inverse
        if k_endog == 1 and inversion_method & INVERT_UNIVARIATE:
            forecast_error_cov_inv = 1.0 / forecast_error_cov[0, 0, t]
            determinant = forecast_error_cov[0, 0, t]
            tmp2 = forecast_error_cov_inv * forecast_error[:, t]
            tmp3 = forecast_error_cov_inv * model.design[:, :, design_t]
        elif inversion_method & SOLVE_CHOLESKY:
            U, info = lapack.dpotrf(forecast_error_cov[:, :, t])
            determinant = np.product(U.diagonal())**2
            tmp2, info = lapack.dpotrs(U, forecast_error[:, t])
            tmp3, info = lapack.dpotrs(U, model.design[:, :, design_t])
        else:
            forecast_error_cov_inv = np.linalg.inv(forecast_error_cov[:, :, t])
            determinant = np.linalg.det(forecast_error_cov[:, :, t])
            tmp2 = np.dot(forecast_error_cov_inv, forecast_error[:, t])
            tmp3 = np.dot(forecast_error_cov_inv, model.design[:, :, design_t])

        # #### Filtered state for time t
        # $a_{t|t} = a_t + P_t Z_t' F_t^{-1} v_t$  
        # $a_{t|t} = 1.0 * \\#_1 \\#_2 + 1.0 a_t$
        filtered_state[:, t] = (
            predicted_state[:, t] + np.dot(tmp1, tmp2)
        )

        # #### Filtered state covariance for time t
        # $P_{t|t} = P_t - P_t Z_t' F_t^{-1} Z_t P_t$  
        # $P_{t|t} = P_t - \\#_1 \\#_3 P_t$  
        filtered_state_cov[:, :, t] = (
            predicted_state_cov[:, :, t] -
            np.dot(
                np.dot(tmp1, tmp3),
                predicted_state_cov[:, :, t]
            )
        )

        # #### Loglikelihood
        loglikelihood[t] = -0.5 * (
            np.log((2*np.pi)**k_endog * determinant) +
            np.dot(forecast_error[:, t], tmp2)
        )

        # #### Predicted state for time t+1
        # $a_{t+1} = T_t a_{t|t} + c_t$
        predicted_state[:, t+1] = (
            np.dot(model.transition[:, :, transition_t],
                   filtered_state[:, t]) +
            model.state_intercept[:, state_intercept_t]
        )

        # #### Predicted state covariance matrix for time t+1
        # $P_{t+1} = T_t P_{t|t} T_t' + Q_t^*$
        predicted_state_cov[:, :, t+1] = (
            np.dot(
                np.dot(model.transition[:, :, transition_t],
                       filtered_state_cov[:, :, t]),
                model.transition[:, :, transition_t].T
            ) + selected_state_cov
        )

        # Enforce symmetry of predicted covariance matrix
        predicted_state_cov[:, :, t+1] = (
            predicted_state_cov[:, :, t+1] + predicted_state_cov[:, :, t+1].T
        ) / 2

    if return_loglike:
            return np.array(loglikelihood)
    else:
        kwargs = dict(
            (k, v) for k, v in locals().items()
            if k in _kalman_filter._fields
        )
        kwargs['model'] = _statespace(
            initial_state=initial_state, initial_state_cov=initial_state_cov
        )
        kfilter = _kalman_filter(**kwargs)
        return FilterResults(model, kfilter)
