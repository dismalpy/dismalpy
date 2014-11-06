"""
Tests for collapsed observation vector

Author: Chad Fulton
License: Simplified-BSD
"""
from __future__ import division, absolute_import, print_function

import numpy as np
import pandas as pd
import os

from dismalpy import ssm
import dismalpy.ssm.tests.results_kalman as results_kalman_filter
from numpy.testing import assert_almost_equal
from nose.exc import SkipTest

current_path = os.path.dirname(os.path.abspath(__file__))


class Clark1989(ssm.Representation):
    """
    Clark's (1989) bivariate unobserved components model of real GDP (as
    presented in Kim and Nelson, 1999)

    Tests two-dimensional observation data.

    Test data produced using GAUSS code described in Kim and Nelson (1999) and
    found at http://econ.korea.ac.kr/~cjkim/SSMARKOV.htm

    See `results.results_kalman_filter` for more information.
    """
    def __init__(self, dtype=float, **kwargs):
        self.true = results_kalman_filter.uc_bi
        self.true_states = pd.DataFrame(self.true['states'])

        # GDP and Unemployment, Quarterly, 1948.1 - 1995.3
        data = pd.DataFrame(
            self.true['data'],
            index=pd.date_range('1947-01-01', '1995-07-01', freq='QS'),
            columns=['GDP', 'UNEMP']
        )[4:]
        data['GDP'] = np.log(data['GDP'])
        data['UNEMP'] = (data['UNEMP']/100)

        k_states = 6
        super(Clark1989, self).__init__(data, k_states=k_states, **kwargs)

        # Statespace representation
        self.design[:, :, 0] = [[1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1]]
        self.transition[
            ([0, 0, 1, 1, 2, 3, 4, 5],
             [0, 4, 1, 2, 1, 2, 4, 5],
             [0, 0, 0, 0, 0, 0, 0, 0])
        ] = [1, 1, 0, 0, 1, 1, 1, 1]
        self.selection = np.eye(self.k_states)

        # Update matrices with given parameters
        (sigma_v, sigma_e, sigma_w, sigma_vl, sigma_ec,
         phi_1, phi_2, alpha_1, alpha_2, alpha_3) = np.array(
            self.true['parameters'],
        )
        self.design[([1, 1, 1], [1, 2, 3], [0, 0, 0])] = [
            alpha_1, alpha_2, alpha_3
        ]
        self.transition[([1, 1], [1, 2], [0, 0])] = [phi_1, phi_2]
        self.obs_cov[1, 1, 0] = sigma_ec**2
        self.state_cov[
            np.diag_indices(k_states)+(np.zeros(k_states, dtype=int),)] = [
            sigma_v**2, sigma_e**2, 0, 0, sigma_w**2, sigma_vl**2
        ]

        # Initialization
        initial_state = np.zeros((k_states,))
        initial_state_cov = np.eye(k_states)*100

        # Initialization: self.modification
        initial_state_cov = np.dot(
            np.dot(self.transition[:, :, 0], initial_state_cov),
            self.transition[:, :, 0].T
        )
        self.initialize_known(initial_state, initial_state_cov)

    def test_forecasts(self):
        assert_almost_equal(
            self.results_a.forecasts[0,:],
            self.results_b.forecasts[0,:], 9
        )

    def test_forecasts_error(self):
        assert_almost_equal(
            self.results_a.forecasts_error[0,:],
            self.results_b.forecasts_error[0,:], 9
        )

    def test_forecasts_error_cov(self):
        assert_almost_equal(
            self.results_a.forecasts_error_cov[0,0,:],
            self.results_b.forecasts_error_cov[0,0,:], 9
        )

    def test_filtered_state(self):
        assert_almost_equal(
            self.results_a.filtered_state,
            self.results_b.filtered_state, 8
        )

    def test_filtered_state_cov(self):
        assert_almost_equal(
            self.results_a.filtered_state_cov,
            self.results_b.filtered_state_cov, 9
        )

    def test_predicted_state(self):
        assert_almost_equal(
            self.results_a.predicted_state,
            self.results_b.predicted_state, 8
        )

    def test_predicted_state_cov(self):
        assert_almost_equal(
            self.results_a.predicted_state_cov,
            self.results_b.predicted_state_cov, 9
        )

    def test_loglike(self):
        assert_almost_equal(
            self.results_a.loglikelihood,
            self.results_b.loglikelihood, 7
        )

    def test_smoothed_states(self):
        assert_almost_equal(
            self.results_a.smoothed_state,
            self.results_b.smoothed_state, 9
        )

    def test_smoothed_states_cov(self):
        assert_almost_equal(
            self.results_a.smoothed_state_cov,
            self.results_b.smoothed_state_cov, 9
        )

    def test_smoothed_measurement_disturbance(self):
        assert_almost_equal(
            self.results_a.smoothed_measurement_disturbance,
            self.results_b.smoothed_measurement_disturbance, 9
        )

    def test_smoothed_measurement_disturbance_cov(self):
        assert_almost_equal(
            self.results_a.smoothed_measurement_disturbance_cov,
            self.results_b.smoothed_measurement_disturbance_cov, 9
        )

    def test_smoothed_state_disturbance(self):
        assert_almost_equal(
            self.results_a.smoothed_state_disturbance,
            self.results_b.smoothed_state_disturbance, 9
        )

    def test_smoothed_state_disturbance_cov(self):
        assert_almost_equal(
            self.results_a.smoothed_state_disturbance_cov,
            self.results_b.smoothed_state_disturbance_cov, 9
        )

    def test_simulation_smoothed_state(self):
        assert_almost_equal(
            self.sim_a.simulated_state,
            self.sim_a.simulated_state, 9
        )

    def test_simulation_smoothed_measurement_disturbance(self):
        assert_almost_equal(
            self.sim_a.simulated_measurement_disturbance,
            self.sim_a.simulated_measurement_disturbance, 9
        )

    def test_simulation_smoothed_state_disturbance(self):
        assert_almost_equal(
            self.sim_a.simulated_state_disturbance,
            self.sim_a.simulated_state_disturbance, 9
        )

class TestClark1989Conventional(Clark1989):

    def __init__(self, dtype=float, **kwargs):
        super(TestClark1989Conventional, self).__init__(dtype, **kwargs)

        # Conventional filtering, smoothing, and simulation smoothing
        self.filter_method = ssm.FILTER_CONVENTIONAL
        self.results_a = self.smooth()
        n_disturbance_variates = (self.k_endog + self.k_posdef) * self.nobs
        self.sim_a = self.simulation_smoother(
            disturbance_variates=np.zeros(n_disturbance_variates),
            initial_state_variates=np.zeros(self.k_states)
        )

        # Univariate filtering, smoothing, and simulation smoothing
        self.filter_method = ssm.FILTER_CONVENTIONAL | ssm.FILTER_COLLAPSED
        self.results_b = self.smooth()
        self.sim_b = self.simulation_smoother(
            disturbance_variates=np.zeros(n_disturbance_variates),
            initial_state_variates=np.zeros(self.k_states)
        )

class TestClark1989Univariate(Clark1989):

    def __init__(self, dtype=float, **kwargs):
        super(TestClark1989Univariate, self).__init__(dtype, **kwargs)

        # Conventional filtering, smoothing, and simulation smoothing
        self.filter_method = ssm.FILTER_UNIVARIATE
        self.results_a = self.smooth()
        n_disturbance_variates = (self.k_endog + self.k_posdef) * self.nobs
        self.sim_a = self.simulation_smoother(
            disturbance_variates=np.zeros(n_disturbance_variates),
            initial_state_variates=np.zeros(self.k_states)
        )

        # Univariate filtering, smoothing, and simulation smoothing
        self.filter_method = ssm.FILTER_UNIVARIATE | ssm.FILTER_COLLAPSED
        self.results_b = self.smooth()
        self.sim_b = self.simulation_smoother(
            disturbance_variates=np.zeros(n_disturbance_variates),
            initial_state_variates=np.zeros(self.k_states)
        )