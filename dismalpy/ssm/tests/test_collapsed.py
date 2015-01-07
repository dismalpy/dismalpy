"""
Tests for collapsed observation vector

These tests cannot be run for the Clark 1989 model since the dimension of
observations (2) is smaller than the number of states (6).

Author: Chad Fulton
License: Simplified-BSD
"""
from __future__ import division, absolute_import, print_function

import numpy as np
import pandas as pd
import os

from dismalpy import ssm
import dismalpy.ssm.tests.results_kalman as results_kalman_filter
from numpy.testing import assert_almost_equal, assert_allclose
from nose.exc import SkipTest

current_path = os.path.dirname(os.path.abspath(__file__))


class Trivariate(ssm.Model):
    """
    Tests collapsing three-dimensional observation data to two-dimensional
    """
    def __init__(self, dtype=float, **kwargs):
        self.results = results_kalman_filter.uc_bi

        # GDP and Unemployment, Quarterly, 1948.1 - 1995.3
        data = pd.DataFrame(
            self.results['data'],
            index=pd.date_range('1947-01-01', '1995-07-01', freq='QS'),
            columns=['GDP', 'UNEMP']
        )[4:]
        data['GDP'] = np.log(data['GDP'])
        data['UNEMP'] = (data['UNEMP']/100)
        data['X'] = np.exp(data['GDP']) * data['UNEMP']

        k_states = 2
        super(Trivariate, self).__init__(data, k_states=k_states, **kwargs)

        # Statespace representation
        self.selection = np.eye(self.k_states)

        # Update matrices with test parameters
        self['design'] = np.array([[0.5, 0.2],
                                   [0,   0.8],
                                   [1,  -0.5]])
        self['transition'] = np.array([[0.4, 0.5],
                                       [1,   0]])
        self['obs_cov'] = np.diag([0.2, 1.1, 0.5])
        self['state_cov'] = np.diag([2., 1])

        # Initialization
        self.initialize_approximate_diffuse()

    def test_forecasts(self):
        assert_allclose(
            self.results_a.forecasts[0,:],
            self.results_b.forecasts[0,:],
        )

    def test_forecasts_error(self):
        assert_allclose(
            self.results_a.forecasts_error[0,:],
            self.results_b.forecasts_error[0,:]
        )

    def test_forecasts_error_cov(self):
        assert_allclose(
            self.results_a.forecasts_error_cov[0,0,:],
            self.results_b.forecasts_error_cov[0,0,:]
        )

    def test_filtered_state(self):
        assert_allclose(
            self.results_a.filtered_state,
            self.results_b.filtered_state
        )

    def test_filtered_state_cov(self):
        assert_allclose(
            self.results_a.filtered_state_cov,
            self.results_b.filtered_state_cov
        )

    def test_predicted_state(self):
        assert_allclose(
            self.results_a.predicted_state,
            self.results_b.predicted_state
        )

    def test_predicted_state_cov(self):
        assert_allclose(
            self.results_a.predicted_state_cov,
            self.results_b.predicted_state_cov
        )

    def test_loglike(self):
        assert_allclose(
            self.results_a.loglikelihood,
            self.results_b.loglikelihood
        )

    def test_smoothed_states(self):
        assert_allclose(
            self.results_a.smoothed_state,
            self.results_b.smoothed_state
        )

    def test_smoothed_states_cov(self):
        assert_allclose(
            self.results_a.smoothed_state_cov,
            self.results_b.smoothed_state_cov,
            rtol=1e-4
        )

    @SkipTest
    def test_smoothed_measurement_disturbance(self):
        assert_allclose(
            self.results_a.smoothed_measurement_disturbance,
            self.results_b.smoothed_measurement_disturbance
        )

    @SkipTest
    def test_smoothed_measurement_disturbance_cov(self):
        assert_allclose(
            self.results_a.smoothed_measurement_disturbance_cov,
            self.results_b.smoothed_measurement_disturbance_cov
        )

    def test_smoothed_state_disturbance(self):
        assert_allclose(
            self.results_a.smoothed_state_disturbance,
            self.results_b.smoothed_state_disturbance
        )

    def test_smoothed_state_disturbance_cov(self):
        assert_allclose(
            self.results_a.smoothed_state_disturbance_cov,
            self.results_b.smoothed_state_disturbance_cov
        )

    def test_simulation_smoothed_state(self):
        assert_allclose(
            self.sim_a.simulated_state,
            self.sim_a.simulated_state
        )

    def test_simulation_smoothed_measurement_disturbance(self):
        assert_allclose(
            self.sim_a.simulated_measurement_disturbance,
            self.sim_a.simulated_measurement_disturbance
        )

    def test_simulation_smoothed_state_disturbance(self):
        assert_allclose(
            self.sim_a.simulated_state_disturbance,
            self.sim_a.simulated_state_disturbance
        )

class TestTrivariateConventional(Trivariate):

    def __init__(self, dtype=float, **kwargs):
        super(TestTrivariateConventional, self).__init__(dtype, **kwargs)
        n_disturbance_variates = (self.k_endog + self.k_posdef) * self.nobs

        # Conventional filtering, smoothing, and simulation smoothing
        self.filter_method = ssm.FILTER_CONVENTIONAL
        self.results_a = self.smooth()
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

class TestTrivariateConventionalPartialMissing(Trivariate):

    def __init__(self, dtype=float, **kwargs):
        super(TestTrivariateConventionalPartialMissing, self).__init__(dtype, **kwargs)
        n_disturbance_variates = (self.k_endog + self.k_posdef) * self.nobs

        # Set partially missing data
        self.endog[:2, 10:180] = np.nan

        # Conventional filtering, smoothing, and simulation smoothing
        self.filter_method = ssm.FILTER_CONVENTIONAL
        self.results_a = self.smooth()
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

class TestTrivariateConventionalAllMissing(Trivariate):

    def __init__(self, dtype=float, **kwargs):
        super(TestTrivariateConventionalAllMissing, self).__init__(dtype, **kwargs)
        n_disturbance_variates = (self.k_endog + self.k_posdef) * self.nobs

        # Set partially missing data
        self.endog[:, 10:180] = np.nan

        # Conventional filtering, smoothing, and simulation smoothing
        self.filter_method = ssm.FILTER_CONVENTIONAL
        self.results_a = self.smooth()
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


class TestTrivariateUnivariate(Trivariate):

    def __init__(self, dtype=float, **kwargs):
        super(TestTrivariateUnivariate, self).__init__(dtype, **kwargs)

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

class TestTrivariateUnivariatePartialMissing(Trivariate):

    def __init__(self, dtype=float, **kwargs):
        super(TestTrivariateUnivariatePartialMissing, self).__init__(dtype, **kwargs)

        # Set partially missing data
        self.endog[:2, 10:180] = np.nan

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

class TestTrivariateUnivariateAllMissing(Trivariate):

    def __init__(self, dtype=float, **kwargs):
        super(TestTrivariateUnivariateAllMissing, self).__init__(dtype, **kwargs)

        # Set partially missing data
        self.endog[:, 10:180] = np.nan

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