"""
Tests for smoothing and estimation of unobserved states and disturbances

- Predicted states: :math:`E(\alpha_t | Y_{t-1})`
- Filtered states: :math:`E(\alpha_t | Y_t)`
- Smoothed states: :math:`E(\alpha_t | Y_n)`
- Smoothed disturbances :math:`E(\varepsilon_t | Y_n), E(\eta_t | Y_n)`
- Simulation smoothing

Tested against R (FKF, KalmanRun / KalmanSmooth), Stata (sspace), and
MATLAB (ssm toolbox)

Author: Chad Fulton
License: Simplified-BSD
"""
from __future__ import division, absolute_import, print_function

import numpy as np
import pandas as pd
import os

from dismalpy.ssm import sarimax
from numpy.testing import assert_almost_equal
from nose.exc import SkipTest

current_path = os.path.dirname(os.path.abspath(__file__))


class TestStatesAR3(object):
    def __init__(self, alternate_timing=False, *args, **kwargs):
        # Dataset / Stata comparison
        path = current_path + os.sep + 'results/results_wpi1_ar3_stata.csv'
        self.stata = pd.read_csv(path)
        self.stata.index = pd.date_range(start='1960-01-01', periods=124,
                                         freq='QS')
        # Matlab comparison
        path = current_path + os.sep+'results/results_wpi1_ar3_matlab_ssm.csv'
        matlab_names = [
            'a1','a2','a3','detP','alphahat1','alphahat2','alphahat3',
            'detV','eps','epsvar','eta','etavar'
        ]
        self.matlab_ssm = pd.read_csv(path, header=None, names=matlab_names)
        # Regression tests data
        path = current_path + os.sep+'results/results_wpi1_ar3_regression.csv'
        self.regression = pd.read_csv(path)

        self.model = sarimax.SARIMAX(
            self.stata['wpi'], order=(3, 1, 0), simple_differencing=True,
            hamilton_representation=True, *args, **kwargs
        )
        if alternate_timing:
            self.model.timing_init_filtered = True

        # Parameters from from Stata's sspace MLE estimation
        params = np.r_[.5270715, .0952613, .2580355, .5307459]
        self.results = self.model.smooth(params, return_ssm=True)

        # Calculate the determinant of the covariance matrices (for easy
        # comparison to other languages without having to store 2-dim arrays)
        self.results.det_predicted_state_cov = np.zeros((1, self.model.nobs))
        self.results.det_smoothed_state_cov = np.zeros((1, self.model.nobs))
        for i in range(self.model.nobs):
            self.results.det_predicted_state_cov[0,i] = np.linalg.det(
                self.results.predicted_state_cov[:,:,i])
            self.results.det_smoothed_state_cov[0,i] = np.linalg.det(
                self.results.smoothed_state_cov[:,:,i])

        # Perform simulation smoothing
        n_disturbance_variates = (
            (self.model.k_endog + self.model.k_posdef) * self.model.nobs
        )
        self.sim = self.model.simulation_smoother()
        self.sim.simulate(
            disturbance_variates=np.zeros(n_disturbance_variates),
            initial_state_variates=np.zeros(self.model.k_states)
        )


    def test_predict_obs(self):
        assert_almost_equal(
            self.results.predict().forecasts[0],
            self.stata.ix[1:, 'dep1'], 4
        )

    def test_standardized_residuals(self):
        assert_almost_equal(
            self.results.standardized_forecasts_error[0],
            self.stata.ix[1:, 'sr1'], 4
        )

    def test_predicted_states(self):
        assert_almost_equal(
            self.results.predicted_state[:,:-1].T,
            self.stata.ix[1:, ['sp1', 'sp2', 'sp3']], 4
        )
        assert_almost_equal(
            self.results.predicted_state[:,:-1].T,
            self.matlab_ssm[['a1', 'a2', 'a3']], 4
        )

    def test_predicted_states_cov(self):
        assert_almost_equal(
            self.results.det_predicted_state_cov.T,
            self.matlab_ssm[['detP']], 4
        )

    def test_filtered_states(self):
        assert_almost_equal(
            self.results.filtered_state.T,
            self.stata.ix[1:, ['sf1', 'sf2', 'sf3']], 4
        )

    def test_smoothed_states(self):
        assert_almost_equal(
            self.results.smoothed_state.T,
            self.stata.ix[1:, ['sm1', 'sm2', 'sm3']], 4
        )
        assert_almost_equal(
            self.results.smoothed_state.T,
            self.matlab_ssm[['alphahat1', 'alphahat2', 'alphahat3']], 4
        )

    def test_smoothed_states_cov(self):
        assert_almost_equal(
            self.results.det_smoothed_state_cov.T,
            self.matlab_ssm[['detV']], 4
        )

    def test_smoothed_measurement_disturbance(self):
        assert_almost_equal(
            self.results.smoothed_measurement_disturbance.T,
            self.matlab_ssm[['eps']], 4
        )

    def test_smoothed_measurement_disturbance_cov(self):
        assert_almost_equal(
            self.results.smoothed_measurement_disturbance_cov[0].T,
            self.matlab_ssm[['epsvar']], 4
        )

    def test_smoothed_state_disturbance(self):
        assert_almost_equal(
            self.results.smoothed_state_disturbance.T,
            self.matlab_ssm[['eta']], 4
        )

    def test_smoothed_state_disturbance_cov(self):
        assert_almost_equal(
            self.results.smoothed_state_disturbance_cov[0].T,
            self.matlab_ssm[['etavar']], 4
        )

    def test_simulation_smoothed_state(self):
        # regression test
        assert_almost_equal(
            self.sim.simulated_state.T,
            self.regression[['state1', 'state2', 'state3']], 4
        )

    def test_simulation_smoothed_measurement_disturbance(self):
        # regression test
        assert_almost_equal(
            self.sim.simulated_measurement_disturbance.T,
            self.regression[['measurement_disturbance']][:-1], 4
        )

    def test_simulation_smoothed_state_disturbance(self):
        # regression test
        assert_almost_equal(
            self.sim.simulated_state_disturbance.T,
            self.regression[['state_disturbance']], 4
        )

class TestStatesAR3Alternate(TestStatesAR3):
    def __init__(self, *args, **kwargs):
        super(TestStatesAR3Alternate, self).__init__(alternate_timing=True, *args, **kwargs)

class TestStatesMissingAR3(object):
    def __init__(self, alternate_timing=True, *args, **kwargs):
        # Dataset
        path = current_path + os.sep + 'results/results_wpi1_ar3_stata.csv'
        self.stata = pd.read_csv(path)
        self.stata.index = pd.date_range(start='1960-01-01', periods=124,
                                         freq='QS')
        # Matlab comparison
        path = current_path + os.sep+'results/results_wpi1_missing_ar3_matlab_ssm.csv'
        matlab_names = [
            'a1','a2','a3','detP','alphahat1','alphahat2','alphahat3',
            'detV','eps','epsvar','eta','etavar'
        ]
        self.matlab_ssm = pd.read_csv(path, header=None, names=matlab_names)
        # Regression tests data
        path = current_path + os.sep+'results/results_wpi1_missing_ar3_regression.csv'
        self.regression = pd.read_csv(path)

        # Create missing observations
        self.stata['dwpi'] = self.stata['wpi'].diff()
        self.stata.ix[10:21, 'dwpi'] = np.nan

        self.model = sarimax.SARIMAX(
            self.stata.ix[1:,'dwpi'], order=(3, 0, 0),
            hamilton_representation=True, *args, **kwargs
        )
        if alternate_timing:
            self.model.timing_init_filtered = True

        # Parameters from from Stata's sspace MLE estimation
        params = np.r_[.5270715, .0952613, .2580355, .5307459]
        self.results = self.model.smooth(params, return_ssm=True)

        # Calculate the determinant of the covariance matrices (for easy
        # comparison to other languages without having to store 2-dim arrays)
        self.results.det_predicted_state_cov = np.zeros((1, self.model.nobs))
        self.results.det_smoothed_state_cov = np.zeros((1, self.model.nobs))
        for i in range(self.model.nobs):
            self.results.det_predicted_state_cov[0,i] = np.linalg.det(
                self.results.predicted_state_cov[:,:,i])
            self.results.det_smoothed_state_cov[0,i] = np.linalg.det(
                self.results.smoothed_state_cov[:,:,i])

        # Perform simulation smoothing
        n_disturbance_variates = (
            (self.model.k_endog + self.model.k_posdef) * self.model.nobs
        )
        self.sim = self.model.simulation_smoother()
        self.sim.simulate(
            disturbance_variates=np.zeros(n_disturbance_variates),
            initial_state_variates=np.zeros(self.model.k_states)
        )

    def test_predicted_states(self):
        assert_almost_equal(
            self.results.predicted_state[:,:-1].T,
            self.matlab_ssm[['a1', 'a2', 'a3']], 4
        )

    def test_predicted_states_cov(self):
        assert_almost_equal(
            self.results.det_predicted_state_cov.T,
            self.matlab_ssm[['detP']], 4
        )

    def test_smoothed_states(self):
        assert_almost_equal(
            self.results.smoothed_state.T,
            self.matlab_ssm[['alphahat1', 'alphahat2', 'alphahat3']], 4
        )

    def test_smoothed_states_cov(self):
        assert_almost_equal(
            self.results.det_smoothed_state_cov.T,
            self.matlab_ssm[['detV']], 4
        )

    def test_smoothed_measurement_disturbance(self):
        assert_almost_equal(
            self.results.smoothed_measurement_disturbance.T,
            self.matlab_ssm[['eps']], 4
        )

    def test_smoothed_measurement_disturbance_cov(self):
        assert_almost_equal(
            self.results.smoothed_measurement_disturbance_cov[0].T,
            self.matlab_ssm[['epsvar']], 4
        )

    # TODO there is a discrepancy between MATLAB ssm toolbox and
    # dismalpy.ssm on the following variables in the case of missing data.
    # Need to find a third implementation to compare against.

    # def test_smoothed_state_disturbance(self):
    #     assert_almost_equal(
    #         self.results.smoothed_state_disturbance.T,
    #         self.matlab_ssm[['eta']], 4
    #     )

    # def test_smoothed_state_disturbance_cov(self):
    #     assert_almost_equal(
    #         self.results.smoothed_state_disturbance_cov[0].T,
    #         self.matlab_ssm[['etavar']], 4
    #     )

    # def test_simulation_smoothed_state(self):
    #     # regression test
    #     assert_almost_equal(
    #         self.sim.simulated_state.T,
    #         self.regression[['state1', 'state2', 'state3']], 4
    #     )

    # def test_simulation_smoothed_measurement_disturbance(self):
    #     # regression test
    #     assert_almost_equal(
    #         self.sim.simulated_measurement_disturbance.T,
    #         self.regression[['measurement_disturbance']][:-1], 4
    #     )

    # def test_simulation_smoothed_state_disturbance(self):
    #     # regression test
    #     assert_almost_equal(
    #         self.sim.simulated_state_disturbance.T,
    #         self.regression[['state_disturbance']], 4
    #     )

class TestStatesMissingAR3Alternate(TestStatesMissingAR3):
    def __init__(self, *args, **kwargs):
        super(TestStatesMissingAR3Alternate, self).__init__(alternate_timing=True, *args, **kwargs)
