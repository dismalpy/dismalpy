"""
Tests for Kalman Filter

Author: Chad Fulton
License: Simplified-BSD

References
----------

Kim, Chang-Jin, and Charles R. Nelson. 1999.
"State-Space Models with Regime Switching:
Classical and Gibbs-Sampling Approaches with Applications".
MIT Press Books. The MIT Press.

Hamilton, James D. 1994.
Time Series Analysis.
Princeton, N.J.: Princeton University Press.
"""

import os
import numpy as np
import pandas as pd
from scipy import optimize
import statespace as ss
import results_kalman
from numpy.testing import assert_allclose, assert_almost_equal
from nose.exc import SkipTest


class TestClark1987(ss.Model):
    """
    Clark's (1987) univariate unobserved components model of real GDP (as
    presented in Kim and Nelson, 1999)

    Test data produced using GAUSS code described in Kim and Nelson (1999) and
    found at http://econ.korea.ac.kr/~cjkim/SSMARKOV.htm

    See `results.results_kalman` for more information.
    """
    def __init__(self):
        self.true = results_kalman.uc_uni
        self.true_states = pd.DataFrame(self.true['states'])

        # GDP, Quarterly, 1947.1 - 1995.3
        data = pd.DataFrame(
            self.true['data'],
            index=pd.date_range('1947-01-01', '1995-07-01', freq='QS'),
            columns=['GDP']
        )
        data['lgdp'] = np.log(data['GDP'])

        super(TestClark1987, self).__init__(data['lgdp'], nstates=4)
        self.H[:, :, 0] = [1, 1, 0, 0]
        self.F[([0, 0, 1, 1, 2, 3], [0, 3, 1, 2, 1, 3])] = [1, 1, 0, 0, 1, 1]

        # Initial Values
        self.initial_state = np.zeros((self.k,))
        self.initial_state_cov = np.eye(self.k)*100

        # Given parameters
        self.result = self.estimate(self.true['parameters'], True)

        # Only run MLE if necessary (long run-time)
        self.mle_fit = None

    def mle(self):
        if self.mle_fit is None:
            # Maximize Likelihood
            # (start parameters match those in uc_uni.opt, although this class
            # uses a different transformation method so they appear different)
            start_params = self.untransform(np.array([
                0.013534, 0.013534, 0.013534, 1.166667, -0.333333
            ]))
            f = lambda params: -est.loglike(params)
            est = ss.Estimator(self, self.true['start'])
            self.mle_fit = optimize.fmin_bfgs(f, start_params,
                                              full_output=True)

    def transform(self, unconstrained):
        constrained = np.zeros(unconstrained.shape)
        constrained[0:3] = unconstrained[0:3]**2
        constrained[3:] = ss.constrain_stationary(unconstrained[3:])
        return constrained

    def untransform(self, constrained):
        unconstrained = np.zeros(constrained.shape)
        unconstrained[0:3] = constrained[0:3]**0.5
        unconstrained[3:] = ss.unconstrain_stationary(constrained[3:])
        return unconstrained

    def update(self, params, transformed=False):
        if not transformed:
            params = self.transform(params)
        (sigma_v, sigma_e, sigma_w, phi_1, phi_2) = params

        # Matrices
        self.F[([1, 1], [1, 2])] = [phi_1, phi_2]
        self.Q[np.diag_indices(self.k)] = [
            sigma_v**2, sigma_e**2, 0, sigma_w**2
        ]

    def test_mle_parameters(self):
        self.mle()
        assert_allclose(
            self.transform(self.mle_fit[0]), self.true['parameters'], rtol=0.05
        )

    def test_mle_standard_errors(self):
        assert False

    def test_loglike(self):
        assert_almost_equal(
            self.result.loglike(self.true['start']), self.true['loglike'], 5
        )

    def test_filtered_state(self):
        assert_almost_equal(
            self.result.state[0][self.true['start']:],
            self.true_states.iloc[:, 0], 4
        )
        assert_almost_equal(
            self.result.state[1][self.true['start']:],
            self.true_states.iloc[:, 1], 4
        )
        assert_almost_equal(
            self.result.state[3][self.true['start']:],
            self.true_states.iloc[:, 2], 4
        )

    def test_smoothed_state(self):
        assert False


class TestClark1989(ss.Model):
    """
    Clark's (1989) bivariate unobserved components model of real GDP (as
    presented in Kim and Nelson, 1999)

    Test data produced using GAUSS code described in Kim and Nelson (1999) and
    found at http://econ.korea.ac.kr/~cjkim/SSMARKOV.htm

    See `results.results_kalman` for more information.
    """
    def __init__(self):
        self.true = results_kalman.uc_bi
        self.true_states = pd.DataFrame(self.true['states'])

        # GDP and Unemployment, Quarterly, 1948.1 - 1995.3
        data = pd.DataFrame(
            self.true['data'],
            index=pd.date_range('1947-01-01', '1995-07-01', freq='QS'),
            columns=['GDP', 'UNEMP']
        )[4:]
        data['GDP'] = np.log(data['GDP'])
        data['UNEMP'] = (data['UNEMP']/100)

        super(TestClark1989, self).__init__(data, nstates=6)
        self.H[:, :, 0] = [[1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1]]
        self.F[([0, 0, 1, 1, 2, 3, 4, 5],
                [0, 4, 1, 2, 1, 2, 4, 5])] = [1, 1, 0, 0, 1, 1, 1, 1]

        # Initial Values
        self.initial_state = np.zeros((self.k,))
        self.initial_state_cov = np.eye(self.k)*100

        # Given parameters
        self.result = self.estimate(self.true['parameters'], True)

        # Only run MLE if necessary (long run-time)
        self.mle_fit = None

    def mle(self):
        if self.mle_fit is None:
            # Maximize Likelihood
            # (start parameters match those in uc_bi.opt, although this class
            # uses a different transformation method so they appear different)
            # (uc_bi.opt starts at the optimizing parameters)
            start_params = self.untransform(np.r_[self.true['parameters']])
            f = lambda params: -est.loglike(params)
            est = ss.Estimator(self, self.true['start'])
            self.mle_fit = optimize.fmin_bfgs(f, start_params,
                                              full_output=True)

    def transform(self, unconstrained):
        constrained = np.zeros(unconstrained.shape)
        constrained[0:5] = unconstrained[0:5]**2
        constrained[5:7] = ss.constrain_stationary(unconstrained[5:7])
        constrained[7:] = unconstrained[7:]
        return constrained

    def untransform(self, constrained):
        unconstrained = np.zeros(constrained.shape)
        unconstrained[0:5] = constrained[0:5]**0.5
        unconstrained[5:7] = ss.unconstrain_stationary(constrained[5:7])
        unconstrained[7:] = constrained[7:]
        return unconstrained

    def update(self, params, transformed=False):
        if not transformed:
            params = self.transform(params)
        (sigma_v, sigma_e, sigma_w, sigma_vl, sigma_ec,
         phi_1, phi_2, alpha_1, alpha_2, alpha_3) = params

        # Matrices
        self.H[([1, 1, 1], [1, 2, 3], [0, 0, 0])] = [alpha_1, alpha_2, alpha_3]
        self.F[([1, 1], [1, 2])] = [phi_1, phi_2]
        self.R[1, 1] = sigma_ec**2
        self.Q[np.diag_indices(self.k)] = [
            sigma_v**2, sigma_e**2, 0, 0, sigma_w**2, sigma_vl**2
        ]

    def test_mle_parameters(self):
        self.mle()
        assert_allclose(
            self.transform(self.mle_fit[0]), self.true['parameters'], rtol=0.05
        )

    def test_mle_standard_errors(self):
        assert False

    def test_loglike(self):
        assert_almost_equal(
            self.result.loglike(self.true['start']), self.true['loglike'], 5
        )

    def test_filtered_state(self):
        assert_almost_equal(
            self.result.state[0][self.true['start']:],
            self.true_states.iloc[:, 0], 4
        )
        assert_almost_equal(
            self.result.state[1][self.true['start']:],
            self.true_states.iloc[:, 1], 4
        )
        assert_almost_equal(
            self.result.state[4][self.true['start']:],
            self.true_states.iloc[:, 2], 4
        )
        assert_almost_equal(
            self.result.state[5][self.true['start']:],
            self.true_states.iloc[:, 3], 4
        )

    def test_smoothed_state(self):
        assert False


class TestKimNelson1989(ss.Model):
    """
    Kim and Nelson's (1989) time-varying parameters model of monetary growth
    (as presented in Kim and Nelson, 1999)

    Test data produced using GAUSS code described in Kim and Nelson (1999) and
    found at http://econ.korea.ac.kr/~cjkim/SSMARKOV.htm

    See `results.results_kalman` for more information.
    """
    def __init__(self):
        self.true = results_kalman.tvp
        self.true_states = pd.DataFrame(self.true['states'])

        # Quarterly, 1959.3--1985.4
        data = pd.DataFrame(
            self.true['data'],
            index=pd.date_range(
                start='1959-07-01', end='1985-10-01', freq='QS'),
            columns=['Qtr', 'm1', 'dint', 'inf', 'surpl', 'm1lag']
        )

        super(TestKimNelson1989, self).__init__(data['m1'], nstates=5)
        self.H = np.c_[
            np.ones(data['dint'].shape),
            data['dint'],
            data['inf'],
            data['surpl'],
            data['m1lag']
        ].T[None, :]
        self.F = np.eye(self.k)

        # Initial Values
        self.initial_state = np.zeros((self.k,))
        self.initial_state_cov = np.eye(self.k)*50

        # Given parameters
        self.result = self.estimate(self.true['parameters'], True)

        # Only run MLE if necessary (long run-time)
        self.mle_fit = None

    def mle(self):
        if self.mle_fit is None:
            # Maximize Likelihood
            # (start parameters match those in tvp.opt)
            start_params = np.array([0.5, 0.1, 0.1, 0.1, 0.1, 0.1])
            f = lambda params: -est.loglike(params)
            est = ss.Estimator(self, self.true['start'])
            self.mle_fit = optimize.fmin_bfgs(f, start_params,
                                              full_output=True)

    def transform(self, unconstrained):
        constrained = unconstrained.copy()**2
        return constrained

    def untransform(self, constrained):
        unconstrained = constrained.copy()**0.5
        return unconstrained

    def update(self, params, transformed=False):
        if not transformed:
            params = self.transform(params)
        (sigma_e, sigma_v0, sigma_v1, sigma_v2, sigma_v3, sigma_v4) = params

        # Matrices
        self.R[0, 0] = sigma_e**2
        self.Q[np.diag_indices(self.k)] = [
            sigma_v0**2, sigma_v1**2,
            sigma_v2**2, sigma_v3**2,
            sigma_v4**2
        ]

    def test_mle_parameters(self):
        self.mle()
        assert_allclose(
            self.transform(self.mle_fit[0]), self.true['parameters'], rtol=0.1
        )

    def test_mle_standard_errors(self):
        assert False

    def test_loglike(self):
        assert_almost_equal(
            self.result.loglike(self.true['start']), self.true['loglike'], 5
        )

    def test_filtered_state(self):
        assert_almost_equal(
            self.result.state[0][self.true['start']-1:-1],
            self.true_states.iloc[:, 0], 3
        )
        assert_almost_equal(
            self.result.state[1][self.true['start']-1:-1],
            self.true_states.iloc[:, 1], 3
        )
        assert_almost_equal(
            self.result.state[2][self.true['start']-1:-1],
            self.true_states.iloc[:, 2], 3
        )
        assert_almost_equal(
            self.result.state[3][self.true['start']-1:-1],
            self.true_states.iloc[:, 3], 3
        )
        assert_almost_equal(
            self.result.state[4][self.true['start']-1:-1],
            self.true_states.iloc[:, 4], 3
        )

    def test_smoothed_state(self):
        assert False
