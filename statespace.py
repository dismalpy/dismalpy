"""
State Space Models

Author: Chad Fulton

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

from __future__ import division
import numpy as np
from scipy.linalg.blas import find_best_blas_type
from kalman.kalman_filter import (
    skalman_filter, dkalman_filter,
    ckalman_filter, zkalman_filter
)

prefix_dtype_map = {
    's': np.float32, 'd': np.float64, 'c': np.complex64, 'z': np.complex128
}
prefix_kalman_filter_map = {
    's': skalman_filter, 'd': dkalman_filter,
    'c': ckalman_filter, 'z': zkalman_filter
}

class Model(object):
    """
    A general state space model

    Parameters
    ----------
    endog : array-like
        The observed endogenous vector or matrix, with shape (nobs, n).
    nstates : integer
        The dimension of the state vector.
    exog : array-like, optional
        The observed (weakly) exogenous variables, with shape (nobs, r).
    nstates : integer, optional
        The length of the state vector.
    init : bool, optional
        Whether or not to initialize the model arrays A,H,mu,F,R,Q to be
        arrays of zeros.
        Defaults to False, so that if one is not specifically set, an exception
        will be raised.
        Requires `nstates` to be set.
    dtype : data-type, optional
        The desired data-type for the arrays.  If not given, then
        the type will be determined from the `endog` matrix.

    Notes
    -----
    In addition to being a holder for the state space representation values, it
    also handles some annoyances:

    - C vs Fortran ordering of arrays in memory. The BLAS/LAPACK functions
      called in the kalman_filter.pyx module require Fortran (column-major)
      ordering, whereas numpy arrays are typically C (row-major) ordered.
    - Maximum Likelihood Estimation of a state space model via `scipy.optimize`
      functions will drift into a complex parameter space to estimate the
      gradient and hessian. This means that the log likelihood function will
      need to intelligently call `kalman_filter.kalman_filter` vs
      `kalman_filter.kalman_filter_complex`, and make sure the arguments (
      including e.g. the data `y` and `z`) are of the appropriate type.
    """
    def __init__(self, endog, exog=None, nstates=None, init=False, dtype=None,
                 A=None, H=None, mu=None, F=None, R=None, Q=None,
                 initial_state=None, initial_state_cov=None):
        self.endog = np.asarray(endog, dtype=dtype, order="C")
        self.dtype = self.endog.dtype
        self.prefix = find_best_blas_type((self.endog,))[0]
        if self.endog.ndim == 1:
            self.endog.shape = (self.endog.shape[0], 1)
        self.endog = self.endog.T
        self.n, self.nobs = self.endog.shape
        self.k = None

        if exog is not None:
            self.exog = np.asarray(exog, dtype=dtype, order="C").T
            self.r = self.exog.shape[0]
        else:
            self.exog = np.zeros((1, self.nobs), dtype=dtype, order="F")
            self.r = 1
            self._A = np.empty((self.n, 0))

        if nstates is not None:
            self.k = nstates

        self._A = None
        self._H = None
        self._mu = None
        self._F = None
        self._R = None
        self._Q = None
        self._initial_state = None
        self._initial_state_cov = None

        if init:
            if nstates is None:
                raise ValueError('Cannot initialize model variables if'
                                 ' `nstates` has not been set.')
            self.A  = np.zeros((self.n, self.r), self.dtype, order="F")
            self.H  = np.zeros((self.n, self.k), self.dtype, order="F")
            self.mu = np.zeros((self.k, ),       self.dtype, order="F")
            self.F  = np.zeros((self.k, self.k), self.dtype, order="F")
            self.R  = np.zeros((self.n, self.n), self.dtype, order="F")
            self.Q  = np.zeros((self.k, self.k), self.dtype, order="F")

        if A is not None:
            self.A = A
        if H is not None:
            self.H = H
        if mu is not None:
            self.mu = mu
        if F is not None:
            self.F = F
        if R is not None:
            self.R = R
        if Q is not None:
            self.Q = Q
        if initial_state is not None:
            self.initial_state = initial_state
        if initial_state_cov is not None:
            self.initial_state_cov = initial_state_cov

    def transform(self, params):
        return params

    def update(self, params, transformed=False):
        pass

    def estimate(self, params):
        params = self.transform(params)
        self.update(params, True)
        kalman_filter = prefix_kalman_filter_map[self.prefix]
        return Results(self, params, *kalman_filter(*self.args))

    @property
    def args(self):
        return (
            self.endog, self.exog, self.A, self.H, self.mu, self.F, self.R,
            self.Q, self.initial_state, self.initial_state_cov
        )

    @property
    def A(self):
        if self._A is not None:
            return self._A
        else:
            raise NotImplementedError
    @A.setter
    def A(self, value):
        _A = np.asarray(value, dtype=self.dtype, order="F")
        if not _A.shape == (self.n, self.r):
            raise ValueError('Invalid dimensions for A matrix. Requires'
                             ' shape (%d, %d), got shape %s' %
                             (self.n, self.r, str(_A.shape)))
        if self._A is None:
            self._A  = np.zeros((self.n, self.r), self.dtype, order="F")
        self._A = _A

    @property
    def H(self):
        if self._H is not None:
            return self._H
        else:
            raise NotImplementedError
    @H.setter
    def H(self, value):
        _H = np.asarray(value, dtype=self.dtype, order="F")
        if not _H.ndim in [2,3]:
            raise ValueError('Invalid value for H matrix. Requires a'
                             ' 2-dimensional array, got %d dimensions' %
                             _H.ndim)
        if _H.ndim == 3 and not _H.shape[2] in [1, self.nobs]:
            raise ValueError('Invalid dimensions for time-varying H matrix.'
                             ' Requires shape (*,*,%d), got %s' %
                             (self.nobs, str(_H.shape)))
        if not _H.shape[0] == self.n:
            raise ValueError('Invalid dimensions for H matrix. Requires'
                             ' %d rows, got %d' % (self.n, _H.shape[0]))
        if self.k is not None and not _H.shape[1] == self.k:
            raise ValueError('Invalid dimensions for H matrix. Requires'
                             ' %d columns, got %d' % (self.k, _H.shape[1]))
        else:
            self.k = _H.shape[1]

        if _H.ndim == 2:
            _H = np.array(_H[:,:,None], order="F")

        if self._H is None:
            self._H  = np.zeros((self.n, self.k), self.dtype, order="F")
        self._H = _H

    @property
    def mu(self):
        if self._mu is not None:
            return self._mu
        else:
            raise NotImplementedError
    @mu.setter
    def mu(self, value):
        _mu = np.asarray(value, dtype=self.dtype, order="F")
        if _mu.ndim > 1:
            raise ValueError('Invalid mu vector. Requires a'
                             ' 1-dimensional array, got %d dimensions'
                             % _mu.ndim)
        if self.k is not None and not _mu.shape[0] == self.k:
            raise ValueError('Invalid dimensions for mu vector. Requires'
                             ' %d rows, got %d' % (self.k, _mu.shape[0]))
        else:
            self.k = _mu.shape[0]

        if self._mu is None:
            self._mu = np.zeros((self.k, ),       self.dtype, order="F")
        self._mu = _mu

    @property
    def F(self):
        if self._F is not None:
            return self._F
        else:
            raise NotImplementedError
    @F.setter
    def F(self, value):
        _F = np.asarray(value, dtype=self.dtype, order="F")
        if not _F.ndim == 2:
            raise ValueError('Invalid value for F matrix. Requires a'
                             ' 2-dimensional array, got %d dimensions' %
                             _F.ndim)
        if not _F.shape[0] == _F.shape[1]:
            raise ValueError('Invalid F matrix. Requires a square matrix, got'
                             ' shape %s.' % str(_F.shape))
        if self.k is not None and not _F.shape[0] == self.k:
            raise ValueError('Invalid dimensions for F matrix. Requires'
                             ' %d rows and columns, got %d' %
                             (self.k, _F.shape[0]))
        else:
            self.k = _F.shape[0]

        if self._F is None:
            self._F  = np.zeros((self.k, self.k), self.dtype, order="F")
        self._F = _F

    @property
    def R(self):
        if self._R is not None:
            return self._R
        else:
            raise NotImplementedError
    @R.setter
    def R(self, value):
        _R = np.asarray(value, dtype=self.dtype, order="F")
        if not _R.ndim == 2:
            raise ValueError('Invalid value for R matrix. Requires a'
                             ' 2-dimensional array, got %d dimensions' %
                             _R.ndim)
        if not _R.shape[0] == _R.shape[1]:
            raise ValueError('Invalid R matrix. Requires a square matrix, got'
                             ' shape %s.' % str(_R.shape))
        if not _R.shape[0] == self.n:
            raise ValueError('Invalid dimensions for R matrix. Requires'
                             ' %d rows and columns, got %d' %
                             (self.n, _R.shape[0]))

        if self._R is None:
            self._R  = np.zeros((self.n, self.n), self.dtype, order="F")
        self._R = _R

    @property
    def Q(self):
        if self._Q is not None:
            return self._Q
        else:
            raise NotImplementedError
    @Q.setter
    def Q(self, value):
        _Q = np.asarray(value, dtype=self.dtype, order="F")
        if not _Q.ndim == 2:
            raise ValueError('Invalid value for Q matrix. Requires a'
                             ' 2-dimensional array, got %d dimensions' %
                             _Q.ndim)
        if not _Q.shape[0] == _Q.shape[1]:
            raise ValueError('Invalid Q matrix. Requires a square matrix, got'
                             ' shape %s.' % str(_Q.shape))
        if self.k is not None and not _Q.shape[0] == self.k:
            raise ValueError('Invalid dimensions for Q matrix. Requires'
                             ' %d rows and columns, got %d' %
                             (self.k, _Q.shape[0]))
        else:
            self.k = _Q.shape[0]

        if self._Q is None:
            self._Q  = np.zeros((self.k, self.k), self.dtype, order="F")
        self._Q = _Q

    @property
    def initial_state(self):
        return self._initial_state

    @initial_state.setter
    def initial_state(self, value):
        _initial_state = np.asarray(value, dtype=self.dtype, order="F")
        if _initial_state.ndim > 1:
            raise ValueError('Invalid initial state vector. Requires a'
                             ' 1-dimensional array, got %d dimensions'
                             % _initial_state.ndim)
        if self.k is not None and not _initial_state.shape[0] == self.k:
            raise ValueError('Invalid dimensions for initial state vector.'
                             ' Requires %d rows, got %d' %
                             (self.k, _initial_state.shape[0]))
        else:
            self.k = _initial_state.shape[0]
        self._initial_state = _initial_state

    @property
    def initial_state_cov(self):
        return self._initial_state_cov

    @initial_state_cov.setter
    def initial_state_cov(self, value):
        _initial_state_cov = np.asarray(value, dtype=self.dtype, order="F")
        if not _initial_state_cov.ndim == 2:
            raise ValueError('Invalid value for initial state covariance'
                             ' matrix. Requires a 2-dimensional array, got %d'
                             ' dimensions' % _initial_state_cov.ndim)
        if not _initial_state_cov.shape[0] == _initial_state_cov.shape[1]:
            raise ValueError('Invalid initial state covariance matrix.'
                             ' Requires a square matrix, got shape %s.' %
                             str(_initial_state_cov.shape))
        if self.k is not None and not _initial_state_cov.shape[0] == self.k:
            raise ValueError('Invalid dimensions for initial state covariance'
                             ' matrix. Requires %d rows and columns, got %d' %
                             (self.k, _initial_state_cov.shape[0]))
        else:
            self.k = _initial_state_cov.shape[0]
        self._initial_state_cov = _initial_state_cov

class Estimator(object):
    """
    A state space estimation class

    Parameters
    ----------
    model : statespace.Model
        A state space model

    Notes
    -----
    This handles some annoyances:
    - Maximum Likelihood Estimation of a state space model via `scipy.optimize`
      functions will drift into a complex parameter space to estimate the
      gradient and hessian. This means that the log likelihood function will
      need to intelligently call the appropriate cython Kalman filter and make
      sure the arguments (including e.g. the data `y` and `z`) are of the
      appropriate type.
    """

    def __init__(self, model, burn=0):
        if not isinstance(model, Model):
            raise ValueError('Invalid model. Must be an object of type Model.')
        self.default_model = model
        self.typed_models = {
            's': None, 'd': None, 'c': None, 'z': None,
        }
        self.default_prefix = find_best_blas_type(
            (self.default_model.endog,)
        )[0]
        self.typed_models[self.default_prefix] = self.default_model
        self.burn = burn

    def typed_model(self, prefix):
        if self.typed_models[prefix] is None:
            dtype = prefix_dtype_map[prefix]
            default = self.default_model
            self.typed_models[prefix] = Model(
                default.endog.T, default.exog.T, default.k, dtype=dtype,
                A=default._A, H=default._H, mu=default._mu, F=default._F,
                R=default._R, Q=default._Q,
                initial_state=default._initial_state,
                initial_state_cov=default._initial_state_cov
            )
        return self.typed_models[prefix]

    def loglike(self, params):
        prefix = find_best_blas_type((params,))[0]
        model = self.typed_model(prefix)
        model.update(params)
        kalman_filter = prefix_kalman_filter_map[prefix]
        ll = np.sum(kalman_filter(*model.args)[-1][self.burn:])
        return ll

class Results(object):
    def __init__(self, model, params, state, state_cov, est_state,
                 est_state_cov, forecast, prediction_error,
                 prediction_error_cov, gain, loglikelihood):
        # Save the inputs
        self.model = model
        self.params = params
        # Save the state space representation at params
        self.model.update(params, True)
        (y, z, A, H, mu, F, R, Q,
         initial_state, initial_state_cov) = self.model.args
        self.y = y
        self.z = z
        self.A = A.copy()
        self.H = H.copy()
        self.mu = mu.copy()
        self.F = F.copy()
        self.R = R.copy()
        self.Q = Q.copy()
        # Save Kalman Filter output
        self.state = np.asarray(state[:,1:])
        self.state_cov = np.asarray(state_cov[:,:,1:])
        self.est_state = np.asarray(est_state[:,1:])
        self.est_state_cov = np.asarray(est_state_cov[:,:,1:])
        self.forecast = np.asarray(forecast[:,1:])
        self.prediction_error = np.asarray(prediction_error[:,1:])
        self.prediction_error_cov = np.asarray(prediction_error_cov[:,:1:])
        self.gain = np.asarray(gain[:,:,1:])
        self.loglikelihood = np.asarray(loglikelihood[1:])

    def loglike(self, burn=0):
        return np.sum(self.loglikelihood[burn:])