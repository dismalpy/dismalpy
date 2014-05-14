"""
State Space Models

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

from __future__ import division
import warnings
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

class Representation(object):
    """
    State space representation of a time series process

    A class to hold and facilitate manipulation of the matrices describing the
    state space process:

    .. math::

        y_t = A z_t + H_t \beta_t + e_t \\
        beta_t = \mu + F \beta_t + G v_t^* \\

    along with the definition of the shocks

    .. math::

        e_t \sim N(0, R) \\
        v_t^* \sim N(0, Q^*) \\

    The dimensions of the matrices are:

    A   : :math:`n \times r`
    H   : :math:`n \times k` or `n \times k \times T`
    mu  : :math:`k \times 1`
    F   : :math:`k \times k`
    R   : :math:`n \times n`
    G   : :math:`k \times g`
    Q^* : :math:`g \times g`

    This class does not deal with the _data_ of the state-space process.

    Parameters
    ----------
    nobs : int
        The length (T) of the described time-series process.
    nendog : int
        The dimension (n) of the endogenous vector.
    nstates : int
        The dimension (k) of the state vector.
    nexog : int
        The dimension (r) of the vector of weakly exogenous explanatory
        variables.
    nposdef : int
        The dimension (g) of a guaranteed positive definite covariance matrix
        describing the shocks in the measurement equation. It must be that
        :math:`g \le k`.
    dtype : data-type, optional
        The desired data-type for the arrays.
    A, H, mu, F, R, G, Q : array_like, optional
        Input arrays describing the state-space process.
    initial_state, initial_state_cov : array_like, optional
        Initializing values for the state and covariance matrix (useful when
        running the Kalman filter).

    """
    def __init__(self, nobs, nendog, nstates, nexog, nposdef, dtype,
                 A=None, H=None, mu=None, F=None, R=None, G=None, Q=None,
                 initial_state=None, initial_state_cov=None):
        # Dimensions
        self.T = self.nobs    = nobs
        self.n = self.nendog  = nendog
        self.k = self.nstates = nstates
        self.r = self.nexog   = nexog
        self.g = self.nposdef = nposdef

        # Data type for representation matrices
        self.dtype = dtype

        # Representation matrices
        self.A = A       # n x r
        self.H = H       # n x k   or   n x k x nobs (for TVP)
        self.mu = mu     # k x 0
        self.F = F       # k x k
        self.R = R       # n x n
        self.G = G       # k x g
        self.Q = Q       # g x g

        # Initial values of the state vector and covariance matrix
        self.initial_state = initial_state
        self.initial_state_cov = initial_state_cov

        # Record the shapes of all of our matrices
        self.shapes = {
            'A':  (self.n, self.r),
            'H':  (self.n, self.k, self.nobs),
            'Ht': (self.n, self.k),
            'mu': (self.k, ),
            'F':  (self.k, self.k),
            'R':  (self.n, self.n),
            'G':  (self.k, self.g),
            'Q':  (self.g, self.g)
        }

    @property
    def A(self):
        # A is special because we often won't have it
        if hasattr(self, '_A'):
            return self._A
        else:
            return None
    @A.setter
    def A(self, value):
        if value is None:
            _A = None
        else:
            _A = np.asarray(value, dtype=self.dtype, order="F")
            if not _A.shape == (self.n, self.r):
                raise ValueError('Invalid dimensions for A matrix. Requires'
                                 ' shape (%d, %d), got shape %s' %
                                 (self.n, self.r, str(_A.shape)))
            if not hasattr(self, '_A') or self._A is None:
                self._A  = np.zeros((self.n, self.r), self.dtype, order="F")
        self._A = _A

    @property
    def H(self):
        if hasattr(self, '_H') and self._H is not None:
            return self._H
        else:
            raise NotImplementedError
    @H.setter
    def H(self, value):
        if value is None:
            _H = None
        else:
            _H = np.asarray(value, dtype=self.dtype, order="F")
            if _H.ndim == 1 and self.n == 1 and _H.shape[0] == self.k:
                _H = _H[None,:]
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
            if not _H.shape[1] == self.k:
                raise ValueError('Invalid dimensions for H matrix. Requires'
                                 ' %d columns, got %d' % (self.k, _H.shape[1]))
            if _H.ndim == 2:
                _H = np.array(_H[:,:,None], order="F")
            if not hasattr(self, '_H') or self._H is None:
                self._H  = np.zeros((self.n, self.k), self.dtype, order="F")
        self._H = _H

    @property
    def mu(self):
        if hasattr(self, '_mu') and self._mu is not None:
            return self._mu
        else:
            raise NotImplementedError
    @mu.setter
    def mu(self, value):
        if value is None:
            _mu = None
        else:
            _mu = np.asarray(value, dtype=self.dtype, order="F")
            if _mu.ndim > 1:
                raise ValueError('Invalid mu vector. Requires a'
                                 ' 1-dimensional array, got %d dimensions'
                                 % _mu.ndim)
            if not _mu.shape[0] == self.k:
                raise ValueError('Invalid dimensions for mu vector. Requires'
                                 ' %d rows, got %d' % (self.k, _mu.shape[0]))
            if not hasattr(self, '_mu') or self._mu is None:
                self._mu = np.zeros((self.k, ), self.dtype, order="F")
        self._mu = _mu

    @property
    def F(self):
        if hasattr(self, '_F') and self._F is not None:
            return self._F
        else:
            raise NotImplementedError
    @F.setter
    def F(self, value):
        if value is None:
            _F = None
        else:
            _F = np.asarray(value, dtype=self.dtype, order="F")
            if not _F.ndim == 2:
                raise ValueError('Invalid value for F matrix. Requires a'
                                 ' 2-dimensional array, got %d dimensions' %
                                 _F.ndim)
            if not _F.shape[0] == _F.shape[1]:
                raise ValueError('Invalid F matrix. Requires a square matrix, got'
                                 ' shape %s.' % str(_F.shape))
            if not _F.shape[0] == self.k:
                raise ValueError('Invalid dimensions for F matrix. Requires'
                                 ' %d rows and columns, got %d' %
                                 (self.k, _F.shape[0]))
            if not hasattr(self, '_F') or self._F is None:
                self._F  = np.zeros((self.k, self.k), self.dtype, order="F")
        self._F = _F

    @property
    def R(self):
        if hasattr(self, '_R') and self._R is not None:
            return self._R
        else:
            raise NotImplementedError
    @R.setter
    def R(self, value):
        if value is None:
            _R = None
        else:
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
            if not hasattr(self, '_R') or self._R is None:
                self._R  = np.zeros((self.n, self.n), self.dtype, order="F")
        self._R = _R

    @property
    def G(self):
        if hasattr(self, '_G') and self._G is not None:
            return self._G
        else:
            raise NotImplementedError
    @G.setter
    def G(self, value):
        if value is None:
            _G = None
        else:
            _G = np.asarray(value, dtype=self.dtype, order="F")
            if not _G.ndim == 2:
                raise ValueError('Invalid value for G matrix. Requires a'
                                 ' 2-dimensional array, got %d dimensions' %
                                 _G.ndim)
            if not _G.shape == (self.k, self.g):
                raise ValueError('Invalid dimensions for G matrix. Requires'
                                 ' (%d, %d) got %s' %
                                 (self.k, self.g, str(_G.shape)))
            if not hasattr(self, '_G') or self._G is None:
                self._G  = np.zeros((self.k, self.g), self.dtype, order="F")
        self._G = _G

    @property
    def Q(self):
        if self._Q is not None:
            return self._Q
        else:
            raise NotImplementedError
    @Q.setter
    def Q(self, value):
        if value is None:
            _Q = None
        else:
            _Q = np.asarray(value, dtype=self.dtype, order="F")
            if not _Q.ndim == 2:
                raise ValueError('Invalid value for Q matrix. Requires a'
                                 ' 2-dimensional array, got %d dimensions' %
                                 _Q.ndim)
            if not _Q.shape[0] == _Q.shape[1]:
                raise ValueError('Invalid Q matrix. Requires a square matrix, got'
                                 ' shape %s.' % str(_Q.shape))
            if not _Q.shape[0] == self.g:
                raise ValueError('Invalid dimensions for Q matrix. Requires'
                                 ' %d rows and columns, got %d' %
                                 (self.g, _Q.shape[0]))
            if not hasattr(self, '_Q') or self._Q is None:
                self._Q  = np.zeros((self.g, self.g), self.dtype, order="F")
        self._Q = _Q

    @property
    def initial_state(self):
        if hasattr(self, '_initial_state'):
            return self._initial_state
        else:
            raise None
    @initial_state.setter
    def initial_state(self, value):
        if value is None:
            _initial_state = None
        else:
            _initial_state = np.asarray(value, dtype=self.dtype, order="F")
            if _initial_state.ndim > 1:
                raise ValueError('Invalid initial state vector. Requires a'
                                 ' 1-dimensional array, got %d dimensions'
                                 % _initial_state.ndim)
            if not _initial_state.shape[0] == self.k:
                raise ValueError('Invalid dimensions for initial state vector.'
                                 ' Requires %d rows, got %d' %
                                 (self.k, _initial_state.shape[0]))
            if not hasattr(self, '_initial_state') or self._initial_state is None:
                self._initial_state = np.zeros((self.k,), self.dtype, order="F")
        self._initial_state = _initial_state

    @property
    def initial_state_cov(self):
        if hasattr(self, '_initial_state_cov'):
            return self._initial_state_cov
        else:
            raise None
    @initial_state_cov.setter
    def initial_state_cov(self, value):
        if value is None:
            _initial_state_cov = None
        else:
            _initial_state_cov = np.asarray(value, dtype=self.dtype, order="F")
            if not _initial_state_cov.ndim == 2:
                raise ValueError('Invalid value for initial state covariance'
                                 ' matrix. Requires a 2-dimensional array, got %d'
                                 ' dimensions' % _initial_state_cov.ndim)
            if not _initial_state_cov.shape[0] == _initial_state_cov.shape[1]:
                raise ValueError('Invalid initial state covariance matrix.'
                                 ' Requires a square matrix, got shape %s.' %
                                 str(_initial_state_cov.shape))
            if not _initial_state_cov.shape[0] == self.k:
                raise ValueError('Invalid dimensions for initial state covariance'
                                 ' matrix. Requires %d rows and columns, got %d' %
                                 (self.k, _initial_state_cov.shape[0]))
            if not hasattr(self, '_initial_state_cov') or self._initial_state_cov is None:
                self._initial_state_cov = np.zeros((self.k, self.k), self.dtype, order="F")
        self._initial_state_cov = _initial_state_cov

class Model(Representation):
    """
    Model for the estimation of the parameters of a state-space representation

    Parameters
    ----------
    endog : array-like
        The observed endogenous vector or matrix, with shape (nobs, n).
    exog : array-like, optional
        The observed (weakly) exogenous variables, with shape (nobs, r).
    nstates : integer, optional
        The length of the state vector.
    dtype : data-type, optional
        The desired data-type for the arrays.  If not given, then
        the type will be determined from the `endog` matrix.
    fill : bool, optional
        Whether or not to pre-fill all of the `Representation` matrices with
        zeros. Default is True so that after instantiation, elements of the
        matrices can be set using slice notation.

    Notes
    -----
    This class handles some annoyances with holding state space data:

    - C vs Fortran ordering of arrays in memory. The BLAS/LAPACK functions
      called in the kalman_filter.pyx module require Fortran (column-major)
      ordering, whereas numpy arrays are typically C (row-major) ordered.
    - Maximum Likelihood Estimation of a state space model via `scipy.optimize`
      functions will drift into a complex parameter space to estimate the
      gradient and hessian. This means that the log likelihood function will
      need to intelligently call `kalman_filter.dkalman_filter` vs
      `kalman_filter.zkalman_filter`, and make sure the arguments (
      including e.g. the data `y` and `z`) are of the appropriate type.
    """
    def __init__(self, endog, nstates, nposdef=None, exog=None, dtype=None,
                 fill=True, **kwargs):

        # Standardize endogenous variables array
        self.endog = np.array(endog, copy=True, dtype=dtype, order="C")
        if self.endog.ndim == 1:
            self.endog.shape = (self.endog.shape[0], 1)
        self.endog = self.endog.T
        nendog, nobs = self.endog.shape
        dtype = self.endog.dtype

        # Get the BLAS prefix associated with the endog datatype: s,d,c,z
        self.prefix = find_best_blas_type((self.endog,))[0]

        # Standardize weakly exogenous explanatory variables array
        if exog is not None:
            self.exog = np.array(exog, copy=True, dtype=dtype, order="C").T
            nexog = self.exog.shape[0]
        else:
            self.exog = None
            nexog = 0

        # If we're not given the size of the positive definite covariance
        # matrix, assume that it is equal to `nstates`, and that the G matrix
        # is then just the identity matrix.
        if nposdef is None and 'G' in kwargs:
            nposdef = kwargs['G'].shape[1]
        if nposdef is None:
            nposdef = nstates
            kwargs['G'] = np.eye(nposdef)

        # Construct the state-space representation
        super(Model, self).__init__(nobs, nendog, nstates, nexog, nposdef,
                                    dtype, **kwargs)

        # By default, set any matrices that are not given to be appropriately
        # sized zero matrices
        if fill:
            for key in ['A', 'mu', 'F', 'R', 'G', 'Q']:
                if not key in kwargs or kwargs[key] is None:
                    setattr(self, key, np.zeros(self.shapes[key], self.dtype,
                                                order="F"))
            if not 'H' in kwargs or kwargs['H'] is None:
                setattr(self, 'H', np.zeros(self.shapes['Ht'], self.dtype,
                                            order="F"))

    def initialize(self):
        pass

    def transform(self, params):
        return params

    def update(self, params, transformed=False):
        pass

    def estimate(self, params, transformed=False):
        if not transformed:
            params = self.transform(params)
        self.update(params, True)
        kalman_filter = prefix_kalman_filter_map[self.prefix]
        return Results(self, params, *kalman_filter(*self.args))

    @property
    def start_params(self):
        raise NotImplementedError

    @property
    def args(self):
        return (
            self.endog, self.H, self.mu, self.F, self.R, self.G, self.Q,
            self.exog, self.A, self.initial_state, self.initial_state_cov
        )

class Estimator(object):
    """
    Helper class for estimation of state space

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

    def loglike(self, params, transformed=False):
        prefix = find_best_blas_type((params,))[0]
        model = self.typed_model(prefix)
        model.update(params, transformed)
        kalman_filter = prefix_kalman_filter_map[prefix]
        ll = np.sum(kalman_filter(*model.args)[-1][self.burn:])
        return ll

class Results(object):
    def __init__(self, model, params, state, state_cov, est_state,
                 est_state_cov, forecast, prediction_error,
                 prediction_error_cov, inverse_prediction_error_cov,
                 gain, loglikelihood):
        # Save the inputs
        self.model = model
        self.params = params
        # Save the model parameters
        self.nobs = model.nobs
        self.n = model.n
        self.k = model.k
        self.r = model.r
        # Save the state space representation at params
        self.model.update(params, True)
        (y, H, mu, F, R, G, Q, z, A,
         initial_state, initial_state_cov) = self.model.args
        self.y = y
        self.H = H.copy()
        self.mu = mu.copy()
        self.F = F.copy()
        self.R = R.copy()
        self.G = G.copy()
        self.Q = Q.copy()
        self.z = z
        self.A = A.copy() if A is not None else None
        # Save Kalman Filter output
        self.state = np.asarray(state[:,1:])
        self.state_cov = np.asarray(state_cov[:,:,1:])
        self.est_state = np.asarray(est_state[:,1:])
        self.est_state_cov = np.asarray(est_state_cov[:,:,1:])
        self.forecast = np.asarray(forecast[:,1:])
        self.prediction_error = np.asarray(prediction_error[:,1:])
        self.prediction_error_cov = np.asarray(prediction_error_cov[:,:1:])
        self.inverse_prediction_error_cov = np.asarray(
            inverse_prediction_error_cov[:,:1:]
        )
        self.gain = np.asarray(gain[:,:,1:])
        self.loglikelihood = np.asarray(loglikelihood[1:])

    def loglike(self, burn=0):
        return np.sum(self.loglikelihood[burn:])

class ARMA(Model):
    def __init__(self, endog, exog=None, order=(1,0), dtype=None, A=None,
                 measurement_error=False):

        self.p = order[0]
        self.q = order[1]
        k = max(self.p, self.q+1)

        self.measurement_error = measurement_error

        # Initialize the G matrix shape
        g = 1

        super(ARMA, self).__init__(endog, k, exog=exog, nposdef=g, dtype=dtype,
                                   A=A)

        # Exclude VARMA cases
        if not self.n == 1:
            raise ValueError('Invalid endogenous variable. ARMA model does not'
                             ' support vector ARMA models. `endog` must have'
                             ' 1 column, got %d.' % self.n)

        # Initialize the H matrix
        H = np.zeros((1, self.k), self.dtype)
        H[0,0] = 1
        self.H = H

        # Initialize the F matrix
        F = np.zeros((self.k, self.k), self.dtype)
        idx = np.diag_indices(self.k-1)
        idx = (idx[0]+1, idx[1])
        F[idx] = 1
        self.F = F

        # Initialize the G matrix as a (k x 1) matrix [1, 0, ..., 0]
        G = np.zeros((k,1), self.dtype)
        G[0,0] = 1
        self.G = G

    @property
    def start_params(self):
        """
        Starting parameters for maximum likelihood estimation
        """
        X = np.concatenate(
            [self.endog[:,self.p-i:-i] for i in range(1, self.p+1)], 0).T
        phi_cmle = np.linalg.pinv(X).dot(self.endog[:,self.p:].T)
        resids = self.endog[:,self.p:].T - X.dot(phi_cmle)
        sigma = resids.std()

        if self.measurement_error:
            sigma = np.r_[1, sigma]

        return np.r_[np.zeros(self.q,), phi_cmle[:,0], sigma**2].astype(self.dtype)

    def is_stationary(self, params=None, transformed=False):
        if params is not None:
            self.update(params, transformed)
        return np.max(np.abs(np.linalg.eigvals(self.F))) < 1

    def transform(self, unconstrained):
        """
        Transform unconstrained parameters used by the optimizer to constrained
        parameters used in likelihood evaluation

        References
        ----------

        Monahan, John F. 1984.
        "A Note on Enforcing Stationarity in
        Autoregressive-moving Average Models."
        Biometrika 71 (2) (August 1): 403-404.
        """
        constrained = np.zeros(unconstrained.shape, self.dtype)

        # Transform the MA parameters (theta) to be invertible
        if self.q > 0:
            constrained[:self.q] = constrain_stationary(
                unconstrained[:self.q]
            )

        # Transform the AR parameters (phi) to be stationary
        if self.p > 0:
            constrained[self.q:self.q+self.p] = constrain_stationary(
                unconstrained[self.q:self.q+self.p]
            )

        # Transform the standard deviation parameters to be positive
        if self.measurement_error:
            constrained[-2] = unconstrained[-2]**2
        constrained[-1] = unconstrained[-1]**2

        return constrained
    constrain = transform

    def untransform(self, constrained):
        """
        Transform constrained parameters used in likelihood evaluation
        to unconstrained parameters used by the optimizer

        References
        ----------

        Monahan, John F. 1984.
        "A Note on Enforcing Stationarity in
        Autoregressive-moving Average Models."
        Biometrika 71 (2) (August 1): 403-404.
        """

        unconstrained = np.zeros(constrained.shape, self.dtype)

        # Untransform the MA parameters (theta) from invertible
        if self.q > 0:
            unconstrained[:self.q] = unconstrain_stationary(
                constrained[:self.q]
            )

        # Untransform the AR parameters (phi) from stationary
        if self.p > 0:
            unconstrained[self.q:self.q+self.p] = unconstrain_stationary(
                constrained[self.q:self.q+self.p]
            )

        # Untransform the standard deviation
        if self.measurement_error:
            unconstrained[-2] = constrained[-2]**0.5
        unconstrained[-1] = constrained[-1]**0.5

        return unconstrained
    unconstrain = untransform

    def update(self, params, transformed=False):
        if not transformed:
            params = self.transform(params)
        theta = params[:self.q]
        self.H[0,1:,0] = np.r_[theta, [0]*(self.k-self.q-1)]

        phi = params[self.q:self.q+self.p]
        self.F[0] = np.r_[phi, [0]*(self.k-self.p)]

        if self.measurement_error:
            self.R[0,0] = params[-2]

        self.Q[0,0] = params[-1]

def constrain_stationary(unconstrained):
    n = unconstrained.shape[0]
    y = np.zeros((n, n))
    r = unconstrained/((1+unconstrained**2)**0.5)
    for k in range(n):
        for i in range(k):
            y[k,i] = y[k-1,i] + r[k] * y[k-1,k-i-1]
        y[k,k] = r[k]
    return -y[n-1,:]

def unconstrain_stationary(constrained):
    n = constrained.shape[0]
    y = np.zeros((n, n))
    y[n-1:] = -constrained
    for k in range(n-1,0,-1):
        for i in range(k):
            y[k-1,i] = (y[k,i] - y[k,k]*y[k,k-i-1]) / (1 - y[k,k]**2)
    r = y.diagonal()
    x = r / ((1 - r**2)**0.5)
    return x