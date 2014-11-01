"""
State Space Representation

Author: Chad Fulton
License: Simplified-BSD
"""
from __future__ import division, absolute_import, print_function

from warnings import warn

import numpy as np
from ..model import Model
from .tools import (
    find_best_blas_type, prefix_dtype_map, prefix_statespace_map,
    prefix_kalman_filter_map, prefix_kalman_smoother_map,
    prefix_simulation_smoother_map,
    validate_matrix_shape, validate_vector_shape
)

# Define constants
FILTER_CONVENTIONAL = 0x01     # Durbin and Koopman (2012), Chapter 4
FILTER_EXACT_INITIAL = 0x02    # ibid., Chapter 5.6
FILTER_AUGMENTED = 0x04        # ibid., Chapter 5.7
FILTER_SQUARE_ROOT = 0x08      # ibid., Chapter 6.3
FILTER_UNIVARIATE = 0x10       # ibid., Chapter 6.4
FILTER_COLLAPSED = 0x20        # ibid., Chapter 6.5
FILTER_EXTENDED = 0x40         # ibid., Chapter 10.2
FILTER_UNSCENTED = 0x80        # ibid., Chapter 10.3

SMOOTHER_STATE = 0x01          # Durbin and Koopman (2012), Chapter 4.4.2
SMOOTHER_STATE_COV = 0x02      # ibid., Chapter 4.4.3
SMOOTHER_DISTURBANCE = 0x04    # ibid., Chapter 4.5
SMOOTHER_DISTURBANCE_COV = 0x08    # ibid., Chapter 4.5
SMOOTHER_ALL = (
    SMOOTHER_STATE | SMOOTHER_STATE_COV | SMOOTHER_DISTURBANCE |
    SMOOTHER_DISTURBANCE_COV
)

SIMULATION_STATE = 0x01
SIMULATION_DISTURBANCE = 0x04
SIMULATION_ALL = (
    SIMULATION_STATE | SIMULATION_DISTURBANCE
)

INVERT_UNIVARIATE = 0x01
SOLVE_LU = 0x02
INVERT_LU = 0x04
SOLVE_CHOLESKY = 0x08
INVERT_CHOLESKY = 0x10
INVERT_NUMPY = 0x20

STABILITY_FORCE_SYMMETRY = 0x01

MEMORY_STORE_ALL = 0
MEMORY_NO_FORECAST = 0x01
MEMORY_NO_PREDICTED = 0x02
MEMORY_NO_FILTERED = 0x04
MEMORY_NO_LIKELIHOOD = 0x08
MEMORY_NO_GAIN = 0x10
MEMORY_NO_SMOOTHING = 0x20
MEMORY_CONSERVE = (
    MEMORY_NO_FORECAST | MEMORY_NO_PREDICTED | MEMORY_NO_FILTERED |
    MEMORY_NO_LIKELIHOOD | MEMORY_NO_GAIN | MEMORY_NO_SMOOTHING
)


class MatrixWrapper(object):
    def __init__(self, name, attribute):
        self.name = name
        self.attribute = attribute
        self._attribute = '_' + attribute

    def __get__(self, obj, objtype):
        matrix = getattr(obj, self._attribute, None)
        # # Remove last dimension if the array is not actually time-varying
        # if matrix is not None and matrix.shape[-1] == 1:
        #     return np.squeeze(matrix, -1)
        return matrix

    def __set__(self, obj, value):
        value = np.asarray(value, order="F")
        shape = obj.shapes[self.attribute]

        if len(shape) == 3:
            value = self._set_matrix(obj, value, shape)
        else:
            value = self._set_vector(obj, value, shape)

        setattr(obj, self._attribute, value)

    def _set_matrix(self, obj, value, shape):
        # Expand 1-dimensional array if possible
        if (value.ndim == 1 and shape[0] == 1
                and value.shape[0] == shape[1]):
            value = value[None, :]

        # Enforce that the matrix is appropriate size
        validate_matrix_shape(
            self.name, value.shape, shape[0], shape[1], obj.nobs
        )

        # Expand time-invariant matrix
        if value.ndim == 2:
            value = np.array(value[:, :, None], order="F")

        return value

    def _set_vector(self, obj, value, shape):
        # Enforce that the vector has appropriate length
        validate_vector_shape(
            self.name, value.shape, shape[0], obj.nobs
        )

        # Expand the time-invariant vector
        if value.ndim == 1:
            value = np.array(value[:, None], order="F")

        return value


class Representation(Model):
    r"""
    State space representation of a time series process

    Parameters
    ----------
    endog : array_like
        The observed time-series process :math:`y`
    k_states : int
        The dimension of the unobserved state process.
    k_posdef : int, optional
        The dimension of a guaranteed positive definite covariance matrix
        describing the shocks in the measurement equation. Must be less than
        or equal to `k_states`. Default is `k_states`.
    design : array_like, optional
        The design matrix, :math:`Z`. Default is set to zeros.
    obs_intercept : array_like, optional
        The intercept for the observation equation, :math:`d`. Default is set
        to zeros.
    obs_cov : array_like, optional
        The covariance matrix for the observation equation :math:`H`. Default
        is set to zeros.
    transition : array_like, optional
        The transition matrix, :math:`T`. Default is set to zeros.
    state_intercept : array_like, optional
        The intercept for the transition equation, :math:`c`. Default is set to
        zeros.
    selection : array_like, optional
        The selection matrix, :math:`R`. Default is set to zeros.
    state_cov : array_like, optional
        The covariance matrix for the state equation :math:`Q`. Default is set
        to zeros.

    Attributes
    ----------
    nobs : int
        The number of observations.
    k_endog : int
        The dimension of the observation series.
    k_states : int
        The dimension of the unobserved state process.
    k_posdef : int
        The dimension of a guaranteed positive definite covariance matrix
        describing the shocks in the measurement equation.
    dtype : dtype
        Datatype of currently active representation matrices
    prefix : str
        BLAS prefix of currently active representation matrices
    shapes : dictionary of name:tuple
        A dictionary recording the initial shapes of each of the representation
        matrices as tuples.
    endog : array
        The observation vector. Alias for `obs`.
    obs : array
        Observation vector: :math:`y~(k\_endog \times nobs)`
    design : array
        Design matrix: :math:`Z~(k\_endog \times k\_states \times nobs)`
    obs_intercept : array
        Observation intercept: :math:`d~(k\_endog \times nobs)`
    obs_cov : array
        Observation covariance matrix: :math:`H~(k\_endog \times k\_endog \times nobs)`
    transition : array
        Transition matrix: :math:`T~(k\_states \times k\_states \times nobs)`
    state_intercept : array
        State intercept: :math:`c~(k\_states \times nobs)`
    selection : array
        Selection matrix: :math:`R~(k\_states \times k\_posdef \times nobs)`
    state_cov : array
        State covariance matrix: :math:`Q~(k\_posdef \times k\_posdef \times nobs)`
    time_invariant : bool
        Whether or not currently active representation matrices are time-invariant
    initialization : str
        Kalman filter initialization method. Default is unset.
    initial_variance : float
        Initial variance for approximate diffuse initialization. Default is
        1e6.
    loglikelihood_burn : int
        The number of initial periods during which the loglikelihood is not
        recorded. Default is 0.
    filter_method : int
        Bitmask representing the Kalman filtering method.
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
        steady-state. Default is 1e-19.
    filter_results_class : class
        The class instantiated and returned as a result of the `filter` method.
        Default is FilterResults.
    simulation_smooth_results_class : class
        The class instantiated and returned as a result of the
        `simulation_smooth` method. Default is SimulationSmoothResults.

    Notes
    -----
    A general state space model is of the form

    .. math::

        y_t & = Z_t \alpha_t + d_t + \varepsilon_t \\
        \alpha_t & = T_t \alpha_{t-1} + c_t + R_t \eta_t \\

    where :math:`y_t` refers to the observation vector at time :math:`t`,
    :math:`\alpha_t` refers to the (unobserved) state vector at time
    :math:`t`, and where the irregular components are defined as

    .. math::

        \varepsilon_t \sim N(0, H_t) \\
        \eta_t \sim N(0, Q_t) \\

    The remaining variables (:math:`Z_t, d_t, H_t, T_t, c_t, R_t, Q_t`) in the
    equations are matrices describing the process. Their variable names and
    dimensions are as follows

    Z : `design`          :math:`(k\_endog \times k\_states \times nobs)`

    d : `obs_intercept`   :math:`(k\_endog \times nobs)`

    H : `obs_cov`         :math:`(k\_endog \times k\_endog \times nobs)`

    T : `transition`      :math:`(k\_states \times k\_states \times nobs)`

    c : `state_intercept` :math:`(k\_states \times nobs)`

    R : `selection`       :math:`(k\_states \times k\_posdef \times nobs)`

    Q : `state_cov`       :math:`(k\_posdef \times k\_posdef \times nobs)`

    In the case that one of the matrices is time-invariant (so that, for
    example, :math:`Z_t = Z_{t+1} ~ \forall ~ t`), its last dimension may
    be of size :math:`1` rather than size `nobs`.

    References
    ----------
    .. [1] Durbin, James, and Siem Jan Koopman. 2012.
       Time Series Analysis by State Space Methods: Second Edition.
       Oxford University Press.
    """

    design = MatrixWrapper('design', 'design')
    obs_intercept = MatrixWrapper('observation intercept', 'obs_intercept')
    obs_cov = MatrixWrapper('observation covariance matrix', 'obs_cov')
    transition = MatrixWrapper('transition', 'transition')
    state_intercept = MatrixWrapper('state intercept', 'state_intercept')
    selection = MatrixWrapper('selection', 'selection')
    state_cov = MatrixWrapper('state covariance matrix', 'state_cov')

    def __init__(self, endog, k_states, k_posdef=None, *args, **kwargs):
        self.shapes = {}

        # Initialize the model
        # (this sets _endog_names, nobs, and endog; if the argument `endog`
        # only consists of names, then self.nobs = None and self.endog = None)
        super(Representation, self).__init__(endog, *args, **kwargs)

        # Get the base datatype
        dtype = self.endog.dtype if self.endog is not None else np.float64

        # Get dimensions from transition equation
        if k_states < 1:
            raise ValueError('Number of states in statespace model must be a'
                             ' positive number.')
        self.k_states = k_states
        self.k_posdef = k_posdef if k_posdef is not None else k_states

        # Record the shapes of all of our matrices
        # Note: these are time-invariant shapes; in practice the last dimension
        # may also be `self.nobs` for any or all of these.
        self.shapes = {
            'obs': (self.k_endog, self.nobs if self.nobs is not None else 0),
            'design': (self.k_endog, self.k_states, 1),
            'obs_intercept': (self.k_endog, 1),
            'obs_cov': (self.k_endog, self.k_endog, 1),
            'transition': (self.k_states, self.k_states, 1),
            'state_intercept': (self.k_states, 1),
            'selection': (self.k_states, self.k_posdef, 1),
            'state_cov': (self.k_posdef, self.k_posdef, 1),
        }

        # Representation matrices
        # These matrices are only used in the Python object as containers,
        # which will be copied to the appropriate _statespace object if a
        # filter is called.
        for name, shape in self.shapes.items():
            if name == 'obs':
                continue
            # Create the initial storage array for each matrix
            setattr(self, '_' + name, np.zeros(shape, dtype=dtype, order="F"))

            # If we were given an initial value for the matrix, set it
            # (notice it is being set via the descriptor)
            if name in kwargs:
                setattr(self, name, kwargs[name])

        # State-space initialization data
        self.initialization = None
        self._initial_state = None
        self._initial_state_cov = None
        self._initial_variance = None

        # Matrix representations storage
        self._representations = {}

        # Setup the underlying statespace object storage
        self._statespaces = {}

        # Setup the underlying Kalman filter storage
        self._kalman_filters = {}

        # Setup the underlying Kalman smoother storage
        self._kalman_smoothers = {}

        # Setup the underlying simulation smoother storage
        self._simulation_smoothers = {}

        # Options
        self.initial_variance = kwargs.get('initial_variance', 1e6)
        self.loglikelihood_burn = kwargs.get('loglikelihood_burn', 0)
        self.filter_results_class = kwargs.get('filter_results_class', FilterResults)
        self.simulation_smooth_results_class = kwargs.get('simulation_smooth_results_class', SimulationSmoothResults)

        self.filter_method = kwargs.get(
            'filter_method', FILTER_CONVENTIONAL
        )
        self.inversion_method = kwargs.get(
            'inversion_method', INVERT_UNIVARIATE | SOLVE_CHOLESKY
        )
        self.stability_method = kwargs.get(
            'stability_method', STABILITY_FORCE_SYMMETRY
        )
        self.conserve_memory = kwargs.get('conserve_memory', 0)
        self.tolerance = kwargs.get('tolerance', 1e-19)

    def __len__(self):
        return self.nobs

    def __contains__(self, key):
        return key in self.shapes.keys()

    # def __repr__(self):
    #     pass

    def __str__(self):
        pass

    def __unicode__(self):
        pass

    def __hash__(self):
        pass

    def __getitem__(self, key):
        _type = type(key)
        # If only a string is given then we must be getting an entire matrix
        if _type is str:
            if not key in self.shapes:
                raise IndexError('"%s" is an invalid state space matrix name' % key)
            matrix = getattr(self, '_' + key)

            # See note on time-varying arrays, below
            if matrix.shape[-1] == 1:
                return np.squeeze(matrix, axis=-1)
        # Otherwise if we have a tuple, we want a slice of a matrix
        elif _type is tuple:
            name, slice_ = key[0], key[1:]
            if not name in self.shapes:
                raise IndexError('"%s" is an invalid state space matrix name' % name)

            matrix = getattr(self, '_' + name)

            # Since the model can support time-varying arrays, but often we
            # will instead have time-invariant arrays, we want to allow setting
            # a matrix slice like mod['transition',0,:] even though technically
            # it should be mod['transition',0,:,0]. Thus if the array in
            # question is time-invariant but the last slice was excluded,
            # add it in as a zero.
            if matrix.shape[-1] == 1 and len(slice_) <= matrix.ndim-1:
                slice_ = slice_ + (0,)

            return matrix[slice_]
        # Otherwise, we have only a single slice index, but it is not a string
        else:
            raise IndexError('first index must the name of a valid state space matrix')

    def __setitem__(self, key, value):
        _type = type(key)
        # If only a string is given then we must be setting an entire matrix
        if _type is str:
            if not key in self.shapes:
                raise IndexError('"%s" is an invalid state space matrix name' % key)
            setattr(self, key, value)
        # If it's a tuple (with a string as the first element) then we must be
        # setting a slice of a matrix
        elif _type is tuple:
            name, slice_ = key[0], key[1:]
            if not name in self.shapes:
                raise IndexError('"%s" is an invalid state space matrix name' % key[0])

            # Change the dtype of the corresponding matrix
            dtype = np.array(value).dtype
            matrix = getattr(self, '_' + name)
            if not matrix.dtype == dtype and dtype.char in ['f','d','F','D']:
                matrix = getattr(self, '_' + name).real.astype(dtype)
                

            # Since the model can support time-varying arrays, but often we
            # will instead have time-invariant arrays, we want to allow setting
            # a matrix slice like mod['transition',0,:] even though technically
            # it should be mod['transition',0,:,0]. Thus if the array in
            # question is time-invariant but the last slice was excluded,
            # add it in as a zero.
            if matrix.shape[-1] == 1 and len(slice_) == matrix.ndim-1:
                slice_ = slice_ + (0,)

            # Set the new value
            matrix[slice_] = value
            setattr(self, name, matrix)
        # Otherwise we got a single non-string key, (e.g. mod[:]), which is
        # invalid
        else:
            raise IndexError('first index must the name of a valid state space matrix')

    @property
    def prefix(self):
        return find_best_blas_type((
            self.endog, self._design, self._obs_intercept, self._obs_cov,
            self._transition, self._state_intercept, self._selection,
            self._state_cov
        ))[0]

    @property
    def dtype(self):
        return prefix_dtype_map[self.prefix]

    @property
    def time_invariant(self):
        return (
            self._design.shape[2] == self._obs_intercept.shape[1] ==
            self._obs_cov.shape[2] == self._transition.shape[2] ==
            self._state_intercept.shape[1] == self._selection.shape[2] ==
            self._state_cov.shape[2]
        )

    @property
    def _statespace(self):
        prefix = self.prefix
        if prefix in self._statespaces:
            return self._statespaces[prefix]
        return None

    @property
    def _kalman_filter(self):
        prefix = self.prefix
        if prefix in self._kalman_filters:
            return self._kalman_filters[prefix]
        return None

    @property
    def _kalman_smoother(self):
        prefix = self.prefix
        if prefix in self._kalman_smoothers:
            return self._kalman_smoothers[prefix]
        return None

    @property
    def _simulation_smoother(self):
        prefix = self.prefix
        if prefix in self._simulation_smoothers:
            return self._simulation_smoothers[prefix]
        return None

    @property
    def obs(self):
        return self.endog

    def bind(self, endog):
        super(Representation, self).bind(endog)
        self.shapes['obs'] = self.endog.shape

    def initialize_known(self, initial_state, initial_state_cov):
        """
        Initialize the statespace model with known distribution for initial
        state.

        These values are assumed to be known with certainty or else
        filled with parameters during, for example, maximum likelihood
        estimation.

        Parameters
        ----------
        initial_state : array_like
            Known mean of the initial state vector.
        initial_state_cov : array_like
            Known covariance matrix of the initial state vector.
        """
        initial_state = np.asarray(initial_state, order="F")
        initial_state_cov = np.asarray(initial_state_cov, order="F")

        if not initial_state.shape == (self.k_states,):
            raise ValueError('Invalid dimensions for initial state vector.'
                             ' Requires shape (%d,), got %s' %
                             (self.k_states, str(initial_state.shape)))
        if not initial_state_cov.shape == (self.k_states, self.k_states):
            raise ValueError('Invalid dimensions for initial covariance'
                             ' matrix. Requires shape (%d,%d), got %s' %
                             (self.k_states, self.k_states,
                              str(initial_state.shape)))

        self._initial_state = initial_state
        self._initial_state_cov = initial_state_cov
        self.initialization = 'known'

    def initialize_approximate_diffuse(self, variance=None):
        """
        Initialize the statespace model with approximate diffuse values.

        Rather than following the exact diffuse treatment (which is developed
        for the case that the variance becomes infinitely large), this assigns
        an arbitrary large number for the variance.

        Parameters
        ----------
        variance : float, optional
            The variance for approximating diffuse initial conditions. Default
            is 1e6.
        """
        if variance is None:
            variance = self.initial_variance

        self._initial_variance = variance
        self.initialization = 'approximate_diffuse'

    def initialize_stationary(self):
        """
        Initialize the statespace model as stationary.
        """
        self.initialization = 'stationary'

    def _initialize_filter(self, filter_method=None, inversion_method=None,
                           stability_method=None, conserve_memory=None,
                           tolerance=None, loglikelihood_burn=None,
                           recreate=True, return_loglike=False,
                           *args, **kwargs):
        if filter_method is None:
            filter_method = self.filter_method
        if inversion_method is None:
            inversion_method = self.inversion_method
        if stability_method is None:
            stability_method = self.stability_method
        if conserve_memory is None:
            conserve_memory = self.conserve_memory
        if loglikelihood_burn is None:
            loglikelihood_burn = self.loglikelihood_burn
        if tolerance is None:
            tolerance = self.tolerance

        # Make sure we have endog
        if self.endog is None:
            raise RuntimeError('Must bind a dataset to the model before'
                               ' filtering or smoothing.')

        # Determine which filter to call
        prefix = kwargs['prefix'] if 'prefix' in kwargs else self.prefix
        dtype = prefix_dtype_map[prefix]

        # If the dtype-specific representation matrices do not exist, create
        # them
        if prefix not in self._representations:
            # Copy the statespace representation matrices
            self._representations[prefix] = {}
            for matrix in self.shapes.keys():
                if matrix == 'obs':
                    self._representations[prefix][matrix] = self.obs.astype(dtype)
                else:
                    # Note: this always makes a copy
                    self._representations[prefix][matrix] = (
                        getattr(self, '_' + matrix).astype(dtype)
                    )
        # If they do exist, update them
        else:
            for matrix in self.shapes.keys():
                if matrix == 'obs':
                    self._representations[prefix][matrix] = self.obs.astype(dtype)[:]
                else:
                    self._representations[prefix][matrix][:] = (
                        getattr(self, '_' + matrix).astype(dtype)[:]
                    )

        # Determine if we need to re-create the _statespace models
        # (if time-varying matrices changed)
        recreate_statespace = False
        if recreate and prefix in self._statespaces:
            ss = self._statespaces[prefix]
            recreate_statespace = (
                not ss.obs.shape[1] == self.endog.shape[1] or
                not ss.design.shape[2] == self.design.shape[2] or
                not ss.obs_intercept.shape[1] == self.obs_intercept.shape[1] or
                not ss.obs_cov.shape[2] == self.obs_cov.shape[2] or
                not ss.transition.shape[2] == self.transition.shape[2] or
                not (ss.state_intercept.shape[1] ==
                     self.state_intercept.shape[1]) or
                not ss.selection.shape[2] == self.selection.shape[2] or
                not ss.state_cov.shape[2] == self.state_cov.shape[2]
            )

        # If the dtype-specific _statespace model does not exist, create it
        if prefix not in self._statespaces or recreate_statespace:
            # Setup the base statespace object
            cls = prefix_statespace_map[prefix]
            self._statespaces[prefix] = cls(
                self._representations[prefix]['obs'],
                self._representations[prefix]['design'],
                self._representations[prefix]['obs_intercept'],
                self._representations[prefix]['obs_cov'],
                self._representations[prefix]['transition'],
                self._representations[prefix]['state_intercept'],
                self._representations[prefix]['selection'],
                self._representations[prefix]['state_cov']
            )

        # Determine if we need to re-create the filter
        # (definitely need to recreate if we recreated the _statespace object)
        recreate_filter = recreate_statespace
        if recreate and not recreate_filter and prefix in self._kalman_filters:
            kalman_filter = self._kalman_filters[prefix]

            recreate_filter = (
                not kalman_filter.k_endog == self.k_endog or
                not kalman_filter.k_states == self.k_states or
                not kalman_filter.k_posdef == self.k_posdef or
                not kalman_filter.k_posdef == self.k_posdef or
                not kalman_filter.conserve_memory == conserve_memory or
                not kalman_filter.loglikelihood_burn == loglikelihood_burn
            )

        # If the dtype-specific _kalman_filter does not exist (or if we need
        # to recreate it), create it
        if prefix not in self._kalman_filters or recreate_filter:
            if recreate_filter:
                # Delete the old filter
                del self._kalman_filters[prefix]
            # Setup the filter
            cls = prefix_kalman_filter_map[prefix]
            self._kalman_filters[prefix] = cls(
                self._statespaces[prefix], filter_method, inversion_method,
                stability_method, conserve_memory, tolerance,
                loglikelihood_burn
            )
        # Otherwise, update the filter parameters
        else:
            self._kalman_filters[prefix].filter_method = filter_method
            self._kalman_filters[prefix].inversion_method = inversion_method
            self._kalman_filters[prefix].stability_method = stability_method
            self._kalman_filters[prefix].tolerance = tolerance

        # (Re-)initialize the statespace model
        if self.initialization == 'known':
            self._statespaces[prefix].initialize_known(
                self._initial_state.astype(dtype),
                self._initial_state_cov.astype(dtype)
            )
        elif self.initialization == 'approximate_diffuse':
            self._statespaces[prefix].initialize_approximate_diffuse(
                self._initial_variance
            )
        elif self.initialization == 'stationary':
            self._statespaces[prefix].initialize_stationary()
        else:
            raise RuntimeError('Statespace model not initialized.')

        return prefix, dtype

    def _initialize_smoother(self, smoother_output, *args, **kwargs):
        # Determine which smoother to call
        prefix = kwargs['prefix'] if 'prefix' in kwargs else self.prefix
        dtype = prefix_dtype_map[prefix]

        # Make sure we have the required Kalman filter
        if prefix not in self._kalman_filters:
            kwargs['prefix'] = prefix
            self._initialize_filter(*args, **kwargs)

        # If the dtype-specific _kalman_smoother does not exist, create it
        if prefix not in self._kalman_smoothers:
            # Setup the filter
            cls = prefix_kalman_smoother_map[prefix]
            self._kalman_smoothers[prefix] = cls(
                self._statespaces[prefix], self._kalman_filters[prefix],
                smoother_output
            )
        # Otherwise, update the smoother parameters
        else:
            self._kalman_smoothers[prefix].smoother_output = smoother_output

        return prefix, dtype

    def _initialize_simulation_smoother(self, simulation_output,
                                        *args, **kwargs):
        # Determine which smoother to call
        prefix = kwargs['prefix'] if 'prefix' in kwargs else self.prefix
        dtype = prefix_dtype_map[prefix]

        # Make sure we have the required Kalman filter
        if prefix not in self._kalman_filters:
            kwargs['prefix'] = prefix
            self._initialize_filter(*args, **kwargs)

        # Make sure we have the required Kalman smoother
        if prefix not in self._kalman_smoothers:
            kwargs.setdefault('smoother_output', simulation_output)
            kwargs['prefix'] = prefix
            self._initialize_smoother(*args, **kwargs)

        # If the dtype-specific _simulation_smoother does not exist, create it
        if prefix not in self._simulation_smoothers:
            # Setup the filter
            cls = prefix_simulation_smoother_map[prefix]
            self._simulation_smoothers[prefix] = cls(
                self._statespaces[prefix], self._kalman_filters[prefix],
                self._kalman_smoothers[prefix], simulation_output
            )
        # Otherwise, update the smoother parameters
        else:
            self._simulation_smoothers[prefix].simulation_output = (
                simulation_output
            )

        return prefix, dtype


    def filter(self, filter_method=None, inversion_method=None,
               stability_method=None, conserve_memory=None, tolerance=None,
               loglikelihood_burn=None,
               recreate=True, return_loglike=False, results_class=None,
               *args, **kwargs):
        """
        Apply the Kalman filter to the statespace model.

        Parameters
        ----------
        filter_method : int, optional
            Determines which Kalman filter to use. Default is conventional.
        inversion_method : int, optional
            Determines which inversion technique to use. Default is by Cholesky
            decomposition.
        stability_method : int, optional
            Determines which numerical stability techniques to use. Default is
            to enforce symmetry of the predicted state covariance matrix.
        conserve_memory : int, optional
            Determines what output from the filter to store. Default is to
            store everything.
        tolerance : float, optional
            The tolerance at which the Kalman filter determines convergence to
            steady-state. Default is 1e-19.
        loglikelihood_burn : int, optional
            The number of initial periods during which the loglikelihood is not
            recorded. Default is 0.
        recreate : bool, optional
            Whether or not to consider re-creating the underlying _statespace
            or filter objects (e.g. due to changing parameters, etc.). Often
            set to false during maximum likelihood estimation. Default is true.
        return_loglike : bool, optional
            Whether to only return the loglikelihood rather than a full
            `FilterResults` object. Default is False.
        """

        if results_class is None:
            results_class = self.filter_results_class

        # Initialize the filter
        prefix, dtype = self._initialize_filter(
            filter_method, inversion_method, stability_method, conserve_memory,
            tolerance, loglikelihood_burn, recreate, return_loglike,
            *args, **kwargs
        )

        # Run the filter
        self._kalman_filters[prefix]()

        if return_loglike:
            return np.array(self._kalman_filters[prefix].loglikelihood, copy=True)
        else:
            return results_class(self, self._kalman_filters[prefix])

    def loglike(self, loglikelihood_burn=None, *args, **kwargs):
        """
        Calculate the loglikelihood associated with the statespace model.

        Parameters
        ----------
        loglikelihood_burn : int, optional
            The number of initial periods during which the loglikelihood is not
            recorded. Default is 0.

        Returns
        -------
        loglike : float
            The joint loglikelihood.
        """
        if loglikelihood_burn is None:
            loglikelihood_burn = self.loglikelihood_burn
        kwargs['return_loglike'] = True
        return np.sum(self.filter(*args, **kwargs)[loglikelihood_burn:])

    def smooth_state(self, refilter=False, *args, **kwargs):
        """
        Perform state smoothing.

        Parameters
        ----------
        refilter : bool, optional
            Whether or not to perform filtering prior to smoothing. Default is
            False, unless filtering objects are not available.

        Returns
        -------
        smoothed_state : array
        smoothed_state_cov : array
        """
        prefix = self.prefix

        if refilter or prefix not in self._kalman_filters:
            kwargs['prefix'] = prefix
            kwargs['return_loglike'] = True
            self.filter(*args, **kwargs)

        results = self._smooth(SMOOTHER_STATE | SMOOTHER_STATE_COV,
                               *args, **kwargs)

        return results[3:5]

    def smooth_disturbance(self, refilter=False, *args, **kwargs):
        """
        Perform measurement and state disturbance smoothing.

        Parameters
        ----------
        refilter : bool, optional
            Whether or not to perform filtering prior to smoothing. Default is
            False, unless filtering objects are not available.

        Returns
        -------
        smoothed_measurement_disturbance : array
        smoothed_state_disturbance : array
        smoothed_measurement_disturbance_cov : array
        smoothed_state_disturbance_cov : array
        """
        prefix = self.prefix

        if refilter or prefix not in self._kalman_filters:
            kwargs['prefix'] = prefix
            kwargs['return_loglike'] = True
            self.filter(*args, **kwargs)

        results = self._smooth(SMOOTHER_DISTURBANCE | SMOOTHER_DISTURBANCE_COV,
                               *args, **kwargs)

        return results[5:9]

    def smooth(self, smoother_output=SMOOTHER_ALL, results_class=None,
               refilter=False, *args, **kwargs):
        """
        Apply the Kalman smoother to the statespace model.

        Parameters
        ----------
        smoother_output : int, optional
            Determines which Kalman smoother output calculate. Default is all
            (including state, disturbances, and all covariances).
        results_class : class
            The class instantiated and returned as a result of the filtering;
            smoothing is called on the resultant object. Default is
            FilterResults.
        refilter : bool, optional
            Whether or not to perform filtering prior to smoothing. Default is
            False, unless filtering objects are not available.
        Returns
        -------
        FilterResults object
        """
        if results_class is None:
            results_class = self.filter_results_class

        prefix = self.prefix

        if refilter or prefix not in self._kalman_filters:
            kwargs['prefix'] = prefix
            kwargs['return_loglike'] = True
            self.filter(*args, **kwargs)

        results = results_class(self, self._kalman_filter)
        results.smooth(smoother_output)

        return results

    def _smooth(self, smoother_output, *args, **kwargs):
        # Initialize the smoother
        prefix, dtype = self._initialize_smoother(smoother_output,
                                                  *args, **kwargs)

        # Run the smoother
        smoother = self._kalman_smoothers[prefix]
        smoother()

        # Return smoothing output
        results = (
            np.array(smoother.scaled_smoothed_estimator, copy=True),
            np.array(smoother.scaled_smoothed_estimator_cov, copy=True),
            np.array(smoother.smoothing_error, copy=True),
        )
        if smoother_output & SMOOTHER_STATE:
            results += (np.array(smoother.smoothed_state, copy=True),)
        else:
            results += (None,)
        if smoother_output & SMOOTHER_STATE_COV:
            results += (np.array(smoother.smoothed_state_cov, copy=True),)
        else:
            results += (None,)
        if smoother_output & SMOOTHER_DISTURBANCE:
            results += (
                np.array(smoother.smoothed_measurement_disturbance, copy=True),
                np.array(smoother.smoothed_state_disturbance, copy=True),
            )
        else:
            results += (None, None)
        if smoother_output & SMOOTHER_DISTURBANCE_COV:
            results += (
                np.array(smoother.smoothed_measurement_disturbance_cov, copy=True),
                np.array(smoother.smoothed_state_disturbance_cov, copy=True),
            )
        else:
            results += (None, None)

        return results

    def simulation_smoother(self, simulation_output=SIMULATION_ALL,
                            results_class=None, *args, **kwargs):
        """
        Retrieve a simulation smoother for the statespace model.

        Parameters
        ----------
        simulation_output : int, optional
            Determines which simulation smoother output is calculated.
            Default is all (including state and disturbances).
        Returns
        -------
        SimulationSmoothResults
        """
        # Check if we've filtered yet
        have_filtered = len(self._kalman_filters) > 0

        # Initialize the results class
        if results_class is None:
            results_class = self.simulation_smooth_results_class

        results = results_class(self, simulation_output)

        # Simulate if we have already filtered
        if have_filtered > 0:
            results.simulate(*args, **kwargs)

        return results
        

class FilterResults(object):
    """
    Results from applying the Kalman filter to a state space model.

    Takes a snapshot of a Statespace model and accompanying Kalman filter,
    saving the model representation and filter output.

    Parameters
    ----------
    model : Representation
        A Statespace representation
    kalman_filter : _statespace.{'s','c','d','z'}KalmanFilter
        A Kalman filter object.

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
    filtered_state : array
        The filtered state vector at each time period.
    filtered_state_cov : array
        The filtered state covariance matrix at each time period.
    predicted_state : array
        The predicted state vector at each time period.
    predicted_state_cov : array
        The predicted state covariance matrix at each time period.
    forecasts : array
        The one-step-ahead forecasts of observations at each time period.
    forecasts_error : array
        The forecast errors at each time period.
    forecasts_error_cov : array
        The forecast error covariance matrices at each time period.
    standardized_forecast_error : array
        The standardized forecast errors
    loglikelihood : array
        The loglikelihood values at each time period.
    time_invariant : bool
        Whether or not the representation matrices are time-invariant
    converged : bool
        Whether or not the Kalman filter converged.
    period_converged : int
        The time period in which the Kalman filter converged.
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
    """
    def __init__(self, model, kalman_filter):
        # Model
        self.model = model

        # Data type
        self.prefix = model.prefix
        self.dtype = model.dtype

        # Copy the model dimensions
        self.nobs = model.nobs
        self.k_endog = model.k_endog
        self.k_states = model.k_states
        self.k_posdef = model.k_posdef
        self.time_invariant = model.time_invariant

        # Save the state space representation at the time
        self.endog = model.endog
        self.design = model._design.copy()
        self.obs_intercept = model._obs_intercept.copy()
        self.obs_cov = model._obs_cov.copy()
        self.transition = model._transition.copy()
        self.state_intercept = model._state_intercept.copy()
        self.selection = model._selection.copy()
        self.state_cov = model._state_cov.copy()

        self.missing = np.array(model._statespaces[self.prefix].missing,
                                copy=True)
        self.nmissing = np.array(model._statespaces[self.prefix].nmissing,
                                 copy=True)

        # Save the final shapes of the matrices
        self.shapes = dict(model.shapes)
        for name in self.shapes.keys():
            if name == 'obs':
                continue
            self.shapes[name] = getattr(self, name).shape
        self.shapes['obs'] = self.endog.shape

        # Save the state space initialization
        self.initialization = model.initialization
        self.initial_state = np.array(kalman_filter.model.initial_state,
                                      copy=True)
        self.initial_state_cov = np.array(
            kalman_filter.model.initial_state_cov, copy=True
        )

        # Save Kalman filter parameters
        self.filter_method = kalman_filter.filter_method
        self.inversion_method = kalman_filter.inversion_method
        self.stability_method = kalman_filter.stability_method
        self.conserve_memory = kalman_filter.conserve_memory
        self.tolerance = kalman_filter.tolerance
        self.loglikelihood_burn = kalman_filter.loglikelihood_burn

        # Save Kalman filter output
        self.converged = bool(kalman_filter.converged)
        self.period_converged = kalman_filter.period_converged

        self.filtered_state = np.array(kalman_filter.filtered_state, copy=True)
        self.filtered_state_cov = np.array(kalman_filter.filtered_state_cov, copy=True)
        self.predicted_state = np.array(kalman_filter.predicted_state, copy=True)
        self.predicted_state_cov = np.array(
            kalman_filter.predicted_state_cov, copy=True
        )
        self.kalman_gain = np.array(kalman_filter.kalman_gain, copy=True)

        self.tmp1 = np.array(kalman_filter.tmp1, copy=True)
        self.tmp2 = np.array(kalman_filter.tmp2, copy=True)
        self.tmp3 = np.array(kalman_filter.tmp3, copy=True)
        self.tmp4 = np.array(kalman_filter.tmp4, copy=True)

        # Setup empty arrays for possible smoothing output
        # (they are not filled in until `smooth` is called)
        self.smoother_output = None
        self.scaled_smoothed_estimator = None
        self.scaled_smoothed_estimator_cov = None
        self.smoothing_error = None
        self.smoothed_state = None
        self.smoothed_state_cov = None
        self.smoothed_measurement_disturbance = None
        self.smoothed_state_disturbance = None
        self.smoothed_measurement_disturbance_cov = None
        self.smoothed_state_disturbance_cov = None

        # Note: use forecasts rather than forecast, so as not to interfer
        # with the `forecast` methods in subclasses
        self.forecasts = np.array(kalman_filter.forecast, copy=True)
        self.forecasts_error = np.array(kalman_filter.forecast_error, copy=True)
        self.forecasts_error_cov = np.array(kalman_filter.forecast_error_cov, copy=True)
        self.loglikelihood = np.array(kalman_filter.loglikelihood, copy=True)

        # Setup caches for uninitialized objects
        self._kalman_gain = None
        self._standardized_forecasts_error = None

        # Fill in missing values in the forecast, forecast error, and
        # forecast error covariance matrix (this is required due to how the
        # Kalman filter implements observations that are completely missing)
        # Construct the predictions, forecasts
        if not (self.conserve_memory & MEMORY_NO_FORECAST or
                self.conserve_memory & MEMORY_NO_PREDICTED):
            for t in range(self.nobs):
                design_t = 0 if self.design.shape[2] == 1 else t
                obs_cov_t = 0 if self.obs_cov.shape[2] == 1 else t
                obs_intercept_t = 0 if self.obs_intercept.shape[1] == 1 else t

                # Skip anything that is less than completely missing
                if self.nmissing[t] < self.k_endog:
                    continue

                self.forecasts[:, t] = np.dot(
                    self.design[:, :, design_t], self.predicted_state[:, t]
                ) + self.obs_intercept[:, obs_intercept_t]
                self.forecasts_error[:, t] = np.nan
                self.forecasts_error_cov[:, :, t] = np.dot(
                    np.dot(self.design[:, :, design_t],
                           self.predicted_state_cov[:, :, t]),
                    self.design[:, :, design_t].T
                ) + self.obs_cov[:, :, obs_cov_t]

    @property
    def py_kalman_gain(self):
        if self._kalman_gain is None:
            # k x n
            self._kalman_gain = np.zeros(
                (self.k_states, self.k_endog, self.nobs), dtype=self.dtype)
            for t in range(self.nobs):
                design_t = 0 if self.design.shape[2] == 1 else t
                transition_t = 0 if self.transition.shape[2] == 1 else t
                self._kalman_gain[:, :, t] = np.dot(
                    np.dot(
                        self.transition[:, :, transition_t],
                        self.predicted_state_cov[:, :, t]
                    ),
                    np.dot(
                        np.transpose(self.design[:, :, design_t]),
                        np.linalg.inv(self.forecasts_error_cov[:, :, t])
                    )
                )
        return self._kalman_gain

    @property
    def standardized_forecasts_error(self):
        if self._standardized_forecasts_error is None:
            from scipy import linalg
            self._standardized_forecasts_error = np.zeros(
                self.forecasts_error.shape, dtype=self.dtype)

            for t in range(self.forecasts_error_cov.shape[2]):
                upper, _ = linalg.cho_factor(self.forecasts_error_cov[:, :, t],
                                         check_finite=False)
                self._standardized_forecasts_error[:, t] = (
                    linalg.solve_triangular(upper, self.forecasts_error[:, t],
                                            check_finite=False))

        return self._standardized_forecasts_error

    def smooth(self, smoother_output=SMOOTHER_ALL):
        """
        Apply the Kalman smoother to the statespace model.

        Parameters
        ----------
        smoother_output : int, optional
            Determines which Kalman smoother output calculate. Default is all
            (including state, disturbances, and all covariances).
        
        Notes
        -----
        Using this method will run the Kalman smoother based on the current
        state of the stored `Representation` object, but will not re-run the
        Kalman filter. This is to avoid unnecessary re-calculations. However
        the `FilterResults` object is not kept in-sync with the stored
        `Representation` object (because the results are intended to be fixed,
        whereas the same `Representation` may be re-used many times, for
        example in parameter estimation). Typically smoothing will be performed
        either directly after filtering (for example in Bayesian approaches) or
        else will be performed only after settling on the final
        parameterization (for example in MLE approaches), and in both of these
        cases the `Representation` object will match the `FilterResults`
        object.

        However, if you are concerned that between the filtering operation
        (that created the `FilterResults` object) and the smoothing operation
        the `Representation` object may have been changed then you should
        either first re-run the filter or else use the `Representation.smooth`
        method (which always first re-runs the filter).
        """
        # Run the smoother
        smoothed = self.model._smooth(smoother_output)

        # Set the output
        self.smoother_output = smoother_output
        # For r_t (and similarly for N_t), what was calculated was
        # r_T, ..., r_{-1}, and stored such that
        # scaled_smoothed_estimator[0] == r_{-1}. We only want r_0, ..., r_T
        # so exclude the zeroth element so that the time index is consistent
        # with the other returned output
        self.scaled_smoothed_estimator = smoothed[0][:,1:]
        self.scaled_smoothed_estimator_cov = smoothed[1][:,:,1:]
        self.smoothing_error = smoothed[2]
        self.smoothed_state = smoothed[3]
        self.smoothed_state_cov = smoothed[4]
        self.smoothed_measurement_disturbance = smoothed[5]
        self.smoothed_state_disturbance = smoothed[6]
        self.smoothed_measurement_disturbance_cov = smoothed[7]
        self.smoothed_state_disturbance_cov = smoothed[8]

    def predict(self, start=None, end=None, dynamic=None, full_results=False,
                *args, **kwargs):
        """
        In-sample and out-of-sample prediction for state space models generally

        Parameters
        ----------
        start : int, optional
            Zero-indexed observation number at which to start forecasting,
            i.e., the first forecast will be at start.
        end : int, optional
            Zero-indexed observation number at which to end forecasting, i.e.,
            the last forecast will be at end.
        dynamic : int or boolean or None, optional
            Specifies the number of steps ahead for each in-sample prediction.
            If not specified, then in-sample predictions are one-step-ahead.
            False and None are interpreted as 0. Default is False.
        full_results : boolean, optional
            If True, returns a FilterResults instance; if False returns a
            tuple with forecasts, the forecast errors, and the forecast error
            covariance matrices. Default is False.

        Returns
        -------
        results : FilterResults or tuple
            Either a FilterResults object (if `full_results=True`) or else a
            tuple of forecasts, the forecast errors, and the forecast error
            covariance matrices otherwise.

        Notes
        -----
        All prediction is performed by applying the deterministic part of the
        measurement equation using the predicted state variables.

        Out-of-sample prediction first applies the Kalman filter to missing
        data for the number of periods desired to obtain the predicted states.
        """
        # Cannot predict if we do not have appropriate arrays
        if (self.conserve_memory & MEMORY_NO_FORECAST or
           self.conserve_memory & MEMORY_NO_PREDICTED):
            raise ValueError('Predict is not possible if memory conservation'
                             ' has been used to avoid storing forecasts or'
                             ' predicted values.')

        # Get the start and the end of the entire prediction range
        if start is None:
            start = 0
        elif start < 0:
            raise ValueError('Cannot predict values previous to the sample.')
        if end is None:
            end = self.nobs

        # Total number of predicted values
        npredicted = end - start

        # Short-circuit if end is before start
        if npredicted < 0:
            return (np.zeros((self.k_endog, 0)),
                    np.zeros((self.k_endog, self.k_endog, 0)))

        # Get the number of forecasts to make after the end of the sample
        # Note: this may be larger than npredicted if the predict command was
        # called, for example, to calculate forecasts for nobs+10 through
        # nobs+20, because the operations below will need to start forecasts no
        # later than the end of the sample and go through `end`. Any
        # calculations prior to `start` will be ignored.
        nforecast = max(0, end - self.nobs)

        # Get the total size of the in-sample prediction component (whether via
        # one-step-ahead or dynamic prediction)
        nsample = npredicted - nforecast

        # Get the number of periods until dynamic forecasting is used
        if dynamic > nsample:
            warn('Dynamic prediction specified for more steps-ahead (%d) than'
                 ' there are observations in the specified range (%d).'
                 ' `dynamic` has been automatically adjusted to %d. If'
                 ' possible, you may want to set `start` to be earlier.'
                 % (dynamic, nsample, nsample))
            dynamic = nsample

        if dynamic is None or dynamic is False:
            dynamic = nsample
        ndynamic = nsample - dynamic

        if dynamic < 0:
            raise ValueError('Prediction cannot be specified with a negative'
                             ' dynamic prediction offset.')

        # Get the number of in-sample, one-step-ahead predictions
        ninsample = nsample - ndynamic

        # Total numer of left-padded zeros
        # Two cases would have this as non-zero. Here are some examples:
        # - If start = 4 and dynamic = 4, then npadded >= 4 so that even the
        #   `start` observation has dynamic of 4
        # - If start = 10 and nobs = 5, then npadded >= 5 because the
        #   intermediate forecasts are required for the desired forecasts.
        npadded = max(0, start - dynamic, start - self.nobs)

        # Construct the design and observation intercept and covariance
        # matrices for start-npadded:end. If not time-varying in the original
        # model, then they will be copied over if none are provided in
        # `kwargs`. Otherwise additional matrices must be provided in `kwargs`.
        representation = {}
        for name, shape in self.shapes.items():
            if name == 'obs':
                continue
            mat = getattr(self, name)
            if shape[-1] == 1:
                representation[name] = mat
            elif len(shape) == 3:
                representation[name] = mat[:, :, start-npadded:]
            else:
                representation[name] = mat[:, start-npadded:]

        # Update the matrices from kwargs for forecasts
        warning = ('Model has time-invariant %s matrix, so the %s'
                   ' argument to `predict` has been ignored.')
        exception = ('Forecasting for models with time-varying %s matrix'
                     ' requires an updated time-varying matrix for the'
                     ' period to be forecasted.')
        if nforecast > 0:
            for name, shape in self.shapes.items():
                if name == 'obs':
                    continue
                if representation[name].shape[-1] == 1:
                    if name in kwargs:
                        warn(warning % (name, name))
                elif name not in kwargs:
                    raise ValueError(exception % name)
                else:
                    mat = np.asarray(kwargs[name])
                    if len(shape) == 2:
                        validate_vector_shape('obs_intercept', mat.shape,
                                              shape[0], nforecast)
                        if mat.ndim < 2 or not mat.shape[1] == nforecast:
                            raise ValueError(exception % name)
                        representation[name] = np.c_[representation[name], mat]
                    else:
                        validate_matrix_shape(name, mat.shape, shape[0],
                                              shape[1], nforecast)
                        if mat.ndim < 3 or not mat.shape[2] == nforecast:
                            raise ValueError(exception % name)
                        representation[name] = np.c_[representation[name], mat]

        # Construct the predicted state and covariance matrix for each time
        # period depending on whether that time period corresponds to
        # one-step-ahead prediction, dynamic prediction, or out-of-sample
        # forecasting.

        # If we only have simple prediction, then we can use the already saved
        # Kalman filter output
        if ndynamic == 0 and nforecast == 0:
            result = self
        else:
            # Construct the new endogenous array - notice that it has
            # npredicted + npadded values (rather than the entire start array,
            # in case the number of observations is large and we don't want to
            # re-run the entire filter just for forecasting)
            endog = np.empty((self.k_endog, nforecast))
            endog.fill(np.nan)
            endog = np.c_[self.endog[:, start-npadded:], endog]

            # Setup the new statespace representation
            model_kwargs = {
                'filter_method': self.filter_method,
                'inversion_method': self.inversion_method,
                'stability_method': self.stability_method,
                'conserve_memory': self.conserve_memory,
                'tolerance': self.tolerance,
                'loglikelihood_burn': self.loglikelihood_burn
            }
            model_kwargs.update(representation)
            model = Representation(
                endog.T, self.k_states, self.k_posdef, **model_kwargs
            )
            model.initialize_known(
                self.predicted_state[:, 0],
                self.predicted_state_cov[:, :, 0]
            )
            model._initialize_filter(*args, **kwargs)

            result = self._predict(ninsample, ndynamic, nforecast, model)

        if full_results:
            return result
        else:
            return (
                result.forecasts[:, npadded:],
                result.forecasts_error[:, npadded:],
                result.forecasts_error_cov[:, :, npadded:]
            )

    def _predict(self, ninsample, ndynamic, nforecast, model, *args, **kwargs):
        # Get the underlying filter
        kfilter = model._kalman_filter

        # Save this (which shares memory with the memoryview on which the
        # Kalman filter will be operating) so that we can replace actual data
        # with predicted data during dynamic forecasting
        endog = model._representations[model.prefix]['obs']

        for t in range(kfilter.model.nobs):
            # Run the Kalman filter for the first `ninsample` periods (for
            # which dynamic computation will not be performed)
            if t < ninsample:
                next(kfilter)
            # Perform dynamic prediction
            elif t < ninsample+ndynamic:
                design_t = 0 if model.design.shape[2] == 1 else t
                obs_intercept_t = 0 if model.obs_intercept.shape[1] == 1 else t

                # Predict endog[:, t] given `predicted_state` calculated in
                # previous iteration (i.e. t-1)
                endog[:, t] = np.dot(
                    model.design[:, :, design_t],
                    kfilter.predicted_state[:, t]
                ) + model.obs_intercept[:, obs_intercept_t]

                # Advance Kalman filter
                next(kfilter)
            # Perform any (one-step-ahead) forecasting
            else:
                next(kfilter)

        # Return the predicted state and predicted state covariance matrices
        return FilterResults(model, kfilter)

class SimulationSmoothResults(object):
    
    def __init__(self, model, simulation_output=SIMULATION_ALL,
                 *args, **kwargs):
        self.model = model

        # Initialize the simulation smoother
        self.prefix, self.dtype = model._initialize_simulation_smoother(
            simulation_output, *args, **kwargs
        )
        self.simulation_smoother = model._simulation_smoothers[self.prefix]

        # Output
        self._generated_obs = None
        self._generated_state = None
        self._simulated_state = None
        self._simulated_measurement_disturbance = None
        self._simulated_state_disturbance = None

    @property
    def generated_obs(self):
        if self._generated_obs is None:
            self._generated_obs = np.array(
                self.simulation_smoother.generated_obs, copy=True
            )
        return self._generated_obs

    @property
    def generated_state(self):
        if self._generated_state is None:
            self._generated_state = np.array(
                self.simulation_smoother.generated_state, copy=True
            )
        return self._generated_state

    @property
    def simulated_state(self):
        if self._simulated_state is None:
            self._simulated_state = np.array(
                self.simulation_smoother.simulated_state, copy=True
            )
        return self._simulated_state

    @property
    def simulated_measurement_disturbance(self):
        if self._simulated_measurement_disturbance is None:
            self._simulated_measurement_disturbance = np.array(
                self.simulation_smoother.simulated_measurement_disturbance,
                copy=True
            )
        return self._simulated_measurement_disturbance

    @property
    def simulated_state_disturbance(self):
        if self._simulated_state_disturbance is None:
            self._simulated_state_disturbance = np.array(
                self.simulation_smoother.simulated_state_disturbance,
                copy=True
            )
        return self._simulated_state_disturbance

    def simulate(self, simulation_output=-1, disturbance_variates=None,
                 initial_state_variates=None):
        # Clear any previous output
        self._generated_obs = None
        self._generated_state = None
        self._simulated_state = None
        self._simulated_measurement_disturbance = None
        self._simulated_state_disturbance = None

        # Draw the (independent) random variates for disturbances in the
        # simulation
        if disturbance_variates is not None:
            self.simulation_smoother.set_disturbance_variates(
                np.array(disturbance_variates, dtype=self.dtype)
            )
        else:
            self.simulation_smoother.draw_disturbance_variates()

        # Draw the (independent) random variates for the initial states in the
        # simulation
        if initial_state_variates is not None:
            self.simulation_smoother.set_initial_state_variates(
                np.array(initial_state_variates, dtype=self.dtype)
            )
        else:
            self.simulation_smoother.draw_initial_state_variates()

        # Perform simulation smoothing
        # Note: simulation_output=-1 corresponds to whatever was setup when
        # the simulation smoother was constructed
        self.simulation_smoother.simulate(simulation_output)
