#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=False
"""
State Space Model Smoother declarations

Author: Chad Fulton  
License: Simplified-BSD
"""

cdef int SMOOTHER_STATE           # Durbin and Koopman (2012), Chapter 4.4.2
cdef int SMOOTHER_STATE_COV       # Durbin and Koopman (2012), Chapter 4.4.3
cdef int SMOOTHER_DISTURBANCE     # Durbin and Koopman (2012), Chapter 4.5
cdef int SMOOTHER_DISTURBANCE_COV # Durbin and Koopman (2012), Chapter 4.5
cdef int SMOOTHER_ALL

# Typical imports
cimport numpy as np

from dismalpy.ssm._statespace cimport (
    sStatespace, dStatespace, cStatespace, zStatespace
)
from dismalpy.ssm._kalman_filter cimport (
    sKalmanFilter, dKalmanFilter, cKalmanFilter, zKalmanFilter
)
from dismalpy.ssm._kalman_smoother cimport (
    sKalmanSmoother, dKalmanSmoother, cKalmanSmoother, zKalmanSmoother
)

# Single precision
cdef class sSimulationSmoother(object):
    # ### Statespace model
    cdef readonly sStatespace model
    # ### Kalman filter
    cdef readonly sKalmanFilter kfilter
    # ### Kalman smoother
    cdef readonly sKalmanSmoother smoother

    # ### Simulated Statespace model
    cdef readonly sStatespace simulated_model
    # ### Simulated Kalman filter
    cdef readonly sKalmanFilter simulated_kfilter
    # ### Simulated Kalman smoother
    cdef readonly sKalmanSmoother simulated_smoother

    # ### Simulation parameters
    cdef public int simulation_output

    # ### Random variates
    cdef int n_disturbance_variates
    cdef readonly np.float32_t [:] disturbance_variates
    cdef int n_initial_state_variates
    cdef readonly np.float32_t [:] initial_state_variates

    # ### Simulated Data
    cdef readonly np.float32_t [::1,:] simulated_measurement_disturbance
    cdef readonly np.float32_t [::1,:] simulated_state_disturbance
    cdef readonly np.float32_t [::1,:] simulated_state

    # ### Generated Data
    cdef readonly np.float32_t [::1,:] generated_obs
    cdef readonly np.float32_t [::1,:] generated_state

    # ### Temporary arrays
    cdef readonly np.float32_t [::1,:] tmp0, tmp1, tmp2

    # ### Pointers
    cdef np.float32_t * _tmp0
    cdef np.float32_t * _tmp1
    cdef np.float32_t * _tmp2

    cpdef draw_disturbance_variates(self)
    cpdef draw_initial_state_variates(self)
    cpdef set_disturbance_variates(self, np.float32_t [:] variates)
    cpdef set_initial_state_variates(self, np.float32_t [:] variates)
    cpdef simulate(self, int simulation_output=*)

    cdef np.float32_t generate_obs(self, int t, np.float32_t * obs, np.float32_t * state, np.float32_t * variates)
    cdef np.float32_t generate_state(self, int t, np.float32_t * state, np.float32_t * input_state, np.float32_t * variates)
    cdef void cholesky(self, np.float32_t * source, np.float32_t * destination, int n)
    cdef void transform_variates(self, np.float32_t * variates, np.float32_t * cholesky_factor, int n)

# Double precision
cdef class dSimulationSmoother(object):
    # ### Statespace model
    cdef readonly dStatespace model
    # ### Kalman filter
    cdef readonly dKalmanFilter kfilter
    # ### Kalman smoother
    cdef readonly dKalmanSmoother smoother

    # ### Simulated Statespace model
    cdef readonly dStatespace simulated_model
    # ### Simulated Kalman filter
    cdef readonly dKalmanFilter simulated_kfilter
    # ### Simulated Kalman smoother
    cdef readonly dKalmanSmoother simulated_smoother

    # ### Simulation parameters
    cdef public int simulation_output

    # ### Random variates
    cdef int n_disturbance_variates
    cdef readonly np.float64_t [:] disturbance_variates
    cdef int n_initial_state_variates
    cdef readonly np.float64_t [:] initial_state_variates

    # ### Simulated Data
    cdef readonly np.float64_t [::1,:] simulated_measurement_disturbance
    cdef readonly np.float64_t [::1,:] simulated_state_disturbance
    cdef readonly np.float64_t [::1,:] simulated_state

    # ### Generated Data
    cdef readonly np.float64_t [::1,:] generated_obs
    cdef readonly np.float64_t [::1,:] generated_state

    # ### Temporary arrays
    cdef readonly np.float64_t [::1,:] tmp0, tmp1, tmp2

    # ### Pointers
    cdef np.float64_t * _tmp0
    cdef np.float64_t * _tmp1
    cdef np.float64_t * _tmp2

    cpdef draw_disturbance_variates(self)
    cpdef draw_initial_state_variates(self)
    cpdef set_disturbance_variates(self, np.float64_t [:] variates)
    cpdef set_initial_state_variates(self, np.float64_t [:] variates)
    cpdef simulate(self, int simulation_output=*)

    cdef np.float64_t generate_obs(self, int t, np.float64_t * obs, np.float64_t * state, np.float64_t * variates)
    cdef np.float64_t generate_state(self, int t, np.float64_t * state, np.float64_t * input_state, np.float64_t * variates)
    cdef void cholesky(self, np.float64_t * source, np.float64_t * destination, int n)
    cdef void transform_variates(self, np.float64_t * variates, np.float64_t * cholesky_factor, int n)

# Single precision complex
cdef class cSimulationSmoother(object):
    # ### Statespace model
    cdef readonly cStatespace model
    # ### Kalman filter
    cdef readonly cKalmanFilter kfilter
    # ### Kalman smoother
    cdef readonly cKalmanSmoother smoother

    # ### Simulated Statespace model
    cdef readonly cStatespace simulated_model
    # ### Simulated Kalman filter
    cdef readonly cKalmanFilter simulated_kfilter
    # ### Simulated Kalman smoother
    cdef readonly cKalmanSmoother simulated_smoother

    # ### Simulation parameters
    cdef public int simulation_output

    # ### Random variates
    cdef int n_disturbance_variates
    cdef readonly np.complex64_t [:] disturbance_variates
    cdef int n_initial_state_variates
    cdef readonly np.complex64_t [:] initial_state_variates

    # ### Simulated Data
    cdef readonly np.complex64_t [::1,:] simulated_measurement_disturbance
    cdef readonly np.complex64_t [::1,:] simulated_state_disturbance
    cdef readonly np.complex64_t [::1,:] simulated_state

    # ### Generated Data
    cdef readonly np.complex64_t [::1,:] generated_obs
    cdef readonly np.complex64_t [::1,:] generated_state

    # ### Temporary arrays
    cdef readonly np.complex64_t [::1,:] tmp0, tmp1, tmp2

    # ### Pointers
    cdef np.complex64_t * _tmp0
    cdef np.complex64_t * _tmp1
    cdef np.complex64_t * _tmp2

    cpdef draw_disturbance_variates(self)
    cpdef draw_initial_state_variates(self)
    cpdef set_disturbance_variates(self, np.complex64_t [:] variates)
    cpdef set_initial_state_variates(self, np.complex64_t [:] variates)
    cpdef simulate(self, int simulation_output=*)

    cdef np.complex64_t generate_obs(self, int t, np.complex64_t * obs, np.complex64_t * state, np.complex64_t * variates)
    cdef np.complex64_t generate_state(self, int t, np.complex64_t * state, np.complex64_t * input_state, np.complex64_t * variates)
    cdef void cholesky(self, np.complex64_t * source, np.complex64_t * destination, int n)
    cdef void transform_variates(self, np.complex64_t * variates, np.complex64_t * cholesky_factor, int n)

# Double precision complex
cdef class zSimulationSmoother(object):
    # ### Statespace model
    cdef readonly zStatespace model
    # ### Kalman filter
    cdef readonly zKalmanFilter kfilter
    # ### Kalman smoother
    cdef readonly zKalmanSmoother smoother

    # ### Simulated Statespace model
    cdef readonly zStatespace simulated_model
    # ### Simulated Kalman filter
    cdef readonly zKalmanFilter simulated_kfilter
    # ### Simulated Kalman smoother
    cdef readonly zKalmanSmoother simulated_smoother

    # ### Simulation parameters
    cdef public int simulation_output

    # ### Random variates
    cdef int n_disturbance_variates
    cdef readonly np.complex128_t [:] disturbance_variates
    cdef int n_initial_state_variates
    cdef readonly np.complex128_t [:] initial_state_variates

    # ### Simulated Data
    cdef readonly np.complex128_t [::1,:] simulated_measurement_disturbance
    cdef readonly np.complex128_t [::1,:] simulated_state_disturbance
    cdef readonly np.complex128_t [::1,:] simulated_state

    # ### Generated Data
    cdef readonly np.complex128_t [::1,:] generated_obs
    cdef readonly np.complex128_t [::1,:] generated_state

    # ### Temporary arrays
    cdef readonly np.complex128_t [::1,:] tmp0, tmp1, tmp2

    # ### Pointers
    cdef np.complex128_t * _tmp0
    cdef np.complex128_t * _tmp1
    cdef np.complex128_t * _tmp2

    cpdef draw_disturbance_variates(self)
    cpdef draw_initial_state_variates(self)
    cpdef set_disturbance_variates(self, np.complex128_t [:] variates)
    cpdef set_initial_state_variates(self, np.complex128_t [:] variates)
    cpdef simulate(self, int simulation_output=*)

    cdef np.complex128_t generate_obs(self, int t, np.complex128_t * obs, np.complex128_t * state, np.complex128_t * variates)
    cdef np.complex128_t generate_state(self, int t, np.complex128_t * state, np.complex128_t * input_state, np.complex128_t * variates)
    cdef void cholesky(self, np.complex128_t * source, np.complex128_t * destination, int n)
    cdef void transform_variates(self, np.complex128_t * variates, np.complex128_t * cholesky_factor, int n)