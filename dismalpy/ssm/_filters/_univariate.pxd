#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=False
"""
State Space Models - Conventional Kalman Filter declarations

Author: Chad Fulton  
License: Simplified-BSD
"""

cimport numpy as np
from dismalpy.ssm._kalman_filter cimport (
    sKalmanFilter, dKalmanFilter, cKalmanFilter, zKalmanFilter
)

# Single precision
cdef int sforecast_univariate(sKalmanFilter kfilter)
cdef int supdating_univariate(sKalmanFilter kfilter)
cdef int sprediction_univariate(sKalmanFilter kfilter)
cdef np.float32_t sinverse_noop_univariate(sKalmanFilter kfilter, np.float32_t determinant) except *
cdef np.float32_t sloglikelihood_univariate(sKalmanFilter kfilter, np.float32_t determinant)

# Double precision
cdef int dforecast_univariate(dKalmanFilter kfilter)
cdef int dupdating_univariate(dKalmanFilter kfilter)
cdef int dprediction_univariate(dKalmanFilter kfilter)
cdef np.float64_t dinverse_noop_univariate(dKalmanFilter kfilter, np.float64_t determinant) except *
cdef np.float64_t dloglikelihood_univariate(dKalmanFilter kfilter, np.float64_t determinant)

# Single precision complex
cdef int cforecast_univariate(cKalmanFilter kfilter)
cdef int cupdating_univariate(cKalmanFilter kfilter)
cdef int cprediction_univariate(cKalmanFilter kfilter)
cdef np.complex64_t cinverse_noop_univariate(cKalmanFilter kfilter, np.complex64_t determinant) except *
cdef np.complex64_t cloglikelihood_univariate(cKalmanFilter kfilter, np.complex64_t determinant)

# Double precision complex
cdef int zforecast_univariate(zKalmanFilter kfilter)
cdef int zupdating_univariate(zKalmanFilter kfilter)
cdef int zprediction_univariate(zKalmanFilter kfilter)
cdef np.complex128_t zinverse_noop_univariate(zKalmanFilter kfilter, np.complex128_t determinant) except *
cdef np.complex128_t zloglikelihood_univariate(zKalmanFilter kfilter, np.complex128_t determinant)
