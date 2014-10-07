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
cdef int sforecast_missing_conventional(sKalmanFilter kfilter)
cdef int supdating_missing_conventional(sKalmanFilter kfilter)
cdef np.float32_t sinverse_missing_conventional(sKalmanFilter kfilter, np.float32_t determinant)  except *
cdef np.float32_t sloglikelihood_missing_conventional(sKalmanFilter kfilter, np.float32_t determinant)

cdef int sforecast_conventional(sKalmanFilter kfilter)
cdef int supdating_conventional(sKalmanFilter kfilter)
cdef int sprediction_conventional(sKalmanFilter kfilter)
cdef np.float32_t sloglikelihood_conventional(sKalmanFilter kfilter, np.float32_t determinant)

# Double precision
cdef int dforecast_missing_conventional(dKalmanFilter kfilter)
cdef int dupdating_missing_conventional(dKalmanFilter kfilter)
cdef np.float64_t dinverse_missing_conventional(dKalmanFilter kfilter, np.float64_t determinant)  except *
cdef np.float64_t dloglikelihood_missing_conventional(dKalmanFilter kfilter, np.float64_t determinant)

cdef int dforecast_conventional(dKalmanFilter kfilter)
cdef int dupdating_conventional(dKalmanFilter kfilter)
cdef int dprediction_conventional(dKalmanFilter kfilter)
cdef np.float64_t dloglikelihood_conventional(dKalmanFilter kfilter, np.float64_t determinant)

# Single precision complex
cdef int cforecast_missing_conventional(cKalmanFilter kfilter)
cdef int cupdating_missing_conventional(cKalmanFilter kfilter)
cdef np.complex64_t cinverse_missing_conventional(cKalmanFilter kfilter, np.complex64_t determinant)  except *
cdef np.complex64_t cloglikelihood_missing_conventional(cKalmanFilter kfilter, np.complex64_t determinant)

cdef int cforecast_conventional(cKalmanFilter kfilter)
cdef int cupdating_conventional(cKalmanFilter kfilter)
cdef int cprediction_conventional(cKalmanFilter kfilter)
cdef np.complex64_t cloglikelihood_conventional(cKalmanFilter kfilter, np.complex64_t determinant)

# Double precision complex
cdef int zforecast_missing_conventional(zKalmanFilter kfilter)
cdef int zupdating_missing_conventional(zKalmanFilter kfilter)
cdef np.complex128_t zinverse_missing_conventional(zKalmanFilter kfilter, np.complex128_t determinant)  except *
cdef np.complex128_t zloglikelihood_missing_conventional(zKalmanFilter kfilter, np.complex128_t determinant)

cdef int zforecast_conventional(zKalmanFilter kfilter)
cdef int zupdating_conventional(zKalmanFilter kfilter)
cdef int zprediction_conventional(zKalmanFilter kfilter)
cdef np.complex128_t zloglikelihood_conventional(zKalmanFilter kfilter, np.complex128_t determinant)
