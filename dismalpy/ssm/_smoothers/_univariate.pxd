#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=False
"""
State Space Models - Conventional Kalman Filter declarations

Author: Chad Fulton  
License: Simplified-BSD
"""

cimport numpy as np
from dismalpy.ssm._kalman_smoother cimport (
    sKalmanSmoother, dKalmanSmoother, cKalmanSmoother, zKalmanSmoother
)

# Single precision
cdef int ssmoothed_estimators_univariate(sKalmanSmoother smoother)
cdef int ssmoothed_disturbances_univariate(sKalmanSmoother smoother)

# Double precision
cdef int dsmoothed_estimators_univariate(dKalmanSmoother smoother)
cdef int dsmoothed_disturbances_univariate(dKalmanSmoother smoother)

# Single precision complex
cdef int csmoothed_estimators_univariate(cKalmanSmoother smoother)
cdef int csmoothed_disturbances_univariate(cKalmanSmoother smoother)

# Double precision complex
cdef int zsmoothed_estimators_univariate(zKalmanSmoother smoother)
cdef int zsmoothed_disturbances_univariate(zKalmanSmoother smoother)
