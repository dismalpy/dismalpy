#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=False
"""
State Space Models - Conventional Kalman Filter declarations

Author: Chad Fulton  
License: Simplified-BSD
"""

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
cdef int ssmoothed_estimators_missing_conventional(sKalmanSmoother smoother, sKalmanFilter kfilter, sStatespace model)
cdef int ssmoothed_disturbances_missing_conventional(sKalmanSmoother smoother, sKalmanFilter kfilter, sStatespace model)

cdef int ssmoothed_estimators_conventional(sKalmanSmoother smoother, sKalmanFilter kfilter, sStatespace model)
cdef int ssmoothed_state_conventional(sKalmanSmoother smoother, sKalmanFilter kfilter, sStatespace model)
cdef int ssmoothed_disturbances_conventional(sKalmanSmoother smoother, sKalmanFilter kfilter, sStatespace model)

# Double precision
cdef int dsmoothed_estimators_missing_conventional(dKalmanSmoother smoother, dKalmanFilter kfilter, dStatespace model)
cdef int dsmoothed_disturbances_missing_conventional(dKalmanSmoother smoother, dKalmanFilter kfilter, dStatespace model)

cdef int dsmoothed_estimators_conventional(dKalmanSmoother smoother, dKalmanFilter kfilter, dStatespace model)
cdef int dsmoothed_state_conventional(dKalmanSmoother smoother, dKalmanFilter kfilter, dStatespace model)
cdef int dsmoothed_disturbances_conventional(dKalmanSmoother smoother, dKalmanFilter kfilter, dStatespace model)

# Single precision complex
cdef int csmoothed_estimators_missing_conventional(cKalmanSmoother smoother, cKalmanFilter kfilter, cStatespace model)
cdef int csmoothed_disturbances_missing_conventional(cKalmanSmoother smoother, cKalmanFilter kfilter, cStatespace model)

cdef int csmoothed_estimators_conventional(cKalmanSmoother smoother, cKalmanFilter kfilter, cStatespace model)
cdef int csmoothed_state_conventional(cKalmanSmoother smoother, cKalmanFilter kfilter, cStatespace model)
cdef int csmoothed_disturbances_conventional(cKalmanSmoother smoother, cKalmanFilter kfilter, cStatespace model)

# Double precision complex
cdef int zsmoothed_estimators_missing_conventional(zKalmanSmoother smoother, zKalmanFilter kfilter, zStatespace model)
cdef int zsmoothed_disturbances_missing_conventional(zKalmanSmoother smoother, zKalmanFilter kfilter, zStatespace model)

cdef int zsmoothed_estimators_conventional(zKalmanSmoother smoother, zKalmanFilter kfilter, zStatespace model)
cdef int zsmoothed_state_conventional(zKalmanSmoother smoother, zKalmanFilter kfilter, zStatespace model)
cdef int zsmoothed_disturbances_conventional(zKalmanSmoother smoother, zKalmanFilter kfilter, zStatespace model)