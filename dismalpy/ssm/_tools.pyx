#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=False
"""
State Space Model - Cython tools

Author: Chad Fulton  
License: Simplified-BSD
"""

# Array shape validation
cdef validate_matrix_shape(str name, Py_ssize_t *shape, int nrows, int ncols, nobs=None):
    if not shape[0] == nrows:
        raise ValueError('Invalid shape for %s matrix: requires %d rows,'
                         ' got %d' % (name, nrows, shape[0]))
    if not shape[1] == ncols:
        raise ValueError('Invalid shape for %s matrix: requires %d columns,'
                         'got %d' % (name, shape[1], shape[1]))
    if nobs is not None and shape[2] not in [1, nobs]:
        raise ValueError('Invalid time-varying dimension for %s matrix:'
                         ' requires 1 or %d, got %d' % (name, nobs, shape[2]))

cdef validate_vector_shape(str name, Py_ssize_t *shape, int nrows, nobs = None):
    if not shape[0] == nrows:
        raise ValueError('Invalid shape for %s vector: requires %d rows,'
                         ' got %d' % (name, nrows, shape[0]))
    if nobs is not None and not shape[1] in [1, nobs]:
        raise ValueError('Invalid time-varying dimension for %s vector:'
                         ' requires 1 or %d got %d' % (name, nobs, shape[1]))
