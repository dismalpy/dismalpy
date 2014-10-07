#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=False
"""
State Space Models - Cython Tools declarations

Author: Chad Fulton  
License: Simplified-BSD
"""

cdef validate_matrix_shape(str name, Py_ssize_t *shape, int nrows, int ncols, object nobs=*)

cdef validate_vector_shape(str name, Py_ssize_t *shape, int nrows, object nobs=*)

