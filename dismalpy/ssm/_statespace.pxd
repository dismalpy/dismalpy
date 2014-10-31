#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=False
"""
State Space Models declarations

Author: Chad Fulton  
License: Simplified-BSD
"""

cimport numpy as np

cdef class sStatespace(object):
    # Statespace dimensions
    cdef readonly int nobs, k_endog, k_states, k_posdef
    
    # Statespace representation matrices
    cdef readonly np.float32_t [::1,:] obs, obs_intercept, state_intercept
    cdef readonly np.float32_t [:] initial_state
    cdef readonly np.float32_t [::1,:] initial_state_cov
    cdef readonly np.float32_t [::1,:,:] design, obs_cov, transition, selection, state_cov, selected_state_cov

    cdef readonly int [::1,:] missing
    cdef readonly int [:] nmissing
    cdef readonly int has_missing

    # Flags
    cdef readonly int time_invariant
    cdef readonly int initialized
    cdef public int diagonal_obs_cov

    # Temporary arrays
    cdef np.float32_t [::1,:] tmp

    # Temporary selection arrays
    cdef readonly np.float32_t [:] selected_obs
    cdef readonly np.float32_t [:] selected_design
    cdef readonly np.float32_t [:] selected_obs_cov

    # Temporary transformation arrays
    cdef readonly np.float32_t [::1,:] transform_ldl_l
    cdef readonly np.float32_t [::1,:] transform_ldl_d
    cdef readonly np.float32_t [::1,:] transform_design

    # Pointers
    cdef np.float32_t * _obs
    cdef np.float32_t * _design
    cdef np.float32_t * _obs_intercept
    cdef np.float32_t * _obs_cov
    cdef np.float32_t * _transition
    cdef np.float32_t * _state_intercept
    cdef np.float32_t * _selection
    cdef np.float32_t * _state_cov
    cdef np.float32_t * _selected_state_cov
    cdef np.float32_t * _initial_state
    cdef np.float32_t * _initial_state_cov

    # Functions
    cdef void initialize_object_pointers(self, unsigned int t) except *
    cdef void select_state_cov(self, unsigned int t)
    cdef int select_missing(self, unsigned int t)
    cdef void _select_missing_entire_obs(self, unsigned int t)
    cdef void _select_missing_partial_obs(self, unsigned int t)
    cdef void transform_cholesky(self, unsigned int t) except *
    cdef void transform_collapse(self, unsigned int t) except *
    cdef void transform_generalized_collapse(self, unsigned int t) except *

cdef class dStatespace(object):
    # Statespace dimensions
    cdef readonly int nobs, k_endog, k_states, k_posdef
    
    # Statespace representation matrices
    cdef readonly np.float64_t [::1,:] obs, obs_intercept, state_intercept
    cdef readonly np.float64_t [:] initial_state
    cdef readonly np.float64_t [::1,:] initial_state_cov
    cdef readonly np.float64_t [::1,:,:] design, obs_cov, transition, selection, state_cov, selected_state_cov

    cdef readonly int [::1,:] missing
    cdef readonly int [:] nmissing
    cdef readonly int has_missing

    # Flags
    cdef readonly int time_invariant
    cdef readonly int initialized
    cdef public int diagonal_obs_cov

    # Temporary arrays
    cdef np.float64_t [::1,:] tmp

    # Temporary selection arrays
    cdef readonly np.float64_t [:] selected_obs
    cdef readonly np.float64_t [:] selected_design
    cdef readonly np.float64_t [:] selected_obs_cov

    # Temporary transformation arrays
    cdef readonly np.float64_t [::1,:] transform_ldl_l
    cdef readonly np.float64_t [::1,:] transform_ldl_d
    cdef readonly np.float64_t [::1,:] transform_design

    # Pointers
    cdef np.float64_t * _obs
    cdef np.float64_t * _design
    cdef np.float64_t * _obs_intercept
    cdef np.float64_t * _obs_cov
    cdef np.float64_t * _transition
    cdef np.float64_t * _state_intercept
    cdef np.float64_t * _selection
    cdef np.float64_t * _state_cov
    cdef np.float64_t * _selected_state_cov
    cdef np.float64_t * _initial_state
    cdef np.float64_t * _initial_state_cov

    # Functions
    cdef void initialize_object_pointers(self, unsigned int t) except *
    cdef void select_state_cov(self, unsigned int t)
    cdef int select_missing(self, unsigned int t)
    cdef void _select_missing_entire_obs(self, unsigned int t)
    cdef void _select_missing_partial_obs(self, unsigned int t)
    cdef void transform_cholesky(self, unsigned int t) except *
    cdef void transform_collapse(self, unsigned int t) except *
    cdef void transform_generalized_collapse(self, unsigned int t) except *

cdef class cStatespace(object):
    # Statespace dimensions
    cdef readonly int nobs, k_endog, k_states, k_posdef
    
    # Statespace representation matrices
    cdef readonly np.complex64_t [::1,:] obs, obs_intercept, state_intercept
    cdef readonly np.complex64_t [:] initial_state
    cdef readonly np.complex64_t [::1,:] initial_state_cov
    cdef readonly np.complex64_t [::1,:,:] design, obs_cov, transition, selection, state_cov, selected_state_cov

    cdef readonly int [::1,:] missing
    cdef readonly int [:] nmissing
    cdef readonly int has_missing

    # Flags
    cdef readonly int time_invariant
    cdef readonly int initialized
    cdef public int diagonal_obs_cov

    # Temporary arrays
    cdef np.complex64_t [::1,:] tmp

    # Temporary selection arrays
    cdef readonly np.complex64_t [:] selected_obs
    cdef readonly np.complex64_t [:] selected_design
    cdef readonly np.complex64_t [:] selected_obs_cov

    # Temporary transformation arrays
    cdef readonly np.complex64_t [::1,:] transform_ldl_l
    cdef readonly np.complex64_t [::1,:] transform_ldl_d
    cdef readonly np.complex64_t [::1,:] transform_design

    # Pointers
    cdef np.complex64_t * _obs
    cdef np.complex64_t * _design
    cdef np.complex64_t * _obs_intercept
    cdef np.complex64_t * _obs_cov
    cdef np.complex64_t * _transition
    cdef np.complex64_t * _state_intercept
    cdef np.complex64_t * _selection
    cdef np.complex64_t * _state_cov
    cdef np.complex64_t * _selected_state_cov
    cdef np.complex64_t * _initial_state
    cdef np.complex64_t * _initial_state_cov

    # Functions
    cdef void initialize_object_pointers(self, unsigned int t) except *
    cdef void select_state_cov(self, unsigned int t)
    cdef int select_missing(self, unsigned int t)
    cdef void _select_missing_entire_obs(self, unsigned int t)
    cdef void _select_missing_partial_obs(self, unsigned int t)
    cdef void transform_cholesky(self, unsigned int t) except *
    cdef void transform_collapse(self, unsigned int t) except *
    cdef void transform_generalized_collapse(self, unsigned int t) except *

cdef class zStatespace(object):
    # Statespace dimensions
    cdef readonly int nobs, k_endog, k_states, k_posdef
    
    # Statespace representation matrices
    cdef readonly np.complex128_t [::1,:] obs, obs_intercept, state_intercept
    cdef readonly np.complex128_t [:] initial_state
    cdef readonly np.complex128_t [::1,:] initial_state_cov
    cdef readonly np.complex128_t [::1,:,:] design, obs_cov, transition, selection, state_cov, selected_state_cov

    cdef readonly int [::1,:] missing
    cdef readonly int [:] nmissing
    cdef readonly int has_missing

    # Flags
    cdef readonly int time_invariant
    cdef readonly int initialized
    cdef public int diagonal_obs_cov

    # Temporary arrays
    cdef np.complex128_t [::1,:] tmp

    # Temporary selection arrays
    cdef readonly np.complex128_t [:] selected_obs
    cdef readonly np.complex128_t [:] selected_design
    cdef readonly np.complex128_t [:] selected_obs_cov

    # Temporary transformation arrays
    cdef readonly np.complex128_t [::1,:] transform_ldl_l
    cdef readonly np.complex128_t [::1,:] transform_ldl_d
    cdef readonly np.complex128_t [::1,:] transform_design

    # Pointers
    cdef np.complex128_t * _obs
    cdef np.complex128_t * _design
    cdef np.complex128_t * _obs_intercept
    cdef np.complex128_t * _obs_cov
    cdef np.complex128_t * _transition
    cdef np.complex128_t * _state_intercept
    cdef np.complex128_t * _selection
    cdef np.complex128_t * _state_cov
    cdef np.complex128_t * _selected_state_cov
    cdef np.complex128_t * _initial_state
    cdef np.complex128_t * _initial_state_cov

    # Functions
    cdef void initialize_object_pointers(self, unsigned int t) except *
    cdef void select_state_cov(self, unsigned int t)
    cdef int select_missing(self, unsigned int t)
    cdef void _select_missing_entire_obs(self, unsigned int t)
    cdef void _select_missing_partial_obs(self, unsigned int t)
    cdef void transform_cholesky(self, unsigned int t) except *
    cdef void transform_collapse(self, unsigned int t) except *
    cdef void transform_generalized_collapse(self, unsigned int t) except *

cdef int sselect_state_cov(int k_states, int k_posdef,
                           np.float32_t * tmp,
                           np.float32_t * selection,
                           np.float32_t * state_cov,
                           np.float32_t * selected_state_cov)

cdef int dselect_state_cov(int k_states, int k_posdef,
                           np.float64_t * tmp,
                           np.float64_t * selection,
                           np.float64_t * state_cov,
                           np.float64_t * selected_state_cov)

cdef int cselect_state_cov(int k_states, int k_posdef,
                           np.complex64_t * tmp,
                           np.complex64_t * selection,
                           np.complex64_t * state_cov,
                           np.complex64_t * selected_state_cov)

cdef int zselect_state_cov(int k_states, int k_posdef,
                           np.complex128_t * tmp,
                           np.complex128_t * selection,
                           np.complex128_t * state_cov,
                           np.complex128_t * selected_state_cov)
