#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=False

# Typical imports
import numpy as np
import warnings
cimport numpy as np
cimport cython

cimport dismalpy.src.blas as blas
cimport dismalpy.src.lapack as lapack

cpdef _rvs_quantities(np.float64_t [::1, :] xTH,
                      np.float64_t [::1, :] xTHx, np.float64_t [:] xTHy,
                      np.float64_t [::1, :] endog, np.float64_t [::1, :, :] exog,
                      np.float64_t [::1, :] precision, int T, int M, int k):
    """
    Calculate sum_t Z_t' H Z_t and sum_t Z_t' H y_t, t = 0, ... T-1

    In place
    """
    cdef:
        int t
        int inc = 1
        int lda
    cdef:
        np.float64_t alpha = 1.0
        np.float64_t beta = 0.0
        np.float64_t gamma = 0.0

    for t in range(T):
        # exog is M x k x T
        # x should be M x k and x'H should be k x M
        blas.dgemm("T", "N", &k, &M, &M,
            &alpha, &exog[0,0,t], &M,
                    &precision[0,0], &M,
            &beta, &xTH[0,0], &k
        )
        # x'H is k x M, x'Hx is k x k
        blas.dgemm("N", "N", &k, &k, &M,
            &alpha, &xTH[0,0], &k,
                    &exog[0,0,t], &M,
            &gamma, &xTHx[0,0], &k
        )
        # endog is T x M
        # x'H is k x M, x'Hy is k x 0
        blas.dgemv("N", &k, &M,
            &alpha, &xTH[0,0], &k,
                    &endog[0,t], &inc,
            &gamma, &xTHy[0], &inc
        )

        if t == 0:
            gamma = 1.0
