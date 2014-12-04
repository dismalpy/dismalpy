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

cpdef _var_posteriors(np.float64_t [:] prior_loc,
                      np.float64_t [::1, :] prior_precision,
                      np.float64_t [::1, :] ZH,
                      np.float64_t [::1, :] ZHZ, np.float64_t [:] ZHy,
                      np.float64_t [::1, :] endog, np.float64_t [::1, :] lagged,
                      np.float64_t [::1, :] precision,
                      int k_endog, int k_ar, int k_var, int recalculate):

    # Storage
    cdef int info, i, j, k_var2=k_var**2, inc=1
    cdef:
        np.float64_t alpha = 1.0
        np.float64_t beta = 0.0

    raise NotImplementedError
    
    # Calculate \sum_t Z_t' H Z_t and \sum_t Z_t' H y_t
    if recalculate:
        _var_quantities(ZH, ZHZ, ZHy,
                        endog, lagged, precision,
                        k_endog, k_ar, k_var)

    blas.daxpy(&k_var, &alpha, &prior_loc[0], &inc, &ZHy[0], &inc)
    blas.daxpy(&k_var2, &alpha, &prior_precision[0,0], &inc, &ZHZ[0,0], &inc)

    # Calculate posterior_scale, posterior_loc, posterior_cholesky

    # Posterior scale = 
    lapack.dpotrf("L", &k_var, &ZHZ[0,0], &k_var, &info)
    lapack.dpotri("L", &k_var, &ZHZ[0,0], &k_var, &info)
    for i in range(k_var): # rows
        for j in range(i+1, k_var): # columns
            ZHZ[i,j] = ZHZ[j,i]

    # np.dot(posterior_scale, self._prior_inv_scale_loc + self._xTHy[:, None])
    # TODO this is not right.
    blas.dcopy(&k_var, &ZHy[0], &inc, &ZHy[0], &inc)
    blas.dgemv("N", &k_var, &k_var,
        &alpha, &ZHZ[0,0], &k_var,
                &ZHy[0], &inc,
        &beta, &ZHy[0], &inc
    )

    # self._posterior_cholesky = np.linalg.cholesky(self.posterior_scale)
    lapack.dpotrf("L", &k_var, &ZHZ[0,0], &k_var, &info)
    for i in range(k_var): # rows
        for j in range(i+1, k_var): # columns
            ZHZ[i,j] = 0

    return np.array(ZHy, copy=True), np.array(ZHZ, copy=True)


cpdef _var_quantities(np.float64_t [::1, :] ZH,
                      np.float64_t [::1, :] ZHZ, np.float64_t [:] ZHy,
                      np.float64_t [::1, :] endog, np.float64_t [::1, :] lagged,
                      np.float64_t [::1, :] precision,
                      int k_endog, int k_ar, int k_var):
    cdef:
        int t, i, nobs = endog.shape[1]
        int inc = 1
        int lda
    cdef:
        np.float64_t alpha = 1.0
        np.float64_t beta = 0.0
        np.float64_t gamma = 0.0

    # Clear Z'HZ, and Z'Hy (because below won't overwrite with zeros)
    # Z'HZ is (k x k) = (k_var x k_var)
    # Z'Hy is (k x 0) = (k_var x 0)
    # for i in range(k_var):  # columns
    #         ZHy[i] = 0
    #         for j in range(k_var):  #rows
    #             ZHZ[j,i] = 0
    ZHy[:] = 0
    ZHZ[:,:] = 0

    for t in range(nobs):
        # Clear (Z' H)' (because below won't overwrite with zeros)
        # (Z'H)' is (M x k) = (k_endog x k_var)
        # for i in range(k_var):  # columns
        #     for j in range(k_endog):  #rows
        #         ZH[j,i] = 0
        ZH[:,:] = 0


        # Create (Z' H)'
        # = [z_{1t}' H_1  z_{2t}' H_2  ...  z_{Mt}' H_M]'
        # where Z is (M x k), H is (M x M), Z'H is (k x M),
        # and (Z'H)' is (M x k)
        # (create the transpose so that we'll be creating blocks across the
        # entire set of rows, which is convenient for arrays that are Fortran
        # ordered in memory)
        for i in range(k_endog):
            # lagged is (d*M x T) = (k_1 x T) = (k_ar x nobs)
            # Taking the outer product of a k_ar x 1 and a 1 x k_endog
            # (i.e. k_i x 1 and 1 x M to get k_i x M), and Z'H is (k x M) so
            # that (Z'H)' is (M x k); thus each outer product fills in the
            # i-th (M x k_i) block (proceeding horizontally across the array)
            blas.dger(&k_endog, &k_ar, &alpha,
                &precision[i,0], &k_endog,
                &lagged[0,t], &inc,
                &ZH[0, i*k_ar], &k_endog
            )

        # Create \sum_t Z' H Z
        # = [ZH_1 z_{1t}  ZH_2 z_{2t}   ...   ZH_M z_{Mt}]
        # ZH_i is the i-th column of Z'H, or the i-th row of (Z'H)' and is
        # (k x 1) or (1 x k)
        # = [(k x 1) (1 x k_i) ... ] = (k x M * k_i) = (k x k)
        # (we previously created (Z'H)' which means that now to access ZH_1,
        # which is a column of the Z'H matrix, we need to access a row of
        # (Z'H)'. Since it is row-ordered, that means we need to have an `inc`
        # argument equal to the number of rows in the matrix, k_endog = M)
        # 
        # The output is stored in the i-th (k x k_i) block of the ZHZ matrix
        # (proceeding horizontally across the array)
        
        # Create \sum_t Z'Hy
        # = ZH_1 y_{1t} + ZH_2 y_{2t} + ... + ZH_M y_{Mt}
        # = (k x 1) (1 x 1) + ... = k x 1
        for i in range(k_endog):
            # Z'HZ
            blas.dger(&k_var, &k_ar, &alpha,
                &ZH[i,0], &k_endog,
                &lagged[0,t], &inc,
                &ZHZ[0, i*k_ar], &k_var
            )

            # Z'Hy
            blas.daxpy(&k_var,
                &endog[i,t], &ZH[i,0], &k_endog,
                             &ZHy[0], &inc
            )
