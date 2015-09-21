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

cpdef _var_quantities_kron(np.float64_t [::1, :] ZH,
                      np.float64_t [::1, :] ZHZ, np.float64_t [:] ZHy,
                      np.float64_t [::1, :] endog, np.float64_t [::1, :] lagged,
                      np.float64_t [::1, :] precision,
                      int k_endog, int k_ar, int k_var):
    cdef:
        int t, i, j, k, l, m, nobs = endog.shape[1]
        int row, col
        int inc = 1
        int lda
    cdef:
        np.float64_t alpha = 1.0
        np.float64_t beta = 0.0
        np.float64_t gamma = 0.0
        np.float64_t scalar

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

        # Add the lower triangle of z'z into the first k_i x k_i block of ZHZ
        blas.dsyr("L", &k_ar,
            &alpha, &lagged[0,t], &inc,
                    &ZHZ[0, 0], &k_var)

        # Add y_t H_i z_t (where H_i is the i-th column of H)
        for i in range(k_endog):
            scalar = blas.ddot(&k_endog, &endog[0,t], &inc, &precision[0,i], &inc)
            blas.daxpy(&k_ar, &scalar, &lagged[0,t], &inc, &ZHy[i*k_ar], &inc)

    # 1. Now the first k_i x k_i block of ZHZ has the lower triangle of
    # z'z = (z_1'z_1 + ... + z_T'z_T)

    if k_endog > 1:

        # 2. Create the full first column of blocks

        # a. Copy the full z'z (lower and upper triangles) into the next block
        #    (i.e. the block directly below the first block)
        for i in range(k_ar):  # i is column
            j = k_ar + i       # row to start copying to

            k = k_ar - i  # number of elements to copy
            # Copy the lower triangular portion + the diagonal (i.e. a column)
            blas.dcopy(&k, &ZHZ[i,i], &inc, &ZHZ[j,i], &inc)
            # Copy the upper triangular portion (i.e. a row)
            k = k - 1
            if k > 0:  # k = 0 last column
                blas.dcopy(&k, &ZHZ[i+1,i], &inc, &ZHZ[j,i+1], &k_var)

        # b. Copy the full z'z into the remaining blocks in the first column
        #    block
        for row in range(2, k_endog):  # row block of the large matrix
            scalar = precision[row,0]
            if scalar == 0:
                continue

            j = row*k_ar  # row to start copying to
            for i in range(k_ar):  # i is column
                if scalar == 1:
                    blas.dcopy(&k_ar, &ZHZ[k_ar,i], &inc, &ZHZ[j,i], &inc)
                else:
                    blas.daxpy(&k_ar, &scalar, &ZHZ[k_ar,i], &inc, &ZHZ[j,i], &inc)

        # 3. Create the remaining column blocks
        # TODO this won't work because we also need to scale all of these
        # elements. so we'll need yet another index so that we can
        # interate over both row and column blocks, and then replace the dcopy
        # with a daxpy
        for col in range(1, k_endog):        # column block of the large matrix
            for row in range(col, k_endog):  # row block of the large matrix
                scalar = precision[row,col]
                if scalar == 0:
                    continue

                j = row * k_ar  # row to start copying to

                # Row to start copying from
                if row == col:
                    k = 0
                else:
                    k = k_ar

                for i in range(k_ar):  # i is column to copy from
                    l = i + col*k_ar   # l is column to copy to
                    if scalar == 1:
                        blas.dcopy(&k_ar, &ZHZ[k,i], &inc, &ZHZ[j,l], &inc)
                    else:
                        blas.daxpy(&k_ar, &scalar, &ZHZ[k,i], &inc, &ZHZ[j,l], &inc)

        # 4. Scale the first two blocks (which we copied from, so they weren't
        #    initially scaled)
        for row in range(2):
            scalar = precision[row,0]
            if scalar == 1:
                continue

            j = row * k_ar  # row to start copying from and to

            for i in range(k_ar):  # i is column to copy from
                blas.dscal(&k_ar, &scalar, &ZHZ[j,i], &inc)

    # 5. Create the upper triangular portion
    # TODO don't actually need to do this, since we could calculate the
    # posterior scale within Cython, and then would only ever need the lower
    # triangular portion
    for row in range(k_var):
        for col in range(row+1,k_var):
            ZHZ[row, col] = ZHZ[col, row]
