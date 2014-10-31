cimport numpy as np

cdef extern from "capsule.h":
    void *Capsule_AsVoidPtr(object ptr)

ctypedef int (*_sselect)(np.float32_t, np.float32_t)
ctypedef int (*_dselect)(np.float64_t, np.float64_t)
ctypedef int (*_cselect)(np.complex64_t, np.complex64_t)
ctypedef int (*_zselect)(np.complex128_t, np.complex128_t)

ctypedef int sgees_t(
    char *jobvs,
    char *sort,
    _sselect select,
    int *n,
    np.float32_t*a,
    int *lda,
    int *sdim,
    np.float32_t *wr,
    np.float32_t *wi,
    np.float32_t *vs,
    int *ldvs,
    np.float32_t *work,
    int *lwork,
    int *bwork,
    int *info
) nogil

ctypedef int strsyl_t(
    char *transa,
    char *transb,
    int *isgn,
    int *m,
    int *n,
    np.float32_t *a,
    int *lda,
    np.float32_t *b,
    int *ldb,
    np.float32_t *c,
    int *ldc,
    np.float32_t *scale,
    int *info
) nogil

ctypedef int sgetrf_t(
    # SGETRF - compute an LU factorization of a general M-by-N
    # matrix A using partial pivoting with row interchanges
    int *m,          # Rows of A
    int *n,          # Columns of A
    np.float32_t *a, # Matrix A: mxn
    int *lda,        # The size of the first dimension of A (in memory)
    int *ipiv,       # Matrix P: mxn (the pivot indices)
    int *info        # 0 if success, otherwise an error code (integer)
) nogil

ctypedef int sgetri_t(
    # SGETRI - compute the inverse of a matrix using the LU fac-
    # torization computed by SGETRF
    int *n,             # Order of A
    np.float32_t *a,    # Matrix A: nxn (the LUP decomposed matrix from dgetrf)
    int *lda,           # The size of the first dimension of A (in memory)
    int *ipiv,          # Matrix P: nxn (the pivot indices from the LUP decomposition)
    np.float32_t *work, # Matrix: nxn (a workspace for the inversion, optimal size=nxn)
    int *lwork,         # Number of elements in the workspace: optimal is n**2
    int *info           # 0 if success, otherwise an error code (integer)
) nogil

ctypedef int sgetrs_t(
    # SGETRS - solve a system of linear equations  A * X = B or A'
    # * X = B with a general N-by-N matrix A using the LU factori-
    # zation computed by SGETRF
    char *trans,        # Specifies the form of the system of equations
    int *n,             # Order of A
    int *nrhs,          # The number of right hand sides
    np.float32_t *a,    # Matrix A: nxn (the LUP decomposed matrix from dgetrf)
    int *lda,           # The size of the first dimension of A (in memory)
    int *ipiv,          # Matrix P: nxn (the pivot indices from the LUP decomposition)
    np.float32_t *b,    # Matrix B: nxnrhs
    int *ldb,           # The size of the first dimension of B (in memory)
    int *info           # 0 if success, otherwise an error code (integer)
) nogil

ctypedef int spotrf_t(
    # Compute the Cholesky factorization of a
    # real  symmetric positive definite matrix A
    char *uplo,       # 'U':  A = U'U and U is stored, 'L': A = LL' and L is stored
    int *n,           # The order of the matrix A.  n >= 0.
    np.float32_t *a,  # Matrix A: nxn
    int *lda,         # The size of the first dimension of A (in memory)
    int *info         # 0 if success, otherwise an error code (integer)
) nogil

ctypedef int spotri_t(
    # SPOTRI - compute the inverse of a real symmetric positive
    # definite matrix A using the Cholesky factorization A =
    # U**T*U or A = L*L**T computed by SPOTRF
    char *uplo,       # 'U':  A = U'U and U is stored, 'L': A = LL' and L is stored
    int *n,           # The order of the matrix A.  n >= 0.
    np.float32_t *a,  # Matrix A: nxn
    int *lda,         # The size of the first dimension of A (in memory)
    int *info         # 0 if success, otherwise an error code (integer)
) nogil

ctypedef int spotrs_t(
    # SPOTRS - solve a system of linear equations A*X = B with a
    # symmetric positive definite matrix A using the Cholesky fac-
    # torization A = U**T*U or A = L*L**T computed by SPOTRF
    char *uplo,       # 'U':  A = U'U and U is stored, 'L': A = LL' and L is stored
    int *n,           # The order of the matrix A.  n >= 0.
    int *nrhs,        # The number of right hand sides
    np.float32_t *a,  # Matrix A: nxn
    int *lda,         # The size of the first dimension of A (in memory)
    np.float32_t *b,  # Matrix B: nxnrhs
    int *ldb,         # The size of the first dimension of B (in memory)
    int *info         # 0 if success, otherwise an error code (integer)
) nogil

ctypedef int strtrs_t(
    # STRTRS solves a triangular system of the form
    # A * X = B  or  A**T * X = B,
    # where A is a triangular matrix of order N, and B is an N-by-NRHS
    # matrix. A check is made to verify that A is nonsingular.
    char *uplo,       # 'U':  A is upper triangular
    char *trans,      # 'N', 'T', 'C'
    char *diag,       # 'N': A is non-unit triangular, 'U': a is unit triangular
    int *n,           # The order of the matrix A.  n >= 0.
    int *nrhs,        # The number of right hand sides
    np.float32_t *a,  # Matrix A: nxn
    int *lda,         # The size of the first dimension of A (in memory)
    np.float32_t *b,  # Matrix B: nxnrhs
    int *ldb,         # The size of the first dimension of B (in memory)
    int *info         # 0 if success, otherwise an error code (integer)
) nogil

ctypedef int dgees_t(
    char *jobvs,
    char *sort,
    _dselect select,
    int *n,
    np.float64_t*a,
    int *lda,
    int *sdim,
    np.float64_t *wr,
    np.float64_t *wi,
    np.float64_t *vs,
    int *ldvs,
    np.float64_t *work,
    int *lwork,
    int *bwork,
    int *info
) nogil

ctypedef int dtrsyl_t(
    char *transa,
    char *transb,
    int *isgn,
    int *m,
    int *n,
    np.float64_t *a,
    int *lda,
    np.float64_t *b,
    int *ldb,
    np.float64_t *c,
    int *ldc,
    np.float64_t *scale,
    int *info
) nogil

ctypedef int dgetrf_t(
    # DGETRF - compute an LU factorization of a general M-by-N
    # matrix A using partial pivoting with row interchanges
    int *m,          # Rows of A
    int *n,          # Columns of A
    np.float64_t *a, # Matrix A: mxn
    int *lda,        # The size of the first dimension of A (in memory)
    int *ipiv,       # Matrix P: mxn (the pivot indices)
    int *info        # 0 if success, otherwise an error code (integer)
) nogil

ctypedef int dgetri_t(
    # DGETRI - compute the inverse of a matrix using the LU fac-
    # torization computed by DGETRF
    int *n,              # Order of A
    np.float64_t *a,     # Matrix A: nxn (the LUP decomposed matrix from dgetrf)
    int *lda,            # The size of the first dimension of A (in memory)
    int *ipiv,           # Matrix P: nxn (the pivot indices from the LUP decomposition)
    np.float64_t *work,  # Matrix: nxn (a workspace for the inversion, optimal size=nxn)
    int *lwork,          # Number of elements in the workspace: optimal is n**2
    int *info            # 0 if success, otherwise an error code (integer)
) nogil

ctypedef int dgetrs_t(
    # DGETRS - solve a system of linear equations  A * X = B or A'
    # * X = B with a general N-by-N matrix A using the LU factori-
    # zation computed by DGETRF
    char *trans,        # Specifies the form of the system of equations
    int *n,             # Order of A
    int *nrhs,          # The number of right hand sides
    np.float64_t *a,    # Matrix A: nxn (the LUP decomposed matrix from dgetrf)
    int *lda,           # The size of the first dimension of A (in memory)
    int *ipiv,          # Matrix P: nxn (the pivot indices from the LUP decomposition)
    np.float64_t *b,    # Matrix B: nxnrhs
    int *ldb,           # The size of the first dimension of B (in memory)
    int *info           # 0 if success, otherwise an error code (integer)
) nogil

ctypedef int dpotrf_t(
    # Compute the Cholesky factorization of a
    # real  symmetric positive definite matrix A
    char *uplo,      # 'U':  A = U'U and U is stored, 'L': A = LL' and L is stored
    int *n,          # The order of the matrix A.  n >= 0.
    np.float64_t *a, # Matrix A: nxn
    int *lda,        # The size of the first dimension of A (in memory)
    int *info        # 0 if success, otherwise an error code (integer)
) nogil

ctypedef int dpotri_t(
    # DPOTRI - compute the inverse of a real symmetric positive
    # definite matrix A using the Cholesky factorization A =
    # U**T*U or A = L*L**T computed by DPOTRF
    char *uplo,      # 'U':  A = U'U and U is stored, 'L': A = LL' and L is stored
    int *n,          # The order of the matrix A.  n >= 0.
    np.float64_t *a, # Matrix A: nxn
    int *lda,        # The size of the first dimension of A (in memory)
    int *info        # 0 if success, otherwise an error code (integer)
) nogil

ctypedef int dpotrs_t(
    # DPOTRS - solve a system of linear equations A*X = B with a
    # symmetric positive definite matrix A using the Cholesky fac-
    # torization A = U**T*U or A = L*L**T computed by DPOTRF
    char *uplo,       # 'U':  A = U'U and U is stored, 'L': A = LL' and L is stored
    int *n,           # The order of the matrix A.  n >= 0.
    int *nrhs,        # The number of right hand sides
    np.float64_t *a,  # Matrix A: nxn
    int *lda,         # The size of the first dimension of A (in memory)
    np.float64_t *b,  # Matrix B: nxnrhs
    int *ldb,         # The size of the first dimension of B (in memory)
    int *info         # 0 if success, otherwise an error code (integer)
) nogil

ctypedef int dtrtrs_t(
    # DTRTRS solves a triangular system of the form
    # A * X = B  or  A**T * X = B,
    # where A is a triangular matrix of order N, and B is an N-by-NRHS
    # matrix. A check is made to verify that A is nonsingular.
    char *uplo,       # 'U':  A is upper triangular
    char *trans,      # 'N', 'T', 'C'
    char *diag,       # 'N': A is non-unit triangular, 'U': a is unit triangular
    int *n,           # The order of the matrix A.  n >= 0.
    int *nrhs,        # The number of right hand sides
    np.float64_t *a,  # Matrix A: nxn
    int *lda,         # The size of the first dimension of A (in memory)
    np.float64_t *b,  # Matrix B: nxnrhs
    int *ldb,         # The size of the first dimension of B (in memory)
    int *info         # 0 if success, otherwise an error code (integer)
) nogil

ctypedef int cgees_t(
    char *jobvs,
    char *sort,
    _cselect select,
    int *n,
    np.complex64_t *a,
    int *lda,
    int *sdim,
    np.complex64_t *w,
    np.complex64_t *vs,
    int *ldvs,
    np.complex64_t *work,
    int *lwork,
    np.float64_t *rwork,
    int *bwork,
    int *info
) nogil

ctypedef int ctrsyl_t(
    char *transa,
    char *transb,
    int *isgn,
    int *m,
    int *n,
    np.complex64_t *a,
    int *lda,
    np.complex64_t *b,
    int *ldb,
    np.complex64_t *c,
    int *ldc,
    np.complex64_t *scale,
    int *info
) nogil

ctypedef int cgetrf_t(
    # CGETRF - compute an LU factorization of a general M-by-N
    # matrix A using partial pivoting with row interchanges
    int *m,             # Rows of A
    int *n,             # Columns of A
    np.complex64_t *a,  # Matrix A: mxn
    int *lda,           # The size of the first dimension of A (in memory)
    int *ipiv,          # Matrix P: mxn (the pivot indices)
    int *info           # 0 if success, otherwise an error code (integer)
) nogil

ctypedef int cgetri_t(
    # CGETRI - compute the inverse of a matrix using the LU fac-
    # torization computed by CGETRF
    int *n,               # Order of A
    np.complex64_t *a,    # Matrix A: nxn (the LUP decomposed matrix from dgetrf)
    int *lda,             # The size of the first dimension of A (in memory)
    int *ipiv,            # Matrix P: nxn (the pivot indices from the LUP decomposition)
    np.complex64_t *work, # Matrix: nxn (a workspace for the inversion, optimal size=nxn)
    int *lwork,           # Number of elements in the workspace: optimal is n**2
    int *info             # 0 if success, otherwise an error code (integer)
) nogil

ctypedef int cgetrs_t(
    # CGETRS - solve a system of linear equations  A * X = B, A**T
    # * X = B, or A**H * X = B with a general N-by-N matrix A
    # using the LU factorization computed by CGETRF
    char *trans,          # Specifies the form of the system of equations
    int *n,               # Order of A
    int *nrhs,            # The number of right hand sides
    np.complex64_t *a,    # Matrix A: nxn (the LUP decomposed matrix from dgetrf)
    int *lda,             # The size of the first dimension of A (in memory)
    int *ipiv,            # Matrix P: nxn (the pivot indices from the LUP decomposition)
    np.complex64_t *b,    # Matrix B: nxnrhs
    int *ldb,             # The size of the first dimension of B (in memory)
    int *info             # 0 if success, otherwise an error code (integer)
) nogil

ctypedef int cpotrf_t(
    # Compute the Cholesky factorization of a
    # np.complex128_t Hermitian positive definite matrix A
    char *uplo,         # 'U':  A = U'U and U is stored, 'L': A = LL' and L is stored
    int *n,             # The order of the matrix A.  n >= 0.
    np.complex64_t *a,  # Matrix A: nxn
    int *lda,           # The size of the first dimension of A (in memory)
    int *info           # 0 if success, otherwise an error code (integer)
) nogil

ctypedef int cpotri_t(
    # CPOTRI - compute the inverse of a np.complex128_t Hermitian positive
    # definite matrix A using the Cholesky factorization A =
    # U**T*U or A = L*L**T computed by CPOTRF
    char *uplo,        # 'U':  A = U'U and U is stored, 'L': A = LL' and L is stored
    int *n,            # The order of the matrix A.  n >= 0.
    np.complex64_t *a, # Matrix A: nxn
    int *lda,          # The size of the first dimension of A (in memory)
    int *info          # 0 if success, otherwise an error code (integer)
) nogil

ctypedef int cpotrs_t(
    # ZPOTRS - solve a system of linear equations A*X = B with a
    # Hermitian positive definite matrix A using the Cholesky fac-
    # torization A = U**H*U or A = L*L**H computed by ZPOTRF
    char *uplo,         # 'U':  A = U'U and U is stored, 'L': A = LL' and L is stored
    int *n,             # The order of the matrix A.  n >= 0.
    int *nrhs,          # The number of right hand sides
    np.complex64_t *a,  # Matrix A: nxn
    int *lda,           # The size of the first dimension of A (in memory)
    np.complex64_t *b,  # Matrix B: nxnrhs
    int *ldb,           # The size of the first dimension of B (in memory)
    int *info           # 0 if success, otherwise an error code (integer)
) nogil

ctypedef int ctrtrs_t(
    # CTRTRS solves a triangular system of the form
    # A * X = B  or  A**T * X = B,
    # where A is a triangular matrix of order N, and B is an N-by-NRHS
    # matrix. A check is made to verify that A is nonsingular.
    char *uplo,       # 'U':  A is upper triangular
    char *trans,      # 'N', 'T', 'C'
    char *diag,       # 'N': A is non-unit triangular, 'U': a is unit triangular
    int *n,           # The order of the matrix A.  n >= 0.
    int *nrhs,        # The number of right hand sides
    np.complex64_t *a,  # Matrix A: nxn
    int *lda,         # The size of the first dimension of A (in memory)
    np.complex64_t *b,  # Matrix B: nxnrhs
    int *ldb,         # The size of the first dimension of B (in memory)
    int *info         # 0 if success, otherwise an error code (integer)
) nogil

ctypedef int zgees_t(
    char *jobvs,
    char *sort,
    _zselect select,
    int *n,
    np.complex128_t *a,
    int *lda,
    int *sdim,
    np.complex128_t *w,
    np.complex128_t *vs,
    int *ldvs,
    np.complex128_t *work,
    int *lwork,
    np.float64_t *rwork,
    int *bwork,
    int *info
) nogil

ctypedef int ztrsyl_t(
    char *transa,
    char *transb,
    int *isgn,
    int *m,
    int *n,
    np.complex128_t *a,
    int *lda,
    np.complex128_t *b,
    int *ldb,
    np.complex128_t *c,
    int *ldc,
    np.complex128_t *scale,
    int *info
) nogil

ctypedef int zgetrf_t(
    # ZGETRF - compute an LU factorization of a general M-by-N
    # matrix A using partial pivoting with row interchanges
    int *m,                # Rows of A
    int *n,                # Columns of A
    np.complex128_t *a,    # Matrix A: mxn
    int *lda,              # The size of the first dimension of A (in memory)
    int *ipiv,             # Matrix P: mxn (the pivot indices)
    int *info              # 0 if success, otherwise an error code (integer)
) nogil

ctypedef int zgetri_t(
    # ZGETRI - compute the inverse of a matrix using the LU fac-
    # torization computed by ZGETRF
    int *n,                # Order of A
    np.complex128_t *a,    # Matrix A: nxn (the LUP decomposed matrix from dgetrf)
    int *lda,              # The size of the first dimension of A (in memory)
    int *ipiv,             # Matrix P: nxn (the pivot indices from the LUP decomposition)
    np.complex128_t *work, # Matrix: nxn (a workspace for the inversion, optimal size=nxn)
    int *lwork,            # Number of elements in the workspace: optimal is n**2
    int *info              # 0 if success, otherwise an error code (integer)
) nogil

ctypedef int zgetrs_t(
    # ZGETRS - solve a system of linear equations  A * X = B, A**T
    # * X = B, or A**H * X = B with a general N-by-N matrix A
    # using the LU factorization computed by ZGETRF
    char *trans,           # Specifies the form of the system of equations
    int *n,                # Order of A
    int *nrhs,             # The number of right hand sides
    np.complex128_t *a,    # Matrix A: nxn (the LUP decomposed matrix from dgetrf)
    int *lda,              # The size of the first dimension of A (in memory)
    int *ipiv,             # Matrix P: nxn (the pivot indices from the LUP decomposition)
    np.complex128_t *b,    # Matrix B: nxnrhs
    int *ldb,              # The size of the first dimension of B (in memory)
    int *info              # 0 if success, otherwise an error code (integer)
) nogil

ctypedef int zpotrf_t(
    # Compute the Cholesky factorization of a
    # np.complex128_t Hermitian positive definite matrix A
    char *uplo,         # 'U':  A = U'U and U is stored, 'L': A = LL' and L is stored
    int *n,             # The order of the matrix A.  n >= 0.
    np.complex128_t *a, # Matrix A: nxn
    int *lda,           # The size of the first dimension of A (in memory)
    int *info           # 0 if success, otherwise an error code (integer)
) nogil

ctypedef int zpotri_t(
    # ZPOTRI - compute the inverse of a np.complex128_t Hermitian positive
    # definite matrix A using the Cholesky factorization A =
    # U**T*U or A = L*L**T computed by ZPOTRF
    char *uplo,          # 'U':  A = U'U and U is stored, 'L': A = LL' and L is stored
    int *n,              # The order of the matrix A.  n >= 0.
    np.complex128_t *a,  # Matrix A: nxn
    int *lda,            # The size of the first dimension of A (in memory)
    int *info            # 0 if success, otherwise an error code (integer)
) nogil

ctypedef int zpotrs_t(
    # ZPOTRS - solve a system of linear equations A*X = B with a
    # Hermitian positive definite matrix A using the Cholesky fac-
    # torization A = U**H*U or A = L*L**H computed by ZPOTRF
    char *uplo,          # 'U':  A = U'U and U is stored, 'L': A = LL' and L is stored
    int *n,              # The order of the matrix A.  n >= 0.
    int *nrhs,           # The number of right hand sides
    np.complex128_t *a,  # Matrix A: nxn
    int *lda,            # The size of the first dimension of A (in memory)
    np.complex128_t *b,  # Matrix B: nxnrhs
    int *ldb,            # The size of the first dimension of B (in memory)
    int *info            # 0 if success, otherwise an error code (integer)
) nogil

ctypedef int ztrtrs_t(
    # ZTRTRS solves a triangular system of the form
    # A * X = B  or  A**T * X = B,
    # where A is a triangular matrix of order N, and B is an N-by-NRHS
    # matrix. A check is made to verify that A is nonsingular.
    char *uplo,       # 'U':  A is upper triangular
    char *trans,      # 'N', 'T', 'C'
    char *diag,       # 'N': A is non-unit triangular, 'U': a is unit triangular
    int *n,           # The order of the matrix A.  n >= 0.
    int *nrhs,        # The number of right hand sides
    np.complex128_t *a,  # Matrix A: nxn
    int *lda,         # The size of the first dimension of A (in memory)
    np.complex128_t *b,  # Matrix B: nxnrhs
    int *ldb,         # The size of the first dimension of B (in memory)
    int *info         # 0 if success, otherwise an error code (integer)
) nogil

cdef:
    cgees_t  *cgees
    ctrsyl_t *ctrsyl
    cgetrf_t *cgetrf
    cgetri_t *cgetri
    cgetrs_t *cgetrs
    cpotrf_t *cpotrf
    cpotri_t *cpotri
    cpotrs_t *cpotrs
    ctrtrs_t *ctrtrs
    sgees_t  *sgees
    strsyl_t *strsyl
    sgetrf_t *sgetrf
    sgetri_t *sgetri
    sgetrs_t *sgetrs
    spotrf_t *spotrf
    spotri_t *spotri
    spotrs_t *spotrs
    strtrs_t *strtrs
    zgees_t  *zgees
    ztrsyl_t *ztrsyl
    zgetrf_t *zgetrf
    zgetri_t *zgetri
    zgetrs_t *zgetrs
    zpotrf_t *zpotrf
    zpotri_t *zpotri
    zpotrs_t *zpotrs
    ztrtrs_t *ztrtrs
    dgees_t  *dgees
    dtrsyl_t *dtrsyl
    dgetrf_t *dgetrf
    dgetri_t *dgetri
    dgetrs_t *dgetrs
    dpotrf_t *dpotrf
    dpotri_t *dpotri
    dpotrs_t *dpotrs
    dtrtrs_t * dtrtrs