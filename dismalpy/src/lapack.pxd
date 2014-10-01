cimport numpy as np

cdef extern from "capsule.h":
    void *Capsule_AsVoidPtr(object ptr)

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

cdef:
    cgetrf_t *cgetrf
    cgetri_t *cgetri
    cgetrs_t *cgetrs
    cpotrf_t *cpotrf
    cpotri_t *cpotri
    cpotrs_t *cpotrs
    sgetrf_t *sgetrf
    sgetri_t *sgetri
    sgetrs_t *sgetrs
    spotrf_t *spotrf
    spotri_t *spotri
    spotrs_t *spotrs
    zgetrf_t *zgetrf
    zgetri_t *zgetri
    zgetrs_t *zgetrs
    zpotrf_t *zpotrf
    zpotri_t *zpotri
    zpotrs_t *zpotrs
    dgetrf_t *dgetrf
    dgetri_t *dgetri
    dgetrs_t *dgetrs
    dpotrf_t *dpotrf
    dpotri_t *dpotri
    dpotrs_t *dpotrs