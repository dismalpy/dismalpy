cimport numpy as np

cdef extern from "capsule.h":
    void *Capsule_AsVoidPtr(object ptr)

ctypedef int sger_t(
    # SGER - perform the rank 1 operation   A := alpha*x*y' + A,
    int *m,               # Rows of A
    int *n,               # Columns of A
    np.float32_t *alpha,  # Scalar multiple
    np.float32_t *x,      # Vector x
    int *incx,            # The increment between elements of x (usually 1)
    np.float32_t *y,      # Vector y
    int *incy,            # The increment between elements of y (usually 1)
    np.float32_t *a,      # Matrix A: mxn
    int *lda,             # The size of the first dimension of A (in memory)
) nogil

ctypedef int ssyr_t(
    # SSYR - perform the symmetric rank 1 operation   A := alpha*x*x' + A,
    char *uplo,           # {'U','L'}, upper, lower
    int *n,               # Order of A
    np.float32_t *alpha,  # Scalar multiple
    np.float32_t *x,      # Vector x
    int *incx,            # The increment between elements of x (usually 1)
    np.float32_t *a,      # Matrix A: mxn
    int *lda,             # The size of the first dimension of A (in memory)
) nogil

ctypedef int sgemm_t(
    # Compute C := alpha*A*B + beta*C
    char *transa,         # {'T','C'}: o(A)=A'; {'N'}: o(A)=A
    char *transb,         # {'T','C'}: o(B)=B'; {'N'}: o(B)=B
    int *m,               # Rows of o(A)    (and of C)
    int *n,               # Columns of o(B) (and of C)
    int *k,               # Columns of o(A) / Rows of o(B)
    np.float32_t *alpha,  # Scalar multiple
    np.float32_t *a,      # Matrix A: mxk
    int *lda,             # The size of the first dimension of A (in memory)
    np.float32_t *b,      # Matrix B: kxn
    int *ldb,             # The size of the first dimension of B (in memory)
    np.float32_t *beta,   # Scalar multiple
    np.float32_t *c,      # Matrix C: mxn
    int *ldc              # The size of the first dimension of C (in memory)
) nogil

ctypedef int sgemv_t(
    # Compute C := alpha*A*x + beta*y
    char *trans,          # {'T','C'}: o(A)=A'; {'N'}: o(A)=A
    int *m,               # Rows of o(A)
    int *n,               # Columns of o(A) / min(len(x))
    np.float32_t *alpha,  # Scalar multiple
    np.float32_t *a,      # Matrix A: mxn
    int *lda,             # The size of the first dimension of A (in memory)
    np.float32_t *x,      # Vector x, min(len(x)) = n
    int *incx,            # The increment between elements of x (usually 1)
    np.float32_t *beta,   # Scalar multiple
    np.float32_t *y,      # Vector y, min(len(y)) = m
    int *incy             # The increment between elements of y (usually 1)
) nogil

ctypedef int ssymm_t(
    # SSYMM - perform one of the matrix-matrix operations   C :=
    # alpha*A*B + beta*C,
    char *side,           # {'L', 'R'}: left, right
    char *uplo,           # {'U','L'}, upper, lower
    int *m,               # Rows of C
    int *n,               # Columns of C
    np.float32_t *alpha,  # Scalar multiple
    np.float32_t *a,      # Matrix A
    int *lda,             # The size of the first dimension of A (in memory)
    np.float32_t *b,      # Matrix B
    int *ldb,             # The size of the first dimension of B (in memory)
    np.float32_t *beta,   # Scalar multiple
    np.float32_t *c,      # Matrix C
    int *ldc,             # The size of the first dimension of C (in memory)
) nogil

ctypedef int ssymv_t(
    # SSYMV - perform the matrix-vector operation   y := alpha*A*x
    # + beta*y,
    char *uplo,           # {'U','L'}, upper, lower
    int *n,               # Order of matrix A
    np.float32_t *alpha,  # Scalar multiple
    np.float32_t *a,      # Matrix A: mxn
    int *lda,             # The size of the first dimension of A (in memory)
    np.float32_t *x,      # Vector x, min(len(x)) = n
    int *incx,            # The increment between elements of x (usually 1)
    np.float32_t *beta,   # Scalar multiple
    np.float32_t *y,      # Vector y, min(len(y)) = n
    int *incy,            # The increment between elements of y (usually 1)
) nogil

ctypedef int strmm_t(
    # STRMM - perform one of the matrix-matrix operations   B :=
    # alpha*op( A )*B, or B := alpha*B*op( A ),
    char *side,           # {'L', 'R'}: left, right
    char *uplo,           # {'U','L'}, upper, lower
    char *transa,         # {'T','C'}: o(A)=A'; {'N'}: o(A)=A
    char *diag,           # {'U','N'}: unit triangular or not
    int *m,               # Rows of B
    int *n,               # Columns of B
    np.float32_t *alpha,  # Scalar multiple
    np.float32_t *a,      # Matrix A
    int *lda,             # The size of the first dimension of A (in memory)
    np.float32_t *b,      # Matrix B
    int *ldb,             # The size of the first dimension of B (in memory)
) nogil

ctypedef int strmv_t(
    # STRMV - perform one of the matrix-vector operations   x :=
    # A*x, or x := A'*x,
    char *uplo,           # {'U','L'}, upper, lower
    char *trans,          # {'T','C'}: o(A)=A'; {'N'}: o(A)=A
    char *diag,           # {'U','N'}: unit triangular or not
    int *n,               # Order of matrix A
    np.float32_t *a,      # Matrix A: mxn
    int *lda,             # The size of the first dimension of A (in memory)
    np.float32_t *x,      # Vector x, min(len(x)) = n
    int *incx,            # The increment between elements of x (usually 1)
) nogil

ctypedef int scopy_t(
    int *n,           # Number of vector elements to be copied.
    np.float32_t *x,  # Vector from which to copy.
    int *incx,        # Increment between elements of x.
    np.float32_t *y,  # array of dimension (n-1) * |incy| + 1, result vector.
    int *incy         # Increment between elements of y.
) nogil

ctypedef int sscal_t(
    # SSCAL - BLAS level one, scales a double precision vector
    int *n,               # Number of elements in the vector.
    np.float32_t *alpha,  # scalar alpha
    np.float32_t *x,      # Array of dimension (n-1) * |incx| + 1. Vector to be scaled.
    int *incx             # Increment between elements of x.
) nogil

ctypedef int sswap_t(
    # Compute y <-> x
    int *n,               # Number of vector elements to be swapped.
    np.float32_t *x,      # Vector x
    int *incx,            # The increment between elements of x (usually 1)
    np.float32_t *y,      # Vector y
    int *incy             # The increment between elements of y (usually 1)
) nogil

ctypedef int saxpy_t(
    # Compute y := alpha*x + y
    int *n,               # Columns of o(A) / min(len(x))
    np.float32_t *alpha,  # Scalar multiple
    np.float32_t *x,      # Vector x, min(len(x)) = n
    int *incx,            # The increment between elements of x (usually 1)
    np.float32_t *y,      # Vector y, min(len(y)) = m
    int *incy             # The increment between elements of y (usually 1)
) nogil

ctypedef np.float64_t sdot_t(
    # Compute DDOT := x.T * y
    int *n,           # Length of vectors
    np.float32_t *x,  # Vector x, min(len(x)) = n
    int *incx,        # The increment between elements of x (usually 1)
    np.float32_t *y,  # Vector y, min(len(y)) = m
    int *incy         # The increment between elements of y (usually 1)
) nogil

ctypedef int dger_t(
    # DGER - perform the rank 1 operation   A := alpha*x*y' + A,
    int *m,               # Rows of A
    int *n,               # Columns of A
    np.float64_t *alpha,  # Scalar multiple
    np.float64_t *x,      # Vector x
    int *incx,            # The increment between elements of x (usually 1)
    np.float64_t *y,      # Vector y
    int *incy,            # The increment between elements of y (usually 1)
    np.float64_t *a,      # Matrix A: mxn
    int *lda,             # The size of the first dimension of A (in memory)
) nogil

ctypedef int dsyr_t(
    # DSYR - perform the symmetric rank 1 operation   A := alpha*x*x' + A,
    char *uplo,           # {'U','L'}, upper, lower
    int *n,               # Order of A
    np.float64_t *alpha,  # Scalar multiple
    np.float64_t *x,      # Vector x
    int *incx,            # The increment between elements of x (usually 1)
    np.float64_t *a,      # Matrix A: mxn
    int *lda,             # The size of the first dimension of A (in memory)
) nogil

ctypedef int dgemm_t(
    # Compute C := alpha*A*B + beta*C
    char *transa,        # {'T','C'}: o(A)=A'; {'N'}: o(A)=A
    char *transb,        # {'T','C'}: o(B)=B'; {'N'}: o(B)=B
    int *m,              # Rows of o(A)    (and of C)
    int *n,              # Columns of o(B) (and of C)
    int *k,              # Columns of o(A) / Rows of o(B)
    np.float64_t *alpha, # Scalar multiple
    np.float64_t *a,     # Matrix A: mxk
    int *lda,            # The size of the first dimension of A (in memory)
    np.float64_t *b,     # Matrix B: kxn
    int *ldb,            # The size of the first dimension of B (in memory)
    np.float64_t *beta,  # Scalar multiple
    np.float64_t *c,     # Matrix C: mxn
    int *ldc             # The size of the first dimension of C (in memory)
) nogil

ctypedef int dgemv_t(
    # Compute y := alpha*A*x + beta*y
    char *trans,         # {'T','C'}: o(A)=A'; {'N'}: o(A)=A
    int *m,              # Rows of A (prior to transpose from *trans)
    int *n,              # Columns of A / min(len(x))
    np.float64_t *alpha, # Scalar multiple
    np.float64_t *a,     # Matrix A: mxn
    int *lda,            # The size of the first dimension of A (in memory)
    np.float64_t *x,     # Vector x, min(len(x)) = n
    int *incx,           # The increment between elements of x (usually 1)
    np.float64_t *beta,  # Scalar multiple
    np.float64_t *y,     # Vector y, min(len(y)) = m
    int *incy            # The increment between elements of y (usually 1)
) nogil

ctypedef int dsymm_t(
    # DSYMM - perform one of the matrix-matrix operations   C :=
    # alpha*A*B + beta*C,
    char *side,           # {'L', 'R'}: left, right
    char *uplo,           # {'U','L'}, upper, lower
    int *m,               # Rows of C
    int *n,               # Columns of C
    np.float64_t *alpha,  # Scalar multiple
    np.float64_t *a,      # Matrix A
    int *lda,             # The size of the first dimension of A (in memory)
    np.float64_t *b,      # Matrix B
    int *ldb,             # The size of the first dimension of B (in memory)
    np.float64_t *beta,   # Scalar multiple
    np.float64_t *c,      # Matrix C
    int *ldc,             # The size of the first dimension of C (in memory)
) nogil

ctypedef int dsymv_t(
    # DSYMV - perform the matrix-vector operation   y := alpha*A*x
    # + beta*y,
    char *uplo,           # {'U','L'}, upper, lower
    int *n,               # Order of matrix A
    np.float64_t *alpha,  # Scalar multiple
    np.float64_t *a,      # Matrix A: mxn
    int *lda,             # The size of the first dimension of A (in memory)
    np.float64_t *x,      # Vector x, min(len(x)) = n
    int *incx,            # The increment between elements of x (usually 1)
    np.float64_t *beta,   # Scalar multiple
    np.float64_t *y,      # Vector y, min(len(y)) = n
    int *incy,            # The increment between elements of y (usually 1)
) nogil

ctypedef int dtrmm_t(
    # DTRMM - perform one of the matrix-matrix operations   B :=
    # alpha*op( A )*B, or B := alpha*B*op( A ),
    char *side,           # {'L', 'R'}: left, right
    char *uplo,           # {'U','L'}, upper, lower
    char *transa,         # {'T','C'}: o(A)=A'; {'N'}: o(A)=A
    char *diag,           # {'U','N'}: unit triangular or not
    int *m,               # Rows of B
    int *n,               # Columns of B
    np.float64_t *alpha,  # Scalar multiple
    np.float64_t *a,      # Matrix A
    int *lda,             # The size of the first dimension of A (in memory)
    np.float64_t *b,      # Matrix B
    int *ldb,             # The size of the first dimension of B (in memory)
) nogil

ctypedef int dtrmv_t(
    # DTRMV - perform one of the matrix-vector operations   x :=
    # A*x, or x := A'*x,
    char *uplo,           # {'U','L'}, upper, lower
    char *trans,          # {'T','C'}: o(A)=A'; {'N'}: o(A)=A
    char *diag,           # {'U','N'}: unit triangular or not
    int *n,               # Order of matrix A
    np.float64_t *a,      # Matrix A: mxn
    int *lda,             # The size of the first dimension of A (in memory)
    np.float64_t *x,      # Vector x, min(len(x)) = n
    int *incx,            # The increment between elements of x (usually 1)
) nogil

ctypedef int dcopy_t(
    int *n,              # Number of vector elements to be copied.
    np.float64_t *x,     # Vector from which to copy.
    int *incx,           # Increment between elements of x.
    np.float64_t *y,     # array of dimension (n-1) * |incy| + 1, result vector.
    int *incy            # Increment between elements of y.
) nogil

ctypedef int dscal_t(
    # DSCAL - BLAS level one, scales a double precision vector
    int *n,               # Number of elements in the vector.
    np.float64_t *alpha,  # scalar alpha
    np.float64_t *x,      # Array of dimension (n-1) * |incx| + 1. Vector to be scaled.
    int *incx             # Increment between elements of x.
) nogil

ctypedef int dswap_t(
    # Compute y <-> x
    int *n,               # Number of vector elements to be swapped.
    np.float64_t *x,      # Vector x
    int *incx,            # The increment between elements of x (usually 1)
    np.float64_t *y,      # Vector y
    int *incy             # The increment between elements of y (usually 1)
) nogil

ctypedef int daxpy_t(
    # Compute y := alpha*x + y
    int *n,              # Columns of o(A) / min(len(x))
    np.float64_t *alpha, # Scalar multiple
    np.float64_t *x,     # Vector x, min(len(x)) = n
    int *incx,           # The increment between elements of x (usually 1)
    np.float64_t *y,     # Vector y, min(len(y)) = m
    int *incy            # The increment between elements of y (usually 1)
) nogil

ctypedef double ddot_t(
    # Compute DDOT := x.T * y
    int *n,              # Length of vectors
    np.float64_t *x,     # Vector x, min(len(x)) = n
    int *incx,           # The increment between elements of x (usually 1)
    np.float64_t *y,     # Vector y, min(len(y)) = m
    int *incy            # The increment between elements of y (usually 1)
) nogil

ctypedef int cgeru_t(
    # CGERU - perform the rank 1 operation   A := alpha*x*y' + A,
    int *m,                 # Rows of A
    int *n,                 # Columns of A
    np.complex64_t *alpha,  # Scalar multiple
    np.complex64_t *x,      # Vector x
    int *incx,              # The increment between elements of x (usually 1)
    np.complex64_t *y,      # Vector y
    int *incy,              # The increment between elements of y (usually 1)
    np.complex64_t *a,      # Matrix A: mxn
    int *lda,               # The size of the first dimension of A (in memory)
) nogil

ctypedef int cher_t(
    # CHER - perform the hermitian rank 1 operation   A := alpha*x*conjg( x' ) + A,
    char *uplo,             # {'U','L'}, upper, lower
    int *n,                 # Order of A
    np.complex64_t *alpha,  # Scalar multiple
    np.complex64_t *x,      # Vector x
    int *incx,              # The increment between elements of x (usually 1)
    np.complex64_t *a,      # Matrix A: mxn
    int *lda,               # The size of the first dimension of A (in memory)
) nogil

ctypedef int cgemm_t(
    # Compute C := alpha*A*B + beta*C
    char *transa,           # {'T','C'}: o(A)=A'; {'N'}: o(A)=A
    char *transb,           # {'T','C'}: o(B)=B'; {'N'}: o(B)=B
    int *m,                 # Rows of o(A)    (and of C)
    int *n,                 # Columns of o(B) (and of C)
    int *k,                 # Columns of o(A) / Rows of o(B)
    np.complex64_t *alpha,  # Scalar multiple
    np.complex64_t *a,      # Matrix A: mxk
    int *lda,               # The size of the first dimension of A (in memory)
    np.complex64_t *b,      # Matrix B: kxn
    int *ldb,               # The size of the first dimension of B (in memory)
    np.complex64_t *beta,   # Scalar multiple
    np.complex64_t *c,      # Matrix C: mxn
    int *ldc                # The size of the first dimension of C (in memory)
) nogil

ctypedef int cgemv_t(
    # Compute C := alpha*A*x + beta*y
    char *trans,            # {'T','C'}: o(A)=A'; {'N'}: o(A)=A
    int *m,                 # Rows of o(A)
    int *n,                 # Columns of o(A) / min(len(x))
    np.complex64_t *alpha,  # Scalar multiple
    np.complex64_t *a,      # Matrix A: mxn
    int *lda,               # The size of the first dimension of A (in memory)
    np.complex64_t *x,      # Vector x, min(len(x)) = n
    int *incx,              # The increment between elements of x (usually 1)
    np.complex64_t *beta,   # Scalar multiple
    np.complex64_t *y,      # Vector y, min(len(y)) = m
    int *incy               # The increment between elements of y (usually 1)
) nogil

ctypedef int csymm_t(
    # CSYMM - perform one of the matrix-matrix operations   C :=
    # alpha*A*B + beta*C,
    char *side,             # {'L', 'R'}: left, right
    char *uplo,             # {'U','L'}, upper, lower
    int *m,                 # Rows of C
    int *n,                 # Columns of C
    np.complex64_t *alpha,  # Scalar multiple
    np.complex64_t *a,      # Matrix A
    int *lda,               # The size of the first dimension of A (in memory)
    np.complex64_t *b,      # Matrix B
    int *ldb,               # The size of the first dimension of B (in memory)
    np.complex64_t *beta,   # Scalar multiple
    np.complex64_t *c,      # Matrix C
    int *ldc,               # The size of the first dimension of C (in memory)
) nogil

ctypedef int csymv_t(
    # CSYMV - perform the matrix-vector operation   y := alpha*A*x
    # + beta*y,
    char *uplo,             # {'U','L'}, upper, lower
    int *n,                 # Order of matrix A
    np.complex64_t *alpha,  # Scalar multiple
    np.complex64_t *a,      # Matrix A: mxn
    int *lda,               # The size of the first dimension of A (in memory)
    np.complex64_t *x,      # Vector x, min(len(x)) = n
    int *incx,              # The increment between elements of x (usually 1)
    np.complex64_t *beta,   # Scalar multiple
    np.complex64_t *y,      # Vector y, min(len(y)) = n
    int *incy,              # The increment between elements of y (usually 1)
) nogil

ctypedef int ctrmm_t(
    # CTRMM - perform one of the matrix-matrix operations   B :=
    # alpha*op( A )*B, or B := alpha*B*op( A ),
    char *side,             # {'L', 'R'}: left, right
    char *uplo,             # {'U','L'}, upper, lower
    char *transa,           # {'T','C'}: o(A)=A'; {'N'}: o(A)=A
    char *diag,             # {'U','N'}: unit triangular or not
    int *m,                 # Rows of B
    int *n,                 # Columns of B
    np.complex64_t *alpha,  # Scalar multiple
    np.complex64_t *a,      # Matrix A
    int *lda,               # The size of the first dimension of A (in memory)
    np.complex64_t *b,      # Matrix B
    int *ldb,               # The size of the first dimension of B (in memory)
) nogil

ctypedef int ctrmv_t(
    # CTRMV - perform one of the matrix-vector operations   x :=
    # A*x, or x := A'*x,
    char *uplo,           # {'U','L'}, upper, lower
    char *trans,          # {'T','C'}: o(A)=A'; {'N'}: o(A)=A
    char *diag,           # {'U','N'}: unit triangular or not
    int *n,               # Order of matrix A
    np.complex64_t *a,    # Matrix A: mxn
    int *lda,             # The size of the first dimension of A (in memory)
    np.complex64_t *x,    # Vector x, min(len(x)) = n
    int *incx,            # The increment between elements of x (usually 1)
) nogil

ctypedef int ccopy_t(
    int *n,             # Number of vector elements to be copied.
    np.complex64_t *x,  # Vector from which to copy.
    int *incx,          # Increment between elements of x.
    np.complex64_t *y,  # array of dimension (n-1) * |incy| + 1, result vector.
    int *incy           # Increment between elements of y.
) nogil

ctypedef int cscal_t(
    # CSCAL - BLAS level one, scales a double precision vector
    int *n,                 # Number of elements in the vector.
    np.complex64_t *alpha,  # scalar alpha
    np.complex64_t *x,      # Array of dimension (n-1) * |incx| + 1. Vector to be scaled.
    int *incx               # Increment between elements of x.
) nogil

ctypedef int cswap_t(
    # Compute y <-> x
    int *n,                 # Number of vector elements to be swapped.
    np.complex64_t *x,      # Vector x
    int *incx,              # The increment between elements of x (usually 1)
    np.complex64_t *y,      # Vector y
    int *incy               # The increment between elements of y (usually 1)
) nogil

ctypedef int caxpy_t(
    # Compute y := alpha*x + y
    int *n,                 # Columns of o(A) / min(len(x))
    np.complex64_t *alpha,  # Scalar multiple
    np.complex64_t *x,      # Vector x, min(len(x)) = n
    int *incx,              # The increment between elements of x (usually 1)
    np.complex64_t *y,      # Vector y, min(len(y)) = m
    int *incy               # The increment between elements of y (usually 1)
) nogil

ctypedef np.complex64_t cdotu_t(
    # Compute CDOTU := x.T * y
    int *n,             # Length of vectors
    np.complex64_t *x,  # Vector x, min(len(x)) = n
    int *incx,          # The increment between elements of x (usually 1)
    np.complex64_t *y,  # Vector y, min(len(y)) = m
    int *incy           # The increment between elements of y (usually 1)
) nogil

ctypedef int zgeru_t(
    # ZGERU - perform the rank 1 operation   A := alpha*x*y' + A,
    int *m,                  # Rows of A
    int *n,                  # Columns of A
    np.complex128_t *alpha,  # Scalar multiple
    np.complex128_t *x,      # Vector x
    int *incx,               # The increment between elements of x (usually 1)
    np.complex128_t *y,      # Vector y
    int *incy,               # The increment between elements of y (usually 1)
    np.complex128_t *a,      # Matrix A: mxn
    int *lda,                # The size of the first dimension of A (in memory)
) nogil

ctypedef int zher_t(
    # ZHER - perform the hermitian rank 1 operation   A := alpha*x*conjg( x' ) + A,
    char *uplo,              # {'U','L'}, upper, lower
    int *n,                  # Order of A
    np.complex128_t *alpha,  # Scalar multiple
    np.complex128_t *x,      # Vector x
    int *incx,               # The increment between elements of x (usually 1)
    np.complex128_t *a,      # Matrix A: mxn
    int *lda,                # The size of the first dimension of A (in memory)
) nogil

ctypedef int zgemm_t(
    # Compute C := alpha*A*B + beta*C
    char *transa,           # {'T','C'}: o(A)=A'; {'N'}: o(A)=A
    char *transb,           # {'T','C'}: o(B)=B'; {'N'}: o(B)=B
    int *m,                 # Rows of o(A)    (and of C)
    int *n,                 # Columns of o(B) (and of C)
    int *k,                 # Columns of o(A) / Rows of o(B)
    np.complex128_t *alpha, # Scalar multiple
    np.complex128_t *a,     # Matrix A: mxk
    int *lda,               # The size of the first dimension of A (in memory)
    np.complex128_t *b,     # Matrix B: kxn
    int *ldb,               # The size of the first dimension of B (in memory)
    np.complex128_t *beta,  # Scalar multiple
    np.complex128_t *c,     # Matrix C: mxn
    int *ldc                # The size of the first dimension of C (in memory)
) nogil

ctypedef int zgemv_t(
    # Compute C := alpha*A*x + beta*y
    char *trans,    # {'T','C'}: o(A)=A'; {'N'}: o(A)=A
    int *m,         # Rows of o(A)
    int *n,         # Columns of o(A) / min(len(x))
    np.complex128_t *alpha, # Scalar multiple
    np.complex128_t *a,     # Matrix A: mxn
    int *lda,       # The size of the first dimension of A (in memory)
    np.complex128_t *x,     # Vector x, min(len(x)) = n
    int *incx,      # The increment between elements of x (usually 1)
    np.complex128_t *beta,  # Scalar multiple
    np.complex128_t *y,     # Vector y, min(len(y)) = m
    int *incy       # The increment between elements of y (usually 1)
) nogil

ctypedef int zsymm_t(
    # ZSYMM - perform one of the matrix-matrix operations   C :=
    # alpha*A*B + beta*C,
    char *side,             # {'L', 'R'}: left, right
    char *uplo,             # {'U','L'}, upper, lower
    int *m,                 # Rows of C
    int *n,                 # Columns of C
    np.complex128_t *alpha, # Scalar multiple
    np.complex128_t *a,     # Matrix A
    int *lda,               # The size of the first dimension of A (in memory)
    np.complex128_t *b,     # Matrix B
    int *ldb,               # The size of the first dimension of B (in memory)
    np.complex128_t *beta,  # Scalar multiple
    np.complex128_t *c,     # Matrix C
    int *ldc,               # The size of the first dimension of C (in memory)
) nogil

ctypedef int zsymv_t(
    # ZSYMV - perform the matrix-vector operation   y := alpha*A*x
    # + beta*y,
    char *uplo,             # {'U','L'}, upper, lower
    int *n,                 # Order of matrix A
    np.complex128_t *alpha, # Scalar multiple
    np.complex128_t *a,     # Matrix A: mxn
    int *lda,               # The size of the first dimension of A (in memory)
    np.complex128_t *x,     # Vector x, min(len(x)) = n
    int *incx,              # The increment between elements of x (usually 1)
    np.complex128_t *beta,  # Scalar multiple
    np.complex128_t *y,     # Vector y, min(len(y)) = n
    int *incy,              # The increment between elements of y (usually 1)
) nogil

ctypedef int ztrmm_t(
    # ZTRMM - perform one of the matrix-matrix operations   B :=
    # alpha*op( A )*B, or B := alpha*B*op( A ),
    char *side,             # {'L', 'R'}: left, right
    char *uplo,             # {'U','L'}, upper, lower
    char *transa,           # {'T','C'}: o(A)=A'; {'N'}: o(A)=A
    char *diag,             # {'U','N'}: unit triangular or not
    int *m,                 # Rows of B
    int *n,                 # Columns of B
    np.complex128_t *alpha, # Scalar multiple
    np.complex128_t *a,     # Matrix A
    int *lda,               # The size of the first dimension of A (in memory)
    np.complex128_t *b,     # Matrix B
    int *ldb,               # The size of the first dimension of B (in memory)
) nogil

ctypedef int ztrmv_t(
    # ZTRMV - perform one of the matrix-vector operations   x :=
    # A*x, or x := A'*x,
    char *uplo,           # {'U','L'}, upper, lower
    char *trans,          # {'T','C'}: o(A)=A'; {'N'}: o(A)=A
    char *diag,           # {'U','N'}: unit triangular or not
    int *n,               # Order of matrix A
    np.complex128_t *a,   # Matrix A: mxn
    int *lda,             # The size of the first dimension of A (in memory)
    np.complex128_t *x,   # Vector x, min(len(x)) = n
    int *incx,            # The increment between elements of x (usually 1)
) nogil

ctypedef int zcopy_t(
    int *n,         # Number of vector elements to be copied.
    np.complex128_t *x,     # Vector from which to copy.
    int *incx,      # Increment between elements of x.
    np.complex128_t *y,     # array of dimension (n-1) * |incy| + 1, result vector.
    int *incy       # Increment between elements of y.
) nogil

ctypedef int zscal_t(
    # ZSCAL - BLAS level one, scales a double np.complex128_t precision vector
    int *n,          # Number of elements in the vector.
    np.complex128_t *alpha,  # scalar alpha
    np.complex128_t *x,      # Array of dimension (n-1) * |incx| + 1. Vector to be scaled.
    int *incx        # Increment between elements of x.
) nogil

ctypedef int zswap_t(
    # Compute y <-> x
    int *n,                  # Number of vector elements to be swapped.
    np.complex128_t *x,      # Vector x
    int *incx,               # The increment between elements of x (usually 1)
    np.complex128_t *y,      # Vector y
    int *incy                # The increment between elements of y (usually 1)
) nogil

ctypedef int zaxpy_t(
    # Compute y := alpha*x + y
    int *n,         # Columns of o(A) / min(len(x))
    np.complex128_t *alpha, # Scalar multiple
    np.complex128_t *x,     # Vector x, min(len(x)) = n
    int *incx,      # The increment between elements of x (usually 1)
    np.complex128_t *y,     # Vector y, min(len(y)) = m
    int *incy       # The increment between elements of y (usually 1)
) nogil

ctypedef np.complex128_t zdotu_t(
    # Compute ZDOTU := x.T * y
    int *n,      # Length of vectors
    np.complex128_t *x,  # Vector x, min(len(x)) = n
    int *incx,   # The increment between elements of x (usually 1)
    np.complex128_t *y,  # Vector y, min(len(y)) = m
    int *incy    # The increment between elements of y (usually 1)
) nogil

cdef:
    sdot_t *sdot
    sger_t *sger
    ssyr_t *ssyr
    sgemm_t *sgemm
    sgemv_t *sgemv
    ssymm_t *ssymm
    ssymv_t *ssymv
    strmm_t *strmm
    strmv_t *strmv
    scopy_t *scopy
    sswap_t *sswap
    saxpy_t *saxpy
    sscal_t *sscal
    ddot_t *ddot
    dger_t *dger
    dsyr_t *dsyr
    dgemm_t *dgemm
    dgemv_t *dgemv
    dsymm_t *dsymm
    dsymv_t *dsymv
    dtrmm_t *dtrmm
    dtrmv_t *dtrmv
    dcopy_t *dcopy
    dswap_t *dswap
    daxpy_t *daxpy
    dscal_t *dscal
    cdotu_t *cdot
    cgeru_t *cgeru
    cher_t *cher
    cgemm_t *cgemm
    cgemv_t *cgemv
    csymm_t *csymm
    csymv_t *csymv
    ctrmm_t *ctrmm
    ctrmv_t *ctrmv
    ccopy_t *ccopy
    cswap_t *cswap
    caxpy_t *caxpy
    cscal_t *cscal
    zdotu_t *zdot
    zgeru_t *zgeru
    zher_t *zher
    zgemm_t *zgemm
    zgemv_t *zgemv
    zsymm_t *zsymm
    zsymv_t *zsymv
    ztrmm_t *ztrmm
    ztrmv_t *ztrmv
    zcopy_t *zcopy
    zswap_t *zswap
    zaxpy_t *zaxpy
    zscal_t *zscal