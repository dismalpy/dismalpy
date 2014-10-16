try:
    # Scipy >= 0.12.0 exposes Fortran BLAS functions directly
    from scipy.linalg.blas import cgerc
except:
    # Scipy < 0.12.0 exposes Fortran BLAS functions in the `fblas` submodule
    from scipy.linalg.blas import fblas as blas
else:
    from scipy.linalg import blas

cdef:
    sdot_t *sdot = <sdot_t*>Capsule_AsVoidPtr(blas.sdot._cpointer)
    sgemm_t *sgemm = <sgemm_t*>Capsule_AsVoidPtr(blas.sgemm._cpointer)
    sgemv_t *sgemv = <sgemv_t*>Capsule_AsVoidPtr(blas.sgemv._cpointer)
    strmv_t *strmv = <strmv_t*>Capsule_AsVoidPtr(blas.strmv._cpointer)
    scopy_t *scopy = <scopy_t*>Capsule_AsVoidPtr(blas.scopy._cpointer)
    sswap_t *sswap = <sswap_t*>Capsule_AsVoidPtr(blas.sswap._cpointer)
    saxpy_t *saxpy = <saxpy_t*>Capsule_AsVoidPtr(blas.saxpy._cpointer)
    sscal_t *sscal = <sscal_t*>Capsule_AsVoidPtr(blas.sscal._cpointer)
    ddot_t *ddot = <ddot_t*>Capsule_AsVoidPtr(blas.ddot._cpointer)
    dgemm_t *dgemm = <dgemm_t*>Capsule_AsVoidPtr(blas.dgemm._cpointer)
    dgemv_t *dgemv = <dgemv_t*>Capsule_AsVoidPtr(blas.dgemv._cpointer)
    dtrmv_t *dtrmv = <dtrmv_t*>Capsule_AsVoidPtr(blas.dtrmv._cpointer)
    dcopy_t *dcopy = <dcopy_t*>Capsule_AsVoidPtr(blas.dcopy._cpointer)
    dswap_t *dswap = <dswap_t*>Capsule_AsVoidPtr(blas.dswap._cpointer)
    daxpy_t *daxpy = <daxpy_t*>Capsule_AsVoidPtr(blas.daxpy._cpointer)
    dscal_t *dscal = <dscal_t*>Capsule_AsVoidPtr(blas.dscal._cpointer)
    cdotu_t *cdot = <cdotu_t*>Capsule_AsVoidPtr(blas.cdotu._cpointer)
    cgemm_t *cgemm = <cgemm_t*>Capsule_AsVoidPtr(blas.cgemm._cpointer)
    cgemv_t *cgemv = <cgemv_t*>Capsule_AsVoidPtr(blas.cgemv._cpointer)
    ctrmv_t *ctrmv = <ctrmv_t*>Capsule_AsVoidPtr(blas.ctrmv._cpointer)
    ccopy_t *ccopy = <ccopy_t*>Capsule_AsVoidPtr(blas.ccopy._cpointer)
    cswap_t *cswap = <cswap_t*>Capsule_AsVoidPtr(blas.cswap._cpointer)
    caxpy_t *caxpy = <caxpy_t*>Capsule_AsVoidPtr(blas.caxpy._cpointer)
    cscal_t *cscal = <cscal_t*>Capsule_AsVoidPtr(blas.cscal._cpointer)
    zdotu_t *zdot = <zdotu_t*>Capsule_AsVoidPtr(blas.zdotu._cpointer)
    zgemm_t *zgemm = <zgemm_t*>Capsule_AsVoidPtr(blas.zgemm._cpointer)
    zgemv_t *zgemv = <zgemv_t*>Capsule_AsVoidPtr(blas.zgemv._cpointer)
    ztrmv_t *ztrmv = <ztrmv_t*>Capsule_AsVoidPtr(blas.ztrmv._cpointer)
    zcopy_t *zcopy = <zcopy_t*>Capsule_AsVoidPtr(blas.zcopy._cpointer)
    zswap_t *zswap = <zswap_t*>Capsule_AsVoidPtr(blas.zswap._cpointer)
    zaxpy_t *zaxpy = <zaxpy_t*>Capsule_AsVoidPtr(blas.zaxpy._cpointer)
    zscal_t *zscal = <zscal_t*>Capsule_AsVoidPtr(blas.zscal._cpointer)