try:
    # Scipy >= 0.12.0 exposes Fortran LAPACK functions directly
    from scipy.linalg.lapack import cgbsv
except:
    # Scipy < 0.12.0 exposes Fortran LAPACK functions in the `flapack` submodule
    from scipy.linalg.lapack import flapack as lapack
else:
    from scipy.linalg import lapack

cdef:
    cgetrf_t *cgetrf = <cgetrf_t*>Capsule_AsVoidPtr(lapack.cgetrf._cpointer)
    cgetri_t *cgetri = <cgetri_t*>Capsule_AsVoidPtr(lapack.cgetri._cpointer)
    cgetrs_t *cgetrs = <cgetrs_t*>Capsule_AsVoidPtr(lapack.cgetrs._cpointer)
    cpotrf_t *cpotrf = <cpotrf_t*>Capsule_AsVoidPtr(lapack.cpotrf._cpointer)
    cpotri_t *cpotri = <cpotri_t*>Capsule_AsVoidPtr(lapack.cpotri._cpointer)
    cpotrs_t *cpotrs = <cpotrs_t*>Capsule_AsVoidPtr(lapack.cpotrs._cpointer)
    sgetrf_t *sgetrf = <sgetrf_t*>Capsule_AsVoidPtr(lapack.sgetrf._cpointer)
    sgetri_t *sgetri = <sgetri_t*>Capsule_AsVoidPtr(lapack.sgetri._cpointer)
    sgetrs_t *sgetrs = <sgetrs_t*>Capsule_AsVoidPtr(lapack.sgetrs._cpointer)
    spotrf_t *spotrf = <spotrf_t*>Capsule_AsVoidPtr(lapack.spotrf._cpointer)
    spotri_t *spotri = <spotri_t*>Capsule_AsVoidPtr(lapack.spotri._cpointer)
    spotrs_t *spotrs = <spotrs_t*>Capsule_AsVoidPtr(lapack.spotrs._cpointer)
    zgetrf_t *zgetrf = <zgetrf_t*>Capsule_AsVoidPtr(lapack.zgetrf._cpointer)
    zgetri_t *zgetri = <zgetri_t*>Capsule_AsVoidPtr(lapack.zgetri._cpointer)
    zgetrs_t *zgetrs = <zgetrs_t*>Capsule_AsVoidPtr(lapack.zgetrs._cpointer)
    zpotrf_t *zpotrf = <zpotrf_t*>Capsule_AsVoidPtr(lapack.zpotrf._cpointer)
    zpotri_t *zpotri = <zpotri_t*>Capsule_AsVoidPtr(lapack.zpotri._cpointer)
    zpotrs_t *zpotrs = <zpotrs_t*>Capsule_AsVoidPtr(lapack.zpotrs._cpointer)
    dgetrf_t *dgetrf = <dgetrf_t*>Capsule_AsVoidPtr(lapack.dgetrf._cpointer)
    dgetri_t *dgetri = <dgetri_t*>Capsule_AsVoidPtr(lapack.dgetri._cpointer)
    dgetrs_t *dgetrs = <dgetrs_t*>Capsule_AsVoidPtr(lapack.dgetrs._cpointer)
    dpotrf_t *dpotrf = <dpotrf_t*>Capsule_AsVoidPtr(lapack.dpotrf._cpointer)
    dpotri_t *dpotri = <dpotri_t*>Capsule_AsVoidPtr(lapack.dpotri._cpointer)
    dpotrs_t *dpotrs = <dpotrs_t*>Capsule_AsVoidPtr(lapack.dpotrs._cpointer)