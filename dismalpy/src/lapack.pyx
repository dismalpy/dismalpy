try:
    # Scipy >= 0.12.0 exposes Fortran LAPACK functions directly
    from scipy.linalg.lapack import cgbsv
except:
    # Scipy < 0.12.0 exposes Fortran LAPACK functions in the `flapack` submodule
    from scipy.linalg.lapack import flapack as lapack
else:
    from scipy.linalg import lapack

cdef:
    cgees_t  *cgees  = <cgees_t *>Capsule_AsVoidPtr(lapack.cgees._cpointer)
    ctrsyl_t *ctrsyl = <ctrsyl_t*>Capsule_AsVoidPtr(lapack.ctrsyl._cpointer)
    cgetrf_t *cgetrf = <cgetrf_t*>Capsule_AsVoidPtr(lapack.cgetrf._cpointer)
    cgetri_t *cgetri = <cgetri_t*>Capsule_AsVoidPtr(lapack.cgetri._cpointer)
    cgetrs_t *cgetrs = <cgetrs_t*>Capsule_AsVoidPtr(lapack.cgetrs._cpointer)
    cpotrf_t *cpotrf = <cpotrf_t*>Capsule_AsVoidPtr(lapack.cpotrf._cpointer)
    cpotri_t *cpotri = <cpotri_t*>Capsule_AsVoidPtr(lapack.cpotri._cpointer)
    cpotrs_t *cpotrs = <cpotrs_t*>Capsule_AsVoidPtr(lapack.cpotrs._cpointer)
    ctrtrs_t *ctrtrs = <ctrtrs_t*>Capsule_AsVoidPtr(lapack.ctrtrs._cpointer)
    sgees_t  *sgees  = <sgees_t *>Capsule_AsVoidPtr(lapack.sgees._cpointer)
    strsyl_t *strsyl = <strsyl_t*>Capsule_AsVoidPtr(lapack.strsyl._cpointer)
    sgetrf_t *sgetrf = <sgetrf_t*>Capsule_AsVoidPtr(lapack.sgetrf._cpointer)
    sgetri_t *sgetri = <sgetri_t*>Capsule_AsVoidPtr(lapack.sgetri._cpointer)
    sgetrs_t *sgetrs = <sgetrs_t*>Capsule_AsVoidPtr(lapack.sgetrs._cpointer)
    spotrf_t *spotrf = <spotrf_t*>Capsule_AsVoidPtr(lapack.spotrf._cpointer)
    spotri_t *spotri = <spotri_t*>Capsule_AsVoidPtr(lapack.spotri._cpointer)
    spotrs_t *spotrs = <spotrs_t*>Capsule_AsVoidPtr(lapack.spotrs._cpointer)
    strtrs_t *strtrs = <strtrs_t*>Capsule_AsVoidPtr(lapack.strtrs._cpointer)
    zgees_t  *zgees  = <zgees_t *>Capsule_AsVoidPtr(lapack.zgees._cpointer)
    ztrsyl_t *ztrsyl = <ztrsyl_t*>Capsule_AsVoidPtr(lapack.ztrsyl._cpointer)
    zgetrf_t *zgetrf = <zgetrf_t*>Capsule_AsVoidPtr(lapack.zgetrf._cpointer)
    zgetri_t *zgetri = <zgetri_t*>Capsule_AsVoidPtr(lapack.zgetri._cpointer)
    zgetrs_t *zgetrs = <zgetrs_t*>Capsule_AsVoidPtr(lapack.zgetrs._cpointer)
    zpotrf_t *zpotrf = <zpotrf_t*>Capsule_AsVoidPtr(lapack.zpotrf._cpointer)
    zpotri_t *zpotri = <zpotri_t*>Capsule_AsVoidPtr(lapack.zpotri._cpointer)
    zpotrs_t *zpotrs = <zpotrs_t*>Capsule_AsVoidPtr(lapack.zpotrs._cpointer)
    ztrtrs_t *ztrtrs = <ztrtrs_t*>Capsule_AsVoidPtr(lapack.ztrtrs._cpointer)
    dgees_t  *dgees  = <dgees_t *>Capsule_AsVoidPtr(lapack.dgees._cpointer)
    dtrsyl_t *dtrsyl = <dtrsyl_t*>Capsule_AsVoidPtr(lapack.dtrsyl._cpointer)
    dgetrf_t *dgetrf = <dgetrf_t*>Capsule_AsVoidPtr(lapack.dgetrf._cpointer)
    dgetri_t *dgetri = <dgetri_t*>Capsule_AsVoidPtr(lapack.dgetri._cpointer)
    dgetrs_t *dgetrs = <dgetrs_t*>Capsule_AsVoidPtr(lapack.dgetrs._cpointer)
    dpotrf_t *dpotrf = <dpotrf_t*>Capsule_AsVoidPtr(lapack.dpotrf._cpointer)
    dpotri_t *dpotri = <dpotri_t*>Capsule_AsVoidPtr(lapack.dpotri._cpointer)
    dpotrs_t *dpotrs = <dpotrs_t*>Capsule_AsVoidPtr(lapack.dpotrs._cpointer)
    dtrtrs_t *dtrtrs = <dtrtrs_t*>Capsule_AsVoidPtr(lapack.dtrtrs._cpointer)