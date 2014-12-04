"""
Wishart Random Variable

Used as a shim for versions of Scipy without the `wishart` and `invwishart`
random variables.
"""
from __future__ import division, print_function, absolute_import

import numpy as np
import scipy.linalg
from scipy.misc import doccer
from scipy.special import psi, multigammaln
from scipy.stats._multivariate import _squeeze_output

_LOG_2 = np.log(2)

_wishart_doc_default_callparams = """\
df : integer
    Degrees of freedom, must be greater than or equal to dimension of the
    scale matrix
scale : array_like
    Symmetric positive definite scale matrix of the distribution
"""

_wishart_doc_callparams_note = ""

_wishart_doc_frozen_callparams = ""

_wishart_doc_frozen_callparams_note = \
    """See class definition for a detailed description of parameters."""

wishart_docdict_params = {
    '_doc_default_callparams': _wishart_doc_default_callparams,
    '_doc_callparams_note': _wishart_doc_callparams_note
}

wishart_docdict_noparams = {
    '_doc_default_callparams': _wishart_doc_frozen_callparams,
    '_doc_callparams_note': _wishart_doc_frozen_callparams_note
}


class wishart_gen(object):
    r"""
    A Wishart random variable.

    The `df` keyword specifies the degrees of freedom. The `scale` keyword
    specifies the scale matrix, which must be symmetric and positive definite.
    In this context, the scale matrix is often interpreted in terms of a
    multivariate normal precision matrix (the inverse of the covariance
    matrix).

    Methods
    -------
    pdf(x, df, scale)
        Probability density function.
    logpdf(x, df, scale)
        Log of the probability density function.
    rvs(df, scale, size=1)
        Draw random samples from a Wishart distribution.
    entropy()
        Compute the differential entropy of the Wishart distribution.

    Parameters
    ----------
    x : array_like
        Quantiles, with the last axis of `x` denoting the components.
    %(_doc_default_callparams)s

    Alternatively, the object may be called (as a function) to fix the degrees
    of freedom and scale parameters, returning a "frozen" Wishart random
    variable:

    rv = wishart(df=1, scale=1)
        - Frozen object with the same methods but holding the given
          degrees of freedom and scale fixed.

    Notes
    -----
    %(_doc_callparams_note)s

    The scale matrix `scale` must be a symmetric positive definite
    matrix. Singular matrices, including the symmetric positive semi-definite
    case, are not supported.

    The Wishart distribution is often denoted

    .. math::

        W_p(\nu, \Sigma)

    where :math:`\nu` is the degrees of freedom and :math:`\Sigma` is the
    :math:`p \times p` scale matrix.

    The probability density function for `wishart` has support over positive
    definite matrices :math:`S`; if :math:`S \sim W_p(\nu, \Sigma)`, then
    its PDF is given by:

    .. math::

        f(S) = \frac{|S|^{\frac{\nu - p - 1}{2}}}{2^{ \frac{\nu p}{2} } |\Sigma|^\frac{\nu}{2} \Gamma_p \left ( \frac{\nu}{2} \right )}
        e^{-tr(\Sigma^{-1} S) / 2}

    If :math:`S \sim W_p(\nu, \Sigma)` (Wishart) then
    :math:`S^{-1} \sim W_p^{-1}(\nu, \Sigma^{-1})` (inverse Wishart).

    If the scale matrix is 1-dimensional and equal to one, then the Wishart
    distribution :math:`W_1(\nu, 1)` collapses to the :math:`\chi^2(\nu)`
    distribution.

    .. versionadded:: 0.15.0

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from scipy.stats import wishart, chi2
    >>> x = np.linspace(1e-5, 8, 100)
    >>> w = wishart.pdf(x, df=3, scale=1); w[:5]
    array([ 0.00126156,  0.10892176,  0.14793434,  0.17400548,  0.1929669 ])
    >>> c = chi2.pdf(x, 3); c[:5]
    array([ 0.00126156,  0.10892176,  0.14793434,  0.17400548,  0.1929669 ])
    >>> plt.plot(x, w)

    The input quantiles can be any shape of array, as long as the last
    axis labels the components.

    References
    ----------
    .. [1] Eaton, Morris L. 2007.
       "Multivariate Statistics: A Vector Space Approach."
       Lecture Notes-Monograph Series 53 (January): i - 512.
    .. [2] Smith, W. B., and R. R. Hocking. 1972.
       "Algorithm AS 53: Wishart Variate Generator."
       Journal of the Royal Statistical Society.
       Series C (Applied Statistics) 21 (3): 341-45. doi:10.2307/2346290.

    """

    def __init__(self):
        self.__doc__ = doccer.docformat(self.__doc__, wishart_docdict_params)

    def __call__(self, df=None, scale=None):
        """
        Create a frozen Wishart distribution.

        See `wishart_frozen` for more information.

        """
        return wishart_frozen(df, scale)

    def _process_parameters(self, df, scale):
        if scale is None:
            scale = 1.0
        scale = np.asarray(scale, dtype=float)

        if scale.ndim == 0:
            scale = scale[np.newaxis,np.newaxis]
        elif scale.ndim == 1:
            scale = np.diag(scale)
        elif scale.ndim == 2 and not scale.shape[0] == scale.shape[1]:
            raise ValueError("Array 'scale' must be square if it is two"
                             " dimensional, but scale.scale = %s."
                             % str(scale.shape))
        elif scale.ndim > 2:
            raise ValueError("Array 'scale' must be at most two-dimensional,"
                             " but scale.ndim = %d" % scale.ndim)

        dim = scale.shape[0]

        if df is None:
            df = dim
        elif not np.isscalar(df):
            raise ValueError("Degrees of freedom must be a scalar.")
        elif df < dim:
            raise ValueError("Degrees of freedom cannot be less than dimension"
                             " of scale matrix, but df = %d" % df)

        return dim, df, scale

    def _process_quantiles(self, x, dim):
        """
        Adjust quantiles array so that last axis labels the components of
        each data point.
        """
        x = np.asarray(x, dtype=float)

        if x.ndim == 0:
            x = x * np.eye(dim)[:, :, np.newaxis]
        if x.ndim == 1:
            if dim == 1:
                x = x[np.newaxis, np.newaxis, :]
            else:
                x = np.diag(x)[:, :, np.newaxis]
        elif x.ndim == 2:
            if not x.shape[0] == x.shape[1]:
                raise ValueError("Quantiles must be square if they are two"
                                 " dimensional, but x.shape = %s."
                                 % str(x.shape))
            x = x[:, :, np.newaxis]
        elif x.ndim == 3:
            if not x.shape[0] == x.shape[1]:
                raise ValueError("Quantiles must be square in the first two"
                                 " dimensions if they are three dimensional"
                                 ", but x.shape = %s." % str(x.shape))
        elif x.ndim > 3:
            raise ValueError("Quantiles must be at most two-dimensional with"
                             " an additional dimension for multiple"
                             "components, but x.ndim = %d" % x.ndim)

        # Now we have 3-dim array; should have shape [dim, dim, *]
        if not x.shape[0:2] == (dim, dim):
            raise ValueError('Quantiles have incompatible dimensions: should'
                             ' be %s, got %s.' % ((dim, dim), x.shape[0:2]))

        return x

    def _process_size(self, size):
        size = np.array(size, dtype=float)

        if size.ndim == 0:
            size = size[np.newaxis]
        elif size.ndim > 1:
            raise ValueError('Size must be an integer or tuple of integers;'
                             ' thus must have dimension <= 1.'
                             ' Got size.ndim = %s' % str(tuple(size)))
        n = size.prod()
        shape = tuple(size)
        
        return n, shape

    def _logpdf(self, x, dim, df, scale, log_det_scale, C):
        """
        Parameters
        ----------
        x : ndarray
            Points at which to evaluate the log of the probability
            density function
        dim : int
            Dimension of the scale matrix
        df : int
            Degrees of freedom
        scale : ndarray
            Scale matrix
        log_det_scale : float
            Logarithm of the determinant of the scale matrix
        C : ndarray
            Cholesky factorization of the scale matrix, lower triagular.

        Notes
        -----
        As this function does no argument checking, it should not be
        called directly; use 'logpdf' instead.

        """
        # log determinant of x
        # Note: x has components along the last axis, so that x.T has
        # components alone the 0-th axis. Then since det(A) = det(A'), this
        # gives us a 1-dim vector of determinants
        # TODO slogdet is unavailable as long as Numpy 1.5.x is supported
        # s, log_det_x = np.linalg.slogdet(x.T)

        # Retrieve tr(scale^{-1} x)
        log_det_x = np.zeros(x.shape[-1])
        scale_inv_x = np.zeros(x.shape)
        tr_scale_inv_x = np.zeros(x.shape[-1])
        for i in range(x.shape[-1]):
            log_det_x[i] = np.log(np.linalg.det(x[:,:,i]))
            scale_inv_x[:,:,i] = scipy.linalg.cho_solve((C, True), x[:,:,i])
            tr_scale_inv_x[i] = scale_inv_x[:,:,i].trace()

        # Log PDF
        out = (
            (0.5 * (df - dim - 1) * log_det_x - 0.5 * tr_scale_inv_x) -
            (0.5 * df * dim * _LOG_2 + 0.5 * df * log_det_scale + multigammaln(0.5*df, dim))
        )

        return out

    def logpdf(self, x, df, scale):
        """
        Log of the Wishart probability density function.

        Parameters
        ----------
        x : array_like
            Quantiles, with the last axis of `x` denoting the components.
            Each quantile must be a symmetric positive definite matrix.
        %(_doc_default_callparams)s

        Notes
        -----
        %(_doc_callparams_note)s

        Returns
        -------
        pdf : ndarray
            Log of the probability density function evaluated at `x`

        """
        dim, df, scale = self._process_parameters(df, scale)
        x = self._process_quantiles(x, dim)

        # Cholesky decomposition of scale, get log(det(scale))
        C = scipy.linalg.cholesky(scale, lower=True)
        log_det_scale = 2 * np.sum(np.log(C.diagonal()))

        out = self._logpdf(x, dim, df, scale, log_det_scale, C)
        return _squeeze_output(out)

    def pdf(self, x, df, scale):
        """
        Wishart probability density function.

        Parameters
        ----------
        x : array_like
            Quantiles, with the last axis of `x` denoting the components.
            Each quantile must be a symmetric positive definite matrix.
        %(_doc_default_callparams)s

        Notes
        -----
        %(_doc_callparams_note)s

        Returns
        -------
        pdf : ndarray
            Probability density function evaluated at `x`

        """
        return np.exp(self.logpdf(x, df, scale))

    def _mean(self, dim, df, scale):
        """
        Parameters
        ----------
        dim : int
            Dimension of the scale matrix
        %(_doc_default_callparams)s

        Notes
        -----
        As this function does no argument checking, it should not be
        called directly; use 'mean' instead.

        """
        return df * scale

    def mean(self, df, scale):
        """
        Mean of the Wishart distribution

        Parameters
        ----------
        %(_doc_default_callparams)s

        Returns
        -------
        mean : float
            The mean of the distribution
        """
        dim, df, scale = self._process_parameters(df, scale)
        out = self._mean(dim, df, scale)
        return _squeeze_output(out)

    def _mode(self, dim, df, scale):
        """
        Parameters
        ----------
        dim : int
            Dimension of the scale matrix
        %(_doc_default_callparams)s

        Notes
        -----
        As this function does no argument checking, it should not be
        called directly; use 'mode' instead.

        """
        if df >= dim + 1:
            out = (df-dim-1) * scale
        else:
            out = None
        return out

    def mode(self, df, scale):
        """
        Mode of the Wishart distribution

        Only valid if the degrees of freedom are greater than the dimension of
        the scale matrix.

        Parameters
        ----------
        %(_doc_default_callparams)s

        Returns
        -------
        mode : float or None
            The Mode of the distribution
        """
        dim, df, scale = self._process_parameters(df, scale)
        out = self._mode(dim, df, scale)
        return _squeeze_output(out) if out is not None else out

    def _var(self, dim, df, scale):
        """
        Parameters
        ----------
        dim : int
            Dimension of the scale matrix
        %(_doc_default_callparams)s

        Notes
        -----
        As this function does no argument checking, it should not be
        called directly; use 'var' instead.

        """
        var = scale**2
        diag = scale.diagonal()  # 1 x dim array
        var += np.outer(diag, diag)
        var *= df
        return var

    def var(self, df, scale):
        """
        Variance of the Wishart distribution

        Parameters
        ----------
        %(_doc_default_callparams)s

        Returns
        -------
        var : float
            The variance of the distribution
        """
        dim, df, scale = self._process_parameters(df, scale)
        out = self._var(dim, df, scale)
        return _squeeze_output(out)

    def _standard_rvs(self, n, shape, dim, df):
        """
        Parameters
        ----------
        n : integer
            Number of variates to generate
        shape : iterable
            Shape of the variates to generate
        dim : int
            Dimension of the scale matrix
        df : int
            Degrees of freedom

        Notes
        -----
        As this function does no argument checking, it should not be
        called directly; use 'rvs' instead.

        """
        # Random normal variates for off-diagonal elements
        n_tril = (dim*(dim-1)/2)
        covariances = np.random.normal(size=n*n_tril).reshape(shape+(n_tril,))

        # Random chi-square variates for diagonal elements
        variances = np.r_[
            [np.random.chisquare(df-(i+1)+1, size=n)**0.5 for i in range(dim)]
        ].reshape((dim,) + shape[::-1]).T

        # Create the A matri(ces) - lower triangular
        A = np.zeros(shape + (dim, dim))

        # Input the covariances
        size_idx = tuple([slice(None,None,None)]*len(shape))
        tril_idx = np.tril_indices(dim, k=-1)
        A[size_idx + tril_idx] = covariances

        # Input the variances
        diag_idx = np.diag_indices(dim)
        A[size_idx + diag_idx] = variances

        return A

    def _rvs(self, n, shape, dim, df, C):
        """
        Parameters
        ----------
        n : integer
            Number of variates to generate
        shape : iterable
            Shape of the variates to generate
        dim : int
            Dimension of the scale matrix
        df : int
            Degrees of freedom
        scale : ndarray
            Scale matrix
        C : ndarray
            Cholesky factorization of the scale matrix, lower triagular.

        Notes
        -----
        As this function does no argument checking, it should not be
        called directly; use 'rvs' instead.

        """
        # Calculate the matrices A, which are actually lower triangular
        # Cholesky factorizations of a matrix B such that B ~ W(df, I)
        A = self._standard_rvs(n, shape, dim, df)

        # Calculate SA = C A A' C', where SA ~ W(df, scale)
        # Note: this is the product of a (lower) (lower) (lower)' (lower)'
        #       or, denoting B = AA', it is C B C' where C is the lower
        #       triangular Cholesky factorization of the scale matrix.
        #       this appears to conflict with the instructions in [1]_, which
        #       suggest that it should be D' B D where D is the lower
        #       triangular factorization of the scale matrix. However, it is
        #       meant to refer to the Bartlett (1933) representation of a
        #       Wishart random variate as L A A' L' where L is lower triangular
        #       so it appears that understanding D' to be upper triangular
        #       is either a typo in or misreading of [1]_.
        for index in np.ndindex(shape):
            CA = np.dot(C, A[index])
            A[index] = np.dot(CA, CA.T)

        return A

    def rvs(self, df, scale, size=1):
        """
        Draw random samples from a Wishart distribution.

        Parameters
        ----------
        %(_doc_default_callparams)s
        size : integer or iterable of integers, optional
            Number of samples to draw (default 1).

        Notes
        -----
        %(_doc_callparams_note)s

        Returns
        -------
        rvs : ndarray
            Random variates of shape (`size`) + (`dim`, `dim), where `dim` is
            the dimension of the scale matrix.

        """
        n, shape = self._process_size(size)
        dim, df, scale = self._process_parameters(df, scale)

        # Cholesky decomposition of scale
        C = scipy.linalg.cholesky(scale, lower=True)

        out = self._rvs(n, shape, dim, df, C)
        return _squeeze_output(out)

    def _entropy(self, dim, df, log_det_scale):
        """
        Parameters
        ----------
        dim : int
            Dimension of the scale matrix
        df : int
            Degrees of freedom
        log_det_scale : float
            Logarithm of the determinant of the scale matrix

        Notes
        -----
        As this function does no argument checking, it should not be
        called directly; use 'entropy' instead.

        """
        return (
            0.5 * (dim+1) * log_det_scale +
            0.5 * dim * (dim+1) * _LOG_2 +
            multigammaln(0.5*df, dim) -
            0.5 * (df - dim - 1) * np.sum(
                [psi(0.5*(df + 1 - (i+1))) for i in range(dim)]
            ) + 
            0.5 * df * dim
        )

    def entropy(self, df, scale):
        """
        Compute the differential entropy of the Wishart.

        Parameters
        ----------
        %(_doc_default_callparams)s

        Notes
        -----
        %(_doc_callparams_note)s

        Returns
        -------
        h : scalar
            Entropy of the Wishart distribution

        """
        dim, df, scale = self._process_parameters(df, scale)

        # TODO replace with np.linalg.slogdet when Numpy 1.5.x not necessary
        log_det_scale = np.log(np.linalg.det(scale))
        
        return self._entropy(dim, df, log_det_scale)
wishart = wishart_gen()


class wishart_frozen(object):
    """
    Create a frozen Wishart distribution.

    Parameters
    ----------
    df : array_like
        Degrees of freedom of the distribution
    scale : array_like
        Scale matrix of the distribution
    """
    def __init__(self, df, scale):
        self._wishart = wishart_gen()
        self.dim, self.df, self.scale = self._wishart._process_parameters(
            df, scale
        )
        self.C = scipy.linalg.cholesky(self.scale, lower=True)
        self.log_det_scale = 2 * np.sum(np.log(self.C.diagonal()))

    def logpdf(self, x):
        x = self._wishart._process_quantiles(x, self.dim)

        out = self._wishart._logpdf(x, self.dim, self.df, self.scale,
                                    self.log_det_scale, self.C)
        return _squeeze_output(out)

    def pdf(self, x):
        return np.exp(self.logpdf(x))

    def mean(self):
        out = self._wishart._mean(self.dim, self.df, self.scale)
        return _squeeze_output(out)

    def mode(self):
        out = self._wishart._mode(self.dim, self.df, self.scale)
        return _squeeze_output(out) if out is not None else out

    def var(self):
        out = self._wishart._var(self.dim, self.df, self.scale)
        return _squeeze_output(out)

    def rvs(self, size=1):
        n, shape = self._wishart._process_size(size)
        out = self._wishart._rvs(n, shape,
                                 self.dim, self.df, self.C)
        return _squeeze_output(out)

    def entropy(self):
        return self._wishart._entropy(self.dim, self.df, self.log_det_scale)

# Set frozen generator docstrings from corresponding docstrings in
# Wishart and fill in default strings in class docstrings
for name in ['logpdf', 'pdf', 'mean', 'mode', 'var', 'rvs', 'entropy']:
    method = wishart_gen.__dict__[name]
    method_frozen = wishart_frozen.__dict__[name]
    method_frozen.__doc__ = doccer.docformat(
        method.__doc__, wishart_docdict_noparams)
    method.__doc__ = doccer.docformat(method.__doc__, wishart_docdict_params)


from numpy import asarray_chkfinite, asarray
from scipy.linalg.misc import LinAlgError
from scipy.linalg.lapack import get_lapack_funcs
def _cho_inv_batch(a, check_finite=True):
    """
    Invert the matrices a_i, using a Cholesky factorization of A, where
    a_i resides in the last two dimensions of a and the other indices describe
    the index i.

    Overwrites the data in a.

    Parameters
    ----------
    a : array
        Array of matrices to invert, where the matrices themselves are stored
        in the last two dimensions.
    check_finite : boolean, optional
        Whether to check that the input matrices contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.
    Returns
    -------
    x : array
        Array of inverses of the matrices a_i
    See also
    --------
    scipy.linalg.cholesky : Cholesky factorization of a matrix
    """
    if check_finite:
        a1 = asarray_chkfinite(a)
    else:
        a1 = asarray(a)
    if len(a1.shape) < 2 or a1.shape[-2] != a1.shape[-1]:
        raise ValueError('expected square matrix in last two dimensions')

    potrf, potri = get_lapack_funcs(('potrf','potri'), (a1,))

    tril_idx = np.tril_indices(a.shape[-2], k=-1)
    triu_idx = np.triu_indices(a.shape[-2], k=1)
    for index in np.ndindex(a1.shape[:-2]):

        # Cholesky decomposition
        a1[index], info = potrf(a1[index], lower=True, overwrite_a=False,
                                clean=False)
        if info > 0:
            raise LinAlgError("%d-th leading minor not positive definite"
                              % info)
        if info < 0:
            raise ValueError('illegal value in %d-th argument of internal'
                             ' potrf' % -info)
        # Inversion
        a1[index], info = potri(a1[index], lower=True, overwrite_c=False)
        if info > 0:
            raise LinAlgError("the inverse could not be computed")
        if info < 0:
            raise ValueError('illegal value in %d-th argument of internal'
                             ' potrf' % -info)

        # Make symmetric (dpotri only fills in the lower triangle)
        a1[index][triu_idx] = a1[index][tril_idx]

    return a1

class invwishart_gen(wishart_gen):
    r"""
    An inverse Wishart random variable.

    The `df` keyword specifies the degrees of freedom. The `scale` keyword
    specifies the scale matrix, which must be symmetric and positive definite.
    In this context, the scale matrix is often interpreted in terms of a
    multivariate normal covariance matrix.

    Methods
    -------
    pdf(x, df, scale)
        Probability density function.
    logpdf(x, df, scale)
        Log of the probability density function.
    rvs(df, scale, size=1)
        Draw random samples from an inverse Wishart distribution.

    Parameters
    ----------
    x : array_like
        Quantiles, with the last axis of `x` denoting the components.
    %(_doc_default_callparams)s

    Alternatively, the object may be called (as a function) to fix the degrees
    of freedom and scale parameters, returning a "frozen" inverse Wishart
    random variable:

    rv = invwishart(df=1, scale=1)
        - Frozen object with the same methods but holding the given
          degrees of freedom and scale fixed.

    Notes
    -----
    %(_doc_callparams_note)s

    The scale matrix `scale` must be a symmetric positive definite
    matrix. Singular matrices, including the symmetric positive semi-definite
    case, are not supported.

    The inverse Wishart distribution is often denoted

    .. math::

        W_p^{-1}(\nu, \Psi)

    where :math:`\nu` is the degrees of freedom and :math:`\Psi` is the
    :math:`p \times p` scale matrix.

    The probability density function for `invwishart` has support over positive
    definite matrices :math:`S`; if :math:`S \sim W^{-1}_p(\nu, \Sigma)`,
    then its PDF is given by:

    .. math::

        f(S) = \frac{|\Sigma|^\frac{\nu}{2}}{2^{ \frac{\nu p}{2} } |S|^{\frac{\nu + p + 1}{2}} \Gamma_p \left ( \frac{\nu}{2} \right )}
        e^{-tr(\Sigma S^{-1}) / 2}

    If :math:`S \sim W_p^{-1}(\nu, \Psi)` (inverse Wishart) then
    :math:`S^{-1} \sim W_p(\nu, \Psi^{-1})` (Wishart).

    If the scale matrix is 1-dimensional and equal to one, then the inverse
    Wishart distribution :math:`W_1(\nu, 1)` collapses to the
    inverse Gamma distribution with parameters :math:`shape = \frac{\nu}{2}`
    and :math:`scale=\frac{1}{2}`.

    .. versionadded:: 0.15.0

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from scipy.stats import invwishart, invgamma
    >>> x = np.linspace(0.01, 1, 100)
    >>> iw = invwishart.pdf(x, df=6, scale=1); iw[:3]
    array([  1.20546865e-15,   5.42497807e-06,   4.45813929e-03])
    >>> ig = invgamma.pdf(x, 6/2., scale=1./2); c[:3]
    array([  1.20546865e-15,   5.42497807e-06,   4.45813929e-03])
    >>> plt.plot(x, iw)

    The input quantiles can be any shape of array, as long as the last
    axis labels the components.

    References
    ----------
    .. [1] Eaton, Morris L. 2007.
       "Multivariate Statistics: A Vector Space Approach."
       Lecture Notes-Monograph Series 53 (January): i - 512.
    .. [2] Jones, M. C. 1985.
       "Generating Inverse Wishart Matrices."
       Communications in Statistics - Simulation and Computation 14 (2):511-14.
    """

    def __init__(self):
        self.__doc__ = doccer.docformat(self.__doc__, wishart_docdict_params)

    def __call__(self, df=None, scale=None):
        """
        Create a frozen inverse Wishart distribution.

        See `invwishart_frozen` for more information.

        """
        return invwishart_frozen(df, scale)

    def _logpdf(self, x, dim, df, scale, log_det_scale):
        """
        Parameters
        ----------
        x : ndarray
            Points at which to evaluate the log of the probability
            density function.
        dim : int
            Dimension of the scale matrix
        df : int
            Degrees of freedom
        scale : ndarray
            Scale matrix
        log_det_scale : float
            Logarithm of the determinant of the scale matrix

        Notes
        -----
        As this function does no argument checking, it should not be
        called directly; use 'logpdf' instead.

        """
        log_det_x = np.zeros(x.shape[-1])
        #scale_x_inv = np.zeros(x.shape)
        x_inv = np.copy(x).T
        if dim > 1:
            _cho_inv_batch(x_inv)  # works in-place
        else:
            x_inv = 1./x_inv
        tr_scale_x_inv = np.zeros(x.shape[-1])

        for i in range(x.shape[-1]):
            C, lower = scipy.linalg.cho_factor(x[:,:,i], lower=True)

            log_det_x[i] = 2 * np.sum(np.log(C.diagonal()))

            #scale_x_inv[:,:,i] = scipy.linalg.cho_solve((C, True), scale).T
            tr_scale_x_inv[i] = np.dot(scale, x_inv[i]).trace()

        # Log PDF
        out = (
            (0.5 * df * log_det_scale - 0.5 * tr_scale_x_inv) -
            (0.5 * df * dim * _LOG_2 + 0.5 * (df + dim + 1) * log_det_x) -
            multigammaln(0.5*df, dim)
        )

        return out

    def logpdf(self, x, df, scale):
        """
        Log of the inverse Wishart probability density function.

        Parameters
        ----------
        x : array_like
            Quantiles, with the last axis of `x` denoting the components.
            Each quantile must be a symmetric positive definite matrix.
        %(_doc_default_callparams)s

        Notes
        -----
        %(_doc_callparams_note)s

        Returns
        -------
        pdf : ndarray
            Log of the probability density function evaluated at `x`

        """
        dim, df, scale = self._process_parameters(df, scale)
        x = self._process_quantiles(x, dim)

        # TODO replace with np.linalg.slogdet when Numpy 1.5.x not necessary
        log_det_scale = np.log(np.linalg.det(scale))

        out = self._logpdf(x, dim, df, scale, log_det_scale)
        return _squeeze_output(out)

    def pdf(self, x, df, scale):
        """
        Inverse Wishart probability density function.

        Parameters
        ----------
        x : array_like
            Quantiles, with the last axis of `x` denoting the components.
            Each quantile must be a symmetric positive definite matrix.

        %(_doc_default_callparams)s

        Notes
        -----
        %(_doc_callparams_note)s

        Returns
        -------
        pdf : ndarray
            Probability density function evaluated at `x`

        """
        return np.exp(self.logpdf(x, df, scale))

    def _mean(self, dim, df, scale):
        """
        Parameters
        ----------
        dim : int
            Dimension of the scale matrix
        %(_doc_default_callparams)s

        Notes
        -----
        As this function does no argument checking, it should not be
        called directly; use 'mean' instead.

        """
        if df > dim + 1:
            out = scale / (df - dim - 1)
        else:
            out = None
        return out

    def mean(self, df, scale):
        """
        Mean of the inverse Wishart distribution

        Only valid if the degrees of freedom are greater than the dimension of
        the scale matrix plus one.

        Parameters
        ----------
        %(_doc_default_callparams)s

        Returns
        -------
        mean : float or None
            The mean of the distribution
        """
        dim, df, scale = self._process_parameters(df, scale)
        out = self._mean(dim, df, scale)
        return _squeeze_output(out) if out is not None else out

    def _mode(self, dim, df, scale):
        """
        Parameters
        ----------
        dim : int
            Dimension of the scale matrix
        %(_doc_default_callparams)s

        Notes
        -----
        As this function does no argument checking, it should not be
        called directly; use 'mode' instead.

        """
        return scale / (df + dim + 1)

    def mode(self, df, scale):
        """
        Mode of the inverse Wishart distribution

        Parameters
        ----------
        %(_doc_default_callparams)s

        Returns
        -------
        mode : float
            The Mode of the distribution
        """
        dim, df, scale = self._process_parameters(df, scale)
        out = self._mode(dim, df, scale)
        return _squeeze_output(out)

    def _var(self, dim, df, scale):
        """
        Parameters
        ----------
        dim : int
            Dimension of the scale matrix
        %(_doc_default_callparams)s

        Notes
        -----
        As this function does no argument checking, it should not be
        called directly; use 'var' instead.
        """
        if df > dim + 3:
            var = (df - dim + 1) * scale**2
            diag = scale.diagonal()  # 1 x dim array
            var += (df - dim - 1) * np.outer(diag, diag)
            var /= (df - dim) * (df - dim - 1)**2 * (df - dim - 3)
        else:
            var = None
        return var

    def var(self, df, scale):
        """
        Variance of the inverse Wishart distribution

        Only valid if the degrees of freedom are greater than the dimension of
        the scale matrix plus three.

        Parameters
        ----------
        %(_doc_default_callparams)s

        Returns
        -------
        var : float
            The variance of the distribution
        """
        dim, df, scale = self._process_parameters(df, scale)
        out = self._var(dim, df, scale)
        return _squeeze_output(out) if out is not None else out

    def _rvs(self, n, shape, dim, df, C):
        """
        Parameters
        ----------
        n : integer
            Number of variates to generate
        shape : iterable
            Shape of the variates to generate
        dim : int
            Dimension of the scale matrix
        df : int
            Degrees of freedom
        C : ndarray
            Cholesky factorization of the scale matrix, lower triagular.

        Notes
        -----
        As this function does no argument checking, it should not be
        called directly; use 'rvs' instead.

        """
        # Get random draws A such that A ~ W(df, I)
        A = super(invwishart_gen, self)._standard_rvs(n, shape, dim, df)

        # Calculate SA = (CA)'^{-1} (CA)^{-1} ~ iW(df, scale)
        eye = np.eye(dim)
        trtrs = get_lapack_funcs(('trtrs'), (A,))

        for index in np.ndindex(A.shape[:-2]):
            # Calculate CA
            CA = np.dot(C, A[index])
            # Get (C A)^{-1} via triangular solver
            if dim > 1:
                CA, info = trtrs(CA, eye, lower=True)
                if info > 0:
                    raise LinAlgError("Singular matrix.")
                if info < 0:
                    raise ValueError('Illegal value in %d-th argument of'
                                     ' internal trtrs' % -info)
            else:
                CA = 1. / CA
            # Get SA
            A[index] = np.dot(CA.T, CA)

        return A

    def rvs(self, df, scale, size=1):
        """
        Draw random samples from an inverse Wishart distribution.

        Parameters
        ----------
        %(_doc_default_callparams)s
        size : integer or iterable of integers, optional
            Number of samples to draw (default 1).

        Notes
        -----
        %(_doc_callparams_note)s

        Returns
        -------
        rvs : ndarray
            Random variates of shape (`size`) + (`dim`, `dim), where `dim` is
            the dimension of the scale matrix.

        """
        n, shape = self._process_size(size)
        dim, df, scale = self._process_parameters(df, scale)

        # Invert the scale
        eye = np.eye(dim)
        L, lower = scipy.linalg.cho_factor(scale, lower=True)
        inv_scale = scipy.linalg.cho_solve((L, lower), eye)
        # Cholesky decomposition of inverted scale
        C = scipy.linalg.cholesky(inv_scale, lower=True)

        out = self._rvs(n, shape, dim, df, C)

        return _squeeze_output(out)

    def entropy(self):
        # Need to find reference for inverse Wishart entropy
        raise AttributeError

invwishart = invwishart_gen()

class invwishart_frozen(object):
    def __init__(self, df, scale):
        """
        Create a frozen inverse Wishart distribution.

        Parameters
        ----------
        df : array_like
            Degrees of freedom of the distribution
        scale : array_like
            Scale matrix of the distribution
        """
        self._invwishart = invwishart_gen()
        self.dim, self.df, self.scale = self._invwishart._process_parameters(
            df, scale
        )

        # Get the determinant via Cholesky factorization
        C, lower = scipy.linalg.cho_factor(self.scale, lower=True)
        self.log_det_scale = 2 * np.sum(np.log(C.diagonal()))

        # Get the inverse using the Cholesky factorization
        eye = np.eye(self.dim)
        self.inv_scale = scipy.linalg.cho_solve((C, lower), eye)

        # Get the Cholesky factorization of the inverse scale
        self.C = scipy.linalg.cholesky(self.inv_scale, lower=True)

    def logpdf(self, x):
        x = self._invwishart._process_quantiles(x, self.dim)
        out = self._invwishart._logpdf(x, self.dim, self.df, self.scale,
                                       self.log_det_scale)
        return _squeeze_output(out)

    def pdf(self, x):
        return np.exp(self.logpdf(x))

    def mean(self):
        out = self._invwishart._mean(self.dim, self.df, self.scale)
        return _squeeze_output(out) if out is not None else out

    def mode(self):
        out = self._invwishart._mode(self.dim, self.df, self.scale)
        return _squeeze_output(out)

    def var(self):
        out = self._invwishart._var(self.dim, self.df, self.scale)
        return _squeeze_output(out) if out is not None else out

    def rvs(self, size=1):
        n, shape = self._invwishart._process_size(size)

        out = self._invwishart._rvs(n, shape, self.dim, self.df, self.C)

        return _squeeze_output(out)

    def entropy(self):
        # Need to find reference for inverse Wishart entropy
        raise AttributeError

# Set frozen generator docstrings from corresponding docstrings in
# inverse Wishart and fill in default strings in class docstrings
for name in ['logpdf', 'pdf', 'mean', 'mode', 'var', 'rvs']:
    method = invwishart_gen.__dict__[name]
    method_frozen = wishart_frozen.__dict__[name]
    method_frozen.__doc__ = doccer.docformat(
        method.__doc__, wishart_docdict_noparams)
    method.__doc__ = doccer.docformat(method.__doc__, wishart_docdict_params)
