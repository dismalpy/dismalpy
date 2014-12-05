"""
Vector Autoregression

Author: Chad Fulton
License: Simplified-BSD
"""

import numpy as np
from _var import _var_quantities, _var_quantities_kron

class VAR(object):
    def __init__(self, nobs=None, k_endog=None, endog=None, order=1,
                 *args, **kwargs):

        # Get the endog dimensions
        self.order = order
        if endog is not None:
            endog = np.array(endog, order='F')
            self.nobs, self.k_endog = endog.shape
            self.endog = endog
        else:
            if nobs is None or k_endog is None:
                raise ValueError('Must either provide endogenous array or'
                                 ' endogenous dimensions.')
            self.nobs = nobs
            self.k_endog = k_endog

        # Set additional dimensions
        # self.k_endog                         # M
        # self.order                           # d
        self.k_ar = self.k_endog * self.order  # k_i
        self.k_var = self.k_ar * self.k_endog  # k = \sum_i k_i

        # Create storage arrays
        # Z is M x k, H is M x M
        # _ZH actually stores (Z'H)', see _var for details
        self._ZH = np.zeros((self.k_endog, self.k_var), order='F')  # M x k
        self._ZHZ = np.zeros((self.k_var, self.k_var), order='F')   # k x k
        self._ZHy = np.zeros((self.k_var,), order='F')              # k x 0
        self._precision = None

        # Create flags
        self._recalculate = True

    @property
    def endog(self):
        return self._endog
    @endog.setter
    def endog(self, value):
        self._endog = np.array(value, order='A')

        # (T x M)
        if (self.nobs, self.k_endog) == self._endog.shape:
            self._endog = self._endog.T
        # (M x T)
        elif (self.k_endog, self.nobs) == self._endog.shape:
            pass
        else:
            raise ValueError('Invalid endogenous array shape. Required'
                             '(%d, %d) or (%d, %d). Got %s'
                             % (self.nobs, self.k_endog, self.k_endog,
                                self.nobs, str(self._endog.shape)))

        if not self._endog.flags['F_CONTIGUOUS']:
                self._endog = np.asfortranarray(self._endog)

        # Create a new lag matrix, shaped (k_ar, nobs) = (k_ar, T)
        self._lagged = np.asfortranarray(np.hstack([
            self.endog[:, self.order-i:-i].T
            for i in range(1, self.order+1)
        ]).T)

        # Set calculation flags
        self._recalculate = True

    @property
    def precision(self):
        return self._precision
    @precision.setter
    def precision(self, value):
        self._precision = np.array(value, order='F')
        if not (self.k_endog, self.k_endog) == self._precision.shape:
            raise ValueError('Invalid precision shape. Required (%d, %d), got'
                             ' %s' % (self.k_endog, self.k_endog,
                             str(self._precision.shape)))

        # Set calculation flags
        self._recalculate = True

    @property
    def quantities(self):
        if self._recalculate:
            if self._precision is None:
                raise RuntimeError('Quantities cannot be calculated unless'
                                   ' precision matrix is set.')

            _var_quantities_kron(
                self._ZH, self._ZHZ, self._ZHy,
                self._endog[:, self.order:], self._lagged, self._precision,
                self.k_endog, self.k_ar, self.k_var
            )
        return np.copy(self._ZHy), np.copy(self._ZHZ)
