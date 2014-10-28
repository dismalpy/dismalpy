"""
Model

Author: Chad Fulton
License: Simplified-BSD
"""

import numpy as np
import pandas as pd

class Model(object):
    """
    `endog` is one of:
    - a name (str)
        k_endog = 1
        nobs = 0
        names = [name]
        endog = np.zeros((k_endog,0))
    - iterable of names (str)
        k_endog = len(iterable)
        nobs = 0
        names = list(iterable)
        endog = np.zeros((k_endog,0))
    - iterable of data (non-str), assumed given shape = (nobs, k_endog) or (nobs,)
        k_endog = np.array(iterable).shape[1] or 1
        nobs = np.array(iterable).shape[0]
        names = [id(iterable)_i for i in range(k_endog)]
        endog = np.array(iterable).T
    - ndarray of data, assumed given shape = (nobs, k_endog) or (nobs,)
        k_endog = ndarray.shape[1] or 1
        nobs = ndarray.shape[0]
        names = [id(ndarray)_i for i in range(k_endog)]
    - Pandas series
        k_endog = 1
        nobs = series.shape[0]
        names = [series.name] or [id(series)]
    - Pandas dataframe
        k_endog = dataframe.shape[1]
        nobs = dataframe.shape[0]
        names = series.columns
    """
    def __init__(self, endog, nobs=None, *args, **kwargs):
        self.endog = None

        if type(endog) == str:
            self._endog_names = [endog]
            self.k_endog = 1
            self.nobs = None
            endog = None
        elif isinstance(endog, pd.DataFrame):
            self._endog_names = endog.columns.tolist()
            self.nobs, self.k_endog = endog.shape
        elif isinstance(endog, pd.Series):
            self._endog_names = [endog.name] if endog.name is not None else [id(endog)]
            self.k_endog = 1
            self.nobs = endog.shape[0]
        else:
            if not isinstance(endog, np.ndarray):
                endog = np.array(endog)
            if np.issubdtype(endog.dtype, str):
                self._endog_names = list(endog)
                self.k_endog = len(self._endog_names)
                self.nobs = None
                endog = None
            else:
                endog_id = id(endog)
                if endog.ndim == 1:
                    self.nobs = endog.shape[0]
                    self.k_endog = 1
                else:
                    self.nobs, self.k_endog = endog.shape
                self._endog_names = ['%s_%s' % (endog_id, i)
                                    for i in range(self.k_endog)]

        # We may already know `nobs`, even if we don't want to bind to data yet
        if nobs is not None:
            if not self.nobs is None and not self.nobs == nobs:
                raise ValueError('Provided `nobs` is inconsistent with given'
                                 ' endogenous array. Got %d and %d,'
                                 ' respectively' % (self.nobs, nobs))
            self.nobs = nobs

        # If we were actually given data, bind the data to this instance
        if endog is not None:
            self.bind(endog)

    def bind(self, endog):
        """
        Bind endogenous data to this instance
        """
        # If we were given a Pandas object, make sure the names match
        # and optionally extract only those columns we need
        if isinstance(endog, pd.Series) and endog.name is not None:
            if not endog.name == self._endog_names[0]:
                raise ValueError('Name of the provided endogenous array does'
                                 ' not match the given endogenous name.'
                                 ' Got %s, required %s'
                                 % (endog.name, self._endog_names[0]))
            if not isinstance(endog.index, pd.DatetimeIndex):
                raise ValueError("Given a pandas object and the index does "
                                 "not contain dates")
            self._dates = endog.index
        elif isinstance(endog, pd.DataFrame):
            # Keep only the required columns, and in the required order
            endog = endog[self._endog_names]
            if not isinstance(endog.index, pd.DatetimeIndex):
                raise ValueError("Given a pandas object and the index does "
                                 "not contain dates")
            self._dates = endog.index

        # Explicitly copy / convert to a new ndarray
        # Note: we assume that the given endog array is nobs x k_endog, but
        # _statespace assumes it is k_endog x nobs. Thus we create it in the
        # transposed shape as order "C" and then transpose to get order "F".
        if np.ndim(endog) == 1:
            endog = np.array(endog, ndmin=2, order="F")
        else:
            endog = np.array(endog, order="C").T

        # Check that this fits the k_endog dimension that we previously had
        if not endog.shape[0] == self.k_endog:
            raise ValueError('Provided endogenous array does has the required'
                             ' number of columns. Got %d, required %d)'
                             % (endog.shape[0], self.k_endog))

        # If we were provided a strict nobs in construction, make sure it
        # matches
        if self.nobs is not None and not endog.shape[1] == self.nobs:
            raise ValueError('Provided endogenous array is inconsistent with'
                             ' given `nobs` . Got %d and %d, respectively'
                             % (endog.shape[1], self.nobs))

        # Set the new dimension data
        self.nobs = endog.shape[1]

        # Set the new data
        self.endog = endog
