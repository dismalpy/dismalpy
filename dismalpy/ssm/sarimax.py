"""
SARIMAX Model

Author: Chad Fulton
License: Simplified-BSD
"""
from __future__ import division, absolute_import, print_function

import numpy as np
from .mlemodel import MLEMixin, MLEResultsMixin
from statsmodels.tsa.statespace import mlemodel, sarimax
import statsmodels.base.wrapper as wrap

class SARIMAX(MLEMixin, sarimax.SARIMAX):
    def filter(self, params, transformed=True, cov_type=None, return_ssm=False,
               **kwargs):
        params = np.array(params)

        # Transform parameters if necessary
        if not transformed:
            params = self.transform_params(params)
            transformed = True

        # Get the state space output
        results = super(SARIMAX, self).filter(params, transformed, cov_type,
                                              return_ssm=True, **kwargs)

        # Wrap in a results object
        if not return_ssm:
            result_kwargs = {}
            if cov_type is not None:
                result_kwargs['cov_type'] = cov_type
            results = SARIMAXResultsWrapper(
                SARIMAXResults(self, params, results, **result_kwargs)
            )

        return results

    def smooth(self, params, transformed=True, cov_type=None, return_ssm=False,
               **kwargs):
        params = np.array(params)

        if not transformed:
            params = self.transform_params(params)
            transformed = True

        # Get the state space output
        results = super(SARIMAX, self).smooth(params, transformed, cov_type,
                                              return_ssm=True, **kwargs)

        # Wrap in a results object
        if not return_ssm:
            result_kwargs = {}
            if cov_type is not None:
                result_kwargs['cov_type'] = cov_type
            results = SARIMAXResultsWrapper(
                SARIMAXResults(self, params, results, **result_kwargs)
            )

        return results


class SARIMAXResults(MLEResultsMixin, sarimax.SARIMAXResults):
    pass


class SARIMAXResultsWrapper(mlemodel.MLEResultsWrapper):
    _attrs = {}
    _wrap_attrs = wrap.union_dicts(mlemodel.MLEResultsWrapper._wrap_attrs,
                                   _attrs)
    _methods = {}
    _wrap_methods = wrap.union_dicts(mlemodel.MLEResultsWrapper._wrap_methods,
                                     _methods)
wrap.populate_wrapper(SARIMAXResultsWrapper, SARIMAXResults)
