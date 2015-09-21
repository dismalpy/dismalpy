"""
VARMAX Model

Author: Chad Fulton
License: Simplified-BSD
"""
from __future__ import division, absolute_import, print_function

import numpy as np
from .mlemodel import MLEMixin, MLEResultsMixin
try:
    from statsmodels.tsa.statespace import varmax
    from statsmodels.tsa.statespace import mlemodel
except ImportError:
    from .compat import mlemodel, varmax
import statsmodels.base.wrapper as wrap

class VARMAX(MLEMixin, varmax.VARMAX):
    def filter(self, params, transformed=True, cov_type=None, return_ssm=False,
               **kwargs):
        params = np.array(params, ndmin=1)

        # Transform parameters if necessary
        if not transformed:
            params = self.transform_params(params)
            transformed = True

        # Get the state space output
        results = super(VARMAX, self).filter(params, transformed, cov_type,
                                             return_ssm=True, **kwargs)

        # Wrap in a results object
        if not return_ssm:
            result_kwargs = {}
            if cov_type is not None:
                result_kwargs['cov_type'] = cov_type
            results = VARMAXResultsWrapper(
                VARMAXResults(self, params, results, **result_kwargs)
            )

        return results

    def smooth(self, params, transformed=True, cov_type=None, return_ssm=False,
               **kwargs):
        params = np.array(params, ndmin=1)

        if not transformed:
            params = self.transform_params(params)
            transformed = True

        # Get the state space output
        results = super(VARMAX, self).smooth(params, transformed, cov_type,
                                             return_ssm=True, **kwargs)

        # Wrap in a results object
        if not return_ssm:
            result_kwargs = {}
            if cov_type is not None:
                result_kwargs['cov_type'] = cov_type
            results = VARMAXResultsWrapper(
                VARMAXResults(self, params, results, **result_kwargs)
            )

        return results


class VARMAXResults(MLEResultsMixin, varmax.VARMAXResults):
    pass


class VARMAXResultsWrapper(mlemodel.MLEResultsWrapper):
    _attrs = {}
    _wrap_attrs = wrap.union_dicts(
        mlemodel.MLEResultsWrapper._wrap_attrs, _attrs)
    _methods = {}
    _wrap_methods = wrap.union_dicts(
        mlemodel.MLEResultsWrapper._wrap_methods, _methods)
wrap.populate_wrapper(VARMAXResultsWrapper, VARMAXResults)
