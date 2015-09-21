"""
SARIMAX Model

Author: Chad Fulton
License: Simplified-BSD
"""
from __future__ import division, absolute_import, print_function

import numpy as np
from .mlemodel import MLEMixin, MLEResultsMixin
try:
    from statsmodels.tsa.statespace import varmax
    from statsmodels.tsa.statespace import mlemodel, structural
except ImportError:
    from .compat import mlemodel, structural

import statsmodels.base.wrapper as wrap

class UnobservedComponents(MLEMixin, structural.UnobservedComponents):
    def filter(self, params, transformed=True, cov_type=None, return_ssm=False,
               **kwargs):
        params = np.array(params, ndmin=1)

        # Transform parameters if necessary
        if not transformed:
            params = self.transform_params(params)
            transformed = True

        # Get the state space output
        result = super(UnobservedComponents, self).filter(
            params, transformed, cov_type, return_ssm=True, **kwargs)

        # Wrap in a results object
        if not return_ssm:
            result_kwargs = {}
            if cov_type is not None:
                result_kwargs['cov_type'] = cov_type
            result = UnobservedComponentsResultsWrapper(
                UnobservedComponentsResults(self, params, result,
                                            **result_kwargs)
            )

        return result

    def smooth(self, params, transformed=True, cov_type=None, return_ssm=False,
               **kwargs):
        params = np.array(params, ndmin=1)

        if not transformed:
            params = self.transform_params(params)
            transformed = True

        # Get the state space output
        result = super(UnobservedComponents, self).smooth(
            params, transformed, cov_type, return_ssm=True, **kwargs)

        # Wrap in a results object
        if not return_ssm:
            result_kwargs = {}
            if cov_type is not None:
                result_kwargs['cov_type'] = cov_type
            result = UnobservedComponentsResultsWrapper(
                UnobservedComponentsResults(self, params, result,
                                            **result_kwargs)
            )

        return result


class UnobservedComponentsResults(MLEResultsMixin,
                                  structural.UnobservedComponentsResults):
    @property
    def level(self):
        """
        Filtered and smoothed values of unobserved level component
        """
        out = super(UnobservedComponentsResults, self).level
        if out is not None:
            res = self.smoother_results
            out.smoothed = res.smoothed_state[out.offset]
            out.smoothed_cov = res.smoothed_state_cov[out.offset, out.offset]
        return out

    @property
    def trend(self):
        """
        Filtered and smoothed values of unobserved trend component
        """
        out = super(UnobservedComponentsResults, self).trend
        if out is not None:
            res = self.smoother_results
            out.smoothed = res.smoothed_state[out.offset]
            out.smoothed_cov = res.smoothed_state_cov[out.offset, out.offset]
        return out

    @property
    def seasonal(self):
        """
        Filtered and smoothed values of unobserved seasonal component
        """
        out = super(UnobservedComponentsResults, self).seasonal
        if out is not None:
            res = self.smoother_results
            out.smoothed = res.smoothed_state[out.offset]
            out.smoothed_cov = res.smoothed_state_cov[out.offset, out.offset]
        return out

    @property
    def cycle(self):
        """
        Filtered and smoothed values of unobserved cycle component
        """
        out = super(UnobservedComponentsResults, self).cycle
        if out is not None:
            res = self.smoother_results
            out.smoothed = res.smoothed_state[out.offset]
            out.smoothed_cov = res.smoothed_state_cov[out.offset, out.offset]
        return out

    @property
    def autoregressive(self):
        """
        Filtered and smoothed values of unobserved autoregressive component
        """
        out = super(UnobservedComponentsResults, self).autoregressive
        if out is not None:
            res = self.smoother_results
            out.smoothed = res.smoothed_state[out.offset]
            out.smoothed_cov = res.smoothed_state_cov[out.offset, out.offset]
        return out

    @property
    def regression_coefficients(self):
        """
        Filtered and smoothed values of unobserved regression coefficients
        """
        out = super(UnobservedComponentsResults, self).regression_coefficients
        if out is not None:
            res = self.filter_results
            start = self.smoother_results
            end = out.offset + self.specification.k_exog
            out.smoothed = res.smoothed_state[start:end]
            out.smoothed_cov = res.smoothed_state_cov[start:end, start:end]
        return out
    
    def plot_components(self, which='smoothed', *args, **kwargs):
        return super(UnobservedComponentsResults, self).plot_components(
            which, *args, **kwargs
        )


class UnobservedComponentsResultsWrapper(
        mlemodel.MLEResultsWrapper):
    _attrs = {}
    _wrap_attrs = wrap.union_dicts(
        mlemodel.MLEResultsWrapper._wrap_attrs, _attrs)
    _methods = {}
    _wrap_methods = wrap.union_dicts(
        mlemodel.MLEResultsWrapper._wrap_methods, _methods)
wrap.populate_wrapper(UnobservedComponentsResultsWrapper,
                      UnobservedComponentsResults)
