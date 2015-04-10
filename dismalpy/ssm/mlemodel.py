"""
State Space Model

Author: Chad Fulton
License: Simplified-BSD
"""
from __future__ import division, absolute_import, print_function

import numpy as np
import pandas as pd
from scipy.stats import norm
from .representation import Representation
from .kalman_filter import KalmanFilter, FilterResults
from .kalman_smoother import KalmanSmoother, SmootherResults
from .simulation_smoother import SimulationSmoother, SimulationSmoothResults

import statsmodels.tsa.base.tsa_model as tsbase
from statsmodels.tools.numdiff import approx_hess_cs, approx_fprime_cs
from statsmodels.tools.decorators import cache_readonly, resettable_cache
from statsmodels.tools.eval_measures import aic, bic, hqic

from .representation import Representation, FrozenRepresentation
from .kalman_filter import KalmanFilter, FilterResults
from .kalman_smoother import KalmanSmoother, SmootherResults
from .simulation_smoother import SimulationSmoother, SimulationSmoothResults

from statsmodels.tsa.statespace import kalman_filter, model, mlemodel, sarimax

class MLEModel(mlemodel.MLEModel, model.Model, SimulationSmoother, KalmanSmoother, KalmanFilter, Representation, kalman_filter.KalmanFilter):
    def __init__(self, *args, **kwargs):
        super(MLEModel, self).__init__(*args, **kwargs)
        self.results_class = MLEResults

class MLEResults(mlemodel.MLEResults, SmootherResults, FilterResults, FrozenRepresentation, kalman_filter.FilterResults):
    @property
    def kalman_gain(self):
        return self._kalman_gain
    @kalman_gain.setter
    def kalman_gain(self, value):
        self._kalman_gain = value
