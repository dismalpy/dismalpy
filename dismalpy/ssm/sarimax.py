"""
SARIMAX Model

Author: Chad Fulton
License: Simplified-BSD
"""
from __future__ import division, absolute_import, print_function

from warnings import warn

import numpy as np
from .mlemodel import MLEModel, MLEResults
from .tools import (
    companion_matrix, diff, is_invertible, constrain_stationary_univariate,
    unconstrain_stationary_univariate
)
from scipy.linalg import solve_discrete_lyapunov
from statsmodels.tools.data import _is_using_pandas
from statsmodels.tsa.tsatools import lagmat
from statsmodels.tools.decorators import cache_readonly

from .representation import Representation, FrozenRepresentation
from .kalman_filter import KalmanFilter, FilterResults
from .kalman_smoother import KalmanSmoother, SmootherResults
from .simulation_smoother import SimulationSmoother, SimulationSmoothResults
from .mlemodel import MLEResults

from statsmodels.tsa.statespace import kalman_filter, model, mlemodel, sarimax

class SARIMAX(sarimax.SARIMAX, mlemodel.MLEModel, model.Model, SimulationSmoother, KalmanSmoother, KalmanFilter, Representation, kalman_filter.KalmanFilter):
    def __init__(self, *args, **kwargs):
        super(SARIMAX, self).__init__(*args, **kwargs)
        self.results_class = SARIMAXResults

class SARIMAXResults(sarimax.SARIMAXResults, MLEResults):
    pass
