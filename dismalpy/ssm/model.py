"""
State Space Model

Author: Chad Fulton
License: Simplified-BSD
"""
from __future__ import division, absolute_import, print_function

import numpy as np
import pandas as pd
from .. import model
from .representation import Representation
from .kalman_smoother import KalmanFilter
from .kalman_smoother import KalmanSmoother, SmootherResults
from .simulation_smoother import SimulationSmoother, SimulationSmoothResults


class Model(model.Model, SimulationSmoother, KalmanSmoother, KalmanFilter,
            Representation):
    """
    State space model

    Parameters
    ----------
    endog : array_like, iterable, str
        Endogenous data or names of endogenous variables.

    See Also
    --------
    dismalpy.model.Model
    """
    def __init__(self, endog, k_states, nobs=None, **kwargs):
        # Initialize the generic Model
        super(Model, self).__init__(endog, nobs)

        # Initialize the Representation
        kwargs['nobs'] = self.nobs
        SimulationSmoother.__init__(self, self.k_endog, k_states, **kwargs)

    def bind(self, endog):
        # Bind the data using the generic Model
        super(Model, self).bind(endog)

        # Call additional bind checks specific to the Representation
        SimulationSmoother.bind(self, self.endog)
