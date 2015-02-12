.. _ssm:

******************
State Space Models
******************

.. currentmodule:: dismalpy.ssm

DismalPy provides functionality for describing, filtering, and parameter
estimation for state space models.

Representation
--------------

.. currentmodule:: dismalpy.ssm.representation

.. autosummary::
   :toctree: _generated/

   Representation
   FrozenRepresentation

Kalman filter
-------------

.. currentmodule:: dismalpy.ssm.kalman_filter

.. autosummary::
   :toctree: _generated/

   KalmanFilter
   FilterResults

Kalman smoother 
---------------

.. currentmodule:: dismalpy.ssm.kalman_smoother

.. autosummary::
   :toctree: _generated/

   KalmanSmoother
   SmootherResults

Simulation smoother
-------------------

.. currentmodule:: dismalpy.ssm.simulation_smoother

.. autosummary::
   :toctree: _generated/

   SimulationSmoother
   SimulationSmoothResults

Model
-----

.. currentmodule:: dismalpy.ssm.model

.. autosummary::
   :toctree: _generated/

   Model

Maximum Likelihood Estimation
-----------------------------

.. currentmodule:: dismalpy.ssm.mlemodel

.. autosummary::
   :toctree: _generated/

   MLEModel
   MLEResults

SARIMAX
-------

.. currentmodule:: dismalpy.ssm.sarimax

.. autosummary::
   :toctree: _generated/

   SARIMAX
   SARIMAXResults

Tools
-----

.. currentmodule:: dismalpy.ssm.tools

.. autosummary::
   :toctree: _generated/

   companion_matrix
   diff
   is_invertible
   constrain_stationary_univariate
   unconstrain_stationary_univariate
   validate_matrix_shape
   validate_vector_shape