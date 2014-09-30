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
   FilterResults

Model
-----

.. currentmodule:: dismalpy.ssm.model

.. autosummary::
   :toctree: _generated/

   Model
   StatespaceResults

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