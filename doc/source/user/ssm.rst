.. _ssm:

******************
State Space Models
******************

DismalPy provides functionality for describing, filtering, and parameter
estimation for state space models. Notation and implementation are largely
based on Durbin and Koopman (2012) [1]_.

Definition
----------

A (linear, Gaussian) state space model is one that can be represented as:

.. math::

  y_t & = Z_t \alpha_t + d_t + \varepsilon_t \\
  \alpha_t & = T_t \alpha_{t-1} + c_t + R_t \eta_t \\

where :math:`y_t` refers to the observation vector at time :math:`t`,
:math:`\alpha_t` refers to the (unobserved) state vector at time
:math:`t`, and where the irregular components are defined as

.. math::

  \varepsilon_t \sim N(0, H_t) \\
  \eta_t \sim N(0, Q_t) \\

The remaining variables (:math:`Z_t, d_t, H_t, T_t, c_t, R_t, Q_t`) in the
equations are matrices describing the process. Their variable names and
dimensions are as follows

Z : `design`          :math:`(k\_endog \times k\_states \times nobs)`

d : `obs_intercept`   :math:`(k\_endog \times nobs)`

H : `obs_cov`         :math:`(k\_endog \times k\_endog \times nobs)`

T : `transition`      :math:`(k\_states \times k\_states \times nobs)`

c : `state_intercept` :math:`(k\_states \times nobs)`

R : `selection`       :math:`(k\_states \times k\_posdef \times nobs)`

Q : `state_cov`       :math:`(k\_posdef \times k\_posdef \times nobs)`

In the case that one of the matrices is time-invariant (so that, for
example, :math:`Z_t = Z_{t+1} ~ \forall ~ t`), its last dimension may
be of size :math:`1` rather than size `nobs`.

Examples
--------

.. toctree::
   :maxdepth: 2

   _notebooks/sarimax_internet.ipynb
   _notebooks/sarimax_stata.ipynb
   _notebooks/local_linear_trend.ipynb

References
----------
.. [1] Durbin, James, and Siem Jan Koopman. 2012.
 Time Series Analysis by State Space Methods: Second Edition.
 Oxford University Press.