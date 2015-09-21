.. _ssm:

******************
State Space Models
******************

.. currentmodule:: dismalpy.ssm

DismalPy provides functionality for describing, filtering, and parameter
estimation for state space models.

Built-in models
===============

The following classes define built-in models:

.. toctree::
   :maxdepth: 2

   ssm/sarimax
   ssm/structural
   ssm/varmax
   ssm/dynamic_factor

Extension starting point
========================

Users wishing to specify and estimate custom state space models will typically
want to extend the following class:

.. toctree::
   :maxdepth: 2

   ssm/mlemodel

Base classes
============

The base state space model classes and tools are the following:

.. toctree::
   :maxdepth: 2

   ssm/representation
   ssm/kalman_filter
   ssm/kalman_smoother
   ssm/simulation_smoother
   ssm/model
   ssm/tools