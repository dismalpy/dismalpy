"""
State Space Representation, Kalman Filter, Smoother, and Simulation Smoother

Author: Chad Fulton
License: Simplified-BSD
"""
from __future__ import division, absolute_import, print_function

import numpy as np
from .kalman_smoother import KalmanSmoother
from .tools import prefix_simulation_smoother_map

SIMULATION_STATE = 0x01
SIMULATION_DISTURBANCE = 0x04
SIMULATION_ALL = (
    SIMULATION_STATE | SIMULATION_DISTURBANCE
)

class SimulationSmoother(KalmanSmoother):
    r"""
    State space representation of a time series process, with Kalman filter
    and smoother, and with simulation smoother.
    """

    simulation_outputs = [
        'simulation_state', 'simulation_disturbance', 'simulation_all'
    ]
    
    def __init__(self, *args, **kwargs):
        super(SimulationSmoother, self).__init__(*args, **kwargs)

        self.simulation_smooth_results_class = kwargs.get('simulation_smooth_results_class', SimulationSmoothResults)

    def get_simulation_output(self, simulation_output=None, **kwargs):
        # If we don't explicitly have simulation_output, try to get it from
        # kwargs
        if simulation_output is None:
            simulation_output = 0

            if 'simulation_state' in kwargs and kwargs['simulation_state']:
                simulation_output |= SIMULATION_STATE
            if 'simulation_disturbance' in kwargs and kwargs['simulation_disturbance']:
                simulation_output |= SIMULATION_DISTURBANCE
            if 'simulation_all' in kwargs and kwargs['simulation_all']:
                simulation_output |= SIMULATION_ALL

            # If no information was in kwargs, set simulation output to be the
            # same as smoother output
            if simulation_output == 0:
                simulation_output = self.smoother_output

        return simulation_output

    def simulation_smoother(self, simulation_output=None,
                            results_class=None, *args, **kwargs):
        """
        Retrieve a simulation smoother for the statespace model.

        Parameters
        ----------
        simulation_output : int, optional
            Determines which simulation smoother output is calculated.
            Default is all (including state and disturbances).
        Returns
        -------
        SimulationSmoothResults
        """

        # Set the class to be the default results class, if None provided
        if results_class is None:
            results_class = self.simulation_smooth_results_class

        # Instantiate a new results object
        if not issubclass(results_class, SimulationSmoothResults):
            raise ValueError('Invalid results class provided.')

        # Make sure we have the required Statespace representation
        prefix, dtype, create_statespace = self._initialize_representation(
            *args, **kwargs
        )

        # Simulation smoother parameters
        simulation_output = self.get_simulation_output(simulation_output, **kwargs)

        # Kalman smoother parameters
        smoother_output = kwargs.get('smoother_output', simulation_output)

        # Kalman filter parameters
        filter_method = kwargs.get('filter_method', self.filter_method)
        inversion_method = kwargs.get('inversion_method', self.inversion_method)
        stability_method = kwargs.get('stability_method', self.stability_method)
        conserve_memory = kwargs.get('conserve_memory', self.conserve_memory)
        loglikelihood_burn = kwargs.get('loglikelihood_burn', self.loglikelihood_burn)
        tolerance = kwargs.get('tolerance', self.tolerance)

        # Create a new simulation smoother object
        cls = prefix_simulation_smoother_map[prefix]
        simulation_smoother = cls(
            self._statespaces[prefix],
            filter_method, inversion_method, stability_method, conserve_memory,
            tolerance, loglikelihood_burn, smoother_output, simulation_output
        )

        # Create results object
        results = results_class(self, simulation_smoother, *args, **kwargs)

        return results


class SimulationSmoothResults(object):
    
    def __init__(self, model, simulation_smoother, *args, **kwargs):
        self.model = model
        self.prefix = model.prefix
        self.dtype = model.dtype
        self._simulation_smoother = simulation_smoother

        # Output
        self._generated_obs = None
        self._generated_state = None
        self._simulated_state = None
        self._simulated_measurement_disturbance = None
        self._simulated_state_disturbance = None

    @property
    def simulation_output(self):
        return self._simulation_smoother.simulation_output
    @simulation_output.setter
    def simulation_output(self, value):
        self._simulation_smoother.simulation_output = value

    @property
    def generated_obs(self):
        if self._generated_obs is None:
            self._generated_obs = np.array(
                self._simulation_smoother.generated_obs, copy=True
            )
        return self._generated_obs

    @property
    def generated_state(self):
        if self._generated_state is None:
            self._generated_state = np.array(
                self._simulation_smoother.generated_state, copy=True
            )
        return self._generated_state

    @property
    def simulated_state(self):
        if self._simulated_state is None:
            self._simulated_state = np.array(
                self._simulation_smoother.simulated_state, copy=True
            )
        return self._simulated_state

    @property
    def simulated_measurement_disturbance(self):
        if self._simulated_measurement_disturbance is None:
            self._simulated_measurement_disturbance = np.array(
                self._simulation_smoother.simulated_measurement_disturbance,
                copy=True
            )
        return self._simulated_measurement_disturbance

    @property
    def simulated_state_disturbance(self):
        if self._simulated_state_disturbance is None:
            self._simulated_state_disturbance = np.array(
                self._simulation_smoother.simulated_state_disturbance,
                copy=True
            )
        return self._simulated_state_disturbance

    def simulate(self, simulation_output=-1, disturbance_variates=None,
                 initial_state_variates=None, *args, **kwargs):
        # Clear any previous output
        self._generated_obs = None
        self._generated_state = None
        self._simulated_state = None
        self._simulated_measurement_disturbance = None
        self._simulated_state_disturbance = None

        # Re-initialize the _statespace representation
        self.model._initialize_representation(prefix=self.prefix, *args, **kwargs)

        # Initialize the state
        self.model._initialize_state(prefix=self.prefix)

        # Draw the (independent) random variates for disturbances in the
        # simulation
        if disturbance_variates is not None:
            self._simulation_smoother.set_disturbance_variates(
                np.array(disturbance_variates, dtype=self.dtype)
            )
        else:
            self._simulation_smoother.draw_disturbance_variates()

        # Draw the (independent) random variates for the initial states in the
        # simulation
        if initial_state_variates is not None:
            self._simulation_smoother.set_initial_state_variates(
                np.array(initial_state_variates, dtype=self.dtype)
            )
        else:
            self._simulation_smoother.draw_initial_state_variates()

        # Perform simulation smoothing
        # Note: simulation_output=-1 corresponds to whatever was setup when
        # the simulation smoother was constructed
        self._simulation_smoother.simulate(simulation_output)
