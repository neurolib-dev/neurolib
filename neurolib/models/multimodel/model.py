import logging

import numpy as np
from chspy import join

from ...utils.collections import dotdict, flat_dict_to_nested, flatten_nested_dict, star_dotdict
from ..model import Model
from .builder.base.constants import NETWORK_CONNECTIVITY, NETWORK_DELAYS
from .builder.base.network import Network, Node

# default run parameters for MultiModels
DEFAULT_RUN_PARAMS = {"duration": 2000, "dt": 0.1, "seed": None, "backend": "jitcdde"}


class MultiModel(Model):
    """
    Base for all MultiModels i.e. heterogeneous networks or network nodes built
    using model builder.
    """

    @classmethod
    def init_node(cls, node):
        """
        Init model class from node.

        :param node: initialised network node from MultiModel builder
        :type node: `neurolib.models.multimodel.builder.base.network.Node`
        """
        assert isinstance(node, Node)
        node.index = 0
        node.idx_state_var = 0
        node.init_node()
        return cls(node)

    def __init__(self, model_instance):
        assert isinstance(model_instance, (Node, Network))
        assert model_instance.initialised
        self.model_instance = model_instance

        # set model attributes
        self.name = self.model_instance.label
        self.state_vars = self.model_instance.state_variable_names
        self.default_output = self.model_instance.default_output
        self.output_vars = self.model_instance.output_vars
        assert isinstance(self.default_output, str), "`default_output` must be a string."

        # create parameters
        self.params = self._set_model_params()

        self.integration = None
        self.init_vars = None

        # create output and state dictionary
        self.outputs = dotdict({})
        self.state = dotdict({})
        self.maxDelay = None
        self.initializeRun()

        self.boldInitialized = False
        self.params["sampling_dt"] = self.params["sampling_dt"] or self.params["dt"]

        logging.info(f"{self.name}: Model initialized.")

    def _set_model_params(self):
        """
        Set all necessary model parameters.
        """
        params = star_dotdict(flatten_nested_dict(self.model_instance.get_nested_params()))
        # all matrices to floats
        for k, v in params.items():
            if isinstance(v, np.ndarray):
                params[k] = v.astype(float)
        params.update(DEFAULT_RUN_PARAMS)
        params["name"] = self.model_instance.label
        params["description"] = self.model_instance.name
        if isinstance(self.model_instance, Node):
            params.update({"N": 1, "Cmat": np.zeros((1, 1))})
        else:
            params.update({"N": len(self.model_instance.nodes), "Cmat": self.model_instance.connectivity.astype(float)})
        return params

    def _sync_model_params(self):
        """
        Pulls params from `model_instance` and updates self.params.
        """
        new_model_params = flatten_nested_dict(self.model_instance.get_nested_params())
        self.params.update(new_model_params)

    def getMaxDelay(self):
        """
        Return max delay in units of dt. In ms, this is given as a property in the model instance.
        """
        return int(np.around(self.model_instance.max_delay / self.params["dt"]))

    def _update_model_params(self):
        params_to_update = {k: v for k, v in self.params.items() if self.model_instance.label in k}
        self.model_instance.update_params(flat_dict_to_nested(params_to_update))
        # re-set the model params
        self._sync_model_params()

    @property
    def num_noise_variables(self):
        return self.model_instance.num_noise_variables

    @property
    def num_state_variables(self):
        return self.model_instance.num_state_variables

    @property
    def noise_input(self):
        return self.model_instance.noise_input

    @noise_input.setter
    def noise_input(self, new_noise):
        # first - update model params to save any changes
        self._update_model_params()
        # change the noise in the model instance
        self.model_instance.noise_input = new_noise
        # re-set the model params - to write new noise params into self.params
        self._sync_model_params()

    def run(
        self,
        chunkwise=False,
        chunksize=None,
        bold=False,
        append_outputs=False,
        continue_run=False,
        noise_input=None,
    ):
        self._update_model_params()

        # if a previous run is not to be continued clear the model's state
        if continue_run:
            self.setInitialValuesToLastState()
        else:
            self.clearModelState()

        self.initializeRun(initializeBold=bold)

        if chunkwise is False:
            self.integrate(append_outputs=append_outputs, simulate_bold=bold, noise_input=noise_input)

        else:
            if chunksize is None:
                chunksize = int(2000 / self.params["dt"])
            # check if model is safe for chunkwise integration
            self.checkChunkwise(chunksize)
            if bold and not self.boldInitialized:
                logging.warn(f"{self.name}: BOLD model not initialized, not simulating BOLD. Use `run(bold=True)`")
                bold = False
            self.integrateChunkwise(chunksize=chunksize, bold=bold, append_outputs=append_outputs)

        # check if there was a problem with the simulated data
        self.checkOutputs()

    def _init_noise_inputs(self, backend):
        """
        Build noise / stimulus input to the model.
        """
        if backend == "jitcdde":
            init_func = lambda noise: noise.as_cubic_splines(
                duration=self.params["duration"], dt=self.params["sampling_dt"]
            )
            join_func = lambda x: join(*x)
        elif backend == "numba":
            init_func = lambda noise: noise.as_array(duration=self.params["duration"], dt=self.params["dt"])
            join_func = lambda x: np.vstack(x)
        else:
            raise ValueError(f"Unknown backend {backend}")
        # initialise each noise / stimulation process and join
        return join_func([init_func(noise) for noise in self.model_instance.noise_input])

    def integrate(self, append_outputs=False, simulate_bold=False, noise_input=None):
        """
        :param noise_input: custom noise input if desired, if None, will use
            default, it's type depends on backend:
            - for `numba` backend as np.ndarray
            - for `jitcdde` backend as interpolated Cubic Hermite Splines
                (`chspy.CubicHermiteSpline`)
        :type noise_input: np.ndarray|chspy.CubicHermiteSpline
        """
        if self.params["backend"] == "jitcdde":
            # jitcdde has adaptive time step, so actually its `dt` is `sampling_dt`
            dt = self.params["sampling_dt"]
            self.sample_every = 1
        else:
            dt = self.params["dt"]
        if noise_input is None:
            noise_input = self._init_noise_inputs(self.params["backend"])
        result = self.model_instance.run(
            duration=self.params["duration"],
            dt=dt,
            noise_input=noise_input,
            backend=self.params["backend"],
            return_xarray=True,
        )
        self.storeOutputsAndStates(result, append=append_outputs)

        # bold simulation after integration
        if simulate_bold and self.boldInitialized:
            self.simulateBold(result[self.default_output].values.T, append=True)

    def setInitialValuesToLastState(self):
        if not hasattr(self, "t"):
            raise ValueError("You tried using continue_run=True on the first run.")
        new_initial_state = np.zeros((self.model_instance.initial_state.shape[0], self.maxDelay + 1))
        total_vars_counter = 0
        for node_idx, node_vars in enumerate(self.state_vars):
            for var in node_vars:
                new_initial_state[total_vars_counter, :] = self.state[var][node_idx, :]
                total_vars_counter += 1
        # set initial state
        self.model_instance.initial_state = new_initial_state

    def integrateChunkwise(self, chunksize, bold, append_outputs):
        raise NotImplementedError("for now...")

    def storeOutputsAndStates(self, results, append):
        # save time array
        self.setOutput("t", results.time.values, append=append, removeICs=False)
        self.setStateVariables("t", results.time.values)
        # save outputs
        for variable in results:
            if variable in self.output_vars:
                self.setOutput(variable, results[variable].values.T, append=append, removeICs=False)
            self.setStateVariables(variable, results[variable].values.T)
