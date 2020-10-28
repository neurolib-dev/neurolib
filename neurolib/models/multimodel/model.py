import logging

import numpy as np

from ...utils.collections import dotdict, flatten_nested_dict, star_dotdict
from ..model import Model
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
        assert isinstance(self.default_output, str), "`default_output` must be a string."

        # create parameters
        self.params = self._set_model_params()

        # TODO resolve how to integrate in neurolib's fashion
        self.integration = None
        self.init_vars = None

        # create output and state dictionary
        self.outputs = dotdict({})
        self.state = dotdict({})
        self.maxDelay = None
        self.initializeRun()

        self.boldInitialized = False

        logging.info(f"{self.name}: Model initialized.")

    def _set_model_params(self):
        """
        Set all necessary model parameters.
        """
        params = star_dotdict(flatten_nested_dict(self.model_instance.get_nested_params()))
        params.update(DEFAULT_RUN_PARAMS)
        params["name"] = self.model_instance.label
        params["description"] = self.model_instance.name
        if isinstance(self.model_instance, Node):
            params.update({"N": 1, "Cmat": np.zeros((1, 1))})
        else:
            params.update({"N": len(self.model_instance.nodes), "Cmat": self.model_instance.connectivity})
        return params

    def initializeRun(self, initializeBold=False):
        pass

    def getMaxDelay(self):
        pass

    def run(self):
        pass

    def integrate(self, append_outputs, simulate_bold):
        result = self.model_instance.run(
            duration=self.params["duration"],
            dt=self.params["dt"],
            noise_input=None,
            backend=self.params["backend"],
            return_xarray=True,
        )
