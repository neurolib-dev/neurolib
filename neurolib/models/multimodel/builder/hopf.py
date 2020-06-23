"""
Hopf normal form model.

References:
    Kuznetsov, Y. A. (2013). Elements of applied bifurcation theory (Vol. 112).
    Springer Science & Business Media.

    Deco, G., Cabral, J., Woolrich, M. W., Stevner, A. B., Van Hartevelt, T. J.,
    & Kringelbach, M. L. (2017). Single or multiple frequency generators in
    on-going brain activity: A mechanistic whole-brain model of empirical MEG
    data. Neuroimage, 152, 538-550.
"""

import numpy as np
import symengine as se
from jitcdde import input as system_input

from ..builder.base.network import Network, Node
from ..builder.base.neural_mass import NeuralMass

DEFAULT_PARAMS = {
    "a": 0.25,
    "omega": 0.2,
    "ext_input_x": 0.0,
    "ext_input_y": 0.0,
}


class HopfMass(NeuralMass):
    """
    Hopf normal form (Landau-Stuart oscillator).
    """

    name = "Hopf normal form mass"
    label = "HopfMass"

    num_state_variables = 2
    num_noise_variables = 2
    coupling_variables = {0: "x", 1: "y"}
    state_variable_names = ["x", "y"]
    required_parameters = ["a", "omega", "ext_input_x", "ext_input_y"]
    required_couplings = ["network_x", "network_y"]

    def __init__(self, parameters=None):
        super().__init__(parameters=parameters or DEFAULT_PARAMS)

    def _initialize_state_vector(self):
        """
        Initialize state vector.
        """
        self.initial_state = [0.5 * np.random.uniform(-1, 1)] * self.num_state_variables

    def _derivatives(self, coupling_variables):
        [x, y] = self._unwrap_state_vector()

        d_x = (
            (self.parameters["a"] - x ** 2 - y ** 2) * x
            - self.parameters["omega"] * y
            + coupling_variables["network_x"]
            + system_input(self.noise_input_idx[0])
            + self.parameters["ext_input_x"]
        )

        d_y = (
            (self.parameters["a"] - x ** 2 - y ** 2) * y
            + self.parameters["omega"] * x
            + coupling_variables["network_y"]
            + system_input(self.noise_input_idx[1])
            + self.parameters["ext_input_y"]
        )

        return [d_x, d_y]


class HopfNetworkNode(Node):
    """
    Default Hopf normal form node with 1 neural mass modelled as Landau-Stuart
    oscillator.
    """

    name = "Hopf normal form node"
    label = "HopfNode"

    default_network_coupling = {"network_x": 0.0, "network_y": 0.0}
    default_output = "x"

    def __init__(self, parameters=None):
        """
        :param parameters: parameters of the Hopf mass
        :type parameters: dict|None
        """
        hopf_mass = HopfMass(parameters)
        hopf_mass.index = 0
        super().__init__(neural_masses=[hopf_mass])

    def _sync(self):
        return []


class HopfNetwork(Network):
    """
    Whole brain network of Hopf normal form oscillators.
    """

    name = "Hopf normal form network"
    label = "HopfNet"

    sync_variables = ["network_x", "network_y"]

    def __init__(
        self, connectivity_matrix, delay_matrix, mass_parameters=None, x_coupling="diffusive", y_coupling="none",
    ):
        """
        :param connectivity_matrix: connectivity matrix for between nodes
            coupling, typically DTI structural connectivity, matrix as [from,
            to]
        :type connectivity_matrix: np.ndarray
        :param delay_matrix: delay matrix between nodes, typically derived from
            length matrix, if None, delays are all zeros, in ms, matrix as
            [from, to]
        :type delay_matrix: np.ndarray|None
        :param mass_parameters: parameters for each Hopf normal form neural
            mass, if None, will use default
        :type mass_parameters: list[dict]|dict|None
        :param x_coupling: how to couple `x` variables in the nodes,
            "diffusive", "additive", or "none"
        :type x_coupling: str
        :param y_coupling: how to couple `y` variables in the nodes,
            "diffusive", "additive", or "none"
        :type y_coupling: str
        """
        mass_parameters = self._prepare_mass_parameters(mass_parameters, connectivity_matrix.shape[0])

        nodes = []
        for i, node_params in enumerate(mass_parameters):
            node = HopfNetworkNode(parameters=node_params)
            node.index = i
            node.idx_state_var = i * node.num_state_variables
            nodes.append(node)

        super().__init__(
            nodes=nodes, connectivity_matrix=connectivity_matrix, delay_matrix=delay_matrix,
        )
        # get all coupling variables
        all_couplings = [mass.coupling_variables for node in self.nodes for mass in node.masses]
        # assert they are the same
        assert all(all_couplings[0] == coupling for coupling in all_couplings)
        # invert as to name: idx
        self.coupling_symbols = {v: k for k, v in all_couplings[0].items()}

        self.x_coupling = x_coupling
        self.y_coupling = y_coupling

    def _couple(self, coupling_type, coupling_variable):
        assert coupling_variable in self.coupling_symbols
        if coupling_type == "additive":
            return self._additive_coupling(self.coupling_symbols[coupling_variable], f"network_{coupling_variable}",)
        elif coupling_type == "diffusive":
            return self._diffusive_coupling(self.coupling_symbols[coupling_variable], f"network_{coupling_variable}",)
        elif coupling_type == "none":
            return self._no_coupling(f"network_{coupling_variable}")
        else:
            raise ValueError(f"Unknown coupling type: {coupling_type}")

    def init_network(self):
        # create symbol for each node for input
        self.sync_symbols = {
            f"{symbol}_{node_idx}": se.Symbol(symbol)
            for symbol in self.sync_variables
            for node_idx in range(self.num_nodes)
        }
        for node_idx, node in enumerate(self.nodes):
            node.init_node(start_idx_for_noise=node_idx * node.num_noise_variables)
        assert all(node.initialised for node in self)
        self.initialised = True

    def _sync(self):
        return self._couple(self.x_coupling, "x") + self._couple(self.y_coupling, "y") + super()._sync()
