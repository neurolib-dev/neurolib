"""
FitzHugh–Nagumo model.

Main references:
    FitzHugh, R. (1955). Mathematical models of threshold phenomena in the
    nerve membrane. The bulletin of mathematical biophysics, 17(4), 257-278.

    Nagumo, J., Arimoto, S., & Yoshizawa, S. (1962). An active pulse
    transmission line simulating nerve axon. Proceedings of the IRE, 50(10),
    2061-2070.

Additional reference:
    Kostova, T., Ravindran, R., & Schonbek, M. (2004). FitzHugh–Nagumo
    revisited: Types of bifurcations, periodical forcing and stability regions
    by a Lyapunov functional. International journal of bifurcation and chaos,
    14(03), 913-925.
"""

import numpy as np
import symengine as se
from jitcdde import input as system_input

from ..builder.base.network import Network, Node
from ..builder.base.neural_mass import NeuralMass

DEFAULT_PARAMS = {
    "alpha": 3.0,
    "beta": 4.0,
    "gamma": -1.5,
    "delta": 0.0,
    "epsilon": 0.5,
    "tau": 20.0,
    "ext_input_x": 1.0,
    "ext_input_y": 0.0,
}


class FitzHughNagumoMass(NeuralMass):
    """
    FitzHugh-Nagumo neural mass.
    """

    name = "FitzHugh-Nagumo mass"
    label = "FHNmass"

    num_state_variables = 2
    num_noise_variables = 2
    coupling_variables = {0: "x", 1: "y"}
    state_variable_names = ["x", "y"]
    required_params = [
        "alpha",
        "beta",
        "gamma",
        "delta",
        "epsilon",
        "tau",
        "ext_input_x",
        "ext_input_y",
    ]
    required_couplings = ["network_x", "network_y"]

    def __init__(self, params=None):
        super().__init__(params=params or DEFAULT_PARAMS)

    def _initialize_state_vector(self):
        """
        Initialize state vector.
        """
        self.initial_state = [0.05 * np.random.uniform(0, 1)] * self.num_state_variables

    def _derivatives(self, coupling_variables):
        [x, y] = self._unwrap_state_vector()

        d_x = (
            -self.params["alpha"] * x ** 3
            + self.params["beta"] * x ** 2
            + self.params["gamma"] * x
            - y
            + coupling_variables["network_x"]
            + system_input(self.noise_input_idx[0])
            + self.params["ext_input_x"]
        )

        d_y = (
            (x - self.params["delta"] - self.params["epsilon"] * y) / self.params["tau"]
            + coupling_variables["network_y"]
            + system_input(self.noise_input_idx[1])
            + self.params["ext_input_y"]
        )

        return [d_x, d_y]


class FitzHughNagumoNetworkNode(Node):
    """
    Default FitzHugh-Nagumo node with 1 neural mass modelled as FitzHugh-Nagumo
    oscillator.
    """

    name = "FitzHugh-Nagumo node"
    label = "FHNnode"

    default_network_coupling = {"network_x": 0.0, "network_y": 0.0}
    default_output = "x"

    def __init__(self, params=None):
        """
        :param params: parameters of the FitzHugh-Nagumo mass
        :type params: dict|None
        """
        fhn_mass = FitzHughNagumoMass(params)
        fhn_mass.index = 0
        super().__init__(neural_masses=[fhn_mass])

    def _sync(self):
        return []


class FitzHughNagumoNetwork(Network):
    """
    Whole brain network of FitzHugh-Nagumo oscillators.
    """

    name = "FitzHugh-Nagumo network"
    label = "FHNnet"

    sync_variables = ["network_x", "network_y"]

    def __init__(
        self, connectivity_matrix, delay_matrix, mass_params=None, x_coupling="diffusive", y_coupling="none",
    ):
        """
        :param connectivity_matrix: connectivity matrix for between nodes
            coupling, typically DTI structural connectivity, matrix as [to,
            from]
        :type connectivity_matrix: np.ndarray
        :param delay_matrix: delay matrix between nodes, typically derived from
            length matrix, if None, delays are all zeros, in ms, matrix as
            [to, from]
        :type delay_matrix: np.ndarray|None
        :param mass_params: parameters for each Hopf normal form neural
            mass, if None, will use default
        :type mass_params: list[dict]|dict|None
        :param x_coupling: how to couple `x` variables in the nodes,
            "diffusive", "additive", or "none"
        :type x_coupling: str
        :param y_coupling: how to couple `y` variables in the nodes,
            "diffusive", "additive", or "none"
        :type y_coupling: str
        """
        mass_params = self._prepare_mass_params(mass_params, connectivity_matrix.shape[0])

        nodes = []
        for i, node_params in enumerate(mass_params):
            node = FitzHughNagumoNetworkNode(params=node_params)
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