import numpy as np
from jitcdde import input as system_input

from ..builder.base.network import Network, Node
from ..builder.base.neural_mass import NeuralMass

HOPF_DEFAULT_PARAMS = {
    "a": 0.25,
    "w": 0.2,
    "x_ext": 0.0,
    "y_ext": 0.0,
}


class HopfMass(NeuralMass):
    """
    Hopf normal form (Landau-Stuart oscillator).

    References:
        Landau, L. D. (1944). On the problem of turbulence. In Dokl. Akad. Nauk USSR
        (Vol. 44, p. 311).

        Stuart, J. T. (1960). On the non-linear mechanics of wave disturbances in
        stable and unstable parallel flows Part 1. The basic behaviour in plane
        Poiseuille flow. Journal of Fluid Mechanics, 9(3), 353-370.
    """

    name = "Hopf normal form mass"
    label = "HopfMass"

    num_state_variables = 2
    num_noise_variables = 2
    coupling_variables = {0: "x", 1: "y"}
    state_variable_names = ["x", "y"]
    required_params = ["a", "w", "x_ext", "y_ext"]
    required_couplings = ["network_x", "network_y"]

    def __init__(self, params=None, seed=None):
        super().__init__(params=params or HOPF_DEFAULT_PARAMS, seed=seed)

    def _initialize_state_vector(self):
        """
        Initialize state vector.
        """
        np.random.seed(self.seed)
        self.initial_state = (0.5 * np.random.uniform(-1, 1, size=(self.num_state_variables,))).tolist()

    def _derivatives(self, coupling_variables):
        [x, y] = self._unwrap_state_vector()

        d_x = (
            (self.params["a"] - x ** 2 - y ** 2) * x
            - self.params["w"] * y
            + coupling_variables["network_x"]
            + system_input(self.noise_input_idx[0])
            + self.params["x_ext"]
        )

        d_y = (
            (self.params["a"] - x ** 2 - y ** 2) * y
            + self.params["w"] * x
            + coupling_variables["network_y"]
            + system_input(self.noise_input_idx[1])
            + self.params["y_ext"]
        )

        return [d_x, d_y]


class HopfNode(Node):
    """
    Default Hopf normal form node with 1 neural mass modelled as Landau-Stuart
    oscillator.
    """

    name = "Hopf normal form node"
    label = "HopfNode"

    default_network_coupling = {"network_x": 0.0, "network_y": 0.0}
    default_output = "x"

    def __init__(self, params=None, seed=None):
        """
        :param params: parameters of the Hopf mass
        :type params: dict|None
        :param seed: seed for random number generator
        :type seed: int|None
        """
        hopf_mass = HopfMass(params, seed=seed)
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
    # define default coupling in Hopf network
    default_coupling = {"network_x": "diffusive", "network_y": "none"}

    def __init__(
        self, connectivity_matrix, delay_matrix, mass_params=None, seed=None,
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
        :param seed: seed for random number generator
        :type seed: int|None
        """
        mass_params = self._prepare_mass_params(mass_params, connectivity_matrix.shape[0])
        seeds = self._prepare_mass_params(seed, connectivity_matrix.shape[0], native_type=int)

        nodes = []
        for i, node_params in enumerate(mass_params):
            node = HopfNode(params=node_params, seed=seeds[i])
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
