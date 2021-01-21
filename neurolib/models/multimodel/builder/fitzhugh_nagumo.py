import numpy as np
from jitcdde import input as system_input

from ....utils.stimulus import OrnsteinUhlenbeckProcess
from ..builder.base.network import Network, Node
from ..builder.base.neural_mass import NeuralMass

FHN_DEFAULT_PARAMS = {
    "alpha": 3.0,
    "beta": 4.0,
    "gamma": -1.5,
    "delta": 0.0,
    "epsilon": 0.5,
    "tau": 20.0,
    "x_ext": 1.0,
    "y_ext": 0.0,
}


class FitzHughNagumoMass(NeuralMass):
    """
    FitzHugh-Nagumo model.
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
        "x_ext",
        "y_ext",
    ]
    required_couplings = ["network_x", "network_y"]
    _noise_input = [
        OrnsteinUhlenbeckProcess(mu=0.0, sigma=0.0, tau=5.0),
        OrnsteinUhlenbeckProcess(mu=0.0, sigma=0.0, tau=5.0),
    ]

    def __init__(self, params=None, seed=None):
        super().__init__(params=params or FHN_DEFAULT_PARAMS, seed=seed)

    def _initialize_state_vector(self):
        """
        Initialize state vector.
        """
        np.random.seed(self.seed)
        self.initial_state = (0.05 * np.random.uniform(-1, 1, size=(self.num_state_variables,))).tolist()

    def _derivatives(self, coupling_variables):
        [x, y] = self._unwrap_state_vector()

        d_x = (
            -self.params["alpha"] * x ** 3
            + self.params["beta"] * x ** 2
            + self.params["gamma"] * x
            - y
            + coupling_variables["network_x"]
            + system_input(self.noise_input_idx[0])
            + self.params["x_ext"]
        )

        d_y = (
            (x - self.params["delta"] - self.params["epsilon"] * y) / self.params["tau"]
            + coupling_variables["network_y"]
            + system_input(self.noise_input_idx[1])
            + self.params["y_ext"]
        )

        return [d_x, d_y]


class FitzHughNagumoNode(Node):
    """
    Default FitzHugh-Nagumo node with 1 neural mass modelled as FitzHugh-Nagumo
    oscillator.
    """

    name = "FitzHugh-Nagumo node"
    label = "FHNnode"

    default_network_coupling = {"network_x": 0.0, "network_y": 0.0}
    default_output = "x"
    output_vars = ["x", "y"]

    def __init__(self, params=None, seed=None):
        """
        :param params: parameters of the FitzHugh-Nagumo mass
        :type params: dict|None
        :param seed: seed for random number generator
        :type seed: int|None
        """
        fhn_mass = FitzHughNagumoMass(params, seed=seed)
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
    # define default coupling in FitzHugh-Nagumo network
    default_coupling = {"network_x": "diffusive", "network_y": "none"}
    output_vars = ["x", "y"]

    def __init__(
        self,
        connectivity_matrix,
        delay_matrix,
        mass_params=None,
        seed=None,
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
        :type y_coupling: str
        :param seed: seed for random number generator
        :type seed: int|None
        """
        mass_params = self._prepare_mass_params(mass_params, connectivity_matrix.shape[0])
        seeds = self._prepare_mass_params(seed, connectivity_matrix.shape[0], native_type=int)

        nodes = []
        for i, node_params in enumerate(mass_params):
            node = FitzHughNagumoNode(params=node_params, seed=seeds[i])
            node.index = i
            node.idx_state_var = i * node.num_state_variables
            nodes.append(node)

        super().__init__(
            nodes=nodes,
            connectivity_matrix=connectivity_matrix,
            delay_matrix=delay_matrix,
        )
        # get all coupling variables
        all_couplings = [mass.coupling_variables for node in self.nodes for mass in node.masses]
        # assert they are the same
        assert all(all_couplings[0] == coupling for coupling in all_couplings)
        # invert as to name: idx
        self.coupling_symbols = {v: k for k, v in all_couplings[0].items()}
