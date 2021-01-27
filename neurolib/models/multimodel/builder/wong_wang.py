import numpy as np
from jitcdde import input as system_input
from symengine import exp

from ....utils.stimulus import OrnsteinUhlenbeckProcess
from ..builder.base.constants import EXC, INH, LAMBDA_SPEED
from ..builder.base.network import Network, Node, SingleCouplingExcitatoryInhibitoryNode
from ..builder.base.neural_mass import NeuralMass

WW_EXC_DEFAULT_PARAMS = {
    "a": 0.31,  # nC^-1
    "b": 0.125,  # kHz
    "d": 160.0,  # ms
    "tau": 100.0,  # ms
    "gamma": 0.641,
    "W": 1.0,
    "exc_current": 0.382,  # nA
    "J_NMDA": 0.15,  # nA
    "J_I": 1.0,  # nA
    "lambda": LAMBDA_SPEED,
}
WW_INH_DEFAULT_PARAMS = {
    "a": 0.615,  # nC^-1
    "b": 0.177,  # kHz
    "d": 87.0,  # ms
    "tau": 10.0,  # ms
    "W": 0.7,
    "exc_current": 0.382,  # nA
    "J_NMDA": 0.15,  # nA
    "lambda": LAMBDA_SPEED,
}
WW_REDUCED_DEFAULT_PARAMS = {
    "a": 0.27,  # nC^-1
    "b": 0.108,  # kHz
    "d": 154.0,  # ms
    "tau": 100.0,  # ms
    "gamma": 0.641,  # kinetic parameter
    "w": 0.6,
    "J": 0.2609,  # nA
    "exc_current": 0.33,  # nA
    "lambda": LAMBDA_SPEED,
}

# matrix as [to, from], masses as (EXC, INH)
WW_NODE_DEFAULT_CONNECTIVITY = np.array([[1.4, 1.0], [1.0, 1.0]])


class WongWangMass(NeuralMass):
    """
    Wong-Wang neural mass. Can be excitatory or inhibitory, depending on the
    parameters. Also a base for reduced Wong-Wang mass.

    Wong-Wang model. Contains both:
        - classical Wong-Wang with one network node containing one excitatory and
            one inhibitory mass
        - Reduced Wong-Wang model with one mass per node

    Main reference:
        [original] Wong, K. F., & Wang, X. J. (2006). A recurrent network mechanism
        of time integration in perceptual decisions. Journal of Neuroscience, 26(4),
        1314-1328.

    Additional references:
        [reduced] Deco, G., Ponce-Alvarez, A., Mantini, D., Romani, G. L., Hagmann,
        P., & Corbetta, M. (2013). Resting-state functional connectivity emerges
        from structurally and dynamically shaped slow linear fluctuations. Journal
        of Neuroscience, 33(27), 11239-11252.

        [original] Deco, G., Ponce-Alvarez, A., Hagmann, P., Romani, G. L., Mantini,
        D., & Corbetta, M. (2014). How local excitation–inhibition ratio impacts the
        whole brain dynamics. Journal of Neuroscience, 34(23), 7886-7898.
    """

    name = "Wong-Wang mass"
    label = "WWmass"

    num_state_variables = 2
    num_noise_variables = 1
    coupling_variables = {0: "S"}
    state_variable_names = ["S", "q_mean"]
    _noise_input = [OrnsteinUhlenbeckProcess(mu=0.0, sigma=0.0, tau=5.0)]

    def _initialize_state_vector(self):
        """
        Initialize state vector.
        """
        np.random.seed(self.seed)
        self.initial_state = (np.random.uniform(0, 1) * np.array([1.0, 0.01])).tolist()

    def _get_firing_rate(self, current):
        return (self.params["a"] * current - self.params["b"]) / (
            1.0 - exp(-self.params["d"] * (self.params["a"] * current - self.params["b"]))
        )


class ExcitatoryWongWangMass(WongWangMass):
    """
    Excitatory Wong-Wang neural mass.
    """

    name = "Wong-Wang excitatory mass"
    label = f"WWmass{EXC}"
    mass_type = EXC
    coupling_variables = {0: f"S_{EXC}"}
    state_variable_names = [f"S_{EXC}", f"q_mean_{EXC}"]
    required_couplings = ["node_exc_exc", "node_exc_inh", "network_exc_exc"]
    required_params = [
        "a",
        "b",
        "d",
        "tau",
        "gamma",
        "W",
        "exc_current",
        "J_NMDA",
        "J_I",
        "lambda",
    ]

    def __init__(self, params=None, seed=None):
        super().__init__(params=params or WW_EXC_DEFAULT_PARAMS, seed=seed)

    def _derivatives(self, coupling_variables):
        [s, firing_rate] = self._unwrap_state_vector()

        current = (
            self.params["W"] * self.params["exc_current"]
            + self.params["J_NMDA"] * coupling_variables["node_exc_exc"]
            + self.params["J_NMDA"] * coupling_variables["network_exc_exc"]
            - self.params["J_I"] * coupling_variables["node_exc_inh"]
        )
        firing_rate_now = self._get_firing_rate(current)
        d_s = (
            (-s / self.params["tau"])
            + (1.0 - s) * self.params["gamma"] * firing_rate_now
            + system_input(self.noise_input_idx[0])
        )
        # firing rate as dummy dynamical variable with infinitely fast
        # fixed-point dynamics
        d_firing_rate = -self.params["lambda"] * (firing_rate - firing_rate_now)

        return [d_s, d_firing_rate]


class InhibitoryWongWangMass(WongWangMass):
    """
    Inhibitory Wong-Wang neural mass.
    """

    name = "Wong-Wang inhibitory mass"
    label = f"WWmass{INH}"
    mass_type = INH
    coupling_variables = {0: f"S_{INH}"}
    state_variable_names = [f"S_{INH}", f"q_mean_{INH}"]
    required_couplings = ["node_inh_exc", "node_inh_inh", "network_inh_exc"]
    required_params = [
        "a",
        "b",
        "d",
        "tau",
        "W",
        "exc_current",
        "J_NMDA",
        "lambda",
    ]

    def __init__(self, params=None, seed=None):
        super().__init__(params=params or WW_INH_DEFAULT_PARAMS, seed=seed)

    def _derivatives(self, coupling_variables):
        [s, firing_rate] = self._unwrap_state_vector()

        current = (
            self.params["W"] * self.params["exc_current"]
            + self.params["J_NMDA"] * coupling_variables["node_inh_exc"]
            - coupling_variables["node_inh_inh"]
            + self.params["J_NMDA"] * coupling_variables["network_inh_exc"]
        )
        firing_rate_now = self._get_firing_rate(current)
        d_s = (-s / self.params["tau"]) + firing_rate_now + system_input(self.noise_input_idx[0])
        # firing rate as dummy dynamical variable with infinitely fast
        # fixed-point dynamics
        d_firing_rate = -self.params["lambda"] * (firing_rate - firing_rate_now)

        return [d_s, d_firing_rate]


class ReducedWongWangMass(WongWangMass):
    """
    Reduced Wong-Wang neural mass - only NMDA gating variable, which dominates
    the evolution of the system.
    """

    name = "Reduced Wong-Wang mass"
    label = "ReducedWWmass"
    required_couplings = ["network_S"]
    required_params = [
        "a",
        "b",
        "d",
        "tau",
        "gamma",
        "w",
        "J",
        "exc_current",
        "lambda",
    ]

    def __init__(self, params=None, seed=None):
        super().__init__(params=params or WW_REDUCED_DEFAULT_PARAMS, seed=seed)

    def _derivatives(self, coupling_variables):
        [s, firing_rate] = self._unwrap_state_vector()

        current = (
            self.params["w"] * self.params["J"] * s
            + self.params["J"] * coupling_variables["network_S"]
            + self.params["exc_current"]
        )
        firing_rate_now = self._get_firing_rate(current)
        d_s = (
            (-s / self.params["tau"])
            + (1.0 - s) * self.params["gamma"] * firing_rate_now
            + system_input(self.noise_input_idx[0])
        )
        # firing rate as dummy dynamical variable with infinitely fast
        # fixed-point dynamics
        d_firing_rate = -self.params["lambda"] * (firing_rate - firing_rate_now)

        return [d_s, d_firing_rate]


class WongWangNode(SingleCouplingExcitatoryInhibitoryNode):
    """
    Default Wong-Wang network node with 1 excitatory and 1 inhibitory popultion.
    """

    name = "Wong-Wang node"
    label = "WWnode"

    default_network_coupling = {"network_exc_exc": 0.0, "network_inh_exc": 0.0}
    default_output = f"S_{EXC}"
    output_vars = [f"S_{EXC}", f"q_mean_{EXC}", f"S_{INH}", f"q_mean_{INH}"]

    def __init__(
        self,
        exc_params=None,
        inh_params=None,
        connectivity=WW_NODE_DEFAULT_CONNECTIVITY,
        exc_seed=None,
        inh_seed=None,
    ):
        """
        :param exc_params: parameters for the excitatory mass
        :type exc_params: dict|None
        :param inh_params: parameters for the inhibitory mass
        :type inh_params: dict|None
        :param connectivity: local connectivity matrix
        :type connectivity: np.ndarray
        :param exc_seed: seed for random number generator for the excitatory
            mass
        :type exc_seed: int|None
        :param inh_seed: seed for random number generator for the inhibitory
            mass
        :type inh_seed: int|None
        """
        excitatory_mass = ExcitatoryWongWangMass(exc_params, seed=exc_seed)
        excitatory_mass.index = 0
        inhibitory_mass = InhibitoryWongWangMass(inh_params, seed=inh_seed)
        inhibitory_mass.index = 1
        super().__init__(
            neural_masses=[excitatory_mass, inhibitory_mass],
            local_connectivity=connectivity,
            # within W-W node there are no local delays
            local_delays=None,
        )


class ReducedWongWangNode(Node):
    """
    Default reduced Wong-Wang network node with 1 neural mass.
    """

    name = "Reduced Wong-Wang node"
    label = "ReducedWWnode"

    default_network_coupling = {"network_S": 0.0}
    default_output = "S"
    output_vars = ["S", "q_mean"]

    def __init__(self, params=None, seed=None):
        """
        :param params: parameters of the reduced Wong-Wang mass
        :type params: dict|None
        :param seed: seed for random number generator
        :type seed: int|None
        """
        reduced_ww_mass = ReducedWongWangMass(params, seed=seed)
        reduced_ww_mass.index = 0
        super().__init__(neural_masses=[reduced_ww_mass])

    def _sync(self):
        return []


class WongWangNetwork(Network):
    """
    Whole brain network of Wong-Wong excitatory and inhibitory nodes.
    """

    name = "Wong-Wang network"
    label = "WWnet"

    sync_variables = ["network_exc_exc", "network_inh_exc"]
    # define default coupling in Wong-Wang network
    default_coupling = {"network_exc_exc": "additive", "network_inh_exc": "additive"}
    output_vars = [f"S_{EXC}", f"q_mean_{EXC}", f"S_{INH}", f"q_mean_{INH}"]

    def __init__(
        self,
        connectivity_matrix,
        delay_matrix,
        exc_mass_params=None,
        inh_mass_params=None,
        local_connectivity=WW_NODE_DEFAULT_CONNECTIVITY,
        exc_seed=None,
        inh_seed=None,
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
        :param exc_mass_params: parameters for each excitatory Wong-Wang
            neural mass, if None, will use default
        :type exc_mass_params: list[dict]|dict|None
        :param inh_mass_params: parameters for each inhibitory Wong-Wang
            neural mass, if None, will use default
        :type inh_mass_params: list[dict]|dict|None
        :param local_connectivity: local within-node connectivity matrix
        :type local_connectivity: list[np.ndarray]|np.ndarray
        :param exc_seed: seed for random number generator for the excitatory
            masses
        :type exc_seed: int|None
        :param inh_seed: seed for random number generator for the excitatory
            masses
        :type inh_seed: int|None
        """
        num_nodes = connectivity_matrix.shape[0]
        exc_mass_params = self._prepare_mass_params(exc_mass_params, num_nodes)
        inh_mass_params = self._prepare_mass_params(inh_mass_params, num_nodes)
        exc_seeds = self._prepare_mass_params(exc_seed, num_nodes, native_type=int)
        inh_seeds = self._prepare_mass_params(inh_seed, num_nodes, native_type=int)
        local_connectivity = self._prepare_mass_params(local_connectivity, num_nodes, native_type=np.ndarray)

        nodes = []
        for i, (exc_params, inh_params, local_conn) in enumerate(
            zip(exc_mass_params, inh_mass_params, local_connectivity)
        ):
            node = WongWangNode(
                exc_params=exc_params,
                inh_params=inh_params,
                connectivity=local_conn,
                exc_seed=exc_seeds[i],
                inh_seed=inh_seeds[i],
            )
            node.index = i
            node.idx_state_var = i * node.num_state_variables
            # set correct indices of noise input
            for mass in node:
                mass.noise_input_idx = [2 * i + mass.index]
            nodes.append(node)

        super().__init__(
            nodes=nodes,
            connectivity_matrix=connectivity_matrix,
            delay_matrix=delay_matrix,
        )
        # assert we have two sync variables
        assert len(self.sync_variables) == 2
        self.coupling_symbols = {"exc_exc": 0, "inh_exc": 0}


class ReducedWongWangNetwork(Network):
    """
    Whole brain network of Reduced Wong-Wang masses.
    """

    name = "Reduced Wong-Wang network"
    label = "ReducedWWnet"

    sync_variables = ["network_S"]
    # define default coupling in Reduced Wong-Wang network
    default_coupling = {"network_S": "additive"}
    output_vars = ["S", "q_mean"]

    def __init__(self, connectivity_matrix, delay_matrix, mass_params=None, seed=None):
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
        :param seed: seed for random number generator
        :type seed: int|None
        """
        mass_params = self._prepare_mass_params(mass_params, connectivity_matrix.shape[0])
        seeds = self._prepare_mass_params(seed, connectivity_matrix.shape[0], native_type=int)

        nodes = []
        for i, node_params in enumerate(mass_params):
            node = ReducedWongWangNode(params=node_params, seed=seeds[i])
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
