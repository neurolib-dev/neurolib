import numpy as np
from jitcdde import input as system_input
from symengine import exp

from ....utils.stimulus import OrnsteinUhlenbeckProcess
from ..builder.base.constants import EXC, INH
from ..builder.base.network import Network, SingleCouplingExcitatoryInhibitoryNode
from ..builder.base.neural_mass import NeuralMass

WC_EXC_DEFAULT_PARAMS = {"a": 1.5, "mu": 3.0, "tau": 2.5, "ext_drive": 0.0}
WC_INH_DEFAULT_PARAMS = {"a": 1.5, "mu": 3.0, "tau": 3.75, "ext_drive": 0.0}
# matrix as [to, from], masses as (EXC, INH)
WC_NODE_DEFAULT_CONNECTIVITY = np.array([[16.0, 12.0], [15.0, 3.0]])


class WilsonCowanMass(NeuralMass):
    """
    Wilson-Cowan neural mass. Can be excitatory or inhibitory, depending on the
    parameters.

    Reference:
        Wilson, H. R., & Cowan, J. D. (1972). Excitatory and inhibitory
        interactions in localized populations of model neurons. Biophysical journal,
        12(1), 1-24.
    """

    name = "Wilson-Cowan mass"
    label = "WCmass"

    num_state_variables = 1
    num_noise_variables = 1
    coupling_variables = {0: "q_mean"}
    state_variable_names = ["q_mean"]
    required_params = ["a", "mu", "tau", "ext_drive"]
    _noise_input = [OrnsteinUhlenbeckProcess(mu=0.0, sigma=0.0, tau=5.0)]

    def _initialize_state_vector(self):
        """
        Initialize state vector.
        """
        np.random.seed(self.seed)
        self.initial_state = [0.05 * np.random.uniform(0, 1)]

    def _sigmoid(self, x):
        return 1.0 / (1.0 + exp(-self.params["a"] * (x - self.params["mu"])))


class ExcitatoryWilsonCowanMass(WilsonCowanMass):
    """
    Excitatory Wilson-Cowan neural mass.
    """

    name = "Wilson-Cowan excitatory mass"
    label = f"WCmass{EXC}"
    coupling_variables = {0: f"q_mean_{EXC}"}
    state_variable_names = [f"q_mean_{EXC}"]
    mass_type = EXC
    required_couplings = ["node_exc_exc", "node_exc_inh", "network_exc_exc"]

    def __init__(self, params=None, seed=None):
        super().__init__(params=params or WC_EXC_DEFAULT_PARAMS, seed=seed)

    def _derivatives(self, coupling_variables):
        [x] = self._unwrap_state_vector()
        d_x = (
            -x
            + (1.0 - x)
            * self._sigmoid(
                coupling_variables["node_exc_exc"]
                - coupling_variables["node_exc_inh"]
                + coupling_variables["network_exc_exc"]
                + self.params["ext_drive"]
            )
            + system_input(self.noise_input_idx[0])
        ) / self.params["tau"]

        return [d_x]


class InhibitoryWilsonCowanMass(WilsonCowanMass):

    name = "Wilson-Cowan inhibitory mass"
    label = f"WCmass{INH}"
    coupling_variables = {0: f"q_mean_{INH}"}
    state_variable_names = [f"q_mean_{INH}"]
    mass_type = INH
    required_couplings = ["node_inh_exc", "node_inh_inh", "network_inh_exc"]

    def __init__(self, params=None, seed=None):
        super().__init__(params=params or WC_INH_DEFAULT_PARAMS, seed=seed)

    def _derivatives(self, coupling_variables):
        [x] = self._unwrap_state_vector()
        d_x = (
            -x
            + (1.0 - x)
            * self._sigmoid(
                coupling_variables["node_inh_exc"]
                - coupling_variables["node_inh_inh"]
                + coupling_variables["network_inh_exc"]
                + self.params["ext_drive"]
            )
            + system_input(self.noise_input_idx[0])
        ) / self.params["tau"]

        return [d_x]


class WilsonCowanNode(SingleCouplingExcitatoryInhibitoryNode):
    """
    Default Wilson-Cowan network node with 1 excitatory and 1 inhibitory
    population.
    """

    name = "Wilson-Cowan node"
    label = "WCnode"

    default_network_coupling = {"network_exc_exc": 0.0, "network_inh_exc": 0.0}
    default_output = f"q_mean_{EXC}"
    output_vars = [f"q_mean_{EXC}", f"q_mean_{INH}"]

    def __init__(
        self, exc_params=None, inh_params=None, connectivity=WC_NODE_DEFAULT_CONNECTIVITY, exc_seed=None, inh_seed=None
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
        excitatory_mass = ExcitatoryWilsonCowanMass(exc_params, seed=exc_seed)
        excitatory_mass.index = 0
        inhibitory_mass = InhibitoryWilsonCowanMass(inh_params, seed=inh_seed)
        inhibitory_mass.index = 1
        super().__init__(
            neural_masses=[excitatory_mass, inhibitory_mass],
            local_connectivity=connectivity,
            # within W-C node there are no local delays
            local_delays=None,
        )


class WilsonCowanNetwork(Network):
    """
    Whole brain network of Wilson-Cowan excitatory and inhibitory nodes.
    """

    name = "Wilson-Cowan network"
    label = "WCnet"

    sync_variables = ["network_exc_exc", "network_inh_exc"]
    # define default coupling in Wilson-Cowan network
    default_coupling = {"network_exc_exc": "additive", "network_inh_exc": "additive"}
    output_vars = [f"q_mean_{EXC}", f"q_mean_{INH}"]

    def __init__(
        self,
        connectivity_matrix,
        delay_matrix,
        exc_mass_params=None,
        inh_mass_params=None,
        local_connectivity=WC_NODE_DEFAULT_CONNECTIVITY,
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
        :param exc_mass_params: parameters for each excitatory Wilson-Cowan
            neural mass, if None, will use default
        :type exc_mass_params: list[dict]|dict|None
        :param inh_mass_params: parameters for each inhibitory Wilson-Cowan
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
            node = WilsonCowanNode(
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
