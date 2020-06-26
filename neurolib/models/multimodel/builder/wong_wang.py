"""
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

import numpy as np
from jitcdde import input as system_input
from symengine import exp

from ..builder.base.constants import EXC, INH, LAMBDA_SPEED
from ..builder.base.network import Node, SingleCouplingExcitatoryInhibitoryNode
from ..builder.base.neural_mass import NeuralMass

# TODO compare with TVB (at least the reduced version - they have it)

DEFAULT_PARAMS_EXC = {
    "a": 310.0,  # nC^-1
    "b": 0.125,  # kHz
    "d": 160.0,  # ms
    "tau": 100.0,  # ms
    "gamma": 0.641,
    "W": 1.0,
    "exc_current": 0.382,  # nA
    "J": 0.15,  # nA
    "lambda": LAMBDA_SPEED,
}
DEFAULT_PARAMS_INH = {
    "a": 615.0,  # nC^-1
    "b": 0.177,  # kHz
    "d": 87.0,  # ms
    "tau": 10.0,  # ms
    "W": 0.7,
    "exc_current": 0.382,  # nA
    "lambda": LAMBDA_SPEED,
}
DEFAULT_PARAMS_REDUCED = {
    "a": 270.0,  # nC^-1
    "b": 0.108,  # kHz
    "d": 154.0,  # ms
    "tau": 100.0,  # ms
    "gamma": 0.641 / 1000.0,
    "w": 0.6,
    "J": 0.2609,  # nA
    "exc_current": 0.3,  # nA
    "lambda": LAMBDA_SPEED,
}

# matrix as [to, from], masses as (EXC, INH)
DEFAULT_WW_NODE_CONNECTIVITY = np.array([[1.4, 1.0], [1.0, 1.0]])


class WongWangMass(NeuralMass):
    """
    Wong-Wang neural mass. Can be excitatory or inhibitory, depending on the
    parameters. Also a base for reduced Wong-Wang mass.
    """

    name = "Wong-Wang mass"
    label = "WWmass"

    num_state_variables = 2
    num_noise_variables = 1
    coupling_variables = {0: "S"}
    state_variable_names = ["S", "q_mean"]

    def _initialize_state_vector(self):
        """
        Initialize state vector.
        """
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
    required_couplings = ["node_exc_exc", "node_inh_exc", "network_exc_exc"]
    required_params = [
        "a",
        "b",
        "d",
        "tau",
        "gamma",
        "W",
        "exc_current",
        "J",
        "lambda",
    ]

    def __init__(self, params=None):
        super().__init__(params=params or DEFAULT_PARAMS_EXC)

    def _derivatives(self, coupling_variables):
        [s, firing_rate] = self._unwrap_state_vector()

        current = (
            self.params["W"] * self.params["exc_current"]
            + coupling_variables["node_exc_exc"]
            + self.params["J"] * coupling_variables["network_exc_exc"]
            - coupling_variables["node_inh_exc"]
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
    required_couplings = ["node_exc_inh", "node_inh_inh"]
    required_params = [
        "a",
        "b",
        "d",
        "tau",
        "W",
        "exc_current",
        "lambda",
    ]

    def __init__(self, params=None):
        super().__init__(params=params or DEFAULT_PARAMS_INH)

    def _derivatives(self, coupling_variables):
        [s, firing_rate] = self._unwrap_state_vector()

        current = (
            self.params["W"] * self.params["exc_current"]
            + coupling_variables["node_exc_inh"]
            - coupling_variables["node_inh_inh"]
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
    required_couplings = ["network_s"]
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

    def __init__(self, params=None):
        super().__init__(params=params or DEFAULT_PARAMS_REDUCED)

    def _derivatives(self, coupling_variables):
        [s, firing_rate] = self._unwrap_state_vector()

        current = (
            self.params["w"] * self.params["J"] * s
            + self.params["J"] * coupling_variables["network_s"]
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


class WongWangNetworkNode(SingleCouplingExcitatoryInhibitoryNode):
    """
    Default Wong-Wang network node with 1 excitatory and 1 inhibitory popultion.
    """

    name = "Wong-Wang node"
    label = "WWnode"

    default_output = f"S_{EXC}"

    def __init__(
        self, exc_params=None, inh_params=None, connectivity=DEFAULT_WW_NODE_CONNECTIVITY,
    ):
        """
        :param exc_params: parameters for the excitatory mass
        :type exc_params: dict|None
        :param inh_params: parameters for the inhibitory mass
        :type inh_params: dict|None
        :param connectivity: local connectivity matrix
        :type connectivity: np.ndarray
        """
        excitatory_mass = ExcitatoryWongWangMass(exc_params)
        excitatory_mass.index = 0
        inhibitory_mass = InhibitoryWongWangMass(inh_params)
        inhibitory_mass.index = 1
        super().__init__(
            neural_masses=[excitatory_mass, inhibitory_mass],
            local_connectivity=connectivity,
            # within W-W node there are no local delays
            local_delays=None,
        )


class ReducedWongWangNetworkNode(Node):
    """
    Default reduced Wong-Wang network node with 1 neural mass.
    """

    name = "Reduced Wong-Wang node"
    label = "ReducedWWnode"

    default_network_coupling = {"network_s": 0.0}
    default_output = "S"

    def __init__(self, params=None):
        """
        :param params: parameters of the reduced Wong-Wang mass
        :type params: dict|None
        """
        reduced_ww_mass = ReducedWongWangMass(params)
        reduced_ww_mass.index = 0
        super().__init__(neural_masses=[reduced_ww_mass])

    def _sync(self):
        return []


# TODO add network instances for both WW versions, after checking with TVB
