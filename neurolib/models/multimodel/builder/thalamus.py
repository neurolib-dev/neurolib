import numpy as np
from jitcdde import input as system_input
from symengine import exp

from ....utils.stimulus import OrnsteinUhlenbeckProcess, ZeroInput
from ..builder.base.constants import EXC, INH, LAMBDA_SPEED
from ..builder.base.network import SingleCouplingExcitatoryInhibitoryNode
from ..builder.base.neural_mass import NeuralMass

TCR_DEFAULT_PARAMS = {
    "tau": 20.0,  # ms
    "Q_max": 400.0e-3,  # 1/ms
    "theta": -58.5,  # mV
    "sigma": 6.0,
    "C1": 1.8137993642,
    "C_m": 1.0,  # muF/cm^2
    "gamma_e": 70.0e-3,  # 1/ms
    "gamma_r": 100.0e-3,  # 1/ms
    "g_L": 1.0,  # AU
    "g_GABA": 1.0,  # ms
    "g_AMPA": 1.0,  # ms
    "g_LK": 0.018,  # mS/cm^2
    "g_T": 3.0,  # mS/cm^2
    "g_h": 0.062,  # mS/cm^2
    "E_AMPA": 0.0,  # mV
    "E_GABA": -70.0,  # mV
    "E_L": -70.0,  # mV
    "E_K": -100.0,  # mV
    "E_Ca": 120.0,  # mV
    "E_h": -40.0,  # mV
    "alpha_Ca": -51.8e-6,  # nmol
    "tau_Ca": 10.0,  # ms
    "Ca_0": 2.4e-4,
    "k1": 2.5e7,
    "k2": 4.0e-4,
    "k3": 1.0e-1,
    "k4": 1.0e-3,
    "n_P": 4.0,
    "g_inc": 2.0,
    "ext_current": 0.0,
    "lambda": LAMBDA_SPEED,
}
TRN_DEFAULT_PARAMS = {
    "tau": 20.0,  # ms
    "Q_max": 400.0e-3,  # 1/ms
    "theta": -58.5,  # mV
    "sigma": 6.0,
    "C1": 1.8137993642,
    "C_m": 1.0,  # muF/cm^2
    "gamma_e": 70.0e-3,  # 1/ms
    "gamma_r": 100.0e-3,  # 1/ms
    "g_L": 1.0,  # AU
    "g_GABA": 1.0,  # ms
    "g_AMPA": 1.0,  # ms
    "g_LK": 0.018,  # mS/cm^2
    "g_T": 2.3,  # mS/cm^2
    "E_AMPA": 0.0,  # mV
    "E_GABA": -70.0,  # mV
    "E_L": -70.0,  # mV
    "E_K": -100.0,  # mV
    "E_Ca": 120.0,  # mV
    "ext_current": 0.0,
    "lambda": LAMBDA_SPEED,
}
# matrix as [to, from], masses as (TCR, TRN)
THALAMUS_NODE_DEFAULT_CONNECTIVITY = np.array([[0.0, 5.0], [3.0, 25.0]])


class ThalamicMass(NeuralMass):
    """
    Base for thalamic neural populations

    Reference:
        Costa, M. S., Weigenand, A., Ngo, H. V. V., Marshall, L., Born, J.,
        Martinetz, T., & Claussen, J. C. (2016). A thalamocortical neural mass
        model of the EEG during NREM sleep and its response to auditory stimulation.
        PLoS computational biology, 12(9).
    """

    name = "Thalamic mass"
    label = "THLM"

    def _get_firing_rate(self, voltage):
        return self.params["Q_max"] / (
            1.0 + exp(-self.params["C1"] * (voltage - self.params["theta"]) / self.params["sigma"])
        )

    # synaptic currents
    def _get_excitatory_current(self, voltage, synaptic_rate):
        return self.params["g_AMPA"] * synaptic_rate * (voltage - self.params["E_AMPA"])

    def _get_inhibitory_current(self, voltage, synaptic_rate):
        return self.params["g_GABA"] * synaptic_rate * (voltage - self.params["E_GABA"])

    # intrinsic currents
    def _get_leak_current(self, voltage):
        return self.params["g_L"] * (voltage - self.params["E_L"])

    def _get_potassium_leak_current(self, voltage):
        return self.params["g_LK"] * (voltage - self.params["E_K"])

    def _get_T_type_current(self, voltage, h_T_value):
        return (
            self.params["g_T"]
            * self._m_inf_T(voltage)
            * self._m_inf_T(voltage)
            * h_T_value
            * (voltage - self.params["E_Ca"])
        )


class ThalamocorticalMass(ThalamicMass):
    """
    Excitatory mass representing thalamocortical relay neurons in the thalamus.
    """

    name = "Thalamocortical relay mass"
    label = "TCR"
    mass_type = EXC

    num_state_variables = 10
    num_noise_variables = 1
    coupling_variables = {9: f"r_mean_{EXC}"}
    required_couplings = ["node_exc_exc", "node_exc_inh", "network_exc_exc"]
    state_variable_names = [
        "V",
        "Ca",
        "h_T",
        "m_h1",
        "m_h2",
        "s_e",
        "s_i",
        "ds_e",
        "ds_i",
        "r_mean",
    ]
    required_params = [
        "tau",
        "Q_max",
        "theta",
        "sigma",
        "C1",
        "C_m",
        "gamma_e",
        "gamma_r",
        "g_L",
        "g_GABA",
        "g_AMPA",
        "g_LK",
        "g_T",
        "g_h",
        "E_AMPA",
        "E_GABA",
        "E_L",
        "E_K",
        "E_Ca",
        "E_h",
        "alpha_Ca",
        "tau_Ca",
        "Ca_0",
        "k1",
        "k2",
        "k3",
        "k4",
        "n_P",
        "g_inc",
        "ext_current",
        "lambda",
    ]
    _noise_input = [OrnsteinUhlenbeckProcess(mu=0.0, sigma=0.0, tau=5.0)]

    def __init__(self, params=None):
        super().__init__(params=params or TCR_DEFAULT_PARAMS)

    def _initialize_state_vector(self):
        """
        Initialize state vector.
        """
        self.initial_state = [
            self.params["E_L"],
            self.params["Ca_0"],
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]

    def _get_anomalous_rectifier_current(self, voltage, m_h1_value, m_h2_value):
        return self.params["g_h"] * (m_h1_value + self.params["g_inc"] * m_h2_value) * (voltage - self.params["E_h"])

    def _m_inf_T(self, voltage):
        return 1.0 / (1.0 + exp(-(voltage + 59.0) / 6.2))

    def _h_inf_T(self, voltage):
        return 1.0 / (1.0 + exp((voltage + 81.0) / 4.0))

    def _tau_h_T(self, voltage):
        return (30.8 + (211.4 + exp((voltage + 115.2) / 5.0)) / (1.0 + exp((voltage + 86.0) / 3.2))) / 3.7371928

    def _m_inf_h(self, voltage):
        return 1.0 / (1.0 + exp((voltage + 75.0) / 5.5))

    def _tau_m_h(self, voltage):
        return 20.0 + 1000.0 / (exp((voltage + 71.5) / 14.2) + exp(-(voltage + 89.0) / 11.6))

    def _P_h(self, ca_conc):
        return (self.params["k1"] * ca_conc ** self.params["n_P"]) / (
            self.params["k1"] * ca_conc ** self.params["n_P"] + self.params["k2"]
        )

    def _derivatives(self, coupling_variables):
        (
            voltage,
            ca_conc,
            h_T,
            m_h1,
            m_h2,
            syn_ext,
            syn_inh,
            dsyn_ext,
            dsyn_inh,
            firing_rate,
        ) = self._unwrap_state_vector()
        # voltage dynamics
        d_voltage = -(
            self._get_leak_current(voltage)
            + self._get_excitatory_current(voltage, syn_ext)
            + self._get_inhibitory_current(voltage, syn_inh)
            + self.params["ext_current"]
        ) / self.params["tau"] - (1.0 / self.params["C_m"]) * (
            self._get_potassium_leak_current(voltage)
            + self._get_T_type_current(voltage, h_T)
            + self._get_anomalous_rectifier_current(voltage, m_h1, m_h2)
        )
        # calcium concetration dynamics
        d_ca_conc = (
            self.params["alpha_Ca"] * self._get_T_type_current(voltage, h_T)
            - (ca_conc - self.params["Ca_0"]) / self.params["tau_Ca"]
        )

        # channel dynamics: T-type and rectifier current
        d_h_T = (self._h_inf_T(voltage) - h_T) / self._tau_h_T(voltage)
        d_m_h1 = (
            (self._m_inf_h(voltage) * (1.0 - m_h2) - m_h1) / self._tau_m_h(voltage)
            - self.params["k3"] * self._P_h(ca_conc) * m_h1
            + self.params["k4"] * m_h2
        )
        d_m_h2 = self.params["k3"] * self._P_h(ca_conc) * m_h1 - self.params["k4"] * m_h2

        # synaptic dynamics
        d_syn_ext = dsyn_ext
        d_syn_inh = dsyn_inh
        d_dsyn_ext = (
            self.params["gamma_e"] ** 2
            * (
                coupling_variables["node_exc_exc"]
                + coupling_variables["network_exc_exc"]
                + system_input(self.noise_input_idx[0])
                - syn_ext
            )
            - 2 * self.params["gamma_e"] * dsyn_ext
        )
        d_dsyn_inh = (
            self.params["gamma_r"] ** 2 * (coupling_variables["node_exc_inh"] - syn_inh)
            - 2 * self.params["gamma_r"] * dsyn_inh
        )
        # firing rate as dummy dynamical variable with infinitely fast
        # fixed-point dynamics
        firing_rate_now = self._get_firing_rate(voltage)
        d_firing_rate = -self.params["lambda"] * (firing_rate - firing_rate_now)

        return [
            d_voltage,
            d_ca_conc,
            d_h_T,
            d_m_h1,
            d_m_h2,
            d_syn_ext,
            d_syn_inh,
            d_dsyn_ext,
            d_dsyn_inh,
            d_firing_rate,
        ]


class ThalamicReticularMass(ThalamicMass):
    """
    Inhibitory mass representing thalamic reticular nuclei neurons in the
    thalamus.
    """

    name = "Thalamic reticular nuclei mass"
    label = "TRN"
    mass_type = INH

    num_state_variables = 7
    num_noise_variables = 1
    coupling_variables = {6: f"r_mean_{INH}"}
    required_couplings = ["node_inh_exc", "node_inh_inh", "network_inh_exc"]
    state_variable_names = [
        "V",
        "h_T",
        "s_e",
        "s_i",
        "ds_e",
        "ds_i",
        "r_mean",
    ]
    required_params = [
        "tau",
        "Q_max",
        "theta",
        "sigma",
        "C1",
        "C_m",
        "gamma_e",
        "gamma_r",
        "g_L",
        "g_GABA",
        "g_AMPA",
        "g_LK",
        "g_T",
        "E_AMPA",
        "E_GABA",
        "E_L",
        "E_K",
        "E_Ca",
        "ext_current",
        "lambda",
    ]
    _noise_input = [ZeroInput()]

    def __init__(self, params=None):
        super().__init__(params=params or TRN_DEFAULT_PARAMS)

    def _m_inf_T(self, voltage):
        return 1.0 / (1.0 + exp(-(voltage + 52.0) / 7.4))

    def _h_inf_T(self, voltage):
        return 1.0 / (1.0 + exp((voltage + 80.0) / 5.0))

    def _tau_h_T(self, voltage):
        return (85.0 + 1.0 / (exp((voltage + 48.0) / 4.0) + exp(-(voltage + 407.0) / 50.0))) / 3.7371928

    def _initialize_state_vector(self):
        """
        Initialize state vector.
        """
        self.initial_state = [
            self.params["E_L"],
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]

    def _derivatives(self, coupling_variables):
        (
            voltage,
            h_T,
            syn_ext,
            syn_inh,
            dsyn_ext,
            dsyn_inh,
            firing_rate,
        ) = self._unwrap_state_vector()
        # voltage dynamics
        d_voltage = -(
            self._get_leak_current(voltage)
            + self._get_excitatory_current(voltage, syn_ext)
            + self._get_inhibitory_current(voltage, syn_inh)
            + self.params["ext_current"]
        ) / self.params["tau"] - (1.0 / self.params["C_m"]) * (
            self._get_potassium_leak_current(voltage) + self._get_T_type_current(voltage, h_T)
        )
        # channel dynamics: T-type
        d_h_T = (self._h_inf_T(voltage) - h_T) / self._tau_h_T(voltage)
        # synaptic dynamics
        d_syn_ext = dsyn_ext
        d_syn_inh = dsyn_inh
        d_dsyn_ext = (
            self.params["gamma_e"] ** 2
            * (
                coupling_variables["node_inh_exc"]
                + coupling_variables["network_inh_exc"]
                + system_input(self.noise_input_idx[0])
                - syn_ext
            )
            - 2 * self.params["gamma_e"] * dsyn_ext
        )
        d_dsyn_inh = (
            self.params["gamma_r"] ** 2 * (coupling_variables["node_inh_inh"] - syn_inh)
            - 2 * self.params["gamma_r"] * dsyn_inh
        )
        # firing rate as dummy dynamical variable with infinitely fast
        # fixed-point dynamics
        firing_rate_now = self._get_firing_rate(voltage)
        d_firing_rate = -self.params["lambda"] * (firing_rate - firing_rate_now)

        return [
            d_voltage,
            d_h_T,
            d_syn_ext,
            d_syn_inh,
            d_dsyn_ext,
            d_dsyn_inh,
            d_firing_rate,
        ]


class ThalamicNode(SingleCouplingExcitatoryInhibitoryNode):
    """
    Thalamic mass model network node with 1 excitatory (TCR) and 1 inhibitory
    (TRN) population due to Costa et al.
    """

    name = "Thalamic mass model node"
    label = "THLMnode"

    default_network_coupling = {"network_exc_exc": 0.0, "network_inh_exc": 0.0}
    default_output = f"r_mean_{EXC}"
    output_vars = [f"r_mean_{EXC}", f"r_mean_{INH}", f"V_{EXC}", f"V_{INH}"]

    def __init__(
        self,
        tcr_params=None,
        trn_params=None,
        connectivity=THALAMUS_NODE_DEFAULT_CONNECTIVITY,
    ):
        """
        :param tcr_params: parameters for the excitatory (TCR) mass
        :type tcr_params: dict|None
        :param trn_params: parameters for the inhibitory (TRN) mass
        :type trn_params: dict|None
        :param connectivity: local connectivity matrix
        :type connectivity: np.ndarray
        """
        tcr_mass = ThalamocorticalMass(params=tcr_params)
        tcr_mass.index = 0
        trn_mass = ThalamicReticularMass(params=trn_params)
        trn_mass.index = 1
        super().__init__(
            neural_masses=[tcr_mass, trn_mass],
            local_connectivity=connectivity,
            # within thalamic node there are no local delays
            local_delays=None,
        )
