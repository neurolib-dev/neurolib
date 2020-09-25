"""
Adaptive exponential integrate-and-fire mean-field approximation.

Main reference:
    Augustin, M., Ladenbauer, J., Baumann, F., & Obermayer, K. (2017).
    Low-dimensional spike rate models derived from networks of adaptive
    integrate-and-fire neurons: comparison and implementation. PLoS Comput Biol,
    13(6), e1005545.

Additional reference:
    Cakan C., Obermayer K. (2020). Biophysically grounded mean-field models of
    neural populations under electrical stimulation. PLoS Comput Biol, 16(4),
    e1007822.
"""

import logging
import os
from copy import deepcopy

import numba
import numpy as np
import symengine as se
from h5py import File
from jitcdde import input as system_input

from ..builder.base.constants import EXC, INH, LAMBDA_SPEED
from ..builder.base.network import Network, SingleCouplingExcitatoryInhibitoryNode
from ..builder.base.neural_mass import NeuralMass

DEFAULT_CASCADE_FILENAME = "quantities_cascade.h5"

DEFAULT_PARAMS_EXC = {
    # number of inputs per neuron from EXC/INH
    "K_exc": 800.0,
    "K_inh": 200.0,
    # postsynaptic potential amplitude for global connectome
    "c_global": 0.4,
    # number of incoming EXC connections (to EXC population) from each area
    "K_exc_global": 250.0,
    # synaptic time constant EXC/INH
    "tau_syn_exc": 2.0,  # ms
    "tau_syn_inh": 5.0,  # ms
    # internal Fokker-Planck noise due to random coupling
    "sigma_ext": 1.5,  # mV/sqrt(ms)
    # maximum synaptic current between EXC/INH nodes in mV/ms
    "J_exc_max": 2.43,
    "J_inh_max": -3.3,
    # single neuron parameters
    "C_m": 200.0,
    "g_L": 10.0,
    # external drives
    "ext_current": 0.0,
    "ext_rate": 0.0,
    # adaptation current model parameters
    # subthreshold adaptation conductance
    "a": 15.0,  # nS
    # spike-triggered adaptation increment
    "b": 40.0,  # pA
    "E_A": -80.0,
    "tau_A": 200.0,
    "lambda": LAMBDA_SPEED,
}

DEFAULT_PARAMS_INH = {
    # number of inputs per neuron from EXC/INH
    "K_exc": 800.0,
    "K_inh": 200.0,
    # postsynaptic potential amplitude for global connectome
    "c_global": 0.4,
    # number of incoming EXC connections (to EXC population) from each area
    "K_exc_global": 250.0,
    # synaptic time constant EXC/INH
    "tau_syn_exc": 2.0,  # ms
    "tau_syn_inh": 5.0,  # ms
    # internal Fokker-Planck noise due to random coupling
    "sigma_ext": 1.5,  # mV/sqrt(ms)
    # maximum synaptic current between EXC/INH nodes in mV/ms
    "J_exc_max": 2.60,
    "J_inh_max": -1.64,
    # single neuron parameters
    "C_m": 200.0,
    "g_L": 10.0,
    # external drives
    "ext_current": 0.0,
    "ext_rate": 0.0,
    "lambda": LAMBDA_SPEED,
}
# matrices as [to, from], masses as (EXC, INH)
# EXC is index 0, INH is index 1
DEFAULT_ADEX_NODE_CONNECTIVITY = np.array([[0.3, 0.5], [0.3, 0.5]])
# same but delays, in ms
DEFAULT_ADEX_NODE_DELAYS = np.array([[4.0, 2.0], [4.0, 2.0]])


@numba.njit()
def _get_interpolation_values(xi, yi, sigma_range, mu_range, d_sigma, d_mu):
    """
    Return values needed for interpolation: bilinear (2D) interpolation
    within ranges, linear (1D) if "one edge" is crossed, corner value if
    "two edges" are crossed. Defined as jitted function due to compatibility
    with numba backend.

    :param xi: interpolation value on x-axis, i.e. I_sigma
    :type xi: float
    :param yi: interpolation value on y-axis, i.e. I_mu
    :type yi: float
    :param sigma_range: range of x-axis, i.e. sigma values
    :type sigma_range: np.ndarray
    :param mu_range: range of y-axis, i.e. mu values
    :type mu_range: np.ndarray
    :param d_sigma: grid coarsness in the x-axis, i.e. sigma values
    :type d_sigma: float
    :param d_mu: grid coarsness in the y-axis, i.e. mu values
    :type d_mu: float
    :return: index of the lower interpolation value on x-axis, index of the
        lower interpolation value on y-axis, distance of xi to the lower
        value, distance of yi to the lower value
    :rtype: (int, int, float, float)
    """
    # within all boundaries
    if xi >= sigma_range[0] and xi < sigma_range[-1] and yi >= mu_range[0] and yi < mu_range[-1]:
        xid = (xi - sigma_range[0]) / d_sigma
        xid1 = np.floor(xid)
        dxid = xid - xid1
        yid = (yi - mu_range[0]) / d_mu
        yid1 = np.floor(yid)
        dyid = yid - yid1
        return int(xid1), int(yid1), dxid, dyid

    # outside one boundary
    if yi < mu_range[0]:
        yid1 = 0
        dyid = 0.0
        if xi >= sigma_range[0] and xi < sigma_range[-1]:
            xid = (xi - sigma_range[0]) / d_sigma
            xid1 = np.floor(xid)
            dxid = xid - xid1
        elif xi < sigma_range[0]:
            xid1 = 0
            dxid = 0.0
        else:  # xi >= x(end)
            xid1 = -1
            dxid = 0.0
        return int(xid1), int(yid1), dxid, dyid

    if yi >= mu_range[-1]:
        yid1 = -1
        dyid = 0.0
        if xi >= sigma_range[0] and xi < sigma_range[-1]:
            xid = (xi - sigma_range[0]) / d_sigma
            xid1 = np.floor(xid)
            dxid = xid - xid1
        elif xi < sigma_range[0]:
            xid1 = 0
            dxid = 0.0
        else:  # xi >= x(end)
            xid1 = -1
            dxid = 0.0
        return int(xid1), int(yid1), dxid, dyid

    if xi < sigma_range[0]:
        xid1 = 0
        dxid = 0.0
        yid = (yi - mu_range[0]) / d_mu
        yid1 = np.floor(yid)
        dyid = yid - yid1
        return int(xid1), int(yid1), dxid, dyid

    if xi >= sigma_range[-1]:
        xid1 = -1
        dxid = 0.0
        yid = (yi - mu_range[0]) / d_mu
        yid1 = np.floor(yid)
        dyid = yid - yid1
        return int(xid1), int(yid1), dxid, dyid


@numba.njit()
def _table_lookup(
    current_mu, current_sigma, sigma_range, mu_range, d_sigma, d_mu, cascade_table,
):
    """
    Translate mean and std. deviation of the current to selected quantity using
    linear-nonlinear lookup table for AdEx. Defined as jitted function due to
    compatibility with numba backend.
    """
    x_idx, y_idx, dx_idx, dy_idx = _get_interpolation_values(
        current_sigma, current_mu, sigma_range, mu_range, d_sigma, d_mu
    )
    return (
        cascade_table[y_idx, x_idx] * (1 - dx_idx) * (1 - dy_idx)
        + cascade_table[y_idx, x_idx + 1] * dx_idx * (1 - dy_idx)
        + cascade_table[y_idx + 1, x_idx] * (1 - dx_idx) * dy_idx
        + cascade_table[y_idx + 1, x_idx + 1] * dx_idx * dy_idx
    )


class AdExMass(NeuralMass):
    """
    Adaptive exponential integrate-and-fire mean-field mass. Can be excitatory
    or inhibitory, depending on the parameters.
    """

    name = "AdEx mean-field mass"
    label = "AdExMass"
    # define python callback function name for table lookup (linear-nonlinear
    # approximation of Fokker-Planck equation)
    python_callbacks = ["firing_rate_lookup", "voltage_lookup", "tau_lookup"]

    num_noise_variables = 1

    @staticmethod
    def _rescale_strengths(params):
        """
        Rescale connection strengths for AdEx.
        """
        params = deepcopy(params)
        assert isinstance(params, dict)
        params["c_global"] = params["c_global"] * params["tau_syn_exc"] / params["J_exc_max"]

        params["tau_m"] = params["C_m"] / params["g_L"]
        return params

    def __init__(self, params, lin_nonlin_cascade_filename=None, seed=None):
        """
        :param lin_nonlin_cascade_filename: filename for precomputed
            linear-nonlinear cascade for AdEx, if None, will look for it in this
            directory
        :type lin_nonlin_cascade_filename: str|None
        :param seed: seed for random number generator
        :type seed: int|None
        """
        params = self._rescale_strengths(params)
        super().__init__(params=params, seed=seed)
        # use the same file as neurolib's native
        lin_nonlin_cascade_filename = lin_nonlin_cascade_filename or os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "..", "..", "aln", "aln-precalc", DEFAULT_CASCADE_FILENAME,
        )
        self._load_lin_nonlin_cascade(lin_nonlin_cascade_filename)

    def _load_lin_nonlin_cascade(self, filename):
        """
        Load precomputed linear-nonlinear cascade from h5 file.
        """
        # load linear-nonlinear cascade quantities from file
        logging.info(f"Reading precomputed quantities from {filename}")
        loaded_h5 = File(filename, "r")
        self.mu_range = np.array(loaded_h5["mu_vals"])
        self.d_mu = self.mu_range[1] - self.mu_range[0]
        self.sigma_range = np.array(loaded_h5["sigma_vals"])
        self.d_sigma = self.sigma_range[1] - self.sigma_range[0]
        self.firing_rate_cascade = np.array(loaded_h5["r_ss"])
        self.voltage_cascade = np.array(loaded_h5["V_mean_ss"])
        self.tau_cascade = np.array(loaded_h5["tau_mu_exp"])
        logging.info("Loading done, all quantities loaded.")
        # close the file
        loaded_h5.close()
        self.lin_nonlin_fname = filename

    def update_params(self, params_dict):
        """
        Update parameters as in the base class but also rescale.
        """
        # if we are changing C_m or g_L, update tau_m as well
        if any(k in params_dict for k in ("C_m", "g_L")):
            C_m = params_dict["C_m"] if "C_m" in params_dict else self.params["C_m"]
            g_L = params_dict["g_L"] if "g_L" in params_dict else self.params["g_L"]
            params_dict["tau_m"] = C_m / g_L

        # if we are changing any of the J_exc_max, tau_syn_exc or c_global, rescale c_global
        if any(k in params_dict for k in ("c_global", "J_exc_max", "tau_syn_exc")):
            # get original c_global
            c_global = (
                params_dict["c_global"]
                if "c_global" in params_dict
                else self.params["c_global"] * (self.params["J_exc_max"] / self.params["tau_syn_exc"])
            )
            tau_syn_exc = params_dict["tau_syn_exc"] if "tau_syn_exc" in params_dict else self.params["tau_syn_exc"]
            J_exc_max = params_dict["J_exc_max"] if "J_exc_max" in params_dict else self.params["J_exc_max"]
            params_dict["c_global"] = c_global * tau_syn_exc / J_exc_max

        # update all parameters finally
        super().update_params(params_dict)

    def describe(self):
        return {
            **super().describe(),
            "lin_nonlin_cascade_filename": self.lin_nonlin_fname,
        }

    def _callbacks(self):
        """
        Construct list of python callbacks for AdEx model.
        """
        callbacks_list = [
            (self.callback_functions["firing_rate_lookup"], self.firing_rate_lookup, 2),
            (self.callback_functions["voltage_lookup"], self.voltage_lookup, 2),
            (self.callback_functions["tau_lookup"], self.tau_lookup, 2),
        ]
        self._validate_callbacks(callbacks_list)
        return callbacks_list

    def _numba_callbacks(self):
        """
        Define numba callbacks - has to be different than jitcdde callbacks
        because of the internals.
        """

        def _table_numba_gen(sigma_range, mu_range, d_sigma, d_mu, cascade):
            """
            Function generator for numba callbacks. This works similarly as
            `functools.partial` (i.e. sets some of the arguments of the inner
            function), but afterwards can be jitted with `numba.njit()`, while
            partial functions cannot.
            """

            def inner(current_mu, current_sigma):
                return _table_lookup(current_mu, current_sigma, sigma_range, mu_range, d_sigma, d_mu, cascade,)

            return inner

        return [
            (
                "firing_rate_lookup",
                numba.njit(
                    _table_numba_gen(
                        self.sigma_range, self.mu_range, self.d_sigma, self.d_mu, self.firing_rate_cascade,
                    )
                ),
            ),
            (
                "voltage_lookup",
                numba.njit(
                    _table_numba_gen(self.sigma_range, self.mu_range, self.d_sigma, self.d_mu, self.voltage_cascade)
                ),
            ),
            (
                "tau_lookup",
                numba.njit(
                    _table_numba_gen(self.sigma_range, self.mu_range, self.d_sigma, self.d_mu, self.tau_cascade)
                ),
            ),
        ]

    def firing_rate_lookup(self, y, current_mu, current_sigma):
        """
        Translate mean and std. deviation of the current to firing rate using
        linear-nonlinear lookup table for AdEx.
        """
        return _table_lookup(
            current_mu,
            current_sigma,
            self.sigma_range,
            self.mu_range,
            self.d_sigma,
            self.d_mu,
            self.firing_rate_cascade,
        )

    def voltage_lookup(self, y, current_mu, current_sigma):
        """
        Translate mean and std. deviation of the current to voltage using
        linear-nonlinear lookup table for AdEx.
        """
        return _table_lookup(
            current_mu, current_sigma, self.sigma_range, self.mu_range, self.d_sigma, self.d_mu, self.voltage_cascade,
        )

    def tau_lookup(self, y, current_mu, current_sigma):
        """
        Translate mean and std. deviation of the current to tau - membrane time
        constant using linear-nonlinear lookup table for AdEx.
        """
        return _table_lookup(
            current_mu, current_sigma, self.sigma_range, self.mu_range, self.d_sigma, self.d_mu, self.tau_cascade,
        )


class ExcitatoryAdExMass(AdExMass):
    """
    Excitatory AdEx neural mass. Contains firing rate adaptation current.
    """

    name = "AdEx mean-field excitatory mass"
    label = f"AdExMass{EXC}"
    num_state_variables = 7
    coupling_variables = {6: f"q_mean_{EXC}"}
    mass_type = EXC
    state_variable_names = [
        "I_mu",
        "I_A",
        "I_syn_mu_exc",
        "I_syn_mu_inh",
        "I_syn_sigma_exc",
        "I_syn_sigma_inh",
        "q_mean",
    ]
    required_couplings = [
        "node_exc_exc",
        "node_exc_exc_sq",
        "node_exc_inh",
        "node_exc_inh_sq",
        "network_exc_exc",
        "network_exc_exc_sq",
    ]
    required_params = [
        "K_exc",
        "K_inh",
        "c_global",
        "K_exc_global",
        "tau_syn_exc",
        "tau_syn_inh",
        "sigma_ext",
        "J_exc_max",
        "J_inh_max",
        "tau_m",
        "C_m",
        "ext_current",
        "ext_rate",
        "a",
        "b",
        "E_A",
        "tau_A",
        "lambda",
    ]

    def __init__(self, params=None, lin_nonlin_cascade_filename=None, seed=None):
        super().__init__(
            params=params or DEFAULT_PARAMS_EXC, lin_nonlin_cascade_filename=lin_nonlin_cascade_filename, seed=seed
        )

    def _initialize_state_vector(self):
        """
        Initialize state vector.
        """
        np.random.seed(self.seed)
        self.initial_state = (
            np.random.uniform(0, 1, self.num_state_variables) * np.array([3.0, 200.0, 0.5, 0.5, 0.001, 0.001, 0.01])
        ).tolist()

    def _compute_couplings(self, coupling_variables):
        """
        Helper that computes coupling from other nodes and network.
        """
        exc_coupling = (
            self.params["K_exc"] * coupling_variables["node_exc_exc"]
            + self.params["c_global"] * self.params["K_exc_global"] * coupling_variables["network_exc_exc"]
            + self.params["c_global"] * self.params["K_exc_global"] * self.params["ext_rate"]
        )
        inh_coupling = self.params["K_inh"] * coupling_variables["node_exc_inh"]
        exc_coupling_squared = (
            self.params["K_exc"] * coupling_variables["node_exc_exc_sq"]
            + self.params["c_global"] ** 2 * self.params["K_exc_global"] * coupling_variables["network_exc_exc_sq"]
            + self.params["c_global"] ** 2 * self.params["K_exc_global"] * self.params["ext_rate"]
        )
        inh_coupling_squared = self.params["K_inh"] * coupling_variables["node_exc_inh_sq"]

        return (
            exc_coupling,
            inh_coupling,
            exc_coupling_squared,
            inh_coupling_squared,
        )

    def _derivatives(self, coupling_variables):
        (
            I_mu,
            I_adaptation,
            I_syn_mu_exc,
            I_syn_mu_inh,
            I_syn_sigma_exc,
            I_syn_sigma_inh,
            firing_rate,
        ) = self._unwrap_state_vector()

        exc_inp, inh_inp, exc_inp_sq, inh_inp_sq = self._compute_couplings(coupling_variables)

        I_sigma = se.sqrt(
            2
            * self.params["J_exc_max"] ** 2
            * I_syn_sigma_exc
            * self.params["tau_syn_exc"]
            * self.params["tau_m"]
            / ((1.0 + exc_inp) * self.params["tau_m"] + self.params["tau_syn_exc"])
            + 2
            * self.params["J_inh_max"] ** 2
            * I_syn_sigma_inh
            * self.params["tau_syn_inh"]
            * self.params["tau_m"]
            / ((1.0 + inh_inp) * self.params["tau_m"] + self.params["tau_syn_inh"])
            + self.params["sigma_ext"] ** 2
        )

        # get values from linear-nonlinear lookup table
        firing_rate_now = self.callback_functions["firing_rate_lookup"](
            I_mu - I_adaptation / self.params["C_m"], I_sigma
        )
        voltage = self.callback_functions["voltage_lookup"](I_mu - I_adaptation / self.params["C_m"], I_sigma)
        tau = self.callback_functions["tau_lookup"](I_mu - I_adaptation / self.params["C_m"], I_sigma)

        d_I_mu = (
            self.params["J_exc_max"] * I_syn_mu_exc
            + self.params["J_inh_max"] * I_syn_mu_inh
            + system_input(self.noise_input_idx[0])
            + self.params["ext_current"]
            - I_mu
        ) / tau

        d_I_adaptation = (
            self.params["a"] * (voltage - self.params["E_A"])
            - I_adaptation
            + self.params["tau_A"] * self.params["b"] * firing_rate_now
        ) / self.params["tau_A"]

        d_I_syn_mu_exc = ((1.0 - I_syn_mu_exc) * exc_inp - I_syn_mu_exc) / self.params["tau_syn_exc"]

        d_I_syn_mu_inh = ((1.0 - I_syn_mu_inh) * inh_inp - I_syn_mu_inh) / self.params["tau_syn_inh"]

        d_I_syn_sigma_exc = (
            (1.0 - I_syn_mu_exc) ** 2 * exc_inp_sq
            + (exc_inp_sq - 2.0 * self.params["tau_syn_exc"] * (exc_inp + 1.0)) * I_syn_sigma_exc
        ) / (self.params["tau_syn_exc"] ** 2)

        d_I_syn_sigma_inh = (
            (1.0 - I_syn_mu_inh) ** 2 * inh_inp_sq
            + (inh_inp_sq - 2.0 * self.params["tau_syn_inh"] * (inh_inp + 1.0)) * I_syn_sigma_inh
        ) / (self.params["tau_syn_inh"] ** 2)
        # firing rate as dummy dynamical variable with infinitely fast
        # fixed-point dynamics
        d_firing_rate = -self.params["lambda"] * (firing_rate - firing_rate_now)

        return [
            d_I_mu,
            d_I_adaptation,
            d_I_syn_mu_exc,
            d_I_syn_mu_inh,
            d_I_syn_sigma_exc,
            d_I_syn_sigma_inh,
            d_firing_rate,
        ]


class InhibitoryAdExMass(AdExMass):
    """
    Inhibitory AdEx neural mass. In contrast to excitatory, inhibitory mass do
    not contain fiting rate adaptation current.
    """

    name = "AdEx mean-field inhibitory mass"
    label = f"AdExMass{INH}"
    num_state_variables = 6
    coupling_variables = {5: f"q_mean_{INH}"}
    mass_type = INH
    state_variable_names = [
        "I_mu",
        "I_syn_mu_exc",
        "I_syn_mu_inh",
        "I_syn_sigma_exc",
        "I_syn_sigma_inh",
        "q_mean",
    ]
    required_couplings = [
        "node_inh_exc",
        "node_inh_exc_sq",
        "node_inh_inh",
        "node_inh_inh_sq",
    ]
    required_params = [
        "K_exc",
        "K_inh",
        "c_global",
        "K_exc_global",
        "tau_syn_exc",
        "tau_syn_inh",
        "sigma_ext",
        "J_exc_max",
        "J_inh_max",
        "tau_m",
        "C_m",
        "ext_current",
        "ext_rate",
        "lambda",
    ]

    def __init__(self, params=None, lin_nonlin_cascade_filename=None, seed=None):
        super().__init__(
            params=params or DEFAULT_PARAMS_INH, lin_nonlin_cascade_filename=lin_nonlin_cascade_filename, seed=seed
        )

    def _initialize_state_vector(self):
        """
        Initialize state vector.
        """
        np.random.seed(self.seed)
        self.initial_state = (
            np.random.uniform(0, 1, self.num_state_variables) * np.array([3.0, 0.5, 0.5, 0.01, 0.01, 0.01])
        ).tolist()

    def _compute_couplings(self, coupling_variables):
        """
        Helper that computes coupling from other nodes and network.
        """
        exc_coupling = (
            self.params["K_exc"] * coupling_variables["node_inh_exc"]
            + self.params["c_global"] * self.params["K_exc_global"] * self.params["ext_rate"]
        )
        inh_coupling = self.params["K_inh"] * coupling_variables["node_inh_inh"]
        exc_coupling_squared = (
            self.params["K_exc"] * coupling_variables["node_inh_exc_sq"]
            + self.params["c_global"] ** 2 * self.params["K_exc_global"] * self.params["ext_rate"]
        )
        inh_coupling_squared = self.params["K_inh"] * coupling_variables["node_inh_inh_sq"]

        return (
            exc_coupling,
            inh_coupling,
            exc_coupling_squared,
            inh_coupling_squared,
        )

    def _derivatives(self, coupling_variables):
        (I_mu, I_syn_mu_exc, I_syn_mu_inh, I_syn_sigma_exc, I_syn_sigma_inh, firing_rate,) = self._unwrap_state_vector()

        exc_inp, inh_inp, exc_inp_sq, inh_inp_sq = self._compute_couplings(coupling_variables)

        I_sigma = se.sqrt(
            2
            * self.params["J_exc_max"] ** 2
            * I_syn_sigma_exc
            * self.params["tau_syn_exc"]
            * self.params["tau_m"]
            / ((1.0 + exc_inp) * self.params["tau_m"] + self.params["tau_syn_exc"])
            + 2
            * self.params["J_inh_max"] ** 2
            * I_syn_sigma_inh
            * self.params["tau_syn_inh"]
            * self.params["tau_m"]
            / ((1.0 + inh_inp) * self.params["tau_m"] + self.params["tau_syn_inh"])
            + self.params["sigma_ext"] ** 2
        )

        # get values from linear-nonlinear lookup table
        firing_rate_now = self.callback_functions["firing_rate_lookup"](I_mu, I_sigma)
        tau = self.callback_functions["tau_lookup"](I_mu, I_sigma)

        d_I_mu = (
            self.params["J_exc_max"] * I_syn_mu_exc
            + self.params["J_inh_max"] * I_syn_mu_inh
            + system_input(self.noise_input_idx[0])
            + self.params["ext_current"]
            - I_mu
        ) / tau

        d_I_syn_mu_exc = ((1.0 - I_syn_mu_exc) * exc_inp - I_syn_mu_exc) / self.params["tau_syn_exc"]

        d_I_syn_mu_inh = ((1.0 - I_syn_mu_inh) * inh_inp - I_syn_mu_inh) / self.params["tau_syn_inh"]

        d_I_syn_sigma_exc = (
            (1.0 - I_syn_mu_exc) ** 2 * exc_inp_sq
            + (exc_inp_sq - 2.0 * self.params["tau_syn_exc"] * (exc_inp + 1.0)) * I_syn_sigma_exc
        ) / (self.params["tau_syn_exc"] ** 2)

        d_I_syn_sigma_inh = (
            (1.0 - I_syn_mu_inh) ** 2 * inh_inp_sq
            + (inh_inp_sq - 2.0 * self.params["tau_syn_inh"] * (inh_inp + 1.0)) * I_syn_sigma_inh
        ) / (self.params["tau_syn_inh"] ** 2)
        # firing rate as dummy dynamical variable with infinitely fast
        # fixed-point dynamics
        d_firing_rate = -self.params["lambda"] * (firing_rate - firing_rate_now)

        return [
            d_I_mu,
            d_I_syn_mu_exc,
            d_I_syn_mu_inh,
            d_I_syn_sigma_exc,
            d_I_syn_sigma_inh,
            d_firing_rate,
        ]


class AdExNode(SingleCouplingExcitatoryInhibitoryNode):
    """
    Default AdEx network node with 1 excitatory (featuring adaptive current) and
    1 inhibitory population.
    """

    name = "AdEx mean-field node"
    label = "AdExNode"

    sync_variables = [
        "node_exc_exc",
        "node_inh_exc",
        "node_exc_inh",
        "node_inh_inh",
        # squared variants
        "node_exc_exc_sq",
        "node_inh_exc_sq",
        "node_exc_inh_sq",
        "node_inh_inh_sq",
    ]

    default_network_coupling = {
        "network_exc_exc": 0.0,
        "network_exc_exc_sq": 0.0,
    }

    default_output = f"q_mean_{EXC}"

    def _rescale_connectivity(self):
        """
        Rescale connection strengths for AdEx. Should work also for AdEx nodes
        with arbitrary number of masses of any type.
        """
        # create tau and J_max matrices for rescaling
        tau_mat = np.zeros_like(self.connectivity)
        J_mat = np.zeros_like(self.connectivity)
        for col, mass_from in enumerate(self.masses):
            # taus are constant per col and depends only on "from" mass
            tau_mat[:, col] = mass_from.params[f"tau_syn_{mass_from.mass_type.lower()}"]
            # Js are specific: take J from "to" mass but of type "from" mass
            for row, mass_to in enumerate(self.masses):
                J_mat[row, col] = mass_to.params[f"J_{mass_from.mass_type.lower()}_max"]

        # multiplication with tau makes the increase of synaptic activity
        # subject to a single input spike invariant to tau and division by J
        # ensures that mu = J*s will result in a PSP of exactly c for a single
        # spike
        self.connectivity = (self.connectivity * tau_mat) / np.abs(J_mat)

    def __init__(
        self,
        exc_params=None,
        inh_params=None,
        exc_lin_nonlin_cascade_filename=None,
        inh_lin_nonlin_cascade_filename=None,
        connectivity=DEFAULT_ADEX_NODE_CONNECTIVITY,
        delays=DEFAULT_ADEX_NODE_DELAYS,
        exc_seed=None,
        inh_seed=None,
    ):
        """
        :param exc_params: parameters for the excitatory mass
        :type exc_params: dict|None
        :param inh_params: parameters for the inhibitory mass
        :type inh_params: dict|None
        :param exc_lin_nonlin_cascade_filename: filename for precomputed
            linear-nonlinear cascade for excitatory AdEx mass, if None, will
            look for it in this directory
        :type exc_lin_nonlin_cascade_filename: str|None
        :param inh_lin_nonlin_cascade_filename: filename for precomputed
            linear-nonlinear cascade for inhibitory AdEx mass, if None, will
            look for it in this directory
        :type inh_lin_nonlin_cascade_filename: str|None
        :param connectivity: local connectivity matrix
        :type connectivity: np.ndarray
        :param delays: local delay matrix
        :type delays: np.ndarray
        :param exc_seed: seed for random number generator for the excitatory
            mass
        :type exc_seed: int|None
        :param inh_seed: seed for random number generator for the inhibitory
            mass
        :type inh_seed: int|None
        """
        excitatory_mass = ExcitatoryAdExMass(
            params=exc_params, lin_nonlin_cascade_filename=exc_lin_nonlin_cascade_filename, seed=exc_seed
        )
        excitatory_mass.index = 0
        inhibitory_mass = InhibitoryAdExMass(
            params=inh_params, lin_nonlin_cascade_filename=inh_lin_nonlin_cascade_filename, seed=inh_seed
        )
        inhibitory_mass.index = 1
        super().__init__(
            neural_masses=[excitatory_mass, inhibitory_mass], local_connectivity=connectivity, local_delays=delays,
        )
        self._rescale_connectivity()

    def update_params(self, params_dict):
        """
        Rescale connectivity after params update if connectivity was updated.
        """
        rescale_flag = "local_connectivity" in params_dict
        super().update_params(params_dict)
        if rescale_flag:
            self._rescale_connectivity()

    def _sync(self):
        """
        Apart from basic EXC<->INH connectivity, construct also squared
        variants.
        """
        connectivity_sq = self.connectivity ** 2 * self.inputs
        sq_connectivity = [
            (
                # exc -> exc squared connectivity
                self.sync_symbols[f"node_exc_exc_sq_{self.index}"],
                sum([connectivity_sq[row, col] for row in self.excitatory_masses for col in self.excitatory_masses]),
            ),
            (
                # exc -> inh squared connectivity
                self.sync_symbols[f"node_inh_exc_sq_{self.index}"],
                sum([connectivity_sq[row, col] for row in self.inhibitory_masses for col in self.excitatory_masses]),
            ),
            (
                # inh -> exc squared connectivity
                self.sync_symbols[f"node_exc_inh_sq_{self.index}"],
                sum([connectivity_sq[row, col] for row in self.excitatory_masses for col in self.inhibitory_masses]),
            ),
            (
                # inh -> inh squared connectivity
                self.sync_symbols[f"node_inh_inh_sq_{self.index}"],
                sum([connectivity_sq[row, col] for row in self.inhibitory_masses for col in self.inhibitory_masses]),
            ),
        ]
        return super()._sync() + sq_connectivity


class AdExNetwork(Network):
    """
    Whole brain network of adaptive exponential integrate-and-fire mean-field
    excitatory and inhibitory nodes.
    """

    name = "AdEx mean-field network"
    label = "AdExNet"

    sync_variables = ["network_exc_exc", "network_exc_exc_sq"]

    def __init__(
        self,
        connectivity_matrix,
        delay_matrix,
        exc_mass_params=None,
        inh_mass_params=None,
        exc_lin_nonlin_cascade_filename=None,
        inh_lin_nonlin_cascade_filename=None,
        local_connectivity=DEFAULT_ADEX_NODE_CONNECTIVITY,
        local_delays=DEFAULT_ADEX_NODE_DELAYS,
        exc_seed=None,
        inh_seed=None,
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
        :param exc_mass_params: parameters for each excitatory AdEx neural
            mass, if None, will use default
        :type exc_mass_params: list[dict]|dict|None
        :param inh_mass_params: parameters for each inhibitory AdEx neural
            mass, if None, will use default
        :type inh_mass_params: list[dict]|dict|None
        param exc_lin_nonlin_cascade_filename: filename for precomputed
            linear-nonlinear cascade for excitatory AdEx mass, if None, will
            look for it in this directory
        :type exc_lin_nonlin_cascade_filename: list[str]|str|None
        :param inh_lin_nonlin_cascade_filename: filename for precomputed
            linear-nonlinear cascade for inhibitory AdEx mass, if None, will
            look for it in this directory
        :type inh_lin_nonlin_cascade_filename: list[str]|str|None
        :param local_connectivity: local within-node connectivity matrix
        :type local_connectivity: np.ndarray
        :param local_delays: local within-node delay matrix
        :type local_delays: list[np.ndarray]|np.ndarray
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
        exc_lin_nonlin_cascade_filename = self._prepare_mass_params(
            exc_lin_nonlin_cascade_filename, num_nodes, native_type=str
        )
        inh_lin_nonlin_cascade_filename = self._prepare_mass_params(
            inh_lin_nonlin_cascade_filename, num_nodes, native_type=str
        )
        local_connectivity = self._prepare_mass_params(local_connectivity, num_nodes, native_type=np.ndarray)
        local_delays = self._prepare_mass_params(local_delays, num_nodes, native_type=np.ndarray)
        exc_seeds = self._prepare_mass_params(exc_seed, num_nodes, native_type=int)
        inh_seeds = self._prepare_mass_params(inh_seed, num_nodes, native_type=int)

        nodes = []
        for (i, (exc_params, inh_params, exc_cascade, inh_cascade, local_conn, local_dels,),) in enumerate(
            zip(
                exc_mass_params,
                inh_mass_params,
                exc_lin_nonlin_cascade_filename,
                inh_lin_nonlin_cascade_filename,
                local_connectivity,
                local_delays,
            )
        ):
            node = AdExNode(
                exc_params=exc_params,
                inh_params=inh_params,
                exc_lin_nonlin_cascade_filename=exc_cascade,
                inh_lin_nonlin_cascade_filename=inh_cascade,
                connectivity=local_conn,
                delays=local_dels,
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
            nodes=nodes, connectivity_matrix=connectivity_matrix, delay_matrix=delay_matrix,
        )
        # assert we have 2 sync variable
        assert len(self.sync_variables) == 2

    def _sync(self):
        """
        Need to redefine vanilla sync, since AdEx network is more involved: it
        contains squared coupling weights and non-trivial coupling indices.
        """
        # get coupling variable index from excitatory mass within each node
        coupling_var_idx = set(sum([list(node[0].coupling_variables.keys()) for node in self], []))
        assert len(coupling_var_idx) == 1
        coupling_var_idx = next(iter(coupling_var_idx))
        return (
            # regular additive coupling
            self._additive_coupling(within_node_idx=coupling_var_idx, symbol="network_exc_exc")
            # additive coupling with squared weights
            + self._additive_coupling(
                within_node_idx=coupling_var_idx,
                symbol="network_exc_exc_sq",
                # multiplier is connectivity again, then they'll be squared
                connectivity_multiplier=self.connectivity,
            )
            + super()._sync()
        )
