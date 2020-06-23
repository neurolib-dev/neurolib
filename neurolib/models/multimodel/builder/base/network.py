"""
Base for "network" and "node" operations on neural masses.
"""
import logging

import numpy as np
import symengine as se
from jitcdde import t as time_vector
from jitcdde import y as state_vector

from .backend import BackendIntegrator
from .constants import (
    EXC,
    MASS_NAME_STR,
    NETWORK_CONNECTIVITY,
    NETWORK_DELAYS,
    NETWORK_NAME_STR,
    NODAL_CONNECTIVITY,
    NODAL_DELAYS,
    NODE_NAME_STR,
)
from .neural_mass import NeuralMass


class Node(BackendIntegrator):
    """
    Base class for all nodes within the whole-brain network.
    """

    name = ""
    label = ""

    # index of this node with respect to the whole network
    index = None

    # list of synchronisation variables that are used to compute input/output
    # from different masses in this node, implemented as `jitcdde` helpers
    sync_variables = []

    # default network coupling - will be used if network coupling is None,
    # usually used for isolated node
    default_network_coupling = {}

    # default output of the model - all nodes need to have this variable defined
    default_output = None

    def __init__(self, neural_masses):
        """
        :param neural_masses: list of neural masses in this node
        :type neural_masses: list[NeuralMass]
        """
        assert all(isinstance(mass, NeuralMass) for mass in neural_masses)
        self.masses = neural_masses
        # number of state variables for whole node, i.e. sum of state variables
        # for all masses
        self.num_state_variables = sum([mass.num_state_variables for mass in self])
        self.num_noise_variables = sum([mass.num_noise_variables for mass in self])
        self.idx_state_var = None
        self.initialised = False
        assert self.default_output in self.state_variable_names[0]

    def __str__(self):
        """
        String representation.
        """
        return (
            f"Network node: {self.name} with {len(self.masses)} neural mass(es)"
            f": {', '.join([mass.name for mass in self])}"
        )

    def describe(self):
        """
        Return description dict.
        """
        return {
            "index": self.index,
            "name": self.name,
            "num_masses": len(self),
            "num_num_state_variables": self.num_state_variables,
            "num_noise_variables": self.num_noise_variables,
            "masses": [mass.describe() for mass in self],
        }

    def __len__(self):
        """
        Get length.
        """
        return len(self.masses)

    def __getitem__(self, index):
        """
        Get item.
        """
        return self.masses[index]

    @property
    def state_variable_names(self):
        state_var_names = sum([mass.state_variable_names for mass in self], [])
        # if state variables are not unique per node, append mass label to them
        if len(state_var_names) != len(set(state_var_names)):
            state_var_names = [
                f"{state_var}_{mass.mass_type}" for mass in self for state_var in mass.state_variable_names
            ]
        return [state_var_names]

    @property
    def max_delay(self):
        return 0.0

    def get_nested_parameters(self):
        """
        Return nested dictionary with parameters from all masses within this
        node.

        :return: nested dictionary with all parameters
        :rtype: dict
        """
        assert self.initialised
        node_key = f"{NODE_NAME_STR}_{self.index}"
        nested_dict = {node_key: {}}
        for mass in self:
            mass_key = f"{MASS_NAME_STR}_{mass.index}"
            nested_dict[node_key][mass_key] = mass.parameters
        return nested_dict

    def init_node(self, **kwargs):
        """
        Initialise node and all the masses within.

        :kwargs: optional keyword arguments to init_mass
        """
        # we need to have the index already assigned
        assert self.index is not None
        # initialise possible sync variables
        self.sync_symbols = {
            f"{symbol}_{self.index}": se.Symbol(f"{symbol}_{self.index}") for symbol in self.sync_variables
        }
        for mass in self:
            mass.init_mass(**kwargs)
        assert all(mass.initialised for mass in self)
        self.initialised = True

    def update_parameters(self, parameters_dict):
        """
        Update parameters of the node, i.e. recursively update all the masses.

        :param parameters_dict: new parameters for this node, same format as
            `get_nested_parameters`, i.e. nested dict
        :type parameters_dict: dict
        """
        for mass_key, mass_params in parameters_dict.items():
            if MASS_NAME_STR in mass_key:
                mass_index = int(mass_key.split("_")[-1])
                assert mass_index == self.masses[mass_index].index
                self.masses[mass_index].update_parameters(mass_params)
            else:
                logging.warning(f"Not sure what to do with {mass_key}...")

    @staticmethod
    def _strip_index(symbol_name):
        """
        Strip index value from the symbol name.
        """
        return "_".join(symbol_name.split("_")[:-1])

    def _callbacks(self):
        """
        Gather callbacks from all masses.
        """
        return sum([mass._callbacks() for mass in self.masses], [])

    def _numba_callbacks(self):
        """
        Gather callbacks from all masses.
        """
        return sum([mass._numba_callbacks() for mass in self.masses], [])

    def _sync(self):
        """
        Define synchronisation step for the node. This should define the helper
        and its equations. Must be symbolic, i.e. defined with basic math
        operators and symengine operations on state vector. Should return list
        as
        [(se.Symbol, <symbolic definition>)]
        """
        raise NotImplementedError

    @property
    def initial_state(self):
        """
        Return initial state of this node, i.e. sum of masses initial states.
        """
        return np.array(sum([mass.initial_state for mass in self], [],))

    def all_couplings(self, mass_indices=None):
        """
        Return all couplings from masses denoted by index.

        :param mass_indices: indices as to which mass to probe
        :type mass_indices: list|None
        :return: coupling from all masses indicated in the mass_indices
        :rtype: dict
        """
        mass_indices = mass_indices or np.arange(len(self.masses)).tolist()
        all_couplings = {}
        var_idx = 0
        for mass_idx, mass in enumerate(self.masses):
            if mass_idx in mass_indices:
                all_couplings.update({var_idx + k: v for k, v in mass.coupling_variables.items()})
            var_idx += mass.num_state_variables
        return all_couplings

    def _derivatives(self, network_coupling=None):
        """
        Gather all derivatives from all masses.

        :param network_coupling: dict of network coupling for this node, if
            None, default network coupling will be used
        :type network_coupling: dict|None
        :return: derivatives of the state vector for this node
        :rtype: list
        """
        if network_coupling is None:
            network_coupling = self.default_network_coupling
        node_system_equations = []
        var_idx = 0
        for mass in self:
            # get coupling variables
            coupling_vars = {
                self._strip_index(key): value
                for key, value in self.sync_symbols.items()
                if self._strip_index(key) in mass.required_couplings
            }
            coupling_vars.update(
                {key: value for key, value in network_coupling.items() if key in mass.required_couplings}
            )
            mass.idx_state_var = self.idx_state_var + var_idx
            node_system_equations += mass._derivatives(coupling_vars)
            var_idx += mass.num_state_variables
        assert len(node_system_equations) == self.num_state_variables
        return node_system_equations


class SingleCouplingExcitatoryInhibitoryNode(Node):
    """
    Basic node with arbitrary number of excitatory and inhibitory populations,
    but the coupling is through one variable - usually firing rate of the
    population. Will compute connectivity within node as per population types.
    This node definition assumes constant delays, i.e. not dependent on time
    nor dynamics (state variables).
    """

    name = "Single coupling excitatory vs inhibitory node"
    label = "1ExcInhNode"

    sync_variables = [
        "node_exc_exc",
        "node_inh_exc",
        "node_exc_inh",
        "node_inh_inh",
    ]

    # default network coupling - only EXC to EXC and defaults to 0.0 - no
    # network
    default_network_coupling = {"network_exc_exc": 0.0}

    def __init__(
        self, neural_masses, local_connectivity, local_delays=None,
    ):
        """
        :param neural_masses: list of neural masses in this node
        :type neural_masses: list[NeuralMass]
        :param local_connectivity: connectivity matrix for within node
            connections - same order as neural masses, matrix as [from, to]
        :type local_connectivity: np.ndarray
        :param local_delays: delay matrix for within node connections - same
            order as neural masses, if None, delays are all zeros, in ms,
            matrix as [from, to]
        :type local_delays: np.ndarray|None
        """
        super().__init__(neural_masses=neural_masses)
        # assert all masses have one coupling variable
        assert all(len(mass.coupling_variables) == 1 for mass in self.masses)
        # check length - should be the len of masses
        assert local_connectivity.shape[0] == len(self.masses)
        if local_delays is None:
            local_delays = np.zeros_like(local_connectivity)
        assert local_connectivity.shape == local_delays.shape
        self.connectivity = local_connectivity
        self.delays = local_delays
        self.excitatory_masses = np.array([mass.mass_type == EXC for mass in self.masses])
        self.inhibitory_masses = ~self.excitatory_masses
        self.excitatory_masses = np.where(self.excitatory_masses)[0]
        self.inhibitory_masses = np.where(self.inhibitory_masses)[0]

    def __str__(self):
        """
        String representation.
        """
        mass_names = ", ".join([f"{mass.name} {mass.mass_type}" for mass in self.masses])
        return f"Network node: {self.name} with {len(self.masses)} neural masses:" f" {mass_names}"

    def describe(self):
        """
        Return description dict.
        """
        return {
            **super().describe(),
            **{NODAL_CONNECTIVITY: self.connectivity, NODAL_DELAYS: self.delays},
        }

    @property
    def max_delay(self):
        return np.max(self.delays)

    def get_nested_parameters(self):
        """
        Add local connectivity and local delays matrices to mass parameters.

        :return: nested parameters containing also connectivity
        :rtype: dict
        """
        nested_params = super().get_nested_parameters()
        node_key = f"{NODE_NAME_STR}_{self.index}"
        nested_params[node_key][NODAL_CONNECTIVITY] = self.connectivity
        nested_params[node_key][NODAL_DELAYS] = self.delays

        return nested_params

    def init_node(self):
        super().init_node()
        assert self.idx_state_var is not None
        # gather inputs form this node - assumes constant delays
        var_idx = 0
        self.inputs = []
        for row, mass in enumerate(self.masses):
            self.inputs.append(
                [
                    state_vector(
                        self.idx_state_var  # starting index of this node
                        + var_idx  # starting index of particular mass
                        + next(iter(mass.coupling_variables)),  # index of coupling variable
                        time=time_vector - self.delays[row, col],
                    )
                    for col in range(len(self.masses))
                ]
            )
            var_idx += mass.num_state_variables
        # inputs as matrix [from, to]
        self.inputs = np.array(self.inputs)

    def update_parameters(self, parameters_dict):
        """
        Update parameters - also update local connectivity and local delays,
        then pass to base class.
        """
        local_connectivity = parameters_dict.pop(NODAL_CONNECTIVITY, None)
        local_delays = parameters_dict.pop(NODAL_DELAYS, None)
        if local_connectivity is not None and isinstance(local_connectivity, np.ndarray):
            assert local_connectivity.shape == self.connectivity.shape
            self.connectivity = local_connectivity
        if local_delays is not None and isinstance(local_delays, np.ndarray):
            assert local_delays.shape == self.delays.shape
            self.delays = local_delays
        super().update_parameters(parameters_dict)

    def _sync(self):
        connectivity = self.connectivity * self.inputs
        return [
            (
                self.sync_symbols[f"node_exc_exc_{self.index}"],
                sum([connectivity[row, col] for row in self.excitatory_masses for col in self.excitatory_masses]),
            ),
            (
                self.sync_symbols[f"node_inh_exc_{self.index}"],
                sum([connectivity[row, col] for row in self.inhibitory_masses for col in self.excitatory_masses]),
            ),
            (
                self.sync_symbols[f"node_exc_inh_{self.index}"],
                sum([connectivity[row, col] for row in self.excitatory_masses for col in self.inhibitory_masses]),
            ),
            (
                self.sync_symbols[f"node_inh_inh_{self.index}"],
                sum([connectivity[row, col] for row in self.inhibitory_masses for col in self.inhibitory_masses]),
            ),
        ]


class Network(BackendIntegrator):
    """
    Base class for brain network.
    """

    name = ""
    label = ""

    # list of synchronisation variables that are used to compute input/output
    # from different nodes in this network, implemented as `jitcdde` helpers
    sync_variables = []

    # default output of the network - e.g. BOLD is computed from this
    default_output = None

    def __init__(self, nodes, connectivity_matrix, delay_matrix=None):
        """
        :param nodes: list of nodes in this network
        :type nodes: list[Node]
        :param connectivity_matrix: connectivity matrix for between nodes
            coupling, typically DTI structural connectivity, matrix as [from,
            to]
        :type connectivity_matrix: np.ndarray
        :param delay_matrix: delay matrix between nodes, typically derived from
            length matrix, if None, delays are all zeros, in ms, matrix as
            [from, to]
        :type delay_matrix: np.ndarray|None
        """
        # assert all(isinstance(node, Node) for node in nodes)
        self.nodes = nodes
        self.num_state_variables = sum([node.num_state_variables for node in self])
        self.num_noise_variables = sum([node.num_noise_variables for node in self])
        assert connectivity_matrix.shape[0] == self.num_nodes
        if delay_matrix is None:
            delay_matrix = np.zeros_like(connectivity_matrix)
        assert connectivity_matrix.shape == delay_matrix.shape
        self.connectivity = connectivity_matrix
        self.delays = delay_matrix
        self.initialised = False

        if self.default_output is None:
            default_output = set([node.default_output for node in self])
            assert len(default_output) == 1
            self.default_output = next(iter(default_output))

        assert all(self.default_output in node_state_vars for node_state_vars in self.state_variable_names)

        self.init_network()

    def __str__(self):
        """
        String representation.
        """
        return f"Brain network {self.name} with {self.num_nodes} nodes"

    def describe(self):
        """
        Return description dict.
        """
        return {
            "name": self.name,
            "num_nodes": self.num_nodes,
            "num_state_variables": self.num_state_variables,
            "num_noise_variables": self.num_noise_variables,
            "nodes": [node.describe() for node in self],
            NETWORK_CONNECTIVITY: self.connectivity,
            NETWORK_DELAYS: self.delays,
        }

    def __len__(self):
        """
        Get length.
        """
        return len(self.nodes)

    def __getitem__(self, index):
        """
        Get item.
        """
        return self.nodes[index]

    @property
    def max_delay(self):
        return np.max(self.delays.flatten().tolist() + [node.max_delay for node in self])

    @property
    def num_nodes(self):
        return len(self.nodes)

    @property
    def state_variable_names(self):
        state_var_names = sum([node.state_variable_names for node in self], [])
        assert len(state_var_names) == self.num_nodes
        return state_var_names

    @property
    def initial_state(self):
        """
        Return initial state for whole network.
        """
        return np.concatenate([node.initial_state for node in self], axis=0)

    @staticmethod
    def _strip_index(symbol_name):
        """
        Strip index value from the symbol name.
        """
        return "_".join(symbol_name.split("_")[:-1])

    @staticmethod
    def _strip_node_idx(symbol_name):
        """
        Keep index value from the symbol name.
        """
        return int(symbol_name.split("_")[-1])

    @staticmethod
    def _prepare_mass_parameters(param, num_nodes, native_type=dict):
        """
        Prepare mass parameters for the network. I.e. extend to list of the
        same length as number of nodes.

        :param param: parameters to check / prepare
        :type param: list[native_type]|native_type|None
        :param num_nodes: number of nodes in the network
        :type num_nodes: int
        """
        if param is None:
            param = [None] * num_nodes
        if isinstance(param, native_type):
            param = [param] * num_nodes
        assert isinstance(param, list)
        assert all(p is None or isinstance(p, native_type) for p in param)
        assert len(param) == num_nodes
        return param

    def get_nested_parameters(self):
        """
        Return nested dictionary with parameters from all nodes and all masses
        within this network.

        :return: nested dictionary with all parameters
        :rtype: dict
        """
        assert self.initialised
        nested_dict = {NETWORK_NAME_STR: {}}
        for node in self:
            nested_dict[NETWORK_NAME_STR].update(node.get_nested_parameters())
        nested_dict[NETWORK_CONNECTIVITY] = self.connectivity
        nested_dict[NETWORK_DELAYS] = self.delays
        return nested_dict

    def init_network(self, **kwargs):
        """
        Initialise network and the nodes within.

        :kwargs: optional keyword arguments to init_node
        """
        # create symbol for each node for input
        self.sync_symbols = {
            f"{symbol}_{node_idx}": se.Symbol(f"{symbol}_{node_idx}")
            for symbol in self.sync_variables
            for node_idx in range(self.num_nodes)
        }
        for node in self:
            node.init_node(**kwargs)
        assert all(node.initialised for node in self)
        self.initialised = True

    def update_parameters(self, parameters_dict):
        """
        Update parameters of this network, i.e. recursively for all nodes and
        all masses.

        :param parameters_dict: new parameters for the network
        :type parameters_dict: dict
        """
        for node_key, node_params in parameters_dict.items():
            if NODE_NAME_STR in node_key:
                node_index = int(node_key.split("_")[-1])
                assert node_index == self.nodes[node_index].index
                self.nodes[node_index].update_parameters(node_params)
            elif NETWORK_CONNECTIVITY == node_key:
                assert node_params.shape == self.connectivity.shape
                self.connectivity = node_params
            elif NETWORK_DELAYS == node_key:
                assert node_params.shape == self.delays.shape
                self.delays = node_params
            else:
                logging.warning(f"Not sure what to do with {node_key}...")

    def _callbacks(self):
        """
        Gather callbacks from all nodes.
        """
        return sum([node._callbacks() for node in self.nodes], [])

    def _numba_callbacks(self):
        """
        Gather callbacks from all nodes.
        """
        return sum([node._numba_callbacks() for node in self.nodes], [])

    def _sync(self):
        """
        Define synchronisation step for whole network. This should define helper
        and its equations. Must be symbolic, i.e. defined with basic math
        operators and symengine operations on state vector. Should return list
        as
        [(se.Symbol, <symbolic definition>)]
        """
        return sum([node._sync() for node in self], [])

    def _construct_input_matrix(self, within_node_idx):
        """
        Construct input matrix as [from, to] with correct state variables and
        delays.

        :param within_node_idx: index of coupling variable within node (! not
            mass), either single index or list of indices
        :type within_node_idx: list[int]|int
        :return: matrix of delayed inputs as [from, to]
        :rtype: np.ndarray
        """
        var_idx = 0
        inputs = []
        if isinstance(within_node_idx, int):
            within_node_idx = [within_node_idx] * self.num_nodes
        assert self.num_nodes == len(within_node_idx)

        for row, (node, node_var_idx) in enumerate(zip(self.nodes, within_node_idx)):
            inputs.append(
                [
                    state_vector(var_idx + node_var_idx, time=time_vector - self.delays[row, col],)
                    for col in range(self.num_nodes)
                ]
            )
            var_idx += node.num_state_variables
        # node inputs as matrix [from, to]
        return np.array(inputs)

    def _no_coupling(self, symbol):
        """
        Turn off coupling for given symbol.

        :param symbol: which symbol to fill with the data
        :type symbol: str
        """
        return [(self.sync_symbols[f"{symbol}_{node_idx}"], 0.0) for node_idx in range(self.num_nodes)]

    def _diffusive_coupling(self, within_node_idx, symbol, connectivity_multiplier=1.0):
        """
        Perform diffusive coupling on the network with given symbol, i.e.
            network_inp = SUM_idx(Cmat[idx, to] * (X[idx](t - Dmat) - X[to](t)))

        :param within_node_idx: index of coupling variable within node (! not
            mass), either single index or list of indices
        :type within_node_idx: list[int]|int
        :param symbol: which symbol to fill with the data
        :type symbol: str
        :param connectivity_multiplier: multiplier for connectivity, either
            scalar, or array broadcastable to connectivity
        :type connectivity_multiplier: float|np.ndarray
        """
        assert symbol in self.sync_variables
        if isinstance(within_node_idx, int):
            within_node_idx = [within_node_idx] * self.num_nodes
        assert self.num_nodes == len(within_node_idx)
        inputs = self._construct_input_matrix(within_node_idx)
        connectivity = self.connectivity * connectivity_multiplier
        var_idx = 0
        helpers = []
        for node_idx, node_var_idx in zip(range(self.num_nodes), within_node_idx):
            helpers.append(
                (
                    self.sync_symbols[f"{symbol}_{node_idx}"],
                    sum(
                        [
                            connectivity[row, node_idx]
                            * (inputs[row, node_idx] - state_vector(var_idx + node_var_idx, time=time_vector))
                            for row in range(self.num_nodes)
                        ]
                    ),
                )
            )
            var_idx += self.nodes[node_idx].num_state_variables
        return helpers

    def _additive_coupling(self, within_node_idx, symbol, connectivity_multiplier=1.0):
        """
        Perform additive coupling on the network within given symbol, i.e.
            network_inp = SUM_idx(Cmat[idx, to] * X[idx](t - Dmat))

        :param within_node_idx: index of coupling variable within node (! not
            mass), either single index or list of indices
        :type within_node_idx: list[int]|int
        :param symbol: which symbol to fill with the data
        :type symbol: str
        :param connectivity_multiplier: multiplier for connectivity, either
            scalar, or array broadcastable to connectivity
        :type connectivity_multiplier: float|np.ndarray
        """
        assert symbol in self.sync_variables
        connectivity = self.connectivity * connectivity_multiplier * self._construct_input_matrix(within_node_idx)
        return [
            (
                self.sync_symbols[f"{symbol}_{node_idx}"],
                sum([connectivity[row, node_idx] for row in range(self.num_nodes)]),
            )
            for node_idx in range(self.num_nodes)
        ]

    def _derivatives(self):
        """
        Gather all derivatives from all nodes.
        """
        network_equations = []
        var_idx = 0
        for node_idx, node in enumerate(self.nodes):
            node.idx_state_var = var_idx
            network_coupling = {
                self._strip_index(key): value
                for key, value in self.sync_symbols.items()
                if self._strip_node_idx(key) == node_idx
            }
            network_equations += node._derivatives(network_coupling)
            var_idx += node.num_state_variables
        assert len(network_equations) == self.num_state_variables
        return network_equations
