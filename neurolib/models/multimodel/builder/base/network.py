import logging
from copy import deepcopy
from itertools import chain, islice

import numpy as np
import symengine as se
import sympy as sp
from jitcdde import t as time_vector
from jitcdde import y as state_vector

from .....utils.collections import flat_dict_to_nested, flatten_nested_dict
from .backend import BackendIntegrator
from .constants import EXC, NETWORK_CONNECTIVITY, NETWORK_DELAYS, NODE_CONNECTIVITY, NODE_DELAYS
from .neural_mass import NeuralMass
from .params import float_params_to_individual_symbolic, float_params_to_vector_symbolic


def _sanitize_matrix(matrix, target_shape):
    """
    Sanitize matrix before assigning to connectivity or delay - check shape
    and cast to float if necessary.

    :param matrix: input matrix to sanitize
    :type matrix: np.ndarray|sp.MatrixSymbol
    :param target_shape: shape of the matrix
    :type target_shape: tuple
    :return: sanitized matrix
    :rtype: np.ndarray|sp.MatrixSymbol
    """
    assert matrix.shape == target_shape
    if isinstance(matrix, np.ndarray) and matrix.dtype.kind == "i":
        return matrix.astype(np.float)
    else:
        return matrix


class Node(BackendIntegrator):
    """
    Base class for all nodes within the network.
    """

    name = ""
    label = ""

    # index of this node with respect to the whole network
    index = None

    # list of coupling variables that are used to compute input/output
    # from different masses in this node, implemented as `jitcdde` helpers
    sync_variables = []

    # default network coupling - will be used if network coupling is None,
    # usually used for isolated node
    default_network_coupling = {}

    # default output of the node - all nodes need to have this variable defined
    default_output = None

    # typical output variables of the model - these will be available in model instance
    output_vars = []

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
        assert len(self.noise_input) == self.num_noise_variables
        self.idx_state_var = None
        self.initialised = False
        assert self.default_output in self.state_variable_names[0]
        assert all(var in self.state_variable_names[0] for var in self.output_vars)
        # mass types needs to be unique for all masses!
        mass_types = [mass.mass_type for mass in self]
        assert len(set(mass_types)) == len(mass_types), f"Mass types needs to be different: {mass_types}"
        self._initial_state = None
        # symbolic vs float params
        self.float_params = None
        self.are_params_floats = True

    def __str__(self):
        """
        String representation.
        """
        return (
            f"Network node: {self.name} with {len(self.masses)} neural mass(es)"
            f": {', '.join([mass.name for mass in self])}"
        )

    def __repr__(self):
        return self.__str__()

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
        # if state variables are not unique per node, append mass type to them
        if len(state_var_names) != len(set(state_var_names)):
            state_var_names = [
                f"{state_var}_{mass.mass_type}" for mass in self for state_var in mass.state_variable_names
            ]
        return [state_var_names]

    @property
    def max_delay(self):
        return 0.0

    @property
    def noise_input(self):
        return sum([mass.noise_input for mass in self], [])

    @noise_input.setter
    def noise_input(self, new_noise):
        assert len(new_noise) == self.num_noise_variables
        masses_noise_length = [mass.num_noise_variables for mass in self.masses]
        new_noise = iter(new_noise)
        for noise_chunk, mass in zip([list(islice(new_noise, 0, i)) for i in masses_noise_length], self.masses):
            assert len(noise_chunk) == mass.num_noise_variables
            mass.noise_input = noise_chunk

    def get_nested_params(self):
        """
        Return nested dictionary with parameters from all masses within this
        node.

        :return: nested dictionary with all parameters
        :rtype: dict
        """
        assert self.initialised
        node_key = f"{self.label}_{self.index}"
        nested_dict = {node_key: {}}
        for mass in self:
            mass_key = f"{mass.label}_{mass.index}"
            nested_dict[node_key][mass_key] = mass.params
        return nested_dict

    def make_params_symbolic(self, vector=True):
        """
        Make all node parameters symbolic, instead of concrete values. Useful
        when caching compiled functions.

        :param vector: create vectorised params
        :type vector: bool
        """
        assert self.are_params_floats
        # save copy of original numeric parameters
        self.float_params = deepcopy(self.get_nested_params())
        if vector:
            symbolic_params = float_params_to_vector_symbolic(flatten_nested_dict(self.get_nested_params()))
        else:
            symbolic_params = float_params_to_individual_symbolic(flatten_nested_dict(self.get_nested_params()))
        # update self with symbolic params
        self.are_params_floats = False
        self.update_params(flat_dict_to_nested(symbolic_params), rescale=False)

    def make_params_floats(self):
        """
        Make all node parameters floats again!
        """
        assert not self.are_params_floats
        self.are_params_floats = True
        self.update_params(self.float_params, rescale=False)
        self.float_params = None

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
        self._initial_state = np.array(sum([mass.initial_state for mass in self], []))
        self.initialised = True

    def _sanitize_update_params(self, params_dict):
        """
        If dictionary with parameters for update have one title level, trim this.
        """
        if len(params_dict) == 1 and self.label in next(iter(params_dict)):
            params_dict = next(iter(params_dict.values()))
        return params_dict

    def update_params(self, params_dict, **kwargs):
        """
        Update parameters of the node, i.e. recursively update all parameters of masses within this node.

        :param params_dict: new parameters for this node, same format as
            `get_nested_params`, i.e. nested dict
        :type params_dict: dict
        """
        mass_labels = [mass.label for mass in self.masses]
        params_dict = self._sanitize_update_params(params_dict)
        for mass_key, mass_params in params_dict.items():
            if any(mass_label in mass_key for mass_label in mass_labels):
                mass_index = self._get_index(mass_key)
                assert mass_index == self.masses[mass_index].index
                self.masses[mass_index].update_params(mass_params, **kwargs)
            else:
                logging.warning(f"Not sure what to do with {mass_key}...")

    @staticmethod
    def _get_index(symbol_name):
        """
        Gets index value from the symbol name.
        """
        return int(symbol_name.split("_")[-1])

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
        Return initial state of this node, i.e. sum of initial states of all masses.
        """
        return self._initial_state

    @initial_state.setter
    def initial_state(self, initial_state):
        """
        Manually set initial state - be sure what you are doing!

        :param initial_state: vector representing the initial state, if 2D pass as nodes x time
        :type initial_state: np.ndarray
        """
        assert isinstance(initial_state, np.ndarray)
        assert initial_state.ndim in [1, 2]
        assert initial_state.shape[0] == self.num_state_variables
        self._initial_state = initial_state

    def all_couplings(self, mass_indices=None):
        """
        Return coupling variable names of all masses within this node, denoted by each masses index.

        :param mass_indices: indices of masses
        :type mass_indices: list|None
        :return: coupling variables from all masses indexed by mass_indices
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
        node_equation_system = []
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
            node_equation_system += mass._derivatives(coupling_vars)
            var_idx += mass.num_state_variables
        assert len(node_equation_system) == self.num_state_variables
        return node_equation_system


class SingleCouplingExcitatoryInhibitoryNode(Node):
    """
    Basic node with arbitrary number of excitatory and inhibitory populations,
    but the coupling is through one variable - usually firing rate of the
    population. Will compute connectivity within node as per mass types.
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
        self,
        neural_masses,
        local_connectivity,
        local_delays=None,
    ):
        """
        :param neural_masses: list of neural masses in this node
        :type neural_masses: list[NeuralMass]
        :param local_connectivity: connectivity matrix for within node
            connections - same order as neural masses, matrix as [to, from]
        :type local_connectivity: np.ndarray
        :param local_delays: delay matrix for within node connections - same
            order as neural masses, if None, delays are all zeros, in ms,
            matrix as [to, from]
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
            **{NODE_CONNECTIVITY: self.connectivity, NODE_DELAYS: self.delays},
        }

    @property
    def max_delay(self):
        return np.max(self.delays)

    def get_nested_params(self):
        """
        Add local connectivity and local delays matrices to mass parameters.

        :return: nested parameters containing also connectivity
        :rtype: dict
        """
        nested_params = super().get_nested_params()
        node_key = f"{self.label}_{self.index}"
        nested_params[node_key][NODE_CONNECTIVITY] = self.connectivity
        nested_params[node_key][NODE_DELAYS] = self.delays

        return nested_params

    def init_node(self, **kwargs):
        """
        Init node and construct input matrix as [to, from] array of state
        vectors with correct delays.
        """
        super().init_node(**kwargs)
        assert self.idx_state_var is not None
        # gather inputs form this node - assumes constant delays
        var_idx = 0
        self.inputs = np.zeros_like(self.connectivity, dtype=np.object)
        # iterate over masses as `from`, hence columns
        for from_mass, mass in enumerate(self.masses):
            # iterate over indices as `to`, hence rows
            for to_mass in range(len(self.masses)):
                # input at `to` x `from`
                self.inputs[to_mass, from_mass] = state_vector(
                    self.idx_state_var  # starting index of this node
                    + var_idx  # starting index of particular mass
                    + next(iter(mass.coupling_variables)),  # index of coupling variable
                    time=time_vector - self.delays[to_mass, from_mass],
                )
            var_idx += mass.num_state_variables

    def update_params(self, params_dict, **kwargs):
        """
        Update params - also update local connectivity and local delays,
        then pass to base class.
        """
        params_dict = self._sanitize_update_params(params_dict)
        local_connectivity = params_dict.pop(NODE_CONNECTIVITY, None)
        local_delays = params_dict.pop(NODE_DELAYS, None)
        if local_connectivity is not None and isinstance(local_connectivity, (np.ndarray, sp.MatrixSymbol)):
            self.connectivity = _sanitize_matrix(local_connectivity, self.connectivity.shape)
        if local_delays is not None and isinstance(local_delays, (np.ndarray, sp.MatrixSymbol)):
            self.delays = _sanitize_matrix(local_delays, self.delays.shape)
        super().update_params(params_dict, **kwargs)

    def _sync(self):
        # connectivity as [to, from]
        connectivity = self.connectivity * self.inputs
        return [
            (
                # exc -> exc connectivity
                self.sync_symbols[f"node_exc_exc_{self.index}"],
                sum([connectivity[row, col] for row in self.excitatory_masses for col in self.excitatory_masses]),
            ),
            (
                # exc -> inh connectivity
                self.sync_symbols[f"node_inh_exc_{self.index}"],
                sum([connectivity[row, col] for row in self.inhibitory_masses for col in self.excitatory_masses]),
            ),
            (
                # inh -> exc connectivity
                self.sync_symbols[f"node_exc_inh_{self.index}"],
                sum([connectivity[row, col] for row in self.excitatory_masses for col in self.inhibitory_masses]),
            ),
            (
                # inh -> inh connectivity
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

    # defines coupling type (usually additive, diffusive or none) per coupling
    # variable
    default_coupling = {}

    # default output of the network - e.g. BOLD is computed from this
    default_output = None

    # typical output variables of the model - these will be available in model instance
    output_vars = []

    def __init__(self, nodes, connectivity_matrix, delay_matrix=None):
        """
        :param nodes: list of nodes in this network
        :type nodes: list[Node]
        :param connectivity_matrix: connectivity matrix for between nodes
            coupling, typically DTI structural connectivity, matrix as [to,
            from]
        :type connectivity_matrix: np.ndarray
        :param delay_matrix: delay matrix between nodes, typically derived from
            length matrix, if None, delays are all zeros, in ms, matrix as
            [to, from]
        :type delay_matrix: np.ndarray|None
        """
        # assert all(isinstance(node, Node) for node in nodes)
        self.nodes = nodes
        self.num_state_variables = sum([node.num_state_variables for node in self])
        self.num_noise_variables = sum([node.num_noise_variables for node in self])
        assert len(self.noise_input) == self.num_noise_variables
        assert connectivity_matrix.shape[0] == self.num_nodes
        if delay_matrix is None:
            delay_matrix = np.zeros_like(connectivity_matrix)
        assert connectivity_matrix.shape == delay_matrix.shape
        self.connectivity = connectivity_matrix
        self.delays = delay_matrix
        self._initial_state = None
        self.initialised = False

        if self.default_output is None:
            default_output = set([node.default_output for node in self])
            assert len(default_output) == 1
            self.default_output = next(iter(default_output))

        assert all(var in chain.from_iterable(self.state_variable_names) for var in self.output_vars)
        assert all(self.default_output in node_state_vars for node_state_vars in self.state_variable_names)

        # symbolic vs float params
        self.float_params = None
        self.are_params_floats = True

        self.init_network()

    def __str__(self):
        """
        String representation.
        """
        return f"Brain network {self.name} with {self.num_nodes} nodes"

    def __repr__(self):
        return self.__str__()

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
        return self._initial_state

    @initial_state.setter
    def initial_state(self, initial_state):
        """
        Manually set initial state - be sure what you are doing!

        :param initial_state: vector representing the initial state, if 2D pass as nodes x time
        :type initial_state: np.ndarray
        """
        assert isinstance(initial_state, np.ndarray)
        assert initial_state.ndim in [1, 2]
        assert initial_state.shape[0] == self.num_state_variables
        self._initial_state = initial_state

    @property
    def noise_input(self):
        return sum([node.noise_input for node in self], [])

    @noise_input.setter
    def noise_input(self, new_noise):
        assert len(new_noise) == self.num_noise_variables
        nodes_noise_length = [node.num_noise_variables for node in self.nodes]
        new_noise = iter(new_noise)
        for noise_chunk, node in zip([list(islice(new_noise, 0, i)) for i in nodes_noise_length], self.nodes):
            assert len(noise_chunk) == node.num_noise_variables
            node.noise_input = noise_chunk

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
    def _prepare_mass_params(param, num_nodes, native_type=dict):
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

    def get_nested_params(self):
        """
        Return nested dictionary with parameters from all nodes and all masses
        within this network.

        :return: nested dictionary with all parameters
        :rtype: dict
        """
        assert self.initialised
        nested_dict = {self.label: {}}
        for node in self:
            nested_dict[self.label].update(node.get_nested_params())
        nested_dict[self.label][NETWORK_CONNECTIVITY] = self.connectivity
        nested_dict[self.label][NETWORK_DELAYS] = self.delays
        return nested_dict

    def make_params_symbolic(self, vector=True):
        """
        Make all node parameters symbolic, instead of concrete values. Useful
        when caching compiled functions.

        :param vector: create vectorised params
        :type vector: bool
        """
        assert self.are_params_floats
        # save copy of original numeric parameters
        self.float_params = deepcopy(self.get_nested_params())
        if vector:
            symbolic_params = float_params_to_vector_symbolic(flatten_nested_dict(self.get_nested_params()))
        else:
            symbolic_params = float_params_to_individual_symbolic(flatten_nested_dict(self.get_nested_params()))
        self.are_params_floats = False
        # update self with symbolic params
        self.update_params(flat_dict_to_nested(symbolic_params), rescale=False)

    def make_params_floats(self):
        """
        Make all node parameters floats again!
        """
        assert not self.are_params_floats
        self.are_params_floats = True
        self.update_params(self.float_params, rescale=False)
        self.float_params = None

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
        for node_idx, node in enumerate(self.nodes):
            node.init_node(start_idx_for_noise=node_idx * node.num_noise_variables, **kwargs)
        assert all(node.initialised for node in self)
        self._initial_state = np.concatenate([node.initial_state for node in self], axis=0)
        self.initialised = True

    def update_params(self, params_dict, **kwargs):
        """
        Update parameters of this network, i.e. recursively for all nodes and
        all masses.

        :param params_dict: new parameters for the network
        :type params_dict: dict
        """
        node_labels = [node.label for node in self.nodes]
        if len(params_dict) == 1 and self.label in next(iter(params_dict)):
            params_dict = next(iter(params_dict.values()))
        for node_key, node_params in params_dict.items():
            if any(node_label in node_key for node_label in node_labels):
                node_index = int(node_key.split("_")[-1])
                assert node_index == self.nodes[node_index].index
                self.nodes[node_index].update_params(node_params, **kwargs)
            elif NETWORK_CONNECTIVITY == node_key:
                self.connectivity = _sanitize_matrix(node_params, self.connectivity.shape)
            elif NETWORK_DELAYS == node_key:
                self.delays = _sanitize_matrix(node_params, self.delays.shape)
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
        # start with summing coupling from all nodes - nodal coupling
        all_couplings = sum([node._sync() for node in self], [])
        for coupling_var, coupling_type in self.default_coupling.items():
            assert coupling_var in self.sync_variables
            assert coupling_var.startswith("network_")
            # add coupling to the list of all coupling definitions
            all_couplings += self._couple(coupling_type, coupling_var[8:])
        return all_couplings

    def _construct_input_matrix(self, within_node_idx):
        """
        Construct input matrix as [to, from] with correct state variables and
        delays.

        :param within_node_idx: index of coupling variable within node (! not
            mass), either single index or list of indices
        :type within_node_idx: list[int]|int
        :return: matrix of delayed inputs as [to, from]
        :rtype: np.ndarray
        """
        var_idx = 0
        inputs = []
        if isinstance(within_node_idx, int):
            within_node_idx = [within_node_idx] * self.num_nodes
        assert self.num_nodes == len(within_node_idx)
        inputs = np.zeros_like(self.connectivity, dtype=np.object)

        # iterate over nodes as `from`, hence columns
        for from_node, (node, node_var_idx) in enumerate(zip(self.nodes, within_node_idx)):
            # iterate over indices as `to`, hence rows
            for to_node in range(self.num_nodes):
                # input at `to` x `from`
                # if None - no input
                if node_var_idx is None:
                    inputs[to_node, from_node] = 0.0
                else:
                    inputs[to_node, from_node] = state_vector(
                        var_idx + node_var_idx,
                        time=time_vector - self.delays[to_node, from_node],
                    )
            var_idx += node.num_state_variables

        return inputs

    def _no_coupling(self, symbol):
        """
        Turn off coupling for given symbol.

        :param symbol: which symbol to fill with the data
        :type symbol: str
        """
        return [(self.sync_symbols[f"{symbol}_{node_idx}"], 0.0) for node_idx in range(self.num_nodes)]

    def _diffusive_coupling(self, within_node_idx, symbol, connectivity=None):
        """
        Perform diffusive coupling on the network with given symbol, i.e.
            network_inp = SUM_idx(Cmat[to, idx] * (X[idx](t - Dmat) - X[to](t)))

        :param within_node_idx: index of coupling variable within node (! not
            mass), either single index or list of indices
        :type within_node_idx: list[int]|int
        :param symbol: which symbol to fill with the data
        :type symbol: str
        :param connectivity: connectivity matrix - if None will use the network
            connectivity from init
        :type connectivity: np.ndarray
        """
        assert symbol in self.sync_variables
        if isinstance(within_node_idx, int):
            within_node_idx = [within_node_idx] * self.num_nodes
        assert self.num_nodes == len(within_node_idx)
        inputs = self._construct_input_matrix(within_node_idx)
        connectivity = self.connectivity if connectivity is None else connectivity
        assert connectivity.shape == (self.num_nodes, self.num_nodes)
        var_idx = 0
        helpers = []
        for node_idx, node_var_idx in zip(range(self.num_nodes), within_node_idx):
            helpers.append(
                (
                    self.sync_symbols[f"{symbol}_{node_idx}"],
                    sum(
                        [
                            connectivity[node_idx, row]
                            * (inputs[node_idx, row] - state_vector(var_idx + node_var_idx, time=time_vector))
                            for row in range(self.num_nodes)
                        ]
                    ),
                )
            )
            var_idx += self.nodes[node_idx].num_state_variables
        return helpers

    def _multiplicative_coupling(self, within_node_idx, symbol, connectivity=None):
        """
        Perform multiplicative coupling on the network with given symbol, i.e.
            network_inp = SUM_idx(Cmat[to, idx]*X[idx](t - Dmat)*X[to](t))
        """
        assert symbol in self.sync_variables
        if isinstance(within_node_idx, int):
            within_node_idx = [within_node_idx] * self.num_nodes
        assert self.num_nodes == len(within_node_idx)
        inputs = self._construct_input_matrix(within_node_idx)
        connectivity = self.connectivity if connectivity is None else connectivity
        assert connectivity.shape == (self.num_nodes, self.num_nodes)
        var_idx = 0
        helpers = []
        for node_idx, node_var_idx in zip(range(self.num_nodes), within_node_idx):
            helpers.append(
                (
                    self.sync_symbols[f"{symbol}_{node_idx}"],
                    sum(
                        [
                            connectivity[node_idx, row]
                            * (inputs[node_idx, row])
                            * state_vector(var_idx + node_var_idx, time=time_vector)
                            for row in range(self.num_nodes)
                        ]
                    ),
                )
            )
            var_idx += self.nodes[node_idx].num_state_variables
        return helpers

    def _additive_coupling(self, within_node_idx, symbol, connectivity=None):
        """
        Perform additive coupling on the network within given symbol, i.e.
            network_inp = SUM_idx(Cmat[to, idx] * X[idx](t - Dmat))

        :param within_node_idx: index of coupling variable within node (! not
            mass), either single index or list of indices
        :type within_node_idx: list[int]|int
        :param symbol: which symbol to fill with the data
        :type symbol: str
        :param connectivity: connectivity matrix - if None will use the network
            connectivity from init
        :type connectivity: np.ndarray
        """
        assert symbol in self.sync_variables
        connectivity = self.connectivity if connectivity is None else connectivity
        assert connectivity.shape == (self.num_nodes, self.num_nodes)
        connectivity = connectivity * self._construct_input_matrix(within_node_idx)
        return [
            (
                self.sync_symbols[f"{symbol}_{node_idx}"],
                sum([connectivity[node_idx, row] for row in range(self.num_nodes)]),
            )
            for node_idx in range(self.num_nodes)
        ]

    def _couple(self, coupling_type, coupling_variable):
        """
        Perform simple coupling in the network based on the type.

        :param coupling_type: type of the coupling - additive, diffusive or none
        :type coupling_type: str
        :param coupling_variable: perform coupling on this coupling coupling
            variable
        :type coupling_variable: str
        """
        assert coupling_variable in self.coupling_symbols
        if coupling_type == "additive":
            return self._additive_coupling(
                self.coupling_symbols[coupling_variable],
                f"network_{coupling_variable}",
            )
        elif coupling_type == "diffusive":
            return self._diffusive_coupling(
                self.coupling_symbols[coupling_variable],
                f"network_{coupling_variable}",
            )
        elif coupling_type == "multiplicative":
            return self._multiplicative_coupling(
                self.coupling_symbols[coupling_variable],
                f"network_{coupling_variable}",
            )
        elif coupling_type == "none":
            return self._no_coupling(f"network_{coupling_variable}")
        else:
            raise ValueError(f"Unknown coupling type: {coupling_type}")

    def _derivatives(self):
        """
        Gather all derivatives from all nodes.
        """
        network_equations = []
        var_idx = 0
        for node in self.nodes:
            node.idx_state_var = var_idx
            network_coupling = {
                self._strip_index(key): value
                for key, value in self.sync_symbols.items()
                if self._strip_node_idx(key) == node.index
            }
            network_equations += node._derivatives(network_coupling)
            var_idx += node.num_state_variables
        assert len(network_equations) == self.num_state_variables
        return network_equations
