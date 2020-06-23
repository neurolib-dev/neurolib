"""
Base for all mass models.
"""
import symengine as se
from jitcdde import y as state_vector


class NeuralMass:
    """
    Represent a neural population with given parameters and equations, in
    particular, the derivatives of the state vector.
    """

    name = ""
    label = ""

    # type of the mass - for now excitatory or inhibitory
    mass_type = None

    # number of state variables that are wrapped in the `y` vector for
    # derivatives
    num_state_variables = 0

    # indexing number of the mass - when coupling with other, one can sort
    # based on this and then use connection matrix and/or delay matrix
    index = None

    # dict as "index: name" in the state variables vector for coupling
    # variables, i.e. variables that are present in other populations; the name
    # represents how are they called in other populations
    coupling_variables = {}

    # list of required couplings to this mass
    required_couplings = []

    # number of noise variables / inputs
    num_noise_variables = 0

    # names for the state variables to link them with results
    state_variable_names = []

    # list of required parameters for this mass
    required_parameters = []

    # list of helper variables that are defined as symengine Symbols - these
    # are passed to the jitc*de integrators as helpers
    helper_variables = []

    # list of callbacks functions in pure python, that are called from the
    # symbolic derivative, useful when part of neural dynamics cannot be
    # expressed as symbolic expression (e.g. table lookups in AdEx models)
    # provide names of the functions here - this name shall be used in
    # definition of the derivative;
    # NOTE callbacks should be defined as jitted function (i.e. decorated with
    # `numba.njit()`, because 1. speed and 2. compatibility with numba backend)
    python_callbacks = []

    # index of noise variable within the full system's input handled by jitcdde
    noise_input_idx = None

    DESCRIPTION_FIELD = [
        "index",
        "name",
        "mass_type",
        "num_state_variables",
        "num_noise_variables",
        "state_variable_names",
        "parameters",
    ]

    def __init__(self, parameters):
        """
        :param parameters: parameters of the neural mass
        :type parameters: dict
        """
        assert isinstance(parameters, dict)
        self.parameters = parameters
        # used in determining portion of the full system's state vector
        self.idx_state_var = None
        self.initialised = False

        # initialise possible helpers
        self.helper_symbols = {symbol: se.Symbol(symbol) for symbol in self.helper_variables}

        # initialise possible callback functions
        self.callback_functions = {function: se.Function(function) for function in self.python_callbacks}

        self._validate_parameters()

    def __str__(self):
        """
        String representation.
        """
        return (
            f"Neural mass: {self.name} with {self.num_state_variables} "
            f"state variables: {', '.join(self.state_variable_names)}"
        )

    def describe(self):
        """
        Return description dict.
        """
        return {field: getattr(self, field) for field in self.DESCRIPTION_FIELD}

    def _initialize_state_vector(self):
        """
        Initialize state vector. By default it is all zeroes.
        """
        self.initial_state = [0.0] * self.num_state_variables

    def _validate_parameters(self):
        """
        Validate parameters - check if self.parameters contains all required
        parameters.
        """
        assert all(key in self.parameters for key in self.required_parameters), set(self.required_parameters) - set(
            self.parameters.keys()
        )

    def _validate_callbacks(self, callback_list):
        """
        Validate callbacks - mainly the length and symbolic function names.

        :param callback_list: list of callbacks
        :type callback_list: list[tuple|list]
        """
        assert len(callback_list) == len(self.callback_functions)
        assert all(
            callback_from_list[0] == callback_from_init
            for callback_from_list, callback_from_init in zip(callback_list, self.callback_functions.values())
        )
        assert all(callable(callback[1]) for callback in callback_list)

    def init_mass(self, start_idx_for_noise=None):
        """
        Initialise neural mass. Usually just initialise the state vector,
        possibly can be subclassed and do other initialisation.
        """
        self._initialize_state_vector()
        assert self.index is not None
        if start_idx_for_noise is None:
            start_idx_for_noise = self.index
        if self.noise_input_idx is None:
            self.noise_input_idx = [start_idx_for_noise + i for i in range(self.num_noise_variables)]
        self.initialised = True

    def update_parameters(self, parameters_dict):
        """
        Update parameters of the mass.

        :param parameters_dict: new parameters for this mass
        :type parameters_dict: dict
        """
        assert isinstance(parameters_dict, dict)
        self.parameters.update(parameters_dict)
        # validate again
        self._validate_parameters()

    def _callbacks(self):
        """
        List of python callbacks within the symbolic derivatives definition. By
        default, return empty list. If needed, redefine in subclass.

        NOTE: if you would like to use `numba` backend, the callbacks need to be
        defined as so-called jitted functions, i.e. they cannot be class methods
        but rather basic functions that are wrapped with `@numba.njit()`
        """
        callbacks_list = []
        self._validate_callbacks(callbacks_list)
        return callbacks_list

    def _numba_callbacks(self):
        """
        List of python callbacks (see above) for numba integrator. By default,
        returns the same callbacks as for symbolic, redefine if this need to be
        different.
        """
        return self._callbacks()

    def _unwrap_state_vector(self):
        """
        Unwrap state vector into individual variables. Uses global
        `state_vector` from `jitc*de`.
        """
        return [state_vector(i) for i in range(self.idx_state_var, self.idx_state_var + self.num_state_variables,)]

    def _derivatives(self, coupling_variables=None):
        """
        Define derivates, i.e. right-hand side of the dynamical equation
        describing dynamics of this neural mass. Optional input is
        `coupling variables`, should return a list of derivatives of the state
        vector (of the same length obviously).

        :param coupling_variables: optional coupling variables for the mass, as
            a dictionary {"coupling variable name": symengine.Function}
        :type coupling_variables: dict|None
        :return: derivatives of the state vector
        :rtype: list
        """
        raise NotImplementedError
