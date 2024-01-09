import abc
import numba
import numpy as np
from neurolib.control.optimal_control import cost_functions
from neurolib.utils.model_utils import computeDelayMatrix, adjustArrayShape
import logging
import copy

from numba.core import types
from numba.typed import Dict


def getdefaultweights():
    weights = Dict.empty(
        key_type=types.unicode_type,
        value_type=types.float64,
    )
    weights["w_p"] = 1.0
    weights["w_2"] = 0.0

    return weights


@numba.njit
def compute_gradient(
    N,
    V,
    dim_out,
    df_du,
    adjoint_state,
    control_matrix,
    d_du,
    control_interval,
):
    """Compute the gradient of the total cost wrt. the control signals (explicitly and implicitly) given the adjoint
       state, the Jacobian of the total cost wrt. explicit control contributions and the Jacobian of the dynamics
       wrt. explicit control contributions.

    :param N:       Number of nodes in the network.
    :type N:        int
    :param V:       Number of  variables of the model.
    :type V:        int
    :param dim_out: Number of 'output variables' of the model.
    :type dim_out:  int
    :param T:       Length of simulation (time dimension).
    :type T:        int
    :param df_du:   Derivative of the cost wrt. the explicit control contributions to cost functionals.
    :type df_du:    np.ndarray of shape N x V x T
    :param adjoint_state:  Solution of the adjoint equation.
    :type adjoint_state:   np.ndarray of shape N x V x T
    :param control_matrix: Binary matrix that defines nodes and variables where control inputs are active, defaults to
                           None.
    :type control_matrix:  np.ndarray of shape N x V
    :param d_du:     Jacobian of systems dynamics wrt. the external inputs (control).
    :type d_du:      np.ndarray of shape V x V
    :return:         The gradient of the total cost wrt. the control.
    :rtype:          np.ndarray of shape N x V x T
    """
    grad = np.zeros(df_du.shape)
    for n in range(N):
        for v in range(dim_out):
            for t in range(control_interval[0], control_interval[1]):
                grad[n, v, t] = df_du[n, v, t]
                for k in range(V):
                    grad[n, v, t] += (
                        control_matrix[n, v] * adjoint_state[n, k, t] * d_du[n, k, v, t]
                    )
    return grad


@numba.njit
def solve_adjoint(
    hx_list,
    del_list,
    hx_nw,
    fx,
    state_dim,
    dt,
    N,
    T,
    dmat_ndt,
    dxdoth,
    state_vars,
    output_vars,
):
    """Backwards integration of the adjoint state.

    :param hx_list:     list of Jacobians of systems dynamics wrt. 'state_vars'
    :type hx_list:      list of np.ndarray
    :param del_list:    list of respective time delay integer
    :type del_list:     list of int
    :param hx_nw:       Jacobians for each time step for the network coupling.
    :type hx_nw:        np.ndarray
    :param fx: df/dx    Derivative of cost function wrt. systems dynamics.
    :type fx:           np.ndarray
    :param state_dim:   Dimensions of state (N, V, T).
    :type state_dim:    tuple
    :param dt:          Time resolution of integration.
    :type dt:           float
    :param N:           Number of nodes in the network.
    :type N:            int
    :param T:           Length of simulation (time dimension).
    :type T:            int
    :param dmat_ndt:    N x N delay matrix (discrete number of delayed time-intervals).
    :type dmat_ndt:     np.ndarray
    :param dxdoth:      derivative of system dynamics wrt x dot
    :type dxdoth:       np.ndarray
    :param state_vars:      list of state variables of model
    :type state_vars:       list
    :param output_vars:     list of output variables of model
    :type output_vars:      list

    :return:            Adjoint state.
    :rtype:             np.ndarray of shape `state_dim`
    """
    # ToDo: generalize, not only precision cost
    adjoint_state = np.zeros(state_dim)
    fx_fullstate = np.zeros(state_dim)

    for sv_ind, sv in enumerate(state_vars):
        for ov_ind, ov in enumerate(output_vars):
            if sv == ov:
                fx_fullstate[:, sv_ind, :] = fx[:, ov_ind, :]

    for t in range(T - 2, -1, -1):  # backwards iteration including 0th index
        for n in range(N):  # iterate through nodes
            for k in range(state_dim[1]):
                if dxdoth[n, k, k] == 0:
                    res = fx_fullstate[n, k, t + 1]
                    res += adjoint_input(
                        hx_list, del_list, t, T, state_dim[1], adjoint_state, n, k
                    )
                    res += adjoint_nw_input(
                        N, n, k, dmat_ndt, t, T, state_dim[1], adjoint_state, hx_nw
                    )

                    adjoint_state[n, k, t] = -res

                elif dxdoth[n, k, k] == 1:
                    der = fx_fullstate[n, k, t + 1]
                    der += adjoint_input(
                        hx_list, del_list, t, T, state_dim[1], adjoint_state, n, k
                    )
                    der += adjoint_nw_input(
                        N, n, k, dmat_ndt, t, T - 1, state_dim[1], adjoint_state, hx_nw
                    )

                    adjoint_state[n, k, t] = adjoint_state[n, k, t + 1] - dt * der

                else:
                    print("WARNING: Case dh_dxdot != 0 or 1 not implemented")
                    raise NotImplementedError

    return adjoint_state


@numba.njit
def solve_adjoint_ww(
    hx_list,
    del_list,
    hx_nw,
    fx,
    state_dim,
    dt,
    N,
    T,
    dmat_ndt,
    dxdoth,
    state_vars,
    output_vars,
):
    """Backwards integration of the adjoint state.

    :param hx_list:     list of Jacobians of systems dynamics wrt. 'state_vars'
    :type hx_list:      list of np.ndarray
    :param del_list:    list of respective time delay integer
    :type del_list:     list of int
    :param hx_nw:       Jacobians for each time step for the network coupling.
    :type hx_nw:        np.ndarray
    :param fx: df/dx    Derivative of cost function wrt. systems dynamics.
    :type fx:           np.ndarray
    :param state_dim:   Dimensions of state (N, V, T).
    :type state_dim:    tuple
    :param dt:          Time resolution of integration.
    :type dt:           float
    :param N:           Number of nodes in the network.
    :type N:            int
    :param T:           Length of simulation (time dimension).
    :type T:            int
    :param dmat_ndt:    N x N delay matrix (discrete number of delayed time-intervals).
    :type dmat_ndt:     np.ndarray
    :param dxdoth:      derivative of system dynamics wrt x dot
    :type dxdoth:       np.ndarray
    :param state_vars:      list of state variables of model
    :type state_vars:       list
    :param output_vars:     list of output variables of model
    :type output_vars:      list

    :return:            Adjoint state.
    :rtype:             np.ndarray of shape `state_dim`
    """
    # ToDo: generalize, not only precision cost
    adjoint_state = np.zeros(state_dim)
    fx_fullstate = np.zeros(state_dim)

    for sv_ind, sv in enumerate(state_vars):
        for ov_ind, ov in enumerate(output_vars):
            if sv == ov:
                fx_fullstate[:, sv_ind, :] = fx[:, ov_ind, :]

    for t in range(T - 1, -1, -1):  # backwards iteration including 0th index
        for n in range(N):  # iterate through nodes
            adjoint_state[n, 2, t] = -fx_fullstate[n, 0, t] / hx_list[0][n, t, 2, 0]
            adjoint_state[n, 3, t] = -fx_fullstate[n, 1, t] / hx_list[0][n, t, 3, 1]

            for k in range(state_dim[1]):
                if dxdoth[n, k, k] == 0:
                    res = fx_fullstate[n, k, t + 1]
                    res += adjoint_input(
                        hx_list, del_list, t, T, state_dim[1], adjoint_state, n, k
                    )
                    res += adjoint_nw_input(
                        N, n, k, dmat_ndt, t, T, state_dim[1], adjoint_state, hx_nw
                    )
                    # if n == 0 and k == 0:
                    #    print(t, n, k, res)
                    #    print(adjoint_state[n,])

                    adjoint_state[n, k, t] = -res

                elif dxdoth[n, k, k] == 1:
                    der = fx_fullstate[n, k, t]
                    der += adjoint_input(
                        hx_list, del_list, t, T, state_dim[1], adjoint_state, n, k
                    )
                    der += adjoint_nw_input(
                        N, n, k, dmat_ndt, t, T - 1, state_dim[1], adjoint_state, hx_nw
                    )

                    adjoint_state[n, k, t] = adjoint_state[n, k, t + 1] - dt * der

                else:
                    print("WARNING: Case dh_dxdot != 0 or 1 not implemented")
                    raise NotImplementedError

    return adjoint_state


@numba.njit
def adjoint_input(hx_list, del_list, t, T_lim, state_dim1, adj, n, k):
    """Compute input to adjoint state for backwards integration

    :param hx_list:     list of Jacobians of systems dynamics wrt. 'state_vars'
    :type hx_list:      list of np.ndarray
    :param del_list:    list of respective time delay integer
    :type del_list:     list of int
    :param t:           current time index
    :type t:            int
    :param T_lim:       Maximum time index
    :type T_lim:        int
    :param state_dim1:  Number of state variables (V)
    :type state_dim1:   int
    :param adj:         adjoint state
    :type adj:          np.ndarray
    :param n:           node index
    :type n:            int
    :param k:           node index
    :type k:            int

    :return:            Adjoint state input
    :rtype:             float
    """
    result = 0.0
    for hx, int_delay in zip(hx_list, del_list):
        if t + 1 + int_delay < T_lim:
            for v in range(state_dim1):
                result += adj[n, v, t + 1 + int_delay] * hx[n, t + 1 + int_delay, v, k]
                if int_delay == -1 and v == 2 and k == 0:
                    print("adjoint = ", adj[n, v, t + 1 + int_delay])
                    print("hx = ", hx[n, t + 1 + int_delay, v, k])
    return result


@numba.njit
def adjoint_nw_input(N, n, k, dmat_ndt, t, T_lim, state_dim1, adj, hxnw):
    """Compute input to adjoint state from network connections for backwards integration

    :param N:           Number of nodes in the network.
    :type N:            int
    :param n:           current node index
    :type n:            int
    :param k:           node index
    :type k:            int
    :param dmat_ndt:    N x N delay matrix (discrete number of delayed time-intervals).
    :type dmat_ndt:     np.ndarray
    :param t:           current time index
    :type t:            int
    :param T_lim:       Maximum time index
    :type T_lim:        int
    :param state_dim1:  Number of state variables (V)
    :type state_dim1:   int
    :param adj:          adjoint state
    :type adj:           np.ndarray
    :param hxnw:        Jacobians for each time step for the network coupling.
    :type hx_w:         np.ndarray

    :return:            Adjoint state input
    :rtype:             float
    """
    result = 0.0
    for n2 in range(N):  # iterate through connectivity of current node "n"
        if t + 1 + dmat_ndt[n2, n] < T_lim:
            for v in range(state_dim1):
                result += (
                    adj[n2, v, t + 1 + dmat_ndt[n2, n]]
                    * hxnw[n2, n, t + 1 + dmat_ndt[n2, n], v, k]
                )
    return result


@numba.njit
def limit_control_to_interval(N, dim_in, T, control, control_interval):
    control_new = control.copy()

    for n in range(N):
        for v in range(dim_in):
            for t in range(0, control_interval[0]):
                control_new[n, v, t] = 0.0
            for t in range(control_interval[1], T):
                control_new[n, v, t] = 0.0

    return control_new


@numba.njit
def update_control_with_limit(N, dim_in, T, control, step, gradient, u_max):
    """Computes the updated control signal. The absolute values of the new control are bounded by +/- 'u_max'. If
       'u_max' is 'None', no limit is applied.

    :param control:         N x V x T array. Control signals.
    :type control:          np.ndarray
    :param step:            Step size along the gradients.
    :type step:             float
    :param gradient:        N x V x T array of the gradients.
    :type gradient:         np.ndarray
    :param u_max:           Maximum absolute value allowed for the strength of the control signal.
    :type u_max:            float or None

    :return:                N x V x T array containing the new control signal, updated according to 'step' and
                            'gradient' with the maximum absolute values being limited by 'u_max'.
    :rtype:                 np.ndarray
    """

    control_new = control + step * gradient

    if u_max is not None:
        control_new = control + step * gradient

        for n in range(N):
            for v in range(dim_in):
                for t in range(T):
                    if np.greater(np.abs(control_new[n, v, t]), u_max):
                        control_new[n, v, t] = np.sign(control_new[n, v, t]) * u_max

    return control_new


def convert_interval(interval, array_length):
    """Turn indices into positive values only. It is assumed in any case, that the first index defines the start and
       the second the stop index, both inclusive.

    :param interval:    Tuple containing start and stop index. May contain negative indices or 'None'.
    :type interval:     tuple
    :param array_length:    Length of the array in the dimension, along which the 'interval' is defining the slice.
    :type array_length:     int
    :return:            Tuple containing two positive 'int' indicating the start- and stop-index (both inclusive) of the
                        interval.
    :rtype:             tuple
    """

    (interval_0, interval_1) = interval

    if interval_0 is None:
        interval_0_new = 0
    elif interval_0 < 0:
        assert interval_0 > -array_length, "Interval is not specified in valid range."
        interval_0_new = array_length + interval_0  # interval entry is negative
    else:
        interval_0_new = interval_0

    if interval_1 is None:
        interval_1_new = array_length
    elif interval_1 < 0:
        assert interval_1 > -array_length, "Interval is not specified in valid range."
        interval_1_new = array_length + interval_1  # interval entry is negative
    else:
        interval_1_new = interval_1

    assert (
        interval_0_new < interval_1_new
    ), "Order of indices for interval is not valid."
    assert interval_1_new <= array_length, "Interval is not specified in valid range."

    return interval_0_new, interval_1_new


class OC:
    def __init__(
        self,
        model,
        target,
        weights=None,
        maximum_control_strength=None,
        print_array=[],
        cost_interval=(None, None),
        control_interval=(None, None),
        cost_matrix=None,
        control_matrix=None,
        M=1,
        M_validation=0,
        validate_per_step=False,
    ):
        """
        Base class for optimal control. Model specific methods should be implemented in derived class for each model.

        :param model:       An instance of neurolib's Model-class. Parameters like '.duration' and methods like '.run()'
                            are used within the optimal control.
        :type model:        neurolib.models.model
        :param target:      Target time series of controllable variables.
        :type target:       np.ndarray
        :param weights:     Dictionary of weight parameters, defaults to 'None'.
        :type weights:      dictionary, optional
        :param maximum_control_strength:    Maximum absolute value a control signal can take. No limitation of the
                                            absolute control strength if 'None'. Defaults to None.
        :type:                              float or None, optional
        :param print_array:                 Array of optimization-iteration-indices (starting at 1) in which cost is printed out.
                                            Defaults to empty list `[]`.
        :type print_array:                  list, optional
        :param cost_interval:               (t_start, t_end). Indices of start and end point (both inclusive) of the
                                            time interval in which the accuracy cost is evaluated. Default is full time
                                            series. Defaults to (None, None).
        :type cost_interval:                tuple, optional
        :param control_interval:            (t_start, t_end). Indices of start and end point (both inclusive) of the
                                            time interval in which control can be applied. Default is full time
                                            series. Defaults to (None, None).
        :type control_interval:              tuple, optional
        :param cost_matrix:                 N x V binary matrix that defines nodes and channels of accuracy measurement, defaults
                                            to None.
        :type cost_matrix:                  np.ndarray
        :param control_matrix:      N x V Binary matrix that defines nodes and variables where control inputs are active,
                                    defaults to None.
        :type control_matrix:       np.ndarray
        :param M:                   Number of noise realizations. M=1 implies deterministic case. Defaults to 1.
        :type M:                    int, optional
        :param M_validation:        Number of noise realizations for validation (only used in stochastic case, M>1).
                                    Defaults to 0.
        :type M_validation:         int, optional
        :param validate_per_step:   True for validation in each iteration of the optimization, False for
                                    validation only after final optimization iteration (only used in stochastic case,
                                    M>1). Defaults to False.
        :type validate_per_step:    bool, optional

        """

        self.model = copy.deepcopy(model)

        self.target = target  # ToDo: dimensions-check
        self.maximum_control_strength = maximum_control_strength

        if type(weights) != type(dict()):
            if weights is not None:
                print(
                    "Weights parameter must be dictionary, use default weights instead."
                )
            self.weights = getdefaultweights()
        else:
            defaultweights = getdefaultweights()
            for k in defaultweights.keys():
                if k in weights.keys():
                    defaultweights[k] = weights[k]
                else:
                    print(
                        "Weight ",
                        k,
                        " not in provided weight dictionary. Use default value.",
                    )

            self.weights = defaultweights

        self.N = self.model.params.N
        self.dt = self.model.params["dt"]  # maybe redundant but for now code clarity
        self.duration = self.model.params[
            "duration"
        ]  # maybe redundant but for now code clarity
        self.T = (
            np.around(self.duration / self.dt, 0).astype(int) + 1
        )  # Total number of time steps

        self.dim_vars = len(self.model.state_vars)
        self.dim_in = len(self.model.input_vars)
        self.dim_out = len(self.model.output_vars)

        self.state_vars_dict = self.get_state_vars_dict()

        self.adjust_init()
        self.simulate_forward()

        if self.N == 1:
            self.Dmat_ndt = np.zeros((self.N, self.N)).astype(int)
        else:
            Dmat = computeDelayMatrix(
                self.model.params.lengthMat, self.model.params.signalV
            )
            if self.model.name != "aln":
                Dmat[np.eye(len(Dmat)) == 1] = np.zeros(len(Dmat))
            else:
                Dmat[np.eye(len(Dmat)) == 1] = np.ones(len(Dmat)) * self.model.params.de
            self.Dmat_ndt = np.around(Dmat / self.dt).astype(int)

        if self.N == 1:
            if isinstance(self.model.Cmat, type(None)):
                self.model.Cmat = np.zeros((self.N, self.N))
            if isinstance(self.model.Dmat, type(None)):
                self.model.Dmat = np.zeros((self.N, self.N))

        self.cost_matrix = cost_matrix
        if isinstance(self.cost_matrix, type(None)):
            self.cost_matrix = np.ones(
                (self.N, self.dim_out)
            )  # default: measure precision in all variables and nodes

        self.control_matrix = control_matrix
        if isinstance(self.control_matrix, type(None)):
            self.control_matrix = np.ones(
                (self.N, self.dim_in)
            )  # default: all channels and all nodes active

        self.M = max(1, M)
        self.M_validation = M_validation

        self.step = 10.0  # Initial step size in first optimization iteration.
        self.count_noisy_step = 10
        self.count_step = 30

        self.factor_down = 0.5  # Factor for adaptive step size reduction.
        self.factor_up = 2.0  # Factor for adaptive step size increment.

        self.cost_validation = 0.0
        self.validate_per_step = validate_per_step

        if self.model.params.sigma_ou != 0.0:  # noisy system
            if self.M <= 1:
                logging.warning(
                    "For noisy system, please chose parameter M larger than 1 (recommendation > 10)."
                    + "\n"
                    + 'If you want to study a deterministic system, please set model parameter "sigma_ou" to zero'
                )
            if self.M > self.M_validation:
                print(
                    'Parameter "M_validation" should be chosen larger than parameter "M".'
                )
        else:  # deterministic system
            if self.M > 1 or self.M_validation != 0 or validate_per_step:
                print(
                    'For deterministic systems, parameters "M", "M_validation" and "validate_per_step" are not relevant.'
                    + "\n"
                    + 'If you want to study a noisy system, please set model parameter "sigma_ou" larger than zero'
                )

        # + forward simulation steps of neurolibs model.run().
        self.state_dim = (
            self.N,
            self.dim_vars,
            self.T,
        )  # dimensions of state. Model has N network nodes, V state variables, T time points

        self.adjoint_state = np.zeros(self.state_dim)
        self.gradient = np.zeros(self.state_dim)

        self.cost_history = []
        self.step_sizes_history = []
        self.step_sizes_loops_history = []

        # save control signals throughout optimization iterations for later analysis
        self.control_history = []

        self.print_array = print_array

        self.zero_step_encountered = (
            False  # deterministic gradient descent cannot further improve
        )

        self.cost_interval = convert_interval(cost_interval, self.T)
        self.control_interval = convert_interval(control_interval, self.T)

        self.ndt_de, self.ndt_di = 0.0, 0.0

        self.adjust_input()

        control = np.zeros((self.N, self.dim_in, self.T))
        for v, iv in enumerate(self.model.input_vars):
            control[:, v, :] = self.model.params[iv]

        self.control = control.copy()
        self.check_params()

        self.control = update_control_with_limit(
            self.N,
            self.dim_in,
            self.T,
            control,
            0.0,
            np.zeros(control.shape),
            self.maximum_control_strength,
        )

        self.model_params = self.get_model_params()

    def check_params(self):
        """Checks a subset of parameters and throws an error if a wrong dimension is found."""
        # check if cost matrix is binary
        assert np.array_equal(self.cost_matrix, self.cost_matrix.astype(bool))

        # check if control matrix is binary
        assert np.array_equal(self.control_matrix, self.control_matrix.astype(bool))

        # check if input has right dimensions
        for input_var in self.model.input_vars:
            assert self.T == self.model.params[input_var].shape[1]

        # check if control agrees with model input
        for v, iv in enumerate(self.model.input_vars):
            for n in range(self.N):
                assert (self.control[n, v, :] == self.model.params[iv][n, :]).all()

    def get_state_vars_dict(self):
        """Creates a numba dictionary which maps the state variable names with their indices."""
        state_vars_dict = Dict.empty(
            key_type=types.unicode_type,
            value_type=types.int8,
        )
        for sv_ind in range(self.dim_vars):
            state_vars_dict[str(self.model.state_vars[sv_ind])] = numba.int8(sv_ind)

        return state_vars_dict

    def adjust_init(self):
        """Adjust the shape of the array provided as init to the model. Use adjustArrayShape function from model_utils."""
        init_dur = self.model.getMaxDelay() + 1
        for init_var in self.model.init_vars:
            if "ou" in init_var:
                continue
            if (
                init_var[:-5] not in self.model.output_vars
                and init_var[:-6] not in self.model.output_vars
            ):
                continue

            iv = self.model.params[init_var]
            self.model.params[init_var] = adjustArrayShape(
                iv, np.ones((self.N, init_dur))
            )

    def adjust_input(self):
        """Adjust the shape of the array provided as input to the model. Use adjustArrayShape function from model_utils."""
        for input_var in self.model.input_vars:
            iv = self.model.params[input_var]
            self.model.params[input_var] = adjustArrayShape(
                iv, np.ones((self.N, self.T))
            )

    def get_xs(self):
        """Extract the complete state of the dynamical system."""
        xs = np.zeros((self.N, self.dim_out, self.T))

        for ind_ov, ov in enumerate(self.model.output_vars):
            xs[:, ind_ov, 1:] = self.model[ov]
            for iv in self.model.init_vars:
                if str(ov) + "_init" == str(iv):
                    xs[:, ind_ov, 0] = self.model.params[iv][:, 0]
                    continue

        return xs

    def get_xs_delay(self):
        """Extract the complete state of the delayed dynamical system."""
        maxdel = self.model.getMaxDelay()
        if maxdel == 0:
            return self.get_xs()

        xs = np.zeros((self.N, self.dim_out, self.T + maxdel))

        for ind_ov, ov in enumerate(self.model.output_vars):
            xs[:, ind_ov, 1:-maxdel] = self.model[ov]
            for iv in self.model.init_vars:
                if str(ov) + "_init" == str(iv):
                    xs[:, ind_ov, 0] = self.model.params[iv][:, 0]
                    if np.shape(self.model.params[iv])[1] == 1:
                        xs[:, ind_ov, -maxdel:] = self.model.params[iv][:, 0]
                    elif np.shape(self.model.params[iv])[1] == maxdel + 1:
                        xs[:, ind_ov, -maxdel:] = self.model.params[iv][:, 1:]
                    else:
                        print("WRONG DIMENSION IN INPUT ARRAY")
                        raise NotImplementedError

                    continue

        return xs

    def update_input(self):
        """Update the parameters in 'self.model' according to the current control such that 'self.simulate_forward'
        operates with the appropriate control signal.
        """
        # TODO: find elegant way to combine the cases
        for ind_iv, iv in enumerate(self.model.input_vars):
            if self.N == 1:
                self.model.params[iv] = self.control[:, ind_iv, :].reshape(1, -1)
            else:
                self.model.params[iv] = self.control[:, ind_iv, :]

    def simulate_forward(self):
        """Updates 'state_vars' of 'self.model' in accordance to the current 'self.control'. Results for the controllable state
        variables can be accessed with self.get_xs()
        """
        self.model.run()

    @abc.abstractmethod
    def get_model_params(self):
        """Model params as an ordered tuple"""
        pass

    @abc.abstractmethod
    def Dxdot(self):
        """V x V Jacobian of systems dynamics wrt. change of all 'state_vars'."""
        pass

    @abc.abstractmethod
    def Duh(self):
        """Jacobian of systems dynamics wrt. external inputs (control signals) to all 'state_vars'."""
        pass

    def compute_total_cost(self):
        """Compute the total cost as weighted sum precision of all contributing cost terms.
        :rtype: float
        """
        xs = self.get_xs()
        accuracy_cost = cost_functions.accuracy_cost(
            xs,
            self.target,
            self.weights,
            self.cost_matrix,
            self.dt,
            self.cost_interval,
        )
        control_strenght_cost = cost_functions.control_strength_cost(
            self.control, self.weights, self.dt
        )
        return accuracy_cost + control_strenght_cost

    @abc.abstractmethod
    def compute_gradient(self):
        """Compute the gradient of the total cost wrt. the control:
        1. solve the adjoint equation backwards in time
        2. compute derivatives of cost wrt. control
        3. compute Jacobians of the dynamics wrt. control
        4. compute gradient of the cost wrt. control(i.e., negative descent direction)

        :return:        The gradient of the total cost wrt. the control.
        :rtype:         np.ndarray of shape N x V x T
        """
        self.solve_adjoint()
        df_du = cost_functions.derivative_control_strength_cost(
            self.control, self.weights
        )
        d_du = self.Duh()

        return compute_gradient(
            self.N,
            self.dim_vars,
            self.dim_in,
            df_du,
            self.adjoint_state,
            self.control_matrix,
            d_du,
            self.control_interval,
        )

    @abc.abstractmethod
    def compute_hx_list(self):
        """Jacobians of model dynamics wrt. its 'state_vars' at each time step."""
        pass

    @abc.abstractmethod
    def compute_hx_nw(self):
        """Jacobians for each time step for the network coupling."""
        pass

    @abc.abstractmethod
    def compute_dxdoth(self):
        pass

    def solve_adjoint(self):
        """Backwards integration of the adjoint state."""

        if self.model.name == "aln":
            self.fullstate = self.get_fullstate()

        hx_nw = self.compute_hx_nw()
        dxdoth = self.compute_dxdoth()
        hx_list, del_list = self.compute_hx_list()

        # Derivative of cost wrt. controllable 'state_vars'.
        df_dx = cost_functions.derivative_accuracy_cost(
            self.get_xs(),
            self.target,
            self.weights,
            self.cost_matrix,
            self.cost_interval,
        )
        self.adjoint_state = solve_adjoint_ww(
            hx_list,
            del_list,
            hx_nw,
            df_dx,
            self.state_dim,
            self.dt,
            self.N,
            self.T,
            self.Dmat_ndt,
            dxdoth,
            numba.typed.List(self.model.state_vars),
            numba.typed.List(self.model.output_vars),
        )

    def decrease_step(self, cost, cost0, step, control0, factor_down, cost_gradient):
        """Find a step size which leads to improved cost given the gradient. The step size is iteratively decreased.
        The control-inputs are updated in place according to the found step size via the
        "self.update_input()" call.

        :param cost:    Cost after applying control update according to gradient with first valid step size (numerically
                        stable).
        :type cost:     float
        :param cost0:   Cost without updating the control.
        :type cost0:    float
        :param step:    Step size initial to the iterative decreasing.
        :type step:     float
        :param control0:    The unchanged control signal.
        :type control0:     np.ndarray N x V x T
        :param factor_down:  Factor the step size is scaled with in each iteration until cost is improved.
        :type factor_down:   float
        :param cost_gradient:   Gradient of the total cost wrt. the control signal.
        :type cost_gradient:    np.ndarray of shape N x V x T

        :return:    The selected step size and the count-variable how often step-adjustment-loop was executed.
        :rtype:     tuple[float, int]
        """
        if self.M > 1:
            noisy = True
        else:
            noisy = False

        counter = 0

        while (
            cost > cost0
        ):  # Decrease the step size until first step size is found where cost is improved.
            step *= factor_down  # Decrease step size.
            counter += 1
            # print(step, cost, cost0)

            # Inplace updating of models control bc. forward-sim relies on models parameters.
            self.control = update_control_with_limit(
                self.N,
                self.dim_in,
                self.T,
                control0,
                step,
                cost_gradient,
                self.maximum_control_strength,
            )
            self.update_input()

            # Simulate with control updated according to new step and evaluate cost.
            self.simulate_forward()

            if noisy:
                cost = self.compute_cost_noisy(self.M)
            else:
                cost = self.compute_total_cost()

            if (
                counter == self.count_step
            ):  # Exit if the maximum search depth is reached without improvement of
                # cost.
                step = 0.0  # For later analysis only.
                self.control = update_control_with_limit(
                    self.N,
                    self.dim_in,
                    self.T,
                    control0,
                    0.0,
                    np.zeros(control0.shape),
                    self.maximum_control_strength,
                )
                self.update_input()

                self.zero_step_encountered = True
                break

        return step, counter

    def increase_step(self, cost, cost0, step, control0, factor_up, cost_gradient):
        """Find the largest step size which leads to the biggest improvement of cost given the gradient. The step size is
        iteratively increased. The control-inputs are updated in place according to the found step size via the
        "self.update_input()" call.

        :param cost:    Cost after applying control update according to gradient with first valid step size (numerically
                        stable).
        :type cost:     float
        :param cost0:   Cost without updating the control.
        :type cost0:    float
        :param step:    Step size initial to the iterative decreasing.
        :type step:     float
        :param control0:    The unchanged control signal.
        :type control0:     np.ndarray N x V x T
        :param factor_up:  Factor the step size is scaled with in each iteration while the cost keeps improving.
        :type factor_up:   float
        :param cost_gradient:   Gradient of the total cost wrt. the control signal.
        :type cost_gradient:    np.ndarray of shape N x V x T

        :return:    The selected step size and the count-variable how often step-adjustment-loop was executed.
        :rtype:     tuple[float, int]
        """
        if self.M > 1:
            noisy = True
        else:
            noisy = False

        cost_prev = cost0
        counter = 0

        while (
            cost < cost_prev
        ):  # Increase the step size as long as the cost is improving.
            step *= factor_up
            counter += 1

            # Inplace updating of models control bc. forward-sim relies on models parameters
            self.control = update_control_with_limit(
                self.N,
                self.dim_in,
                self.T,
                control0,
                step,
                cost_gradient,
                self.maximum_control_strength,
            )
            self.update_input()

            self.simulate_forward()
            if np.isnan(
                self.get_xs()
            ).any():  # Go back to last step (that was numerically stable and improved cost)
                # and exit.
                logging.info("Increasing step encountered NAN.")
                step /= factor_up  # Undo the last step update by inverse operation.
                self.control = update_control_with_limit(
                    self.N,
                    self.dim_in,
                    self.T,
                    control0,
                    step,
                    cost_gradient,
                    self.maximum_control_strength,
                )
                self.update_input()
                break

            else:
                if noisy:
                    cost = self.compute_cost_noisy(self.M)
                else:
                    cost = self.compute_total_cost()

                if (
                    cost > cost_prev
                ):  # If the cost increases: go back to last step (that resulted in best cost until
                    # then) and exit.
                    step /= factor_up  # Undo the last step update by inverse operation.
                    self.control = update_control_with_limit(
                        self.N,
                        self.dim_in,
                        self.T,
                        control0,
                        step,
                        cost_gradient,
                        self.maximum_control_strength,
                    )
                    self.update_input()
                    break

                else:
                    cost_prev = cost  # Memorize cost with this step size for comparison in next step-update.

                if counter == self.count_step:
                    # Terminate step size search at count limit, exit with the best performing step size.
                    break

        return step, counter

    def step_size(self, cost_gradient):
        """Adaptively choose a step size for control update.

        :param cost_gradient:   N x V x T gradient of the total cost wrt. control.
        :type cost_gradient:    np.ndarray

        :return:    Step size that got multiplied with the 'cost_gradient'.
        :rtype:     float
        """
        if self.M > 1:
            noisy = True
        else:
            noisy = False

        self.simulate_forward()
        if noisy:
            cost0 = self.compute_cost_noisy(self.M)
        else:
            cost0 = (
                self.compute_total_cost()
            )  # Current cost without updating the control according to the "cost_gradient".

        step = (
            self.step
        )  # Load step size of last optimization-iteration as initial guess.

        control0 = (
            self.control
        )  # Memorize unchanged control throughout step-size computation.

        while (
            True
        ):  # Reduce the step size, if numerical instability occurs in the forward-simulation.
            # inplace updating of models control bc. forward-sim relies on models parameters
            self.control = update_control_with_limit(
                self.N,
                self.dim_in,
                self.T,
                control0,
                step,
                cost_gradient,
                self.maximum_control_strength,
            )
            self.update_input()

            # Input signal might be too high and produce diverging values in simulation.
            self.simulate_forward()

            if np.isnan(
                self.get_xs()
            ).any():  # Detect numerical instability due to too large control update.
                step *= (
                    self.factor_down**2
                )  # Double the step for faster search of stable region.
                self.step = step
                print(f"Diverging model output, decrease step size to {step}.")
            else:
                break
        if noisy:
            cost = self.compute_cost_noisy(self.M)
        else:
            cost = (
                self.compute_total_cost()
            )  # Cost after applying control update according to gradient with first valid
        # step size (numerically stable).
        # print(cost, cost0)
        if (
            cost > cost0
        ):  # If the cost choosing the first (stable) step size is no improvement, reduce step size by bisection.
            step, counter = self.decrease_step(
                cost, cost0, step, control0, self.factor_down, cost_gradient
            )

        elif (
            cost < cost0
        ):  # If the cost is improved with the first (stable) step size, search for larger steps with even better
            # reduction of cost.

            step, counter = self.increase_step(
                cost, cost0, step, control0, self.factor_up, cost_gradient
            )

        else:  # Remark: might be included as part of adaptive search for further improvement.
            step = 0.0  # For later analysis only.
            counter = 0
            self.zero_step_encountered = True

        self.step = step  # Memorize the last step size for the next optimization step with next gradient.

        self.step_sizes_loops_history.append(counter)
        self.step_sizes_history.append(step)

        return step

    def optimize(self, n_max_iterations):
        """Optimization method
            Choose deterministic (M=1 noise realizations) or stochastic (M>1 noise realizations) approach.
            The control-inputs are updated in place throughout the optimization.

        :param n_max_iterations: Maximum number of iterations of gradient descent.
        :type n_max_iterations:  int
        """

        # If changed between repeated calls of ".optimize()".
        self.cost_interval = convert_interval(self.cost_interval, self.T)
        self.control_interval = convert_interval(self.control_interval, self.T)

        self.control = update_control_with_limit(
            self.N,
            self.dim_in,
            self.T,
            self.control,
            0.0,
            np.zeros(self.control.shape),
            self.maximum_control_strength,
        )  # To avoid issues in repeated executions.
        self.control = limit_control_to_interval(
            self.N, self.dim_in, self.T, self.control, self.control_interval
        )

        if self.M == 1:
            print("Compute control for a deterministic system")
            return self.optimize_deterministic(n_max_iterations)
        else:
            print("Compute control for a noisy system")
            return self.optimize_noisy(n_max_iterations)

    def optimize_deterministic(self, n_max_iterations):
        """Compute the optimal control signal for noise averaging method 0 (deterministic, M=1).

        :param n_max_iterations: maximum number of iterations of gradient descent
        :type n_max_iterations: int
        """

        # (I) forward simulation
        self.simulate_forward()  # yields x(t)

        cost = self.compute_total_cost()
        print(f"Cost in iteration 0: %s" % (cost))
        if (
            len(self.cost_history) == 0
        ):  # add only if control model has not yet been optimized
            self.cost_history.append(cost)

        for i in range(1, n_max_iterations + 1):
            self.gradient = self.compute_gradient()

            if np.isnan(self.gradient).any():
                print("nan in gradient, break")
                break

            self.step_size(-self.gradient)
            self.simulate_forward()

            cost = self.compute_total_cost()
            if i in self.print_array:
                print(f"Cost in iteration %s: %s" % (i, cost))
            self.cost_history.append(cost)

            if self.zero_step_encountered:
                print(f"Converged in iteration %s with cost %s" % (i, cost))
                break

        print(f"Final cost : %s" % (cost))

    def optimize_noisy(self, n_max_iterations):
        """Compute the optimal control signal for noise averaging method 3.

        :param n_max_iterations: maximum number of iterations of gradient descent
        :type n_max_iterations: int
        """

        # initialize array containing M gradients (one per noise realization) for each iteration
        grad_m = np.zeros((self.M, self.N, self.dim_in, self.T))
        consecutive_zero_step = 0

        if len(self.control_history) == 0:
            self.control_history.append(self.control)

        if (
            self.validate_per_step
        ):  # if cost is computed for M_validation realizations in every step
            for m in range(self.M):
                self.simulate_forward()
                grad_m[m, :] = self.compute_gradient()
            cost = self.compute_cost_noisy(self.M_validation)
        else:
            cost_m = 0.0
            for m in range(self.M):
                self.simulate_forward()
                cost_m += self.compute_total_cost()
                grad_m[m, :] = self.compute_gradient()
            cost = cost_m / self.M

        print(f"Mean cost in iteration 0: %s" % (cost))

        if len(self.cost_history) == 0:
            self.cost_history.append(cost)

        for i in range(1, n_max_iterations + 1):
            self.gradient = np.mean(grad_m, axis=0)

            count = 0
            while count < self.count_noisy_step:
                count += 1
                self.zero_step_encountered = False
                self.step_size(-self.gradient)
                if not self.zero_step_encountered:
                    consecutive_zero_step = 0
                    break

            if self.zero_step_encountered:
                consecutive_zero_step += 1
                print("Failed to improve further for noisy system.")

                if consecutive_zero_step > 2:
                    print(
                        "Failed to improve further for noisy system three times in a row, stop optimization."
                    )
                    break

            self.control_history.append(self.control)

            if (
                self.validate_per_step
            ):  # if cost is computed for M_validation realizations in every step
                for m in range(self.M):
                    self.simulate_forward()
                    grad_m[m, :] = self.compute_gradient()
                cost = self.compute_cost_noisy(self.M_validation)
            else:
                cost_m = 0.0
                for m in range(self.M):
                    self.simulate_forward()
                    cost_m += self.compute_total_cost()
                    grad_m[m, :] = self.compute_gradient()
                cost = cost_m / self.M

            if i in self.print_array:
                print(f"Mean cost in iteration %s: %s" % (i, cost))
            self.cost_history.append(cost)

        # take most successful control as optimal control
        min_index = np.argmin(self.cost_history)
        oc = self.control_history[min_index]

        print(f"Minimal cost found at iteration %s" % (min_index))

        self.control = oc
        self.update_input()
        self.cost_validation = self.compute_cost_noisy(self.M_validation)
        print(
            f"Final cost validated with %s noise realizations : %s"
            % (self.M_validation, self.cost_validation)
        )

    def compute_cost_noisy(self, M):
        """Computes the average cost from 'M_validation' noise realizations.

        :param M:                   Number of noise realizations. M=1 implies deterministic case. Defaults to 1.
        :type M:                    int, optional

        :rtype: float
        """
        cost_validation = 0.0
        m = 0
        while m < M:
            self.simulate_forward()
            if np.isnan(self.get_xs()).any():
                continue
            cost_validation += self.compute_total_cost()
            m += 1
        return cost_validation / M
