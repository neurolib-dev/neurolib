import abc
import numba
import numpy as np
from neurolib.optimal_control import cost_functions
import logging
import copy


# compared loops agains "@", "np.matmul" and "np.dot": loops ~factor 3.5 faster
@numba.njit
def solve_adjoint(hx, hx_nw, fx, state_dim, dt, N, T):
    """Backwards integration of the adjoint state.
    :param hx: dh/dx    Jacobians for each time step.
    :type hx:           np.ndarray of shape Tx2x2
    :param hx_nw:       Jacobians for each time step for the network coupling.
    :type hx_nw:        np.ndarray
    :param fx: df/dx    Derivative of cost function wrt. to systems dynamics.
    :type fx:           np.ndarray
    :param state_dim:   Dimensions of state (NxVxT).
    :type state_dim:    tuple
    :param dt:          Time resolution of integration.
    :type dt:           float
    :param N:           number of nodes in the network
    :type N:            int
    :param T:           length of simulation (time dimension)
    :type T:            int

    :return:            adjoint state.
    :rtype:             np.ndarray of shape `state_dim`
    """
    # ToDo: generalize, not only precision cost
    adjoint_state = np.zeros(state_dim)
    fx_fullstate = np.zeros(state_dim)
    fx_fullstate[:, :2, :] = fx.copy()

    for ind in range(T - 2, 0, -1):
        for n in range(N):
            der = fx_fullstate[n, :, ind + 1].copy()
            for k in range(len(der)):
                for i in range(len(der)):
                    der[k] += adjoint_state[n, i, ind + 1] * hx[n, ind + 1][i, k]
            for n2 in range(N):
                for k in range(len(der)):
                    for i in range(len(der)):
                        der[k] += adjoint_state[n2, i, ind + 1] * hx_nw[n2, n, ind + 1][i, k]
            adjoint_state[n, :, ind] = adjoint_state[n, :, ind + 1] - der * dt

    return adjoint_state


class OC:
    def __init__(
        self,
        model,
        target,
        w_p=1,
        w_2=1,
        print_array=[],
        precision_cost_interval=(0, None),
        precision_matrix=None,
        control_matrix=None,
        M=1,
        M_validation=0,
        validate_per_step=False,
        method=None,
    ):
        """
        Base class for optimal control. Model specific methods, like

        :param model:   An instance of neurolib's Model-class. Parameters like ".duration" and methods like ".run()"
                            are used within the optimal control.
        :type model:    neurolib.models.model

        :param target:      2xT matrix with [0, :] target of x-population and [1, :] target of y-population.
        :type target:       np.ndarray

        :param w_p:         Weight of the precision cost term, defaults to 1.
        :type w_p:          float, optional

        :param w_2:         Weight of the L2 cost term, defaults to 1.
        :type w_2:          float, optional

        :param print_array: Array of optimization-iteration-indices (starting at 1) in which cost is printed out.
                            Defaults to empty list `[]`.
        :type print_array:  list, optional

        :param precision_cost_interval: [t_start, t_end]. Indices of start and end point (both inclusive) of the
                                        time interval in which the precision cost is evaluated. Default is full time
                                        series. Defaults to (0, None).
        :type precision_cost_interval:  tuple, optional

        :param precision_matrix: NxV binary matrix that defines nodes and channels of precision measurement, defaults to
                                 None
        :type precision_matrix:  np.ndarray

        :param control_matrix: NxV binary matrix that defines active control inputs, defaults to None
        :type control_matrix:  np.ndarray

        :param M :                  Number of noise realizations. M=1 implies deterministic case. Defaults to 1.
        :type M:                    int, optional

        :param M_validation:        Number of noise realizations for validation (only used in stochastic case, M>1).
                                    Defaults to 0.
        :type M_validation:         int, optional

        :param validate_per_step:   True for validation in each iteration of the optimization, False for
                                    validation only after final optimization iteration (only used in stochastic case,
                                    M>1). Defaults to False.
        :type validate_per_step:    bool, optional

        :param method:              Noise averaging method (only used in stochastic case, M>1), defaults to None.
        :type method:               None or str, optional

        """

        self.model = copy.deepcopy(model)

        if self.model.getMaxDelay() > 0.0:
            print("Delay not yet implemented, please set delays to zero")

        self.target = target

        self.w_p = w_p
        self.w_2 = w_2

        self.step = 10.0

        self.N = self.model.params.N

        self.dim_vars = len(self.model.state_vars)
        self.dim_out = len(self.model.output_vars)

        if self.N > 1:  # check that coupling matrix has zero diagonal
            assert np.all(np.diag(self.model.params["Cmat"]) == 0.0)

        self.precision_matrix = precision_matrix
        if isinstance(self.precision_matrix, type(None)):
            self.precision_matrix = np.ones(
                (self.N, self.dim_out)
            )  # default: measure precision in all variables in all nodes

        # check if matrix is binary
        assert np.array_equal(self.precision_matrix, self.precision_matrix.astype(bool))

        self.control_matrix = control_matrix
        if isinstance(self.control_matrix, type(None)):
            self.control_matrix = np.ones((self.N, self.dim_vars))  # default: all channels in all nodes active

        # check if matrix is binary
        assert np.array_equal(self.control_matrix, self.control_matrix.astype(bool))

        self.M = max(1, M)
        self.M_validation = M_validation
        self.cost_validation = 0.0
        self.validate_per_step = validate_per_step
        self.method = method

        if self.model.params.sigma_ou != 0.0:  # noisy system
            if self.M <= 1:
                logging.warning(
                    "For noisy system, please chose parameter M larger than 1 (recommendation > 10)."
                    + "\n"
                    + 'If you want to study a deterministic system, please set model parameter "sigma_ou" to zero'
                )
            if self.M > self.M_validation:
                print('Parameter "M_validation" should be chosen larger than parameter "M".')
        else:  # deterministic system
            if self.M > 1 or self.M_validation != 0 or validate_per_step or method != None:
                print(
                    'For deterministic systems, parameters "M", "M_validation", "validate_per_step" and "method" are not relevant.'
                    + "\n"
                    + 'If you want to study a noisy system, please set model parameter "sigma_ou" larger than zero'
                )

        self.dt = self.model.params["dt"]  # maybe redundant but for now code clarity
        self.duration = self.model.params["duration"]  # maybe redundant but for now code clarity

        self.T = np.around(self.duration / self.dt, 0).astype(int) + 1  # Total number of time steps is
        # initial condition.

        # + forward simulation steps of neurolibs model.run().
        self.state_dim = (
            self.N,
            self.dim_vars,
            self.T,
        )  # dimensions of state. Model has N network nodes, V state variables, T time points

        self.adjoint_state = np.zeros(self.state_dim)

        # check correct specification of inputs
        # ToDo: different models have different inputs
        self.control = None

        self.cost_history = np.array([])
        self.step_sizes_history = np.array([])
        self.step_sizes_loops_history = np.array([])
        self.cost_history_index = 0

        # ToDo: not "x" in other models
        self.x_controls = None  # save control signals throughout optimization iterations for
        # later analysis
        self.x_grads = np.array([])  # save gradients throughout optimization iterations for
        # later analysis

        self.print_array = print_array

        self.zero_step_encountered = False  # deterministic gradient descent cannot further improve

        self.precision_cost_interval = precision_cost_interval

    def add_cost_to_history(self, cost):
        """For later analysis."""
        self.cost_history[self.cost_history_index] = cost
        self.cost_history_index += 1

    @abc.abstractmethod
    def get_xs(self):
        """Stack the initial condition with the simulation results for both populations."""
        # ToDo: different for different models
        pass

    def simulate_forward(self):
        """Updates self.xs in accordance to the current self.control.
        Results can be accessed with self.get_xs()
        """
        self.model.run()

    @abc.abstractmethod
    def update_input(self):
        """Update the parameters in self.model according to the current control such that self.simulate_forward
        operates with the appropriate control signal.
        """
        pass

    @abc.abstractmethod
    def Dxdot(self):
        """2x2 Jacobian of systems dynamics wrt. to change of systems variables."""
        # ToDo: model dependent
        pass

    @abc.abstractmethod
    def Du(self):
        """2x2 Jacobian of systems dynamics wrt. to I_ext (external control input)"""
        # ToDo: model dependent
        pass

    def compute_total_cost(self):
        """Compute the total cost as weighted sum precision of precision and L2 term."""
        precision_cost = cost_functions.precision_cost(
            self.target,
            self.get_xs(),
            self.w_p,
            self.N,
            self.precision_matrix,
            interval=self.precision_cost_interval,
        )
        energy_cost = cost_functions.energy_cost(self.control, w_2=self.w_2)
        return precision_cost + energy_cost

    @abc.abstractmethod
    def compute_gradient(self):
        """Du @ fk + adjoint_k.T @ Du @ h"""
        # ToDo: model dependent due to slicing '[:2, :]'
        pass

    @abc.abstractmethod
    def compute_hx(self):
        """Jacobians for each time step.

        :return: Array of length self.T containing 2x2-matrices
        :rtype: np.ndarray
        """
        # ToDo: model dependent
        pass

    @abc.abstractmethod
    def compute_hx_nw(self):
        """Jacobians for each time step for the network coupling

        :return: (N x self.T x (2x2) array
        :rtype: np.ndarray
        """
        # ToDo: model dependent
        pass

    def solve_adjoint(self):
        """Backwards integration of the adjoint state."""
        hx = self.compute_hx()
        hx_nw = self.compute_hx_nw()

        # ToDo: generalize, not only precision cost
        fx = cost_functions.derivative_precision_cost(
            self.target,
            self.get_xs(),
            self.w_p,
            self.precision_matrix,
            interval=self.precision_cost_interval,
        )

        self.adjoint_state = solve_adjoint(hx, hx_nw, fx, self.state_dim, self.dt, self.N, self.T)

    def step_size(self, cost_gradient):
        """Use cost_gradient to avoid unnecessary re-computations (also of the adjoint state)
        :param cost_gradient:
        :type cost_gradient:

        :return:    Step size that got multiplied with the cost_gradient.
        :rtype:     float
        """
        self.simulate_forward()
        cost0 = self.compute_total_cost()
        factor = 0.5
        step = self.step
        counter = 0.0

        control0 = self.control

        while True:
            # inplace updating of models x_ext bc. forward-sim relies on models parameters
            self.control = control0 + step * cost_gradient
            self.update_input()

            # input signal might be too high and produce diverging values in simulation
            self.simulate_forward()
            if np.isnan(self.get_xs()).any():
                step *= factor
                self.step = step
                print("diverging model output, decrease step size to ", step)
                self.control = control0 + step * cost_gradient
                self.update_input()
            else:
                break

        cost = self.compute_total_cost()
        while cost > cost0:
            step *= factor
            counter += 1

            # "step size loop", cost, step)

            # inplace updating of models x_ext bc. forward-sim relies on models parameters
            self.control = control0 + step * cost_gradient
            self.update_input()

            # time_series = model(x0, duration, dt, control1)
            self.simulate_forward()
            # cost = total_cost(control1, time_series, target)
            cost = self.compute_total_cost()

            if counter == 20.0:
                step = 0.0  # for later analysis only
                self.control = control0
                self.update_input()
                # logging.warning("Zero step encoutered, stop bisection")
                if self.M == 1:
                    self.zero_step_encountered = True
                break

        self.step_sizes_loops_history[self.cost_history_index - 1] = counter
        self.step_sizes_history[self.cost_history_index - 1] = step

        return step

    def optimize(self, n_max_iterations):
        """Optimization method
            chose from deterministic (M=1) or one of several stochastic (M>1) approaches

        :param n_max_iterations: maximum number of iterations of gradient descent
        :type n_max_iterations: int

        """
        if self.M == 1:
            print("Compute control for a deterministic system")
            return self.optimize_M0(n_max_iterations)
        else:
            print("Compute control for a noisy system")
            if self.method == "3":
                return self.optimize_M3(n_max_iterations)
            else:
                logging.error("Optimization method not implemented.")
                return

    def optimize_M0(self, n_max_iterations):
        """Compute the optimal control signal for noise averaging method 0 (deterministic, M=1).

        :param n_max_iterations: maximum number of iterations of gradient descent
        :type n_max_iterations: int
        """
        self.cost_history = np.hstack((self.cost_history, np.zeros(n_max_iterations)))
        self.step_sizes_history = np.hstack((self.step_sizes_history, np.zeros(n_max_iterations)))
        self.step_sizes_loops_history = np.hstack((self.step_sizes_loops_history, np.zeros(n_max_iterations)))
        # (I) forward simulation
        self.simulate_forward()  # yields x(t)

        cost = self.compute_total_cost()
        if 1 in self.print_array:
            print(f"Cost in iteration 1: %s" % (cost))
        self.add_cost_to_history(cost)

        # (II) gradient descent takes place within "step_size"
        # (III) step size and control update
        grad = self.compute_gradient()

        self.step_size(-grad)
        # (IV) forward simulation
        self.simulate_forward()

        for i in range(2, n_max_iterations + 1):
            cost = self.compute_total_cost()
            if i in self.print_array:
                print(f"Cost in iteration %s: %s" % (i, cost))
            self.add_cost_to_history(cost)
            # (V.I) gradient descent takes place within "step_size"
            # (V.II) step size and control update
            grad = self.compute_gradient()
            self.step_size(-grad)

            # (V.III) forward simulation
            self.simulate_forward()  # yields x(t)

            if self.zero_step_encountered:
                print(f"Converged in iteration %s with cost %s" % (i, cost))
                break

    def optimize_M3(self, n_max_iterations):
        """Compute the optimal control signal for noise averaging method 3.

        :param n_max_iterations: maximum number of iterations of gradient descent
        :type n_max_iterations: int
        """
        self.cost_history = np.hstack((self.cost_history, np.zeros((n_max_iterations))))
        self.step_sizes_history = np.hstack((self.step_sizes_history, np.zeros(n_max_iterations)))
        self.step_sizes_loops_history = np.hstack((self.step_sizes_loops_history, np.zeros(n_max_iterations)))

        # initialize array containing M gradients (one per noise realization) for each iteration
        grad_m = np.zeros((self.M, self.N, self.dim_out, self.T))

        for i in range(1, n_max_iterations + 1):

            # (I) compute gradient and mean cost
            if self.validate_per_step:  # if cost is computed for M_validation realizations in every step
                for m in range(self.M):
                    self.simulate_forward()
                    grad_m[m, :] = self.compute_gradient()
                cost = self.compute_cost_validation()
            else:
                cost_m = 0.0
                for m in range(self.M):
                    self.simulate_forward()
                    cost_m += self.compute_total_cost()
                    grad_m[m, :] = self.compute_gradient()
                cost = cost_m / self.M

            grad = np.mean(grad_m, axis=0)

            if i in self.print_array:
                print(f"Mean cost in iteration %s: %s" % (i, cost))
            self.add_cost_to_history(cost)

            control0 = self.control
            step = self.get_step_noisy(grad)

            self.control = control0 + step * (-grad)
            self.update_input()
            self.simulate_forward()

            if self.zero_step_encountered:
                break

        if self.validate_per_step:
            self.cost_validation = cost
        else:
            self.cost_validation = self.compute_cost_validation()
            print(f"Final cost validated with %s noise realizations : %s" % (self.M_validation, self.cost_validation))

    def compute_cost_validation(self):
        """Computes the average cost from M_validation noise realizations."""
        cost_validation = 0.0
        for m in range(self.M_validation):
            self.simulate_forward()
            cost_validation += self.compute_total_cost()
        return cost_validation / self.M_validation

    def get_step_noisy(self, gradient):
        """Computes the mean descent step for M noise realizations.

        :param gradient: (mean) gradient of the respective iteration
        :type: np.ndarray

        :return:    Mean descent step size for M noise realizations.
        :rtype:     float
        """

        step = 0.0
        n_steps = 0.0
        control0 = self.control

        for m in range(self.M):
            step_m = self.step_size(-gradient)
            self.control = control0.copy()
            self.update_input()

            # sort out zero step sizes and average only over rest
            if step_m > 0.0:
                step += step_m
                n_steps += 1

        # if step=0 in all M realizations, this will interrupt the optimization
        if n_steps == 0.0:
            self.zero_step_encountered = True

        return step / n_steps
