import numpy as np
import numba
from neurolib.optimal_control import cost_functions
from .oc_fhn_jit import compute_hx, solve_adjoint
import logging


class OcFhn:

    def __init__(self, fhn_model, target, w_p=1, w_2=1, print_array=[], precision_cost_interval=(0, None)):
        """
            :param fhn_model
            :param target:
            :param w_p: weight of the precision cost term
            :param w_2: weight of the L2 cost term
            :param target: 2xT matrix with [0, :] target of x-population and [1, :] target of y-population
        """

        self.model = fhn_model

        self.target = target

        self.w_p = w_p
        self.w_2 = w_2

        self.step = 10.

        self.dt = self.model.params["dt"]  # maybe redundant but for now code clarity
        self.duration = self.model.params["duration"]  # maybe redundant but for now code clarity

        self.T = np.around(self.duration / self.dt, 0).astype(int) + 1  # Total number of time steps is
                                                                        # initial condition.
        # + forward simulation steps of neurolibs model.run().
        self.output_dim = (2, self.T)  # FHN has two variables

        # check correct specification of inputs
        assert self.T == self.model.params["x_ext"].shape[1]
        assert self.T == self.model.params["y_ext"].shape[1]

        self.xs_init = np.vstack((self.model.params["xs_init"], self.model.params["ys_init"]))  # maybe redundant,
                                                                                                # but convenient
        assert self.xs_init.shape == (2, 1), ("Specification of initial conditions does not match the current OC "
                                              "implementation.")

        self.adjoint_state = np.zeros(self.output_dim)

        # ToDo: HUGE REMARK ON MODELS SPECIFIED FOR MULTIPLE NODES!!!!!
        # ToDo: maybe add input to both?
        self.control = np.vstack((self.model.params["x_ext"], self.model.params["y_ext"]))

        self.cost_history = np.array([])
        self.step_sizes_history = np.array([])
        self.step_sizes_loops_history = np.array([])
        self.cost_history_index = 0

        self.x_controls = self.model.params["x_ext"]    # save control signals throughout optimization iterations for
                                                        # later analysis

        self.x_grads = np.array(())  # save gradients throughout optimization iterations for
                                     # later analysis

        self.print_array = print_array

        self.zero_step_encountered = False  # deterministic gradient descent cannot further improve

        self.precision_cost_interval = precision_cost_interval

    def add_cost_to_history(self, cost):
        """ For later analysis.
        """
        self.cost_history[self.cost_history_index] = cost
        self.cost_history_index += 1

    def get_xs(self):
        """ Stack the initial condition with the simulation results for both populations.
        """
        # ToDo: assert to make sure single node setting is fullfilled (bc. of incorporation of init conditions)
        return np.hstack((np.vstack((self.model.params["xs_init"], self.model.params["ys_init"])),
                          np.vstack((self.model.getOutputs()["x"], self.model.getOutputs()["y"]))))

    def simulate_forward(self):
        """ Updates self.xs in accordance to the current self.control.
            Results can be accessed with self.get_xs()
        """
        self.model.run()

    def update_input(self):
        """ Update the parameters in self.model according to the current control such that self.simulate_forward
            operates with the appropriate control signal.
        """
        self.model.params["x_ext"] = self.control[0, :].reshape(1, -1)  # Reshape as row vector to match access
        self.model.params["y_ext"] = self.control[1, :].reshape(1, -1)  # in model's time integration.

        self.x_controls = np.vstack((self.x_controls, self.control[0, :].reshape(1, -1)))

    def Dxdot(self):
        """ 2x2 Jacobian of systems dynamics wrt. to change of systems variables.
        """
        return(np.array([[1, 0],
                         [0, 1]]))

    def Du(self):
        """ 2x2 Jacobian of systems dynamics wrt. to I_ext (external control input)
        """
        return(np.array([[-1, 0],
                         [0, -1]]))

    def compute_total_cost(self):
        """
        """
        precision_cost = cost_functions.precision_cost(self.target,
                                                       self.get_xs(),
                                                       w_p=self.w_p,
                                                       interval=self.precision_cost_interval)
        energy_cost = cost_functions.energy_cost(self.control, w_2=self.w_2)
        return precision_cost + energy_cost

    def compute_gradient(self):
        """
        Du @ fk + adjoint_k.T @ Du @ h
        """
        self.solve_adjoint()
        fk = cost_functions.derivative_energy_cost(self.control, self.w_2)

        return fk + (self.adjoint_state.T @ self.Du()).T

    def compute_hx(self):
        """ Jacobians for each time step.
            :return: array containing self.T 2x2-matrices
        """
        return compute_hx(self.model.params["alpha"],
                          self.model.params["beta"],
                          self.model.params["gamma"],
                          self.model.params["tau"],
                          self.model.params["epsilon"],
                          self.T,
                          self.get_xs()[0, :])

    def solve_adjoint(self):
        """ Backwards integration.
            :param fx: df/dx
            :param hx: dh/dx
        """
        hx = self.compute_hx()

        # ToDo: generalize, not only precision cost
        fx = cost_functions.derivative_precision_cost(self.target,
                                                      self.get_xs(),
                                                      self.w_p,
                                                      interval=self.precision_cost_interval)

        self.adjoint_state = solve_adjoint(hx, fx, self.output_dim, self.dt, self.T)

    def step_size(self, cost_gradient):
        """
            use cost_gradient to avoid unnecessary re-computations (also of the adjoint state)
        """
        self.simulate_forward()
        cost0 = self.compute_total_cost()
        factor = 0.7
        step = self.step
        counter = 0.

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

            # inplace updating of models x_ext bc. forward-sim relies on models parameters
            self.control = control0 + step * cost_gradient
            self.update_input()

            # time_series = model(x0, duration, dt, control1)
            self.simulate_forward()
            # cost = total_cost(control1, time_series, target)
            cost = self.compute_total_cost()

            if counter == 30.:
                step = 0.   # for later analysis only
                self.control = control0
                self.update_input()
                logging.WARNING("Zero step size encountered! Optimization got to a halt.")
                self.zero_step_encountered = True
                break

        self.step_sizes_loops_history[self.cost_history_index - 1] = counter
        self.step_sizes_history[self.cost_history_index - 1] = step

    def optimize(self, n_max_iterations):
        """ Compute the optimal control signal.
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

        # (II) control gradient happens within "step_size"
        # (III) step size and control update
        grad = self.compute_gradient()
        self.step_size(-grad)
        self.x_grads = grad[0, :]

        # (IV) forward simulation
        self.simulate_forward()

        for i in range(2, n_max_iterations+1):
            cost = self.compute_total_cost()
            if i in self.print_array:
                print(f"Cost in iteration %s: %s" % (i, cost))
            self.add_cost_to_history(cost)
            # (V.I) control gradient happens within "step_size"
            # (V.II) step size and control update
            grad = self.compute_gradient()
            self.step_size(-grad)
            self.x_grads = np.vstack((self.x_grads, grad))

            # (V.III) forward simulation
            self.simulate_forward()  # yields x(t)

            if self.zero_step_encountered:
                break
