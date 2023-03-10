from neurolib.control.optimal_control.oc import OC, update_control_with_limit
from neurolib.control.optimal_control import cost_functions
import numpy as np
import numba
from neurolib.models.fhn.timeIntegration import compute_hx, compute_hx_nw


@numba.njit
def compute_gradient(N, dim_out, T, fk, adjoint_state, control_matrix, duh):
    """Compute the gradient of the total cost wrt. to the control signals.
    :param N:       number of nodes in the network
    :type N:        int
    :param dim_out: number of 'output variables' of the model
    :type dim_out:  int
    :param T:       length of simulation (time dimension)
    :type T:        int
    :param fk:      Derivative of the cost functionals wrt. to the control signal.
    :type fk:   np.ndarray of shape N x V x T
    :param adjoint_state:
    :type adjoint_state: np.ndarray of shape N x V x T
    :param control_matrix: Binary matrix that defines nodes and variables where control inputs are active, defaults to None.
    :type control_matrix:  np.ndarray of shape N x V
    :param duh: Jacobian of systems dynamics wrt. to I_ext (external control input)
    :type duh:  np.ndarray of shape V x V
    :return: The gradient of the total cost wrt. to the control.
    :rtype: np.ndarray of shape N x V x T
    """
    grad = np.zeros(fk.shape)
    for n in range(N):
        for v in range(dim_out):
            for t in range(T):
                grad[n, v, t] = fk[n, v, t] + adjoint_state[n, v, t] * control_matrix[n, v] * duh[v, v]
    return grad


class OcFhn(OC):
    """
    :param model:
    :type model: neurolib.models.fhn.model.FHNModel
    """

    def __init__(
        self,
        model,
        target,
        w_p=1.0,
        w_2=1.0,
        maximum_control_strength=None,
        print_array=[],
        precision_cost_interval=(None, None),
        precision_matrix=None,
        control_matrix=None,
        M=1,
        M_validation=0,
        validate_per_step=False,
    ):
        super().__init__(
            model,
            target,
            w_p=w_p,
            w_2=w_2,
            maximum_control_strength=maximum_control_strength,
            print_array=print_array,
            precision_cost_interval=precision_cost_interval,
            precision_matrix=precision_matrix,
            control_matrix=control_matrix,
            M=M,
            M_validation=M_validation,
            validate_per_step=validate_per_step,
        )

        assert self.model.name == "fhn"

        assert self.T == self.model.params["x_ext"].shape[1]
        assert self.T == self.model.params["y_ext"].shape[1]

        # ToDo: here, a method like neurolib.model_utils.adjustArrayShape() should be applied!
        if self.N == 1:  # single-node model
            if self.model.params["x_ext"].ndim == 1:
                print("not implemented yet")
            else:
                control = np.concatenate((self.model.params["x_ext"], self.model.params["y_ext"]), axis=0)[
                    np.newaxis, :, :
                ]
        else:
            control = np.stack((self.model.params["x_ext"], self.model.params["y_ext"]), axis=1)

        for n in range(self.N):
            assert (control[n, 0, :] == self.model.params["x_ext"][n, :]).all()
            assert (control[n, 1, :] == self.model.params["y_ext"][n, :]).all()

        self.control = update_control_with_limit(control, 0.0, np.zeros(control.shape), self.maximum_control_strength)

    def get_xs(self):
        """Stack the initial condition with the simulation results for both populations."""
        if self.model.params["xs_init"].shape[1] == 1:
            p1 = np.concatenate((self.model.params["xs_init"], self.model.params["ys_init"]), axis=1)[:, :, np.newaxis]
            xs = np.concatenate(
                (
                    p1,
                    np.stack((self.model.x, self.model.y), axis=1),
                ),
                axis=2,
            )
        else:
            p1 = np.stack((self.model.params["xs_init"][:, -1], self.model.params["ys_init"][:, -1]), axis=1)[
                :, :, np.newaxis
            ]
            xs = np.concatenate(
                (
                    p1,
                    np.stack((self.model.x, self.model.y), axis=1),
                ),
                axis=2,
            )

        return xs

    def update_input(self):
        """Update the parameters in self.model according to the current control such that self.simulate_forward
        operates with the appropriate control signal.
        """
        # ToDo: find elegant way to combine the cases
        if self.N == 1:
            self.model.params["x_ext"] = self.control[:, 0, :].reshape(1, -1)  # Reshape as row vector to match access
            self.model.params["y_ext"] = self.control[:, 1, :].reshape(1, -1)  # in model's time integration.

        else:
            self.model.params["x_ext"] = self.control[:, 0, :]
            self.model.params["y_ext"] = self.control[:, 1, :]

    def Dxdot(self):
        """4x4 Jacobian of systems dynamics wrt. to change of systems variables."""
        raise NotImplementedError  # return np.eye(4)

    def Duh(self):
        """4x4 Jacobian of systems dynamics wrt. to I_ext (external control input)"""
        return np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

    def compute_hx(self):
        """Jacobians for each time step.

        :return: Array of length self.T containing 4x4-matrices
        :rtype: np.ndarray
        """
        return compute_hx(
            self.model.params["alpha"],
            self.model.params["beta"],
            self.model.params["gamma"],
            self.model.params["tau"],
            self.model.params["epsilon"],
            self.N,
            self.dim_vars,
            self.T,
            self.get_xs(),
        )

    def compute_hx_nw(self):
        """Jacobians for each time step for the network coupling.

        :return: N x N x T x (4x4) array
        :rtype: np.ndarray
        """
        return compute_hx_nw(
            self.model.params["K_gl"],
            self.model.params["Cmat"],
            self.model.params["coupling"],
            self.N,
            self.dim_vars,
            self.T,
        )

    def compute_gradient(self):
        """
        Du @ fk + adjoint_k.T @ Du @ h
        """
        self.solve_adjoint()
        fk = cost_functions.derivative_energy_cost(self.control, self.w_2)
        duh = self.Duh()

        return compute_gradient(self.N, self.dim_out, self.T, fk, self.adjoint_state, self.control_matrix, duh)
