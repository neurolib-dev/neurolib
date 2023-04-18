from neurolib.control.optimal_control.oc import OC, update_control_with_limit
from neurolib.control.optimal_control import cost_functions
import numba
import numpy as np
from neurolib.models.hopf.timeIntegration import compute_hx, compute_hx_nw


@numba.njit
def compute_gradient(N, dim_out, T, df_du, adjoint_state, control_matrix, d_du):
    """Compute the gradient of the total cost wrt. to the control signals (explicitly and implicitly) given the adjoint
       state, the Jacobian of the total cost wrt. to explicit control contributions and the Jacobian of the dynamics
       wrt. to explicit control contributions.

    :param N:       Number of nodes in the network.
    :type N:        int
    :param dim_out: Number of 'output variables' of the model.
    :type dim_out:  int
    :param T:       Length of simulation (time dimension).
    :type T:        int
    :param df_du:   Derivative of the cost wrt. to the explicit control contributions to cost functionals.
    :type df_du:    np.ndarray of shape N x V x T
    :param adjoint_state:  Solution of the adjoint equation.
    :type adjoint_state:   np.ndarray of shape N x V x T
    :param control_matrix: Binary matrix that defines nodes and variables where control inputs are active, defaults to
                           None.
    :type control_matrix:  np.ndarray of shape N x V
    :param d_du:     Jacobian of systems dynamics wrt. to the external inputs (control).
    :type d_du:      np.ndarray of shape V x V
    :return:         The gradient of the total cost wrt. to the control.
    :rtype:          np.ndarray of shape N x V x T
    """
    grad = np.zeros(df_du.shape)
    for n in range(N):
        for v in range(dim_out):
            for t in range(T):
                grad[n, v, t] = df_du[n, v, t] + adjoint_state[n, v, t] * control_matrix[n, v] * d_du[v, v]

    return grad


class OcHopf(OC):
    """Class for optimal control specific to neurolib's implementation of the Stuart-Landau model with Hopf
        bifurcation ("Hopf model").

    :param model: Instance of Hopf model (can describe a single Hopf node or a network of coupled Hopf nodes.
    :type model: neurolib.models.hopf.model.HopfModel
    """

    # Remark: very similar to FHN!
    def __init__(
        self,
        model,
        target,
        weights=None,
        maximum_control_strength=None,
        print_array=[],
        cost_interval=(None, None),
        cost_matrix=None,
        control_matrix=None,
        M=1,
        M_validation=0,
        validate_per_step=False,
    ):
        super().__init__(
            model,
            target,
            weights=weights,
            maximum_control_strength=maximum_control_strength,
            print_array=print_array,
            cost_interval=cost_interval,
            cost_matrix=cost_matrix,
            control_matrix=control_matrix,
            M=M,
            M_validation=M_validation,
            validate_per_step=validate_per_step,
        )

        assert self.model.name == "hopf"

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

        # save control signals throughout optimization iterations for later analysis
        # self.control_history.append(self.control)

    def get_xs(self):
        """Stack the initial condition with the simulation results for dynamic variables 'x' and 'y' of Hopf model.

        :rtype:     np.ndarray of shape N x V x T
        """
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
        """Update the parameters in 'self.model' according to the current control such that 'self.simulate_forward'
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
        """4 x 4 Jacobian of systems dynamics wrt. to change of systems variables."""
        # Currently not explicitly required since it is identity matrix.
        raise NotImplementedError  # return np.eye(4)

    def Duh(self):
        """4 x 4 Jacobian of systems dynamics wrt. to external inputs (control signals) to all 'state_vars'. There are no
           inputs to the noise variables 'x_ou' and 'y_ou' in the model.

        :rtype:     np.ndarray of shape 4 x 4
        """
        return np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

    def compute_hx(self):
        """Jacobians of Hopf model wrt. to its 'state_vars' at each time step.

        :return:        Array that contains Jacobians for all nodes in all time steps.
        :rtype:         np.ndarray of shape N x T x 4 x 4
        """
        return compute_hx(
            self.model.params.a,
            self.model.params.w,
            self.N,
            self.dim_vars,
            self.T,
            self.get_xs(),
        )

    def compute_hx_nw(self):
        """Jacobians for each time step for the network coupling.

        :return:    Jacobians for network connectivity in all time steps.
        :rtype:     np.ndarray of shape N x N x T x (4x4)
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
        """Compute the gradient of the total cost wrt. to the control signals. This is achieved by first, solving the
           adjoint equation backwards in time. Second, derivatives of the cost wrt. to explicit control variables are
           evaluated as well as the Jacobians of the dynamics wrt. to explicit control. Then the decent direction /
           gradient of the cost wrt. to control (in its explicit form AND IMPLICIT FORM) is computed.

        :return:         The gradient of the total cost wrt. to the control.
        :rtype:          np.ndarray of shape N x V x T
        """
        self.solve_adjoint()
        df_du = cost_functions.derivative_control_strength_cost(self.control, self.weights)
        d_du = self.Duh()

        return compute_gradient(self.N, self.dim_out, self.T, df_du, self.adjoint_state, self.control_matrix, d_du)
