from neurolib.optimal_control.oc import OC
from neurolib.optimal_control import cost_functions
import numpy as np
import numba


@numba.njit
def jacobian_fhn(alpha, beta, gamma, tau, epsilon, x, V):
    """Jacobian of the FHN dynamical system.
    :param alpha:   FHN model parameter.
    :type alpha:    float

    :param beta:    FHN model parameter.
    :type beta:     float

    :param gamma:   FHN model parameter.
    :type gamma:    float

    :param tau:     FHN model parameter.
    :type tau:      float

    :param epsilon: FHN model parameter.
    :type epsilon:  float

    :param x:       Value of the x-population in the FHN model at a specific time step.
    :type x:        float

    :param V:           number of system variables
    :type V:            int

    :return:        Jacobian matrix.
    :rtype:         np.ndarray of dimensions 2x2
    """
    jacobian = np.zeros((V, V))
    jacobian[0, :2] = [3 * alpha * x**2 - 2 * beta * x - gamma, 1]
    jacobian[1, :2] = [-1 / tau, epsilon / tau]
    return jacobian


@numba.njit
def compute_hx(alpha, beta, gamma, tau, epsilon, N, V, T, xs):
    """Jacobians for each time step.

    :param alpha:   FHN model parameter.
    :type alpha:    float

    :param beta:    FHN model parameter.
    :type beta:     float

    :param gamma:   FHN model parameter.
    :type gamma:    float

    :param tau:     FHN model parameter.
    :type tau:      float

    :param epsilon: FHN model parameter.
    :type epsilon:  float

    :param N:           number of nodes in the network
    :type N:            int
    :param V:           number of system variables
    :type V:            int
    :param T:           length of simulation (time dimension)
    :type T:            int

    :param xs:  The jacobian of the FHN systems dynamics depends only on the constant parameters and the values of
                    the x-population.
    :type xs:   np.ndarray of shape 1xT

    :return: array of length T containing 2x2-matrices
    :rtype: np.ndarray of shape Tx2x2
    """
    hx = np.zeros((N, T, V, V))

    for n in range(N):
        for ind, x in enumerate(xs[n, 0, :]):
            hx[n, ind, :, :] = jacobian_fhn(alpha, beta, gamma, tau, epsilon, x, V)
    return hx


@numba.njit
def compute_hx_nw(K_gl, cmat, coupling, N, V, T):
    """Jacobians for network connectivity in all time steps.

    :param K_gl:    FHN model parameter.
    :type K_gl:     float

    :param cmat:    FHN model parameter, connectivity matrix.
    :type cmat:     ndarray

    :param coupling: FHN model parameter.
    :type coupling:  string

    :param N:           number of nodes in the network
    :type N:            int
    :param V:           number of system variables
    :type V:            int
    :param T:           length of simulation (time dimension)
    :type T:            int

    :return: Jacobians for network connectivity in all time steps.
    :rtype: np.ndarray of shape NxNxTx4x4
    """
    hx_nw = np.zeros((N, N, T, V, V))

    for n1 in range(N):
        for n2 in range(N):
            hx_nw[n1, n2, :, 0, 0] = K_gl * cmat[n1, n2]
            if coupling == "diffusive":
                hx_nw[n1, n1, :, 0, 0] += -K_gl * cmat[n1, n2]

    return -hx_nw


@numba.njit
def compute_gradient(N, dim_out, T, fk, adjoint_state, control_matrix, duh):
    """

    :param N:       number of nodes in the network
    :type N:        int
    :param dim_out: number of 'output variables' of the model
    :type dim_out:  int
    :param T:       length of simulation (time dimension)
    :type T:        int
    :param fk:      Derivative of the cost functionals wrt. to the control signal.
    :type fk:   np.ndarray of shape N x V x T

    :type adjoint_state: np.ndarray of shape N x V x T

    :type control_matrix: np.ndarray of shape N x V
    :param duh: Jacobian of systems dynamics wrt. to I_ext (external control input)
    :type duh:  np.ndarray of shape V x V

    """
    grad = np.zeros(fk.shape)
    for n in range(N):
        for v in range(dim_out):
            for t in range(T):
                grad[n, v, t] = fk[n, v, t] + adjoint_state[n, v, t] * control_matrix[n, v] * duh[v, v]
    return grad


class OcFhn(OC):
    def __init__(
        self,
        model,
        target,
        w_p=1,
        w_2=1,
        maximum_control_strength=None,
        print_array=[],
        precision_cost_interval=(0, None),
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

        if self.N == 1:  # single-node model
            if self.model.params["x_ext"].ndim == 1:
                print("not implemented yet")
            else:
                self.control = np.concatenate((self.model.params["x_ext"], self.model.params["y_ext"]), axis=0)[
                    np.newaxis, :, :
                ]
        else:
            self.control = np.stack((self.model.params["x_ext"], self.model.params["y_ext"]), axis=1)

        for n in range(self.N):
            assert (self.control[n, 0, :] == self.model.params["x_ext"][n, :]).all()
            assert (self.control[n, 1, :] == self.model.params["y_ext"][n, :]).all()

        # save control signals throughout optimization iterations for later analysis
        # self.control_history.append(self.control)

    def get_xs(self):
        """Stack the initial condition with the simulation results for both populations."""
        return np.concatenate(
            (
                np.concatenate((self.model.params["xs_init"], self.model.params["ys_init"]), axis=1)[:, :, np.newaxis],
                np.stack((self.model.x, self.model.y), axis=1),
            ),
            axis=2,
        )

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
