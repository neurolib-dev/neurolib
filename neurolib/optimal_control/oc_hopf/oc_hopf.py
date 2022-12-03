from neurolib.optimal_control.oc import OC, update_control_with_limit
from neurolib.optimal_control import cost_functions
import numba
import numpy as np


@numba.njit
def jacobian_hopf(a, w, V, x, y):
    """Jacobian of systems dynamics for Hopf model.
    :param a:   Bifrucation parameter
    :type a :   float
    :param w:   Oscillation frequency parameter.
    :type w:    float
    :param V:   Number of state variables.
    :type V:    int
    :param x:   Activity of x-population at this time instance.
    :type x:    float
    :param y:   Activity of y-population at this time instance.
    :type y:    float
    """
    jacobian = np.zeros((V, V))

    jacobian[0, :2] = [-a + 3 * x**2 + y**2, 2 * x * y + w]
    jacobian[1, :2] = [2 * x * y - w, -a + x**2 + 3 * y**2]

    return jacobian


@numba.njit
def compute_hx(a, w, N, V, T, xs):
    """Jacobians for each time step.
    :param a:   Bifrucation parameter of the Hopf model.
    :type a :   float
    :param w:   Oscillation frequency parameter of the Hopf model.
    :type w:    float
    :param N:   Number of network nodes.
    :type N:    int
    :param V:   Number of state variables.
    :type V:    int
    :param T:   Number of time points.
    :type T:    int
    :param xs:  Time series of the activities (x and y population) in all nodes. x in Nx0xT and y in Nx1xT dimensions.
    :type xs:   np.ndarray of shape Nx2xT
    :return:    array of length T containing 2x2-matrices
    :rtype:     np.ndarray of shape Tx2x2
    """
    hx = np.zeros((N, T, V, V))

    for n in range(N):
        for t in range(T):
            x = xs[n, 0, t]
            y = xs[n, 1, t]
            hx[n, t, :, :] = jacobian_hopf(a, w, V, x, y)
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


class OcHopf(OC):
    """
    :param model:
    :type model: neurolib.models.hopf.model.HopfModel
    """

    # Remark: very similar to FHN!
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
            self.model.params.a,
            self.model.params.w,
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
