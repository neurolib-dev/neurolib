from neurolib.control.optimal_control.oc import OC
from neurolib.control.optimal_control import cost_functions
import numpy as np
import numba
from neurolib.models.wc.timeIntegration import compute_hx, compute_nw_input, compute_hx_nw, Duh


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
                grad[n, v, t] = fk[n, v, t] + adjoint_state[n, v, t] * control_matrix[n, v] * duh[n, v, v, t]

    return grad


class OcWc(OC):
    """
    :param model:
    :type model: neurolib.models.wc.model.WCModel
    """

    def __init__(
        self,
        model,
        target,
        w_p=1,
        w_2=1,
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
            print_array=print_array,
            precision_cost_interval=precision_cost_interval,
            precision_matrix=precision_matrix,
            control_matrix=control_matrix,
            M=M,
            M_validation=M_validation,
            validate_per_step=validate_per_step,
        )

        assert self.model.name == "wc"

        assert self.T == self.model.params["exc_ext"].shape[1]
        assert self.T == self.model.params["inh_ext"].shape[1]

        # ToDo: here, a method like neurolib.model_utils.adjustArrayShape() should be applied!
        if self.N == 1:  # single-node model
            if self.model.params["exc_ext"].ndim == 1:
                print("not implemented yet")
            else:
                self.background = np.concatenate((self.model.params["exc_ext"], self.model.params["inh_ext"]), axis=0)[
                    np.newaxis, :, :
                ]
        else:
            self.background = np.stack((self.model.params["exc_ext"], self.model.params["inh_ext"]), axis=1)

        for n in range(self.N):
            assert (self.background[n, 0, :] == self.model.params["exc_ext"][n, :]).all()
            assert (self.background[n, 1, :] == self.model.params["inh_ext"][n, :]).all()

        self.control = np.zeros((self.background.shape))

    def get_xs_delay(self):
        """Concatenates the initial conditions with simulated values and pads delay contributions at end. In the models
        timeIntegration, these values can be accessed in a circular fashion in the time-indexing.
        """

        if self.model.params["exc_init"].shape[1] == 1:  # no delay
            xs_begin = np.concatenate((self.model.params["exc_init"], self.model.params["inh_init"]), axis=1)[
                :, :, np.newaxis
            ]
            xs = np.concatenate(
                (
                    xs_begin,
                    np.stack((self.model.exc, self.model.inh), axis=1),
                ),
                axis=2,
            )
        else:
            xs_begin = np.stack((self.model.params["exc_init"][:, -1], self.model.params["inh_init"][:, -1]), axis=1)[
                :, :, np.newaxis
            ]
            xs_end = np.stack((self.model.params["exc_init"][:, :-1], self.model.params["inh_init"][:, :-1]), axis=1)
            xs = np.concatenate(
                (
                    xs_begin,
                    np.stack((self.model.exc, self.model.inh), axis=1),
                ),
                axis=2,
            )
            xs = np.concatenate(  # initial conditions for delay-steps are concatenated to the end of the array
                (xs, xs_end),
                axis=2,
            )

        return xs

    def get_xs(self):
        """Stack the initial condition with the simulation results for both populations."""
        if self.model.params["exc_init"].shape[1] == 1:
            xs_begin = np.concatenate((self.model.params["exc_init"], self.model.params["inh_init"]), axis=1)[
                :, :, np.newaxis
            ]
            xs = np.concatenate(
                (
                    xs_begin,
                    np.stack((self.model.exc, self.model.inh), axis=1),
                ),
                axis=2,
            )
        else:
            xs_begin = np.stack((self.model.params["exc_init"][:, -1], self.model.params["inh_init"][:, -1]), axis=1)[
                :, :, np.newaxis
            ]
            xs = np.concatenate(
                (
                    xs_begin,
                    np.stack((self.model.exc, self.model.inh), axis=1),
                ),
                axis=2,
            )

        return xs

    def update_input(self):
        """Update the parameters in self.model according to the current control such that self.simulate_forward
        operates with the appropriate control signal.
        """
        input = self.background + self.control
        # ToDo: find elegant way to combine the cases
        if self.N == 1:
            self.model.params["exc_ext"] = input[:, 0, :].reshape(1, -1)  # Reshape as row vector to match access
            self.model.params["inh_ext"] = input[:, 1, :].reshape(1, -1)  # in model's time integration.

        else:
            self.model.params["exc_ext"] = input[:, 0, :]
            self.model.params["inh_ext"] = input[:, 1, :]

    def Dxdot(self):
        """4x4 Jacobian of systems dynamics wrt. to change of systems variables."""
        raise NotImplementedError  # return np.eye(4)

    def Duh(self):
        """Nx4x4xT Jacobian of systems dynamics wrt. to external control input"""

        xs = self.get_xs()
        e = xs[:, 0, :]
        i = xs[:, 1, :]
        xsd = self.get_xs_delay()
        ed = xsd[:, 0, :]
        nw_e = compute_nw_input(self.N, self.T, self.model.params.K_gl, self.model.Cmat, self.Dmat_ndt, ed)

        input = self.background + self.control
        ue = input[:, 0, :]
        ui = input[:, 1, :]

        return Duh(
            self.N,
            self.dim_out,
            self.T,
            self.model.params.c_excexc,
            self.model.params.c_inhexc,
            self.model.params.c_excinh,
            self.model.params.c_inhinh,
            self.model.params.a_exc,
            self.model.params.a_inh,
            self.model.params.mu_exc,
            self.model.params.mu_inh,
            self.model.params.tau_exc,
            self.model.params.tau_inh,
            nw_e,
            ue,
            ui,
            e,
            i,
        )

    def compute_hx(self):
        """Jacobians for each time step.

        :return: Array of length self.T containing 4x4-matrices
        :rtype: np.ndarray
        """
        return compute_hx(
            self.model.params.tau_exc,
            self.model.params.tau_inh,
            self.model.params.a_exc,
            self.model.params.a_inh,
            self.model.params.mu_exc,
            self.model.params.mu_inh,
            self.model.params.c_excexc,
            self.model.params.c_inhexc,
            self.model.params.c_excinh,
            self.model.params.c_inhinh,
            self.model.params.K_gl,
            self.model.Cmat,
            self.Dmat_ndt,
            self.N,
            self.dim_vars,
            self.T,
            self.get_xs(),
            self.get_xs_delay(),
            self.background + self.control,
        )

    def compute_hx_nw(self):
        """Jacobians for each time step for the network coupling.

        :return: N x N x T x (4x4) array
        :rtype: np.ndarray
        """

        xs = self.get_xs()
        e = xs[:, 0, :]
        i = xs[:, 1, :]
        xsd = self.get_xs_delay()
        e_delay = xsd[:, 0, :]
        ue = self.background[:, 0, :] + self.control[:, 0, :]

        return compute_hx_nw(
            self.model.params.K_gl,
            self.model.Cmat,
            self.Dmat_ndt,
            self.N,
            self.dim_vars,
            self.T,
            e,
            i,
            e_delay,
            ue,
            self.model.params.tau_exc,
            self.model.params.a_exc,
            self.model.params.mu_exc,
            self.model.params.c_excexc,
            self.model.params.c_inhexc,
        )

    def compute_gradient(self):
        """
        Du @ fk + adjoint_k.T @ Du @ h
        """
        self.solve_adjoint()
        fk = cost_functions.derivative_energy_cost(self.control, self.w_2)
        duh = self.Duh()

        return compute_gradient(self.N, self.dim_out, self.T, fk, self.adjoint_state, self.control_matrix, duh)
