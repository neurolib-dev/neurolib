from neurolib.optimal_control.oc import OC
from neurolib.optimal_control import cost_functions
import numpy as np
import numba


@numba.njit
def S(x, a, mu):
    """Logistic function.

    :type x:    float
    :param a:   Slope parameter.
    :typa a:    float
    :param mu:  Inflection point.
    :type mu:   float
    :rtype:     float
    """
    return 1.0 / (1.0 + np.exp(-a * (x - mu)))


@numba.njit
def S_der(x, a, mu):
    """Derivative of logistic function

    :type x:    float
    :param a:   Slope parameter.
    :typa a:    float
    :param mu:  Inflection point.
    :type mu:   float
    :rtype:     float
    """
    return (a * np.exp(-a * (x - mu))) / (1.0 + np.exp(-a * (x - mu))) ** 2


@numba.njit
def Duh(
    N,
    V,
    T,
    c_excexc,
    c_inhexc,
    c_excinh,
    c_inhinh,
    a_exc,
    a_inh,
    mu_exc,
    mu_inh,
    tau_exc,
    tau_inh,
    nw_e,
    ue,
    ui,
    e,
    i,
):
    """Jacobian of systems dynamics wrt. to external control input

    :rtype:     np.ndarray of shape N x V x V x T
    """
    duh = np.zeros((N, V, V, T))
    for t in range(T):
        for n in range(N):
            input_exc = c_excexc * e[n, t] - c_inhexc * i[n, t] + nw_e[n, t] + ue[n, t]
            duh[n, 0, 0, t] = -(1.0 - e[n, t]) * S_der(input_exc, a_exc, mu_exc) / tau_exc
            input_inh = c_excinh * e[n, t] - c_inhinh * i[n, t] + ui[n, t]
            duh[n, 1, 1, t] = -(1.0 - i[n, t]) * S_der(input_inh, a_inh, mu_inh) / tau_inh
    return duh


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


@numba.njit
def jacobian_wc(
    tau_exc, tau_inh, a_exc, a_inh, mu_exc, mu_inh, c_excexc, c_inhexc, c_excinh, c_inhinh, nw_e, e, i, ue, ui, V
):
    """Jacobian of the WC dynamical system.
    :param tau_exc, tau_inh, a_exc, a_inh, mu_exc, mu_inh:   WC model parameter.
    :type tau_exc, tau_inh, a_exc, a_inh, mu_exc, mu_inh:    float


    :param e, i:       Value of the E-/ I-variable at specific time
    :type e, i:        float

    :param V:           number of system variables
    :type V:            int

    :return:        Jacobian matrix.
    :rtype:         np.ndarray of dimensions 2x2
    """
    jacobian = np.zeros((V, V))
    input_exc = c_excexc * e - c_inhexc * i + nw_e + ue
    jacobian[0, 0] = (
        -(-1.0 - S(input_exc, a_exc, mu_exc) + (1.0 - e) * c_excexc * S_der(input_exc, a_exc, mu_exc)) / tau_exc
    )
    jacobian[0, 1] = -((1.0 - e) * (-c_inhexc) * S_der(input_exc, a_exc, mu_exc)) / tau_exc
    input_inh = c_excinh * e - c_inhinh * i + ui
    jacobian[1, 0] = -((1.0 - e) * c_excinh * S_der(input_inh, a_inh, mu_inh)) / tau_inh
    jacobian[1, 1] = (
        -(-1.0 - S(input_inh, a_inh, mu_inh) + (1.0 - i) * (-c_inhinh) * S_der(input_inh, a_inh, mu_inh)) / tau_inh
    )
    return jacobian


@numba.njit
def compute_hx(
    tau_exc,
    tau_inh,
    a_exc,
    a_inh,
    mu_exc,
    mu_inh,
    c_excexc,
    c_inhexc,
    c_excinh,
    c_inhinh,
    K_gl,
    cmat,
    dmat_ndt,
    N,
    V,
    T,
    xs,
    xs_delay,
    control,
):
    """Jacobians for each time step.

    :param tau_exc,
            tau_inh,
            a_exc,
            a_inh,
            mu_exc,
            mu_inh,
            c_excexc,
            c_inhexc,
            c_excinh,
            c_inhinh,
            K_gl,
            cmat:   Wilson-Cowan model parameters
    :type :    float

    :param dmat_ndt:   Wilson-Cowan model parameters, delay matrix in multiples of dt.
    :type dmat_ndt:    np.ndarray of shape NxN

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
    nw_e = compute_nw_input(N, T, K_gl, cmat, dmat_ndt, xs_delay[:, 0, :])

    for n in range(N):
        for t, e in enumerate(xs[n, 0, :]):
            i = xs[n, 1, t]
            ue = control[n, 0, t]
            ui = control[n, 1, t]
            hx[n, t, :, :] = jacobian_wc(
                tau_exc,
                tau_inh,
                a_exc,
                a_inh,
                mu_exc,
                mu_inh,
                c_excexc,
                c_inhexc,
                c_excinh,
                c_inhinh,
                nw_e[n, t],
                e,
                i,
                ue,
                ui,
                V,
            )
    return hx


@numba.njit
def compute_nw_input(N, T, K_gl, cmat, dmat_ndt, E):
    """Compute input by other nodes of network into each node at every timestep.

    :rytpe: np.ndarray of shape N x T
    """
    nw_input = np.zeros((N, T))

    for t in range(1, T):
        for n in range(N):
            for l in range(N):
                nw_input[n, t] += K_gl * cmat[n, l] * (E[l, t - dmat_ndt[n, l] - 1])
                # nw_input[n, t] += K_gl * cmat[n, l] * (E[l, t - 1])
    return nw_input


@numba.njit
def compute_hx_nw(
    K_gl,
    cmat,
    dmat_ndt,
    N,
    V,
    T,
    e,
    i,
    e_delay,
    ue,
    tau_exc,
    a_exc,
    mu_exc,
    c_excexc,
    c_inhexc,
):
    """Jacobians for network connectivity in all time steps.

    :param K_gl:    Global coupling.
    :type K_gl:     float

    :param cmat:    model parameter, connectivity matrix.
    :type cmat:     ndarray

    :param coupling: model parameter.
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

    nw_e = compute_nw_input(N, T, K_gl, cmat, dmat_ndt, e_delay)
    exc_input = c_excexc * e - c_inhexc * i + nw_e + ue

    for n1 in range(N):
        for n2 in range(N):
            for t in range(T - 1):
                hx_nw[n1, n2, t, 0, 0] = (S_der(exc_input[n1, t], a_exc, mu_exc) * K_gl * cmat[n1, n2]) / tau_exc

    return -hx_nw


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
        """Stack the initial condition (last index) with the simulation results, with the initial contidion (all except last index) for both populations."""

        if self.model.params["exc_init"].shape[1] == 1:
            p1 = np.concatenate((self.model.params["exc_init"], self.model.params["inh_init"]), axis=1)[
                :, :, np.newaxis
            ]
            xs = np.concatenate(
                (
                    p1,
                    np.stack((self.model.exc, self.model.inh), axis=1),
                ),
                axis=2,
            )
        else:
            p1 = np.stack((self.model.params["exc_init"][:, -1], self.model.params["inh_init"][:, -1]), axis=1)[
                :, :, np.newaxis
            ]
            p3 = np.stack((self.model.params["exc_init"][:, :-1], self.model.params["inh_init"][:, :-1]), axis=1)
            xs0 = np.concatenate(
                (
                    p1,
                    np.stack((self.model.exc, self.model.inh), axis=1),
                ),
                axis=2,
            )
            xs = np.concatenate(
                (xs0, p3),
                axis=2,
            )

        return xs

    def get_xs(self):
        """Stack the initial condition with the simulation results for both populations."""
        if self.model.params["exc_init"].shape[1] == 1:
            p1 = np.concatenate((self.model.params["exc_init"], self.model.params["inh_init"]), axis=1)[
                :, :, np.newaxis
            ]
            xs = np.concatenate(
                (
                    p1,
                    np.stack((self.model.exc, self.model.inh), axis=1),
                ),
                axis=2,
            )
        else:
            p1 = np.stack((self.model.params["exc_init"][:, -1], self.model.params["inh_init"][:, -1]), axis=1)[
                :, :, np.newaxis
            ]
            xs = np.concatenate(
                (
                    p1,
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

        # print("duh = ", duh[0, 0, 0])

        return compute_gradient(self.N, self.dim_out, self.T, fk, self.adjoint_state, self.control_matrix, duh)
