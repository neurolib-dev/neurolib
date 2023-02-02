from neurolib.optimal_control.oc import OC
from neurolib.optimal_control import cost_functions
import numpy as np
import numba


@numba.njit
def logistic(x, a, mu):
    """Logistic function evaluated at point 'x'.

    :type x:    float
    :param a:   Slope parameter.
    :type a:    float
    :param mu:  Inflection point.
    :type mu:   float
    :rtype:     float
    """
    return 1.0 / (1.0 + np.exp(-a * (x - mu)))


@numba.njit
def logistic_der(x, a, mu):
    """Derivative of logistic function, evaluated at point 'x'.

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
    """Jacobian of systems dynamics wrt. to external inputs (control signals).

    :rtype:     np.ndarray of shape N x V x V x T
    """
    duh = np.zeros((N, V, V, T))
    for t in range(T):
        for n in range(N):
            input_exc = c_excexc * e[n, t] - c_inhexc * i[n, t] + nw_e[n, t] + ue[n, t]
            duh[n, 0, 0, t] = -(1.0 - e[n, t]) * logistic_der(input_exc, a_exc, mu_exc) / tau_exc
            input_inh = c_excinh * e[n, t] - c_inhinh * i[n, t] + ui[n, t]
            duh[n, 1, 1, t] = -(1.0 - i[n, t]) * logistic_der(input_inh, a_inh, mu_inh) / tau_inh
    return duh


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
    :param df_du:      Derivative of the cost wrt. to the explicit control contributions to cost functionals.
    :type df_du:       np.ndarray of shape N x V x T
    :param adjoint_state:   Solution of the adjoint equation.
    :type adjoint_state:    np.ndarray of shape N x V x T
    :param control_matrix:  Binary matrix that defines nodes and variables where control inputs are active, defaults to
                            None.
    :type control_matrix:   np.ndarray of shape N x V
    :param d_du:    Jacobian of systems dynamics wrt. to I_ext (external control input)
    :type d_du:     np.ndarray of shape V x V
    :return:        The gradient of the total cost wrt. to the control.
    :rtype:         np.ndarray of shape N x V x T
    """
    grad = np.zeros(df_du.shape)

    for n in range(N):
        for v in range(dim_out):
            for t in range(T):
                grad[n, v, t] = df_du[n, v, t] + adjoint_state[n, v, t] * control_matrix[n, v] * d_du[n, v, v, t]

    return grad


@numba.njit
def jacobian_wc(wc_model_params, nw_e, e, i, ue, ui, V):
    """Jacobian of the WC dynamical system.

    :param wc_model_params: Tuple of parameters in the WCModel in order (tau_exc, tau_inh, a_exc, a_inh, mu_exc, mu_inh,
                            c_excexc, c_inhexc, c_excinh, c_inhinh). All parameters of type 'float'.
    :type wc_model_params: tuple
    :param  nw_e:   N x T input of network into each node's 'exc'
    :type  nw_e:    np.ndarray
    :param e:       Value of the E-variable at specific time.
    :type e:        float
    :param i:       Value of the I-variable at specific time.
    :type i:        float
    :param ue:      N x T combined input of 'background' and 'control' into 'exc'.
    :type ue:       np.ndarray
    :param ui:      N x T combined input of 'background' and 'control' into 'inh'.
    :type ui:       np.ndarray
    :param V:       Number of system variables.
    :type V:        int
    :return:        4 x 4 Jacobian matrix.
    :rtype:         np.ndarray
    """

    (tau_exc, tau_inh, a_exc, a_inh, mu_exc, mu_inh, c_excexc, c_inhexc, c_excinh, c_inhinh) = wc_model_params

    jacobian = np.zeros((V, V))
    input_exc = c_excexc * e - c_inhexc * i + nw_e + ue
    jacobian[0, 0] = (
        -(-1.0 - logistic(input_exc, a_exc, mu_exc) + (1.0 - e) * c_excexc * logistic_der(input_exc, a_exc, mu_exc))
        / tau_exc
    )
    jacobian[0, 1] = -((1.0 - e) * (-c_inhexc) * logistic_der(input_exc, a_exc, mu_exc)) / tau_exc
    input_inh = c_excinh * e - c_inhinh * i + ui
    jacobian[1, 0] = -((1.0 - i) * c_excinh * logistic_der(input_inh, a_inh, mu_inh)) / tau_inh
    jacobian[1, 1] = (
        -(-1.0 - logistic(input_inh, a_inh, mu_inh) + (1.0 - i) * (-c_inhinh) * logistic_der(input_inh, a_inh, mu_inh))
        / tau_inh
    )
    return jacobian


@numba.njit
def compute_hx(
    wc_model_params: tuple[float, float, float, float, float, float, float, float, float, float],
    K_gl,
    cmat,
    dmat_ndt,
    N,
    V,
    T,
    dyn_vars,
    control,
):
    """Jacobians of WCModel wrt. to the 'e'- and 'i'-variable for each time step.

    :param wc_model_params: Tuple of parameters in the WCModel in order (tau_exc, tau_inh, a_exc, a_inh, mu_exc, mu_inh,
                            c_excexc, c_inhexc, c_excinh, c_inhinh). All parameters of type 'float'.
    :type wc_model_params: tuple
    :param K_gl:        Model parameter of global coupling strength.
    :type K_gl:         float
    :param cmat:        Model parameter, connectivity matrix.
    :type cmat:         ndarray
    :param dmat_ndt:    N x N delay matrix in multiples of dt.
    :type dmat_ndt:     np.ndarray
    :param N:           Number of nodes in the network.
    :type N:            int
    :param V:           Number of system variables.
    :type V:            int
    :param T:           Length of simulation (time dimension).
    :type T:            int
    :param dyn_vars:    N x V x T array containing all values of 'exc' and 'inh'.
    :type dyn_vars:     np.ndarray
    :param control:     N x 2 x T control inputs to 'exc' and 'inh'.
    :type control:      np.ndarray
    :return:            N x T x 4 x 4 Jacobians.
    :rtype:             np.ndarray
    """
    hx = np.zeros((N, T, V, V))
    nw_e = compute_nw_input(N, T, K_gl, cmat, dmat_ndt, dyn_vars[:, 0, :])

    for n in range(N):
        for t, e in enumerate(dyn_vars[n, 0, :]):
            i = dyn_vars[n, 1, t]
            ue = control[n, 0, t]
            ui = control[n, 1, t]
            hx[n, t, :, :] = jacobian_wc(
                wc_model_params,
                nw_e[n, t],
                e,
                i,
                ue,
                ui,
                V,
            )
    return hx


@numba.njit
def compute_nw_input(N, T, K_gl, cmat, dmat_ndt, exc_values):
    """Compute input by other nodes of network into each node's 'exc' population at every timestep.

    :param N:           Number of nodes in the network.
    :type N:            int
    :param T:           Length of simulation (time dimension).
    :type T:            int
    :param K_gl:        Model parameter of global coupling strength.
    :type K_gl:         float
    :param cmat:        Model parameter, connectivity matrix.
    :type cmat:         ndarray
    :param dmat_ndt:    N x N delay matrix in multiples of dt.
    :type dmat_ndt:     np.ndarray
    :param exc_values:  N x T array containing values of 'exc' of all nodes through time.
    :type exc_values:   np.ndarray
    :return:            N x T network inputs.
    :rytpe:             np.ndarray
    """
    nw_input = np.zeros((N, T))

    for t in range(1, T):
        for n in range(N):
            for l in range(N):
                nw_input[n, t] += K_gl * cmat[n, l] * (exc_values[l, t - dmat_ndt[n, l] - 1])
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
    ue,
    tau_exc,
    a_exc,
    mu_exc,
    c_excexc,
    c_inhexc,
):
    """Jacobians for network connectivity in all time steps.

    :param K_gl:        Model parameter of global coupling strength.
    :type K_gl:         float
    :param cmat:        Model parameter, connectivity matrix.
    :type cmat:         ndarray
    :param dmat_ndt:    N x N delay matrix in multiples of dt.
    :type dmat_ndt:     np.ndarray
    :param N:           Number of nodes in the network.
    :type N:            int
    :param V:           Number of system variables.
    :type V:            int
    :param T:           Length of simulation (time dimension).
    :type T:            int
    :param e:       Value of the E-variable at specific time.
    :type e:        float
    :param i:       Value of the I-variable at specific time.
    :type i:        float
    :param ue:      N x T array of the total input received by 'exc' population in every node at any time.
    :type ue:       np.ndarray
    :param tau_exc: Excitatory time constant.
    :type tau_exc:  float
    :param a_exc:   Excitatory gain.
    :type a_exc:    float
    :param mu_exc:  Excitatory firing threshold.
    :type mu_exc:   float
    :param c_excexc: Local E-E coupling.
    :type c_excexc:  float
    :param c_inhexc: Local I-E coupling.
    :type c_inhexc:  float
    :return:         Jacobians for network connectivity in all time steps.
    :rtype:          np.ndarray of shape N x N x T x 4 x 4
    """
    hx_nw = np.zeros((N, N, T, V, V))

    nw_e = compute_nw_input(N, T, K_gl, cmat, dmat_ndt, e)
    exc_input = c_excexc * e - c_inhexc * i + nw_e + ue

    for t in range(T):
        for n1 in range(N):
            for n2 in range(N):
                hx_nw[n1, n2, t, 0, 0] = (logistic_der(exc_input[n1, t], a_exc, mu_exc) * K_gl * cmat[n1, n2]) / tau_exc

    return -hx_nw


class OcWc(OC):
    """Class for optimal control specific to neurolib's implementation of the two-population Wilson-Cowan model
        ("WCmodel").

    :param model: Instance of Wilson-Cowan model (can describe a single Wilson-Cowan node or a network of coupled
                  Wilson-Cowan nodes. Remark: Currently only delay-free networks are supported.
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

        self.control = np.zeros((self.background.shape))  # control is of shape N x 2 x T, controls of 'exc' and 'inh'

    def get_xs(self):
        """Stack the initial condition with the simulation results for both ('exc' and 'inh') populations.

        :return: N x V x T array containing all values of 'exc' and 'inh'.
        :rtype:  np.ndarray
        """
        return np.concatenate(
            (
                np.concatenate((self.model.params["exc_init"], self.model.params["inh_init"]), axis=1)[
                    :, :, np.newaxis
                ],
                np.stack((self.model.exc, self.model.inh), axis=1),
            ),
            axis=2,
        )

    def update_input(self):
        """Update the parameters in 'self.model' according to the current control such that 'self.simulate_forward'
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
        """4 x 4 Jacobian of systems dynamics wrt. to change of systems variables."""
        # Currently not explicitly required since it is identity matrix.
        raise NotImplementedError  # return np.eye(4)

    def Duh(self):
        """Jacobian of systems dynamics wrt. to external control input.

        :return:    N x 4 x 4 x T Jacobians.
        :rtype:     np.ndarray
        """

        xs = self.get_xs()
        e = xs[:, 0, :]
        i = xs[:, 1, :]
        nw_e = compute_nw_input(self.N, self.T, self.model.params.K_gl, self.model.Cmat, self.Dmat_ndt, e)

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
        """Jacobians of WCModel wrt. to the 'e'- and 'i'-variable for each time step.

        :return:    N x T x 4 x 4 Jacobians.
        :rtype:     np.ndarray
        """
        return compute_hx(
            (
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
            ),
            self.model.params.K_gl,
            self.model.Cmat,
            self.Dmat_ndt,
            self.N,
            self.dim_vars,
            self.T,
            self.get_xs(),
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
            ue,
            self.model.params.tau_exc,
            self.model.params.a_exc,
            self.model.params.mu_exc,
            self.model.params.c_excexc,
            self.model.params.c_inhexc,
        )

    def compute_gradient(self):
        """Compute the gradient of the total cost wrt. to the control signals. This is achieved by first, solving the
           adjoint equation backwards in time. Second, derivatives of the cost wrt. to explicit control variables are
           evaluated as well as the Jacobians of the dynamics wrt. to explicit control. Then the decent direction /
           gradient of the cost wrt. to control (in its explicit form AND IMPLICIT FORM) is computed.

        :return:        The gradient of the total cost wrt. to the control.
        :rtype:         np.ndarray of shape N x V x T
        """
        self.solve_adjoint()

        df_du = cost_functions.derivative_energy_cost(self.control, self.w_2)
        duh = self.Duh()

        return compute_gradient(self.N, self.dim_out, self.T, df_du, self.adjoint_state, self.control_matrix, duh)
