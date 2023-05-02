from neurolib.control.optimal_control.oc import OC, update_control_with_limit
from neurolib.control.optimal_control import cost_functions
import numpy as np
import numba
from neurolib.models.aln.timeIntegration import (
    compute_hx,
    compute_nw_input,
    compute_hx_nw,
    Duh,
    Dxdoth,
    compute_hx_de,
    compute_hx_di,
)


@numba.njit
def compute_gradient(N, dim_in, T, df_du, adjoint_state, control_matrix, d_du):
    """Compute the gradient of the total cost wrt. the control signals (explicitly and implicitly) given the adjoint
       state, the Jacobian of the total cost wrt. explicit control contributions and the Jacobian of the dynamics
       wrt. explicit control contributions.

    :param N:       Number of nodes in the network.
    :type N:        int
    :param dim_in: Number of 'input variables' of the model.
    :type dim_in:  int
    :param T:       Length of simulation (time dimension).
    :type T:        int
    :param df_du:      Derivative of the cost wrt. the explicit control contributions to cost functionals.
    :type df_du:       np.ndarray of shape N x V x T
    :param adjoint_state:   Solution of the adjoint equation.
    :type adjoint_state:    np.ndarray of shape N x V x T
    :param control_matrix:  Binary matrix that defines nodes and variables where control inputs are active, defaults to
                            None.
    :type control_matrix:   np.ndarray of shape N x V
    :param d_du:    Jacobian of systems dynamics wrt. I_ext (external control input)
    :type d_du:     np.ndarray of shape V x V
    :return:        The gradient of the total cost wrt. the control.
    :rtype:         np.ndarray of shape N x V x T
    """
    grad = np.zeros(df_du.shape)

    # print("lambda mue =", adjoint_state[0, 2, :])
    # print("duh = ", d_du[0, :4, :4, 0])

    for n in range(N):
        for v in range(dim_in):
            for t in range(T):
                grad[n, v, t] = df_du[n, v, t]
                for k in range(adjoint_state.shape[1]):
                    grad[n, v, t] += control_matrix[n, v] * adjoint_state[n, k, t] * d_du[n, k, v, t]

    return grad


class OcAln(OC):
    """Class for optimal control specific to neurolib's implementation of the two-population ALN model
            ("ALNmodel").

    :param model: Instance of ALN model (can describe a single ALN node or a network of coupled
                  ALN nodes.
    :type model: neurolib.models.aln.model.ALNModel
    """

    def __init__(
        self,
        model,
        target,
        weights=None,
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
            print_array=print_array,
            cost_interval=cost_interval,
            cost_matrix=cost_matrix,
            control_matrix=control_matrix,
            M=M,
            M_validation=M_validation,
            validate_per_step=validate_per_step,
        )

        assert self.model.name == "aln"

        assert self.T == self.model.params["ext_exc_current"].shape[1]
        assert self.T == self.model.params["ext_inh_current"].shape[1]

        # ToDo: here, a method like neurolib.model_utils.adjustArrayShape() should be applied!
        if self.N == 1:  # single-node model
            if self.model.params["ext_exc_current"].ndim == 1:
                print("not implemented yet")
            else:
                control = np.concatenate(
                    (self.model.params["ext_exc_current"], self.model.params["ext_inh_current"]), axis=0
                )[np.newaxis, :, :]
        else:
            control = np.stack((self.model.params["ext_exc_current"], self.model.params["ext_inh_current"]), axis=1)

        for n in range(self.N):
            assert (control[n, 0, :] == self.model.params["ext_exc_current"][n, :]).all()
            assert (control[n, 1, :] == self.model.params["ext_inh_current"][n, :]).all()

            # in aln model, t=0 control does not affect the system
            control[n, 0, 0] = 0.0
            control[n, 1, 0] = 0.0

        self.control = update_control_with_limit(control, 0.0, np.zeros(control.shape), self.maximum_control_strength)
        self.fullstate = self.get_fullstate()

        if self.model.params.filter_sigma:
            print("NOT IMPLEMENTED FOR FILTER_SIGMA=TRUE")

        self.ndt_de = np.around(self.model.params.de / self.dt).astype(int)
        self.ndt_di = np.around(self.model.params.di / self.dt).astype(int)

    def get_xs_delay(self):
        """Concatenates the initial conditions with simulated values and pads delay contributions at end. In the models
        timeIntegration, these values can be accessed in a circular fashion in the time-indexing.
        """

        if self.model.params["rates_exc_init"].shape[1] == 1:  # no delay
            xs_begin = np.concatenate(
                (
                    self.model.params["rates_exc_init"],
                    self.model.params["rates_inh_init"],
                    self.model.params["IA_init"],
                ),
                axis=1,
            )[:, :, np.newaxis]
            xs = np.concatenate(
                (
                    xs_begin,
                    np.stack((self.model.rates_exc, self.model.rates_inh, self.model.IA), axis=1),
                ),
                axis=2,
            )
        else:
            xs_begin = np.stack(
                (
                    self.model.params["rates_exc_init"][:, -1],
                    self.model.params["rates_inh_init"][:, -1],
                    self.model.params["IA_init"][:, -1],
                ),
                axis=1,
            )[:, :, np.newaxis]
            xs_end = np.stack(
                (
                    self.model.params["rates_exc_init"][:, :-1],
                    self.model.params["rates_inh_init"][:, :-1],
                    self.model.params["IA_init"][:, :-1],
                ),
                axis=1,
            )
            xs = np.concatenate(
                (
                    xs_begin,
                    np.stack((self.model.rates_exc, self.model.rates_inh, self.model.IA), axis=1),
                ),
                axis=2,
            )
            xs = np.concatenate(  # initial conditions for delay-steps are concatenated to the end of the array
                (xs, xs_end),
                axis=2,
            )

        return xs

    def get_xs(self):
        """Stack the initial condition with the simulation results for both ('exc' and 'inh') populations.

        :return: N x V x T array containing all values of 'exc' and 'inh'.
        :rtype:  np.ndarray
        """
        if self.model.params["rates_exc_init"].shape[1] == 1:
            xs_begin = np.concatenate(
                (
                    self.model.params["rates_exc_init"],
                    self.model.params["rates_inh_init"],
                    self.model.params["IA_init"],
                ),
                axis=1,
            )[:, :, np.newaxis]
            xs = np.concatenate(
                (
                    xs_begin,
                    np.stack((self.model.rates_exc, self.model.rates_inh, self.model.IA), axis=1),
                ),
                axis=2,
            )
        else:
            xs_begin = np.stack(
                (
                    self.model.params["rates_exc_init"][:, -1],
                    self.model.params["rates_inh_init"][:, -1],
                    self.model.params["IA_init"][:, -1],
                ),
                axis=1,
            )[:, :, np.newaxis]
            xs = np.concatenate(
                (
                    xs_begin,
                    np.stack((self.model.rates_exc, self.model.rates_inh, self.model.IA), axis=1),
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
            self.model.params["ext_exc_current"] = self.control[:, 0, :].reshape(
                1, -1
            )  # Reshape as row vector to match access
            self.model.params["ext_inh_current"] = self.control[:, 1, :].reshape(1, -1)  # in model's time integration.

        else:
            self.model.params["ext_exc_current"] = self.control[:, 0, :]
            self.model.params["ext_inh_current"] = self.control[:, 1, :]

    def compute_dxdoth(self):
        """Derivative of systems dynamics wrt. change of systems variables."""
        return Dxdoth(self.N, self.dim_vars)

    def Duh(self):
        """Jacobian of systems dynamics wrt. external control input.

        :return:    N x 4 x 4 x T Jacobians.
        :rtype:     np.ndarray
        """

        return Duh(
            (
                self.model.params.sigmarange,
                self.model.params.ds,
                self.model.params.Irange,
                self.model.params.dI,
                self.model.params.C,
                self.model.params.precalc_r,
                self.model.params.precalc_V,
                self.model.params.precalc_tau_mu,
                self.model.params.Ke,
                self.model.params.Ki,
                self.model.params.cee,
                self.model.params.cei,
                self.model.params.cie,
                self.model.params.cii,
                self.model.params.Jee_max,
                self.model.params.Jei_max,
                self.model.params.Jie_max,
                self.model.params.Jii_max,
                self.model.params.tau_se,
                self.model.params.tau_si,
                self.model.params.tauA,
                self.model.params.C / self.model.params.gL,
                self.model.params.sigmae_ext,
                self.model.params.sigmai_ext,
                self.model.params.a,
                self.model.params.b,
                self.model.params.EA,
            ),
            self.N,
            self.dim_in,
            self.dim_vars,
            self.T,
            self.fullstate,
        )

    def compute_hx_list(self):
        """List of Jacobians without and with time delays (e.g. in the ALN model) and list of respective time step delays as integers (0 for undelayed)

        :return:        List of Jacobian matrices, list of time step delays
        : rtype:        List of np.ndarray, List of integers

        """
        hx = self.compute_hx()
        hx_de = self.compute_hx_de()
        hx_di = self.compute_hx_di()

        return numba.typed.List([hx, hx_de, hx_di]), numba.typed.List([0, self.ndt_de, self.ndt_di])

    def compute_hx(self):
        """Jacobians of ALNModel wrt. the 'e'- and 'i'-variable for each time step.

        :return:    N x T x 4 x 4 Jacobians.
        :rtype:     np.ndarray
        """
        return compute_hx(
            (
                self.model.params.sigmarange,
                self.model.params.ds,
                self.model.params.Irange,
                self.model.params.dI,
                self.model.params.C,
                self.model.params.precalc_r,
                self.model.params.precalc_V,
                self.model.params.precalc_tau_mu,
                self.model.params.Ke,
                self.model.params.Ki,
                self.model.params.cee,
                self.model.params.cei,
                self.model.params.cie,
                self.model.params.cii,
                self.model.params.Jee_max,
                self.model.params.Jei_max,
                self.model.params.Jie_max,
                self.model.params.Jii_max,
                self.model.params.tau_se,
                self.model.params.tau_si,
                self.model.params.tauA,
                self.model.params.C / self.model.params.gL,
                self.model.params.sigmae_ext,
                self.model.params.sigmai_ext,
                self.model.params.a,
                self.model.params.b,
                self.model.params.EA,
            ),
            self.ndt_de,
            self.ndt_di,
            self.N,
            self.dim_vars,
            self.T,
            self.fullstate,
            self.control,
            self.model.params.Cmat,
            self.model.params.Dmat,
        )

    def compute_hx_de(self):
        return compute_hx_de(
            (
                self.model.params.sigmarange,
                self.model.params.ds,
                self.model.params.Irange,
                self.model.params.dI,
                self.model.params.C,
                self.model.params.precalc_r,
                self.model.params.precalc_V,
                self.model.params.precalc_tau_mu,
                self.model.params.Ke,
                self.model.params.Ki,
                self.model.params.cee,
                self.model.params.cei,
                self.model.params.cie,
                self.model.params.cii,
                self.model.params.Jee_max,
                self.model.params.Jei_max,
                self.model.params.Jie_max,
                self.model.params.Jii_max,
                self.model.params.tau_se,
                self.model.params.tau_si,
                self.model.params.tauA,
                self.model.params.C / self.model.params.gL,
                self.model.params.sigmae_ext,
                self.model.params.sigmai_ext,
                self.model.params.a,
                self.model.params.b,
                self.model.params.EA,
            ),
            self.N,
            self.dim_vars,
            self.T,
            self.fullstate,
            self.control,
        )

    def compute_hx_di(self):
        return compute_hx_di(
            (
                self.model.params.sigmarange,
                self.model.params.ds,
                self.model.params.Irange,
                self.model.params.dI,
                self.model.params.C,
                self.model.params.precalc_r,
                self.model.params.precalc_V,
                self.model.params.precalc_tau_mu,
                self.model.params.Ke,
                self.model.params.Ki,
                self.model.params.cee,
                self.model.params.cei,
                self.model.params.cie,
                self.model.params.cii,
                self.model.params.Jee_max,
                self.model.params.Jei_max,
                self.model.params.Jie_max,
                self.model.params.Jii_max,
                self.model.params.tau_se,
                self.model.params.tau_si,
                self.model.params.tauA,
                self.model.params.C / self.model.params.gL,
                self.model.params.sigmae_ext,
                self.model.params.sigmai_ext,
                self.model.params.a,
                self.model.params.b,
                self.model.params.EA,
            ),
            self.N,
            self.dim_vars,
            self.T,
            self.fullstate,
            self.control,
        )

    def compute_hx_nw(self):
        """Jacobians for each time step for the network coupling.

        :return: N x N x T x (4x4) array
        :rtype: np.ndarray
        """

        return compute_hx_nw(
            self.N,
            self.dim_vars,
            self.T,
        )

    def compute_gradient(self):
        """Compute the gradient of the total cost wrt. the control:
        1. solve the adjoint equation backwards in time
        2. compute derivatives of cost wrt. control
        3. compute Jacobians of the dynamics wrt. control
        4. compute gradient of the cost wrt. control(i.e., negative descent direction)

        :return:        The gradient of the total cost wrt. the control.
        :rtype:         np.ndarray of shape N x V x T
        """
        self.fullstate = self.get_fullstate()
        self.solve_adjoint()
        self.adjoint_state[:, :, 0] = 0.0
        df_du = cost_functions.derivative_control_strength_cost(self.control, self.weights)
        d_du = self.Duh()

        return compute_gradient(self.N, self.dim_in, self.T, df_du, self.adjoint_state, self.control_matrix, d_du)

    def get_fullstate(self):
        T, N = self.T, self.N
        self.simulate_forward()
        maxdel = self.model.getMaxDelay()
        control = self.control
        duration = self.model.params.duration

        fullstate = np.zeros((self.N, self.dim_vars, self.T + 2 * maxdel))

        initstate = self.getinitstate()
        if maxdel > 0:
            fullstate[:, :, -maxdel:] = initstate[:, :, :-1]
        fullstate[:, :, 0] = initstate[:, :, -1]

        self.model.params.duration = self.dt
        self.model.run()
        finalstate = self.getfinalstate()
        fullstate[:, :, 1] = finalstate

        for t in range(0, T - 2 + maxdel):

            if t != 0:
                self.setasinit(fullstate, t)
            self.model.params.duration = 2.0 * self.dt
            if t <= T - 2:
                self.model.params.ext_exc_current = control[:, 0, t : t + 2]
                self.model.params.ext_inh_current = control[:, 1, t : t + 2]
            elif t == T - 1:
                ec = np.concatenate((control[:, 0, t:], np.zeros((N, 1))), axis=1)
                self.model.params.ext_exc_current = np.concatenate((control[:, 0, t:], np.zeros((N, 1))), axis=1)
                self.model.params.ext_inh_current = np.concatenate((control[:, 1, t:], np.zeros((N, 1))), axis=1)
            else:
                self.model.params.ext_exc_current = 0.0
                self.model.params.ext_inh_current = 0.0
            self.model.run()

            finalstate = self.getfinalstate()
            fullstate[:, :, t + 2] = finalstate

        # reset to starting values
        self.model.params.duration = duration
        self.model.params.ext_exc_current = control[:, 0, :]
        self.model.params.ext_inh_current = control[:, 1, :]
        self.setinitstate(initstate)
        self.simulate_forward()

        return fullstate

    def setasinit(self, fullstate, t):
        N = len(self.model.params.Cmat)
        V = len(self.model.init_vars)
        T = self.model.getMaxDelay() + 1

        for n in range(N):
            for v in range(V):
                if "rates" in self.model.init_vars[v] or "IA" in self.model.init_vars[v]:
                    if t >= T:
                        self.model.params[self.model.init_vars[v]] = fullstate[:, v, t - T : t + 1]
                    else:
                        init = np.concatenate((fullstate[:, v, -T + t + 1 :], fullstate[:, v, : t + 1]), axis=1)
                        self.model.params[self.model.init_vars[v]] = init
                else:
                    self.model.params[self.model.init_vars[v]] = fullstate[:, v, t]

    def getinitstate(self):
        N = len(self.model.params.Cmat)
        V = len(self.model.init_vars)
        T = self.model.getMaxDelay() + 1
        initstate = np.zeros((N, V, T))

        for n in range(N):
            for v in range(V):
                if "rates" in self.model.init_vars[v] or "IA" in self.model.init_vars[v]:
                    initstate[n, v, :] = self.model.params[self.model.init_vars[v]][n, -T:]

                else:
                    initstate[n, v, :] = self.model.params[self.model.init_vars[v]][n]
        return initstate

    def getfinalstate(self):
        N = len(self.model.params.Cmat)
        V = len(self.model.state_vars)
        state = np.zeros((N, V))
        for n in range(N):
            for v in range(V):
                if "rates" in self.model.state_vars[v] or "IA" in self.model.state_vars[v]:
                    state[n, v] = self.model.state[self.model.state_vars[v]][n, -1]

                else:
                    state[n, v] = self.model.state[self.model.state_vars[v]][n]
        return state

    def setinitstate(self, state):
        N = len(self.model.params.Cmat)
        V = len(self.model.init_vars)
        T = self.model.getMaxDelay() + 1

        for n in range(N):
            for v in range(V):
                if "rates" in self.model.init_vars[v] or "IA" in self.model.init_vars[v]:
                    self.model.params[self.model.init_vars[v]] = state[:, v, -T:]
                else:
                    self.model.params[self.model.init_vars[v]] = state[:, v, -1]

        return
