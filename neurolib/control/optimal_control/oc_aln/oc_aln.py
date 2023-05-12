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

        self.control = update_control_with_limit(
            self.N, self.dim_in, self.T, control, 0.0, np.zeros(control.shape), self.maximum_control_strength
        )
        self.fullstate = self.get_fullstate()

        if self.model.params.filter_sigma:
            print("NOT IMPLEMENTED FOR FILTER_SIGMA=TRUE")

        self.ndt_de = np.around(self.model.params.de / self.dt).astype(int)
        self.ndt_di = np.around(self.model.params.di / self.dt).astype(int)

        self.model_params = self.get_model_params()
        self.precomp_factors = self.get_precomp_factors()

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

    def get_model_params(self):
        """Model params as an ordered tuple"""
        return (
            self.model.params.sigmarange,
            self.model.params.ds,
            self.model.params.Irange,
            self.model.params.dI,
            self.model.params.C,
            self.model.params.precalc_r,
            self.model.params.precalc_V,
            self.model.params.precalc_tau_mu,
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
            self.model.params.c_gl,
            self.model.params.Ke_gl,
        )

    def get_precomp_factors(self):
        """Precomputed factors as an ordered tuple"""
        return (
            self.model.params.cee
            * self.model.params.Ke
            * self.model.params.tau_se
            / np.abs(self.model.params.Jee_max)
            * 1e-3,
            self.model.params.cee**2
            * self.model.params.Ke
            * self.model.params.tau_se**2
            / np.abs(self.model.params.Jee_max) ** 2
            * 1e-3,
            self.model.params.cei
            * self.model.params.Ki
            * self.model.params.tau_si
            / np.abs(self.model.params.Jei_max)
            * 1e-3,
            self.model.params.cei**2
            * self.model.params.Ki
            * self.model.params.tau_si**2
            / np.abs(self.model.params.Jei_max) ** 2
            * 1e-3,
            self.model.params.cie
            * self.model.params.Ke
            * self.model.params.tau_se
            / np.abs(self.model.params.Jie_max)
            * 1e-3,
            self.model.params.cie**2
            * self.model.params.Ke
            * self.model.params.tau_se**2
            / np.abs(self.model.params.Jie_max) ** 2
            * 1e-3,
            self.model.params.cii
            * self.model.params.Ki
            * self.model.params.tau_si
            / np.abs(self.model.params.Jii_max)
            * 1e-3,
            self.model.params.cii**2
            * self.model.params.Ki
            * self.model.params.tau_si**2
            / np.abs(self.model.params.Jii_max) ** 2
            * 1e-3,
            2.0
            * self.model.params.Jee_max**2
            * self.model.params.tau_se
            * (self.model.params.C / self.model.params.gL),
            2.0
            * self.model.params.Jei_max**2
            * self.model.params.tau_si
            * (self.model.params.C / self.model.params.gL),
            2.0
            * self.model.params.Jie_max**2
            * self.model.params.tau_se
            * (self.model.params.C / self.model.params.gL),
            2.0
            * self.model.params.Jii_max**2
            * self.model.params.tau_si
            * (self.model.params.C / self.model.params.gL),
        )

    def Duh(self):
        """Jacobian of systems dynamics wrt. external control input.

        :return:    N x 4 x 4 x T Jacobians.
        :rtype:     np.ndarray
        """

        return Duh(
            self.model_params,
            self.precomp_factors,
            self.N,
            self.dim_in,
            self.dim_vars,
            self.T,
            self.fullstate,
            self.model.params.Cmat,
            self.Dmat_ndt,
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

        :return:    N x T x V x V Jacobians.
        :rtype:     np.ndarray
        """

        return compute_hx(
            self.model_params,
            self.precomp_factors,
            self.N,
            self.dim_vars,
            self.T,
            self.fullstate,
            self.control,
            self.model.params.Cmat,
            self.Dmat_ndt,
            self.ndt_de,
            self.ndt_di,
        )

    def compute_hx_de(self):
        """Jacobians of ALNModel wrt. the variables delayed by de

        :return:    N x T x V x V Jacobians.
        :rtype:     np.ndarray
        """
        return compute_hx_de(
            self.model_params,
            self.precomp_factors,
            self.N,
            self.dim_vars,
            self.T,
            self.fullstate,
            self.control,
            self.model.params.Cmat,
            self.Dmat_ndt,
            self.ndt_de,
            self.ndt_di,
        )

    def compute_hx_di(self):
        """Jacobians of ALNModel wrt. the variables delayed by di

        :return:    N x T x V x V Jacobians.
        :rtype:     np.ndarray
        """
        return compute_hx_di(
            self.model_params,
            self.precomp_factors,
            self.N,
            self.dim_vars,
            self.T,
            self.fullstate,
            self.control,
            self.model.params.Cmat,
            self.Dmat_ndt,
            self.ndt_de,
            self.ndt_di,
        )

    def compute_hx_nw(self):
        """Jacobians for each time step for the network coupling.

        :return: N x N x T x V x V array
        :rtype: np.ndarray
        """

        return compute_hx_nw(
            self.model_params,
            self.precomp_factors,
            self.N,
            self.dim_vars,
            self.T,
            self.fullstate,
            self.control,
            self.model.params.Cmat,
            self.Dmat_ndt,
            self.ndt_de,
            self.ndt_di,
        )

    def get_fullstate(self):
        """Compute the full state (all 16 variables) of the ALN model by stepwise forward integration.

        :return:    N x V x T state vector
        :rtype:     np.ndarray
        """
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
        """Set the initial state of the ALN model as defined by input 'fullstate'

        :param fullstate:   state vector to read initial state from
        :type fullstate:    np.ndarray
        :param t:           time index
        :type t:            int
        """

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
        """Read the initial state of the ALN model

        :return:            initial state of the ALN model as N x V x N_maxdelay array
        :rtype t:           np.ndarray
        """
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
        """Read the final state of the ALN model (only last timestep)

        :return:            final state of the ALN model as N x V matrix
        :rtype t:           np.ndarray
        """
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
        """Set the initial state of the ALN model as defined by final values of state

        :param state:       state vector to read initial state from
        :type state:        np.ndarray
        """
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
