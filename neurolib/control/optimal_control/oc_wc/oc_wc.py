from neurolib.control.optimal_control.oc import OC
from neurolib.control.optimal_control import cost_functions
import numpy as np
import numba
from neurolib.models.wc.timeIntegration import compute_hx, compute_nw_input, compute_hx_nw, Duh, Dxdoth


class OcWc(OC):
    """Class for optimal control specific to neurolib's implementation of the two-population Wilson-Cowan model
            ("WCmodel").

    :param model: Instance of Wilson-Cowan model (can describe a single Wilson-Cowan node or a network of coupled
                  Wilson-Cowan nodes.
    :type model: neurolib.models.wc.model.WCModel
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
        self.model_params = self.get_model_params()

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
        """Stack the initial condition with the simulation results for both ('exc' and 'inh') populations.

        :return: N x V x T array containing all values of 'exc' and 'inh'.
        :rtype:  np.ndarray
        """
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

    def compute_dxdoth(self):
        """Derivative of systems dynamics wrt. change of systems variables."""
        return Dxdoth(self.N, self.dim_vars)

    def get_model_params(self):
        """Model params as an ordered tuple"""
        return (
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
        )

    def Duh(self):
        """Jacobian of systems dynamics wrt. external control input.

        :return:    N x 4 x 4 x T Jacobians.
        :rtype:     np.ndarray
        """

        xs = self.get_xs()
        e = xs[:, 0, :]
        i = xs[:, 1, :]
        xsd = self.get_xs_delay()
        ed = xsd[:, 0, :]

        input = self.background + self.control
        ue = input[:, 0, :]
        ui = input[:, 1, :]

        return Duh(
            self.model_params,
            self.N,
            self.dim_in,
            self.dim_vars,
            self.T,
            ue,
            ui,
            e,
            i,
            self.model.params.K_gl,
            self.model.params.Cmat,
            self.Dmat_ndt,
            ed,
        )

    def compute_hx_list(self):
        """List of Jacobians without and with time delays (e.g. in the ALN model) and list of respective time step delays as integers (0 for undelayed)

        :return:        List of Jacobian matrices, list of time step delays
        : rtype:        List of np.ndarray, List of integers
        """
        hx = self.compute_hx()
        return numba.typed.List([hx]), numba.typed.List([0])

    def compute_hx(self):
        """Jacobians of WCModel wrt. the 'e'- and 'i'-variable for each time step.

        :return:    N x T x 4 x 4 Jacobians.
        :rtype:     np.ndarray
        """
        return compute_hx(
            self.model_params,
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
            self.model_params,
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
        )
