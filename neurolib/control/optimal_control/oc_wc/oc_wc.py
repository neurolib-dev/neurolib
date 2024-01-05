import numba

from neurolib.control.optimal_control.oc import OC
from neurolib.models.wc.timeIntegration import compute_hx, compute_hx_nw, Duh, Dxdoth


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
        control_interval=(None, None),
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
            control_interval=control_interval,
            cost_matrix=cost_matrix,
            control_matrix=control_matrix,
            M=M,
            M_validation=M_validation,
            validate_per_step=validate_per_step,
        )

        assert self.model.name == "wc"

    def compute_dxdoth(self):
        """Derivative of systems dynamics wrt. change of systems variables."""
        return Dxdoth(self.N, self.dim_vars)

    def get_model_params(self):
        """Model params as an ordered tuple.

        :rtype:     tuple
        """
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
            self.model.params.exc_ext_baseline,
            self.model.params.inh_ext_baseline,
        )

    def Duh(self):
        """Jacobian of systems dynamics wrt. external control input.

        :return:    N x 4 x 4 x T Jacobians.
        :rtype:     np.ndarray
        """

        xs = self.get_xs()
        xsd = self.get_xs_delay()

        return Duh(
            self.model_params,
            self.N,
            self.dim_in,
            self.dim_vars,
            self.T,
            self.control[:, self.state_vars_dict["exc"], :],
            self.control[:, self.state_vars_dict["inh"], :],
            xs[:, self.state_vars_dict["exc"], :],
            xs[:, self.state_vars_dict["inh"], :],
            self.model.params.K_gl,
            self.model.params.Cmat,
            self.Dmat_ndt,
            xsd[:, self.state_vars_dict["exc"], :],
            self.state_vars_dict,
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
            self.control,
            self.state_vars_dict,
        )

    def compute_hx_nw(self):
        """Jacobians for each time step for the network coupling.

        :return: N x N x T x (4x4) array
        :rtype: np.ndarray
        """

        xs = self.get_xs()

        return compute_hx_nw(
            self.model_params,
            self.model.params.K_gl,
            self.model.Cmat,
            self.Dmat_ndt,
            self.N,
            self.dim_vars,
            self.T,
            xs[:, self.state_vars_dict["exc"], :],
            xs[:, self.state_vars_dict["inh"], :],
            self.get_xs_delay()[:, self.state_vars_dict["exc"], :],
            self.control[:, self.state_vars_dict["exc"], :],
            self.state_vars_dict,
        )
