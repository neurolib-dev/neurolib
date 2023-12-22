import numba

from neurolib.control.optimal_control.oc import OC
from neurolib.models.fhn.timeIntegration import compute_hx, compute_hx_nw, Dxdoth, Duh


class OcFhn(OC):
    """Class for optimal control specific to neurolib's implementation of the FitzHugh-Nagumo (FHN) model.

    :param model: Instance of FHN model (can describe a single FHN node or a network of coupled FHN nodes.
    :type model:  neurolib.models.fhn.model.FHNModel
    """

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

        assert self.model.name == "fhn"

    def compute_dxdoth(self):
        """Derivative of systems dynamics wrt. change of systems variables."""
        return Dxdoth(self.N, self.dim_vars)

    def get_model_params(self):
        """Model params as an ordered tuple

        :rtype:        tuple"""
        return (
            self.model.params.alpha,
            self.model.params.beta,
            self.model.params.gamma,
            self.model.params.tau,
            self.model.params.epsilon,
        )

    def Duh(self):
        """4 x 4 Jacobian of systems dynamics wrt. external inputs (control signals) to all 'state_vars'. There are no
           inputs to the noise variables 'x_ou' and 'y_ou' in the model.

        :rtype:     np.ndarray of shape 4 x 4
        """
        return Duh(
            self.N,
            self.dim_in,
            self.dim_vars,
            self.T,
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
        """Jacobians of FHN model wrt. its 'state_vars' at each time step.

        :return:        Array that contains Jacobians for all nodes in all time steps.
        :rtype:         np.ndarray of shape N x T x 4 x 4
        """
        return compute_hx(
            self.model_params,
            self.N,
            self.dim_vars,
            self.T,
            self.get_xs(),
            self.state_vars_dict,
        )

    def compute_hx_nw(self):
        """Jacobians for each time step for the network coupling.

        :return:    Jacobians for network connectivity in all time steps.
        :rtype:     np.ndarray of shape N x N x T x 4 x 4
        """
        return compute_hx_nw(
            self.model.params["K_gl"],
            self.model.params["Cmat"],
            self.model.params["coupling"],
            self.N,
            self.dim_vars,
            self.T,
            self.state_vars_dict,
        )
