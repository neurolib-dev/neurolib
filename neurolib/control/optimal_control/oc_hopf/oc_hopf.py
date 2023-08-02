from neurolib.control.optimal_control.oc import OC, update_control_with_limit
from neurolib.control.optimal_control import cost_functions
import numba
import numpy as np
from neurolib.models.hopf.timeIntegration import compute_hx, compute_hx_nw, Dxdoth, Duh


class OcHopf(OC):
    """Class for optimal control specific to neurolib's implementation of the Stuart-Landau model with Hopf
        bifurcation ("Hopf model").

    :param model: Instance of Hopf model (can describe a single Hopf node or a network of coupled Hopf nodes.
    :type model: neurolib.models.hopf.model.HopfModel
    """

    # Remark: very similar to FHN!
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

        self.control = update_control_with_limit(
            self.N, self.dim_in, self.T, control, 0.0, np.zeros(control.shape), self.maximum_control_strength
        )

        self.model_params = self.get_model_params()

    def get_xs(self):
        """Stack the initial condition with the simulation results for dynamic variables 'x' and 'y' of Hopf model.

        :rtype:     np.ndarray of shape N x V x T
        """
        if self.model.params["xs_init"].shape[1] == 1:
            p1 = np.concatenate((self.model.params["xs_init"], self.model.params["ys_init"]), axis=1)[:, :, np.newaxis]
            xs = np.concatenate(
                (
                    p1,
                    np.stack((self.model.x, self.model.y), axis=1),
                ),
                axis=2,
            )
        else:
            p1 = np.stack((self.model.params["xs_init"][:, -1], self.model.params["ys_init"][:, -1]), axis=1)[
                :, :, np.newaxis
            ]
            xs = np.concatenate(
                (
                    p1,
                    np.stack((self.model.x, self.model.y), axis=1),
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
            self.model.params["x_ext"] = self.control[:, 0, :].reshape(1, -1)  # Reshape as row vector to match access
            self.model.params["y_ext"] = self.control[:, 1, :].reshape(1, -1)  # in model's time integration.

        else:
            self.model.params["x_ext"] = self.control[:, 0, :]
            self.model.params["y_ext"] = self.control[:, 1, :]

    def compute_dxdoth(self):
        """Derivative of systems dynamics wrt. change of systems variables."""
        return Dxdoth(self.N, self.dim_vars)

    def get_model_params(self):
        """Model params as an ordered tuple"""
        return (
            self.model.params.a,
            self.model.params.w,
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
        )

    def compute_hx_list(self):
        """List of Jacobians without and with time delays (e.g. in the ALN model) and list of respective time step delays as integers (0 for undelayed)

        :return:        List of Jacobian matrices, list of time step delays
        : rtype:        List of np.ndarray, List of integers
        """
        hx = self.compute_hx()
        return numba.typed.List([hx]), numba.typed.List([0])

    def compute_hx(self):
        """Jacobians of Hopf model wrt. its 'state_vars' at each time step.

        :return:        Array that contains Jacobians for all nodes in all time steps.
        :rtype:         np.ndarray of shape N x T x 4 x 4
        """
        return compute_hx(
            self.model_params,
            self.N,
            self.dim_vars,
            self.T,
            self.get_xs(),
        )

    def compute_hx_nw(self):
        """Jacobians for each time step for the network coupling.

        :return:    Jacobians for network connectivity in all time steps.
        :rtype:     np.ndarray of shape N x N x T x (4x4)
        """
        return compute_hx_nw(
            self.model.params["K_gl"],
            self.model.params["Cmat"],
            self.model.params["coupling"],
            self.N,
            self.dim_vars,
            self.T,
        )
