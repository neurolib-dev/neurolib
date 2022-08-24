from neurolib.optimal_control.oc import OC
import numpy as np
from .oc_fhn_jit import compute_hx, compute_hx_nw


class OcFhn(OC):
    def __init__(
        self,
        model,
        target,
        w_p=1,
        w_2=1,
        print_array=[],
        precision_cost_interval=(0, None),
        precision_matrix=None,
        control_matrix=None,
        M=1,
        M_validation=0,
        validate_per_step=False,
        method=None,
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
            method=method,
        )
        assert self.T == self.model.params["x_ext"].shape[1]
        assert self.T == self.model.params["y_ext"].shape[1]

        if self.N == 1:  # single-node model
            if self.model.params["x_ext"].ndim == 1:
                print("not implemented yet")
            else:
                self.control = np.concatenate((self.model.params["x_ext"], self.model.params["y_ext"]), axis=0)[
                    np.newaxis, :, :
                ]
        else:
            self.control = np.stack((self.model.params["x_ext"], self.model.params["y_ext"]), axis=1)

        for n in range(self.N):
            assert (self.control[n, 0, :] == self.model.params["x_ext"][n, :]).all()
            assert (self.control[n, 1, :] == self.model.params["y_ext"][n, :]).all()

        self.x_controls = self.model.params["x_ext"]  # save control signals throughout optimization iterations for
        # later analysis

        self.x_grads = np.array([])  # save gradients throughout optimization iterations for
        # later analysis

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
        # ToDo: model dependent
        # ToDo: find elegant way to combine the cases
        if self.N == 1:
            self.model.params["x_ext"] = self.control[:, 0, :].reshape(1, -1)  # Reshape as row vector to match access
            self.model.params["y_ext"] = self.control[:, 1, :].reshape(1, -1)  # in model's time integration.

            self.x_controls = np.vstack((self.x_controls, self.control[:, 0, :].reshape(1, -1)))

        else:
            self.model.params["x_ext"] = self.control[:, 0, :]
            self.model.params["y_ext"] = self.control[:, 1, :]

            self.x_controls = np.vstack((self.x_controls, self.control[:, 0, :]))

    def Dxdot(self):
        """2x2 Jacobian of systems dynamics wrt. to change of systems variables."""
        # ToDo: model dependent
        # Remark: do the dimensions need to be expanded accourding to x_ou and y_ou here?
        return np.array([[1, 0], [0, 1]])

    def Du(self):
        """2x2 Jacobian of systems dynamics wrt. to I_ext (external control input)"""
        # ToDo: model dependent
        return np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

    def compute_hx(self):
        """Jacobians for each time step.

        :return: Array of length self.T containing 2x2-matrices
        :rtype: np.ndarray
        """
        # ToDo: model dependent
        return compute_hx(
            self.model.params["alpha"],
            self.model.params["beta"],
            self.model.params["gamma"],
            self.model.params["tau"],
            self.model.params["epsilon"],
            self.N,
            self.dim_vars,
            self.T,
            self.get_xs(),
        )

    def compute_hx_nw(self):
        """Jacobians for each time step for the network coupling

        :return: (N x self.T x (2x2) array
        :rtype: np.ndarray
        """
        # ToDo: model dependent
        return compute_hx_nw(
            self.model.params["K_gl"],
            self.model.params["Cmat"],
            self.model.params["coupling"],
            self.N,
            self.dim_vars,
            self.T,
        )
