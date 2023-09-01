import unittest
import numpy as np

from neurolib.models.hopf import HopfModel
from neurolib.control.optimal_control import oc_hopf

import test_oc_params

p = test_oc_params.params


class TestHopf(unittest.TestCase):
    """
    Test hopf in neurolib/optimal_control/
    """

    # tests if the control from OC computation coincides with a random input used for target forward-simulation
    # single-node case
    def test_1n(self):
        print("Test OC in single-node system")
        model = HopfModel()
        test_oc_params.setinitzero_1n(model)
        model.params["duration"] = p.TEST_DURATION_6

        for input_channel in [0, 1]:
            cost_mat = np.zeros((model.params.N, len(model.output_vars)))
            control_mat = np.zeros((model.params.N, len(model.state_vars)))
            control_mat[0, input_channel] = 1.0  # only allow inputs to input_channel
            cost_mat[0, np.abs(input_channel - 1).astype(int)] = 1.0  # only measure other channel
            if input_channel == 0:
                print("Input to x channel, measure in y channel")
                model.params["x_ext"] = p.TEST_INPUT_1N_6
                model.params["y_ext"] = p.ZERO_INPUT_1N_6
            elif input_channel == 1:
                print("Input to y channel, measure in x channel")
                model.params["x_ext"] = p.ZERO_INPUT_1N_6
                model.params["y_ext"] = p.TEST_INPUT_1N_6

            model.run()
            target = test_oc_params.gettarget_1n(model)

            model.params["y_ext"] = p.ZERO_INPUT_1N_6
            model.params["x_ext"] = p.ZERO_INPUT_1N_6

            model_controlled = oc_hopf.OcHopf(model, target)

            if input_channel == 0:
                model_controlled.control = np.concatenate(
                    [p.INIT_INPUT_1N_6[:, np.newaxis, :], p.ZERO_INPUT_1N_6[:, np.newaxis, :]], axis=1
                )
            elif input_channel == 1:
                model_controlled.control = np.concatenate(
                    [p.ZERO_INPUT_1N_6[:, np.newaxis, :], p.INIT_INPUT_1N_6[:, np.newaxis, :]], axis=1
                )
            model_controlled.update_input()

            control_coincide = False

            for i in range(p.LOOPS):
                model_controlled.optimize(p.ITERATIONS)
                control = model_controlled.control

                if input_channel == 0:
                    c_diff = (np.abs(control[0, 0, :] - p.TEST_INPUT_1N_6[0, :]),)

                elif input_channel == 1:
                    c_diff = (np.abs(control[0, 1, :] - p.TEST_INPUT_1N_6[0, :]),)

                if np.amax(c_diff) < p.LIMIT_DIFF:
                    control_coincide = True
                    break

                if model_controlled.zero_step_encountered:
                    break

            self.assertTrue(control_coincide)

    # tests if the control from OC computation coincides with a random input used for target forward-simulation
    # network case
    def test_2n(self):
        print("Test OC in 2-node network")

        dmat = np.array([[0.0, 0.0], [0.0, 0.0]])  # no delay
        cmat = np.array([[0.0, 1.0], [1.0, 0.0]])

        model = HopfModel(Cmat=cmat, Dmat=dmat)
        test_oc_params.setinitzero_2n(model)

        cost_mat = np.zeros((model.params.N, len(model.output_vars)))
        control_mat = np.zeros((model.params.N, len(model.state_vars)))
        control_mat[0, 0] = 1.0
        cost_mat[1, 0] = 1.0

        model.params.duration = p.TEST_DURATION_8

        for coupling in ["additive", "diffusive"]:
            model.params.coupling = coupling

            model.params["x_ext"] = p.TEST_INPUT_2N_8
            model.params["y_ext"] = p.ZERO_INPUT_2N_8

            model.run()
            target = test_oc_params.gettarget_2n(model)
            model.params["x_ext"] = p.ZERO_INPUT_2N_8

            model_controlled = oc_hopf.OcHopf(
                model,
                target,
                control_matrix=control_mat,
                cost_matrix=cost_mat,
            )

            model_controlled.control = np.concatenate(
                [p.INIT_INPUT_2N_8[:, np.newaxis, :], p.ZERO_INPUT_2N_8[:, np.newaxis, :]], axis=1
            )
            model_controlled.update_input()

            control_coincide = False

            for i in range(p.LOOPS):
                model_controlled.optimize(p.ITERATIONS)
                control = model_controlled.control

                c_diff_max = np.amax(np.abs(control[0, 0, :] - p.TEST_INPUT_2N_8[0, :]))

                if c_diff_max < p.LIMIT_DIFF:
                    control_coincide = True
                    break

            if model_controlled.zero_step_encountered:
                break

            self.assertTrue(control_coincide)

    # tests if the control from OC computation coincides with a random input used for target forward-simulation
    # delayed network case
    def test_2n_delay(self):
        print("Test OC in delayed 2-node network")

        cmat = np.array([[0.0, 1.0], [1.0, 0.0]])
        dmat = np.array([[0.0, 0.0], [p.TEST_DELAY, 0.0]])

        model = HopfModel(Cmat=cmat, Dmat=dmat)
        test_oc_params.setinitzero_2n(model)

        cost_mat = np.zeros((model.params.N, len(model.output_vars)))
        control_mat = np.zeros((model.params.N, len(model.state_vars)))
        control_mat[0, 0] = 1.0
        cost_mat[1, 0] = 1.0

        model.params.duration = p.TEST_DURATION_10
        model.params.signalV = 1.0

        model.params["x_ext"] = p.TEST_INPUT_2N_10
        model.params["y_ext"] = p.ZERO_INPUT_2N_10

        model.run()
        target = test_oc_params.gettarget_2n(model)
        model.params["x_ext"] = p.ZERO_INPUT_2N_10

        model_controlled = oc_hopf.OcHopf(
            model,
            target,
            control_matrix=control_mat,
            cost_matrix=cost_mat,
        )
        model_controlled.control = np.concatenate(
            [p.INIT_INPUT_2N_10[:, np.newaxis, :], p.ZERO_INPUT_2N_10[:, np.newaxis, :]], axis=1
        )
        model_controlled.update_input()

        control_coincide = False

        for i in range(p.LOOPS):
            model_controlled.optimize(p.ITERATIONS)
            control = model_controlled.control

            # last few entries of adjoint_state[0,0,:] are zero
            self.assertTrue(np.amax(np.abs(model_controlled.adjoint_state[0, 0, -model.getMaxDelay() :])) == 0.0)

            c_diff_max = np.amax(np.abs(control[0, 0, :] - p.TEST_INPUT_2N_10[0, :]))
            if c_diff_max < p.LIMIT_DIFF:
                control_coincide = True
                break

            if model_controlled.zero_step_encountered:
                break

        self.assertTrue(control_coincide)

    # Arbitrary network and control setting, get_xs() returns correct array shape (despite initial values array longer than 1)
    def test_get_xs(self):
        print("Test state shape agrees with target shape")

        cmat = np.array([[0.0, 1.0], [1.0, 0.0]])
        dmat = np.array([[0.0, 0.0], [0.0, 0.0]])  # no delay
        model = HopfModel(Cmat=cmat, Dmat=dmat)
        model.params.duration = p.TEST_DURATION_6

        model.params["x_ext"] = p.TEST_INPUT_2N_6
        model.params["y_ext"] = -p.TEST_INPUT_2N_6

        target = np.ones((model.params.N, 2, p.TEST_INPUT_2N_6.shape[1]))

        model_controlled = oc_hopf.OcHopf(
            model,
            target,
        )

        model_controlled.optimize(1)
        xs = model_controlled.get_xs()
        self.assertTrue(xs.shape == target.shape)


if __name__ == "__main__":
    unittest.main()
