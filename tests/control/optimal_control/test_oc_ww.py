import unittest
import numpy as np

from neurolib.models.ww import WWModel
from neurolib.control.optimal_control import oc_ww

import test_oc_utils as test_oc_utils

p = test_oc_utils.params


class TestWW(unittest.TestCase):
    """
    Test ww in neurolib/optimal_control/
    """

    # tests if the control from OC computation coincides with a random input used for target forward-simulation
    # single-node case
    def test_1n(self):
        print("Test OC in single-node system")
        model = WWModel()
        test_oc_utils.setinitzero_1n(model)
        model.params["duration"] = p.TEST_DURATION_6
        # decrease time scale of sigmoidal function
        # model.params["d_exc"] = 1.0
        # model.params["d_inh"] = 1.0

        for input_channel in [0, 1]:
            for measure_channel in range(4):
                print(
                    "input_channel, measure_channel = ", input_channel, measure_channel
                )

                cost_mat = np.zeros((model.params.N, len(model.output_vars)))
                control_mat = np.zeros((model.params.N, len(model.state_vars)))
                control_mat[
                    0, input_channel
                ] = 1.0  # only allow inputs to input_channel
                cost_mat[0, measure_channel] = 1.0  # only measure other channel

                test_oc_utils.set_input(model, p.ZERO_INPUT_1N_6)
                model.params[model.input_vars[input_channel]] = p.TEST_INPUT_1N_6
                model.run()
                target = test_oc_utils.gettarget_1n_ww(model)

                test_oc_utils.set_input(model, p.ZERO_INPUT_1N_6)

                model_controlled = oc_ww.OcWw(model, target)
                model_controlled.maximum_control_strength = 2.0

                model_controlled.control = np.concatenate(
                    [
                        control_mat[0, 0] * p.INIT_INPUT_1N_6[:, np.newaxis, :],
                        control_mat[0, 1] * p.INIT_INPUT_1N_6[:, np.newaxis, :],
                    ],
                    axis=1,
                )

                model_controlled.update_input()

                control_coincide = False

                for i in range(p.LOOPS):
                    model_controlled.optimize(p.ITERATIONS)

                    c_diff = np.abs(
                        model_controlled.control[0, input_channel, :]
                        - p.TEST_INPUT_1N_6[0, :]
                    )

                    if np.amax(c_diff) < p.LIMIT_DIFF:
                        control_coincide = True
                        break

                    if model_controlled.zero_step_encountered:
                        break

                self.assertTrue(control_coincide)

    def test_2n(self):
        print("Test OC in 2-node network")
        ### communication between E and I is validated in test_onenode_oc. Test only E-E communication
        ### Because of symmetry, test only inputs to 0 node, precision measuement in 1 node

        dmat = np.array([[0.0, 0.0], [0.0, 0.0]])  # no delay
        cmat = np.array([[0.0, 1.0], [1.0, 0.0]])

        model = WWModel(Cmat=cmat, Dmat=dmat)
        test_oc_utils.setinitzero_2n(model)
        model.params.duration = p.TEST_DURATION_10

        # decrease time scale of sigmoidal function
        # model.params["d_exc"] = 1.0
        # model.params["d_inh"] = 1.0

        cost_mat = np.zeros((model.params.N, len(model.output_vars)))
        control_mat = np.zeros((model.params.N, len(model.state_vars)))
        control_mat[0, 0] = 1.0
        cost_mat[1, 0] = 1.0

        model.params["exc_current"] = p.TEST_INPUT_2N_10
        model.params["inh_current"] = p.ZERO_INPUT_2N_10
        model.run()

        target = test_oc_utils.gettarget_2n_ww(model)
        model.params["exc_current"] = p.ZERO_INPUT_2N_10

        model_controlled = oc_ww.OcWw(
            model,
            target,
            control_matrix=control_mat,
            cost_matrix=cost_mat,
        )
        model_controlled.maximum_control_strength = 2.0

        model_controlled.control = np.concatenate(
            [
                p.INIT_INPUT_2N_10[:, np.newaxis, :],
                p.ZERO_INPUT_2N_10[:, np.newaxis, :],
            ],
            axis=1,
        )
        model_controlled.update_input()

        control_coincide = False

        for i in range(p.LOOPS):
            model_controlled.optimize(p.ITERATIONS)
            c_diff = np.abs(
                model_controlled.control[0, 0, :] - p.TEST_INPUT_2N_10[0, :]
            )
            if np.amax(c_diff) < p.LIMIT_DIFF:
                control_coincide = True
                break

            if model_controlled.zero_step_encountered:
                break

        self.assertTrue(control_coincide)

    # tests if the control from OC computation coincides with a random input used for target forward-simulation
    # delayed network case
    def test_2n_delay(self):
        print("Test OC in delayed 2-node network")

        cmat = np.array([[0.0, 0.0], [1.0, 0.0]])
        dmat = np.array([[0.0, 0.0], [p.TEST_DELAY, 0.0]])

        model = WWModel(Cmat=cmat, Dmat=dmat)
        test_oc_utils.setinitzero_2n(model)
        model.params.duration = p.TEST_DURATION_8
        model.params.signalV = 1.0

        cost_mat = np.zeros((model.params.N, len(model.output_vars)))
        control_mat = np.zeros((model.params.N, len(model.state_vars)))
        control_mat[0, 0] = 1.0
        cost_mat[1, 0] = 1.0

        model.params["exc_current"] = p.TEST_INPUT_2N_8
        model.params["inh_current"] = p.ZERO_INPUT_2N_8

        model.run()

        target = test_oc_utils.gettarget_2n_ww(model)
        model.params["exc_current"] = p.ZERO_INPUT_2N_8

        model_controlled = oc_ww.OcWw(
            model,
            target,
            control_matrix=control_mat,
            cost_matrix=cost_mat,
        )
        model_controlled.maximum_control_strength = 2.0

        model_controlled.control = np.concatenate(
            [p.INIT_INPUT_2N_8[:, np.newaxis, :], p.ZERO_INPUT_2N_8[:, np.newaxis, :]],
            axis=1,
        )
        model_controlled.update_input()

        control_coincide = False

        for i in range(p.LOOPS):
            model_controlled.optimize(p.ITERATIONS)

            # last entries of adjoint_state[0,0,:] are zero
            self.assertTrue(
                np.amax(
                    np.abs(model_controlled.adjoint_state[0, 0, -model.getMaxDelay() :])
                )
                == 0.0
            )

            c_diff_max = np.amax(
                np.abs(model_controlled.control[0, 0, :] - p.TEST_INPUT_2N_8[0, :])
            )
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
        model = WWModel(Cmat=cmat, Dmat=dmat)
        model.params.duration = p.TEST_DURATION_6
        test_oc_utils.set_input(model, p.TEST_INPUT_2N_6)

        target = np.ones((2, len(model.output_vars), p.TEST_INPUT_2N_6.shape[1]))

        model_controlled = oc_ww.OcWw(
            model,
            target,
        )

        model_controlled.optimize(1)
        xs = model_controlled.get_xs()
        self.assertTrue(xs.shape == target.shape)


if __name__ == "__main__":
    unittest.main()
