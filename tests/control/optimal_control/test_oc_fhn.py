import unittest
import numpy as np

from neurolib.models.fhn import FHNModel
from neurolib.control.optimal_control import oc_fhn

import test_oc_utils as test_oc_utils

p = test_oc_utils.params


class TestFHN(unittest.TestCase):
    """
    Test fhn in neurolib/optimal_control/
    """

    # tests if the control from OC computation coincides with a random input used for target forward-simulation
    # single-node case
    def test_1n(self):
        print("Test OC in single-node system")
        model = FHNModel()
        model.params["duration"] = p.TEST_DURATION_6
        test_oc_utils.setinitzero_1n(model)

        for input_channel in [0, 1]:
            cost_mat = np.zeros((model.params.N, len(model.output_vars)))
            control_mat = np.zeros((model.params.N, len(model.state_vars)))
            control_mat[0, input_channel] = 1.0  # only allow inputs to input_channel
            cost_mat[0, np.abs(input_channel - 1).astype(int)] = 1.0  # only measure other channel

            test_oc_utils.set_input(model, p.ZERO_INPUT_1N_6)
            model.params[model.input_vars[input_channel]] = p.TEST_INPUT_1N_6
            model.run()
            target = test_oc_utils.gettarget_1n(model)

            test_oc_utils.set_input(model, p.ZERO_INPUT_1N_6)

            model_controlled = oc_fhn.OcFhn(model, target)

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

        model = FHNModel(Cmat=cmat, Dmat=dmat)
        test_oc_utils.setinitzero_2n(model)

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
            target = test_oc_utils.gettarget_2n(model)
            model.params["x_ext"] = p.ZERO_INPUT_2N_8

            model_controlled = oc_fhn.OcFhn(
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

        model = FHNModel(Cmat=cmat, Dmat=dmat)
        test_oc_utils.setinitzero_2n(model)

        cost_mat = np.zeros((model.params.N, len(model.output_vars)))
        control_mat = np.zeros((model.params.N, len(model.state_vars)))
        control_mat[0, 0] = 1.0
        cost_mat[1, 0] = 1.0

        model.params.duration = p.TEST_DURATION_10
        model.params.signalV = 1.0

        model.params["x_ext"] = p.TEST_INPUT_2N_10
        model.params["y_ext"] = p.ZERO_INPUT_2N_10

        model.run()
        target = test_oc_utils.gettarget_2n(model)
        model.params["x_ext"] = p.ZERO_INPUT_2N_10

        model_controlled = oc_fhn.OcFhn(
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
        model = FHNModel(Cmat=cmat, Dmat=dmat)
        model.params.duration = p.TEST_DURATION_6

        model.params["x_ext"] = p.TEST_INPUT_2N_6
        model.params["y_ext"] = p.TEST_INPUT_2N_6
        model.run()
        target = test_oc_utils.gettarget_2n(model)

        model_controlled = oc_fhn.OcFhn(model, target)

        model_controlled.optimize(1)
        xs = model_controlled.get_xs()
        self.assertTrue(xs.shape == target.shape)

    # test whether the control matrix restricts the computed control output
    def test_control_matrix(self):
        print("Test control matrix in 2-node network")

        dmat = np.array([[0.0, 0.0], [0.0, 0.0]])  # no delay
        cmat = np.array([[0.0, 1.0], [1.0, 0.0]])
        model = FHNModel(Cmat=cmat, Dmat=dmat)

        model.params.duration = p.TEST_DURATION_6

        test_oc_utils.set_input(model, p.TEST_INPUT_2N_6)
        model.run()

        target = test_oc_utils.gettarget_2n(model)

        for c_node in [0, 1]:
            for c_channel in [0, 1]:
                control_mat = np.zeros((model.params.N, len(model.state_vars)))
                control_mat[c_node, c_channel] = 1.0

                test_oc_utils.set_input(model, p.ZERO_INPUT_2N_6)

                model_controlled = oc_fhn.OcFhn(
                    model,
                    target,
                    control_matrix=control_mat,
                )

                model_controlled.optimize(1)
                control = model_controlled.control

                for n in range(model.params.N):
                    for v in range(len(model.output_vars)):
                        if n == c_node and v == c_channel:
                            continue
                        self.assertTrue(np.all(control[n, v, :] == 0))

    # tests if the OC computation returns zero control when w_p = 0
    # single-node case
    def test_1n_wp0(self):
        print("Test OC for w_p = 0 in single-node model")
        model = FHNModel()

        model.params["duration"] = p.TEST_DURATION_6
        test_oc_utils.set_input(model, p.TEST_INPUT_1N_6)
        model.run()
        target = test_oc_utils.gettarget_1n(model)

        model_controlled = oc_fhn.OcFhn(model, target)
        model_controlled.weights["w_p"] = 0.0
        model_controlled.weights["w_2"] = 1.0
        control_is_zero = False

        for i in range(p.LOOPS):
            model_controlled.optimize(p.ITERATIONS)
            control = model_controlled.control

            c_max = np.amax(np.abs(control))
            if c_max < p.LIMIT_DIFF_ID:
                control_is_zero = True
                break

        self.assertTrue(control_is_zero)

    # tests if the OC computation returns zero control when w_p = 0
    # 3-node network case
    def test_2n_wp0(self):
        print("Test OC for w_p = 0 in 2-node model")

        dmat = np.array([[0.0, 0.0], [0.0, 0.0]])  # no delay
        cmat = np.array([[0.0, 1.0], [1.0, 0.0]])

        model = FHNModel(Cmat=cmat, Dmat=dmat)
        model.params["duration"] = p.TEST_DURATION_6

        test_oc_utils.set_input(model, p.TEST_INPUT_2N_6)
        model.run()
        target = test_oc_utils.gettarget_2n(model)

        model_controlled = oc_fhn.OcFhn(model, target)
        model_controlled.weights["w_p"] = 0.0
        model_controlled.weights["w_2"] = 1.0
        control_is_zero = False

        for i in range(p.LOOPS):
            model_controlled.optimize(p.ITERATIONS)
            control = model_controlled.control

            c_max = np.amax(np.abs(control))
            if c_max < p.LIMIT_DIFF_ID:
                control_is_zero = True
                break

        self.assertTrue(control_is_zero)

    # Arbitrary network and control setting, initial control violates the maximum absolute criterion.
    def test_u_max_no_optimizations(self):
        print("Test maximum control strength in initialization.")

        cmat = np.array([[0.0, 1.0], [1.0, 0.0]])
        dmat = np.array([[0.0, 0.0], [0.0, 0.0]])  # no delay
        model = FHNModel(Cmat=cmat, Dmat=dmat)
        model.params.duration = p.TEST_DURATION_6

        test_oc_utils.set_input(model, 10.0 * p.TEST_INPUT_2N_6)
        model.run()
        target = test_oc_utils.gettarget_2n(model)

        cost_mat = np.ones((model.params.N, len(model.output_vars)))
        control_mat = np.ones((model.params.N, len(model.state_vars)))

        maximum_control_strength = 0.5

        model_controlled = oc_fhn.OcFhn(
            model,
            target,
            maximum_control_strength=maximum_control_strength,
            cost_matrix=cost_mat,
            control_matrix=control_mat,
        )

        self.assertTrue(np.max(np.abs(model_controlled.control) <= maximum_control_strength))

    # Arbitrary network and control setting, initial control violates the maximum absolute criterion.
    def test_u_max_after_optimizations(self):
        print("Test maximum control strength after optimization.")
        cmat = np.array([[0.0, 1.0], [1.0, 0.0]])
        dmat = np.array([[0.0, 0.0], [0.0, 0.0]])  # no delay
        model = FHNModel(Cmat=cmat, Dmat=dmat)
        model.params.duration = p.TEST_DURATION_6

        test_oc_utils.set_input(model, 10.0 * p.TEST_INPUT_2N_6)
        model.run()
        target = test_oc_utils.gettarget_2n(model)

        cost_mat = np.ones((model.params.N, len(model.output_vars)))
        control_mat = np.ones((model.params.N, len(model.output_vars)))

        maximum_control_strength = 0.5

        model_controlled = oc_fhn.OcFhn(
            model,
            target,
            maximum_control_strength=maximum_control_strength,
            cost_matrix=cost_mat,
            control_matrix=control_mat,
        )

        model_controlled.optimize(1)
        self.assertTrue(np.max(np.abs(model_controlled.control) <= maximum_control_strength))

    def test_adjust_init(self):
        print("Test adjust_init function of OC class")

        cmat = np.array([[0.0, 1.0], [1.0, 0.0]])
        dmat = np.array([[0.0, 10.0], [10.0, 0.0]])  # large delay
        model = FHNModel(Cmat=cmat, Dmat=dmat)
        model.params.duration = p.TEST_DURATION_6

        test_oc_utils.set_input(model, p.ZERO_INPUT_2N_6)
        model.run()
        target = test_oc_utils.gettarget_2n(model)
        intmaxdel = model.getMaxDelay()
        targetinitshape = (2, intmaxdel + 1)

        for test_init in [
            1.0,
            [1.0],
            np.array([1.0]),
            np.ones((2,)),
            np.ones((2, 1)),
            np.ones((2, intmaxdel - 2)),
            np.ones((2, intmaxdel + 1)),
            np.ones((2, intmaxdel + 3)),
        ]:
            for init_var in model.init_vars:
                if "ou" in init_var:
                    continue
                model.params[init_var] = test_init
                model_controlled = oc_fhn.OcFhn(
                    model,
                    target,
                )

                for init_var0 in model.init_vars:
                    if "ou" in init_var0:
                        continue
                    self.assertTrue(model_controlled.model.params[init_var0].shape == targetinitshape)

    def test_adjust_input(self):
        print("Test test_adjust_input function of OC class")

        cmat = np.array([[0.0, 1.0], [1.0, 0.0]])
        dmat = np.array([[0.0, 0.0], [0.0, 0.0]])  # no delay
        model = FHNModel(Cmat=cmat, Dmat=dmat)
        model.params.duration = p.TEST_DURATION_6

        target = np.zeros((model.params.N, len(model.state_vars), p.TEST_INPUT_2N_6.shape[1]))
        targetinputshape = (target.shape[0], target.shape[2])

        for test_input in [
            1.0,
            [1.0],
            np.array([1.0]),
            np.ones((2,)),
            np.ones((2, 1)),
            np.ones((2, target.shape[2] - 2)),
            np.ones((2, target.shape[2])),
            np.ones((2, target.shape[2] + 2)),
        ]:
            for input_var in model.input_vars:
                model.params[input_var] = test_input
                model_controlled = oc_fhn.OcFhn(
                    model,
                    target,
                )

                for input_var0 in model.input_vars:
                    self.assertTrue(model_controlled.model.params[input_var0].shape == targetinputshape)


if __name__ == "__main__":
    unittest.main()
