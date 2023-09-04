import unittest
import numpy as np

from neurolib.models.fhn import FHNModel
from neurolib.utils.stimulus import ZeroInput
from neurolib.control.optimal_control import oc_fhn

import test_oc_utils as test_oc_utils

p = test_oc_utils.params


class TestFHNNoisy(unittest.TestCase):
    """
    Test fhn in neurolib/optimal_control/
    """

    def test_noisy_1n_fhn_with_noise_approaches_input(self):
        model = FHNModel()

        test_oc_utils.set_input(model, p.TEST_INPUT_1N_6)

        model.params["duration"] = p.TEST_DURATION_6
        model.params["xs_init"] = np.array([[0.0]])
        model.params["ys_init"] = np.array([[0.0]])

        model.run()
        target = test_oc_utils.gettarget_1n(model)

        test_oc_utils.set_input(model, p.ZERO_INPUT_1N_6)

        model_controlled_noisy = oc_fhn.OcFhn(model, target, M=2, M_validation=1000)
        control_coincide = False

        for i in range(100):
            model_controlled_noisy.optimize(1000)
            control = model_controlled_noisy.control

            c_diff = np.vstack(
                [
                    np.abs(control[0, 0, :] - p.TEST_INPUT_1N_6[0, :]),
                    np.abs(control[0, 1, :] - p.TEST_INPUT_1N_6[0, :]),
                ]
            )
            if np.amax(c_diff) < p.LIMIT_DIFF:
                control_coincide = True
                break

            if model_controlled_noisy.zero_step_encountered:
                break

        self.assertTrue(control_coincide)

    # tests if we get the same output from running the optimization for the deterministic setting and the noisy setting with sigma=0
    # single node case
    def test_noisy_1n_fhn_with_without_noise_same_output(self):
        model = FHNModel()

        test_iterations = 6

        test_oc_utils.set_input(model, p.TEST_INPUT_1N_6)

        model.params["duration"] = p.TEST_DURATION_6
        model.params["xs_init"] = np.array([[0.0]])
        model.params["ys_init"] = np.array([[0.0]])

        model.run()
        target = test_oc_utils.gettarget_1n(model)

        test_oc_utils.set_input(model, p.ZERO_INPUT_1N_6)
        model_controlled_noisy = oc_fhn.OcFhn(model, target, M=2, M_validation=1000)

        model_controlled_noisy.optimize(test_iterations)
        control_noisy = model_controlled_noisy.control

        model_controlled_det = oc_fhn.OcFhn(model, target)
        model_controlled_det.optimize(test_iterations)
        control_det = model_controlled_det.control

        maxdiff = np.amax(np.abs(control_noisy - control_det))
        self.assertLess(maxdiff, p.LIMIT_DIFF_ID)

    # tests if we get the same output from running the optimization for the deterministic setting and the noisy setting with sigma=0
    # network case
    def test_noisy_2n_fhn_with_without_noise_same_output(self):
        N = 2
        dmat = np.zeros((N, N))  # no delay
        cmat = np.array([[0.0, 1.0], [1.0, 0.0]])

        model = FHNModel(Cmat=cmat, Dmat=dmat)
        test_iterations = 6

        model.params["duration"] = p.TEST_DURATION_6
        model.params["xs_init"] = np.vstack([0.0, 0.0])
        model.params["ys_init"] = np.vstack([0.0, 0.0])

        test_oc_utils.set_input(model, p.TEST_INPUT_2N_6)

        model.run()
        target = test_oc_utils.gettarget_2n(model)
        test_oc_utils.set_input(model, p.ZERO_INPUT_2N_6)

        model_controlled_noisy = oc_fhn.OcFhn(model, target, M=2, M_validation=10)
        model_controlled_noisy.optimize(test_iterations)
        control_noisy = model_controlled_noisy.control

        model_controlled_det = oc_fhn.OcFhn(model, target)
        model_controlled_det.optimize(test_iterations)
        control_det = model_controlled_det.control

        maxdiff = np.amax(np.abs(control_noisy - control_det))
        self.assertLess(maxdiff, p.LIMIT_DIFF_ID)

    # tests if the OC computation returns zero control when w_p = 0
    # single-node case
    def test_1n_wp0(self):
        print("Test OC for w_p = 0 in single-node model")
        model = FHNModel()

        model.params["duration"] = p.TEST_DURATION_6

        test_oc_utils.set_input(model, p.TEST_INPUT_1N_6)

        model.run()
        target = test_oc_utils.gettarget_1n(model)
        model.params.sigma_ou = 1.0

        model_controlled = oc_fhn.OcFhn(model, target, M=2, M_validation=1000)
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
        print("Test OC for w_p = 0 in single-node model")

        N = 2
        dmat = np.zeros((N, N))  # no delay
        cmat = np.array([[0.0, 1.0], [1.0, 0.0]])

        model = FHNModel(Cmat=cmat, Dmat=dmat)

        model.params["duration"] = p.TEST_DURATION_6
        test_oc_utils.set_input(model, p.TEST_INPUT_2N_6)

        model.run()
        target = test_oc_utils.gettarget_2n(model)
        model.params.sigma_ou = 1.0

        model_controlled = oc_fhn.OcFhn(model, target, M=2, M_validation=1000)
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


if __name__ == "__main__":
    unittest.main()
