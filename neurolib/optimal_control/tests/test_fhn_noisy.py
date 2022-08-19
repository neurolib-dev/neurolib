import unittest

import numpy as np

from neurolib.models.fhn import FHNModel
from neurolib.utils.stimulus import ZeroInput
from neurolib.optimal_control import oc_fhn
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence

global limit_diff
limit_diff = 1e-4


class TestFHNNoisy(unittest.TestCase):
    """
    Test fhn in neurolib/optimal_control/
    """

    def test_noisy_1n_fhn_with_noise_approaches_input(self):
        fhn = FHNModel()

        duration = 3.0
        a = 10.0

        zero_input = ZeroInput().generate_input(duration=duration + fhn.params.dt, dt=fhn.params.dt)
        input_x = np.copy(zero_input)
        input_y = np.copy(zero_input)

        rs = RandomState(MT19937(SeedSequence(0)))  # work with fixed seed for reproducibility

        for t in range(1, input_x.shape[1] - 2):
            input_x[0, t] = rs.uniform(-a, a)
        fhn.params["x_ext"] = input_x
        for t in range(1, input_y.shape[1] - 2):
            input_y[0, t] = rs.uniform(-a, a)
        fhn.params["y_ext"] = input_y

        fhn.params["duration"] = duration
        fhn.params["xs_init"] = np.array([[0.0]])
        fhn.params["ys_init"] = np.array([[0.0]])

        fhn.run()
        target = np.concatenate(
            (
                np.concatenate((fhn.params["xs_init"], fhn.params["ys_init"]), axis=1)[:, :, np.newaxis],
                np.stack((fhn.x, fhn.y), axis=1),
            ),
            axis=2,
        )

        fhn.params["y_ext"] = zero_input
        fhn.params["x_ext"] = zero_input

        fhn_controlled_noisy = oc_fhn.OcFhn(fhn, target, w_p=1, w_2=0, M=2, M_validation=1000, method="3")

        control_coincide = False

        for i in range(100):
            fhn_controlled_noisy.optimize(1000)
            control = fhn_controlled_noisy.control

            c_diff = np.vstack([np.abs(control[0, 0, :] - input_x), np.abs(control[0, 1, :] - input_y)])
            if np.amax(c_diff) < limit_diff:
                control_coincide = True
                break

        self.assertTrue(control_coincide)

    # tests if we get the same output from running the optimization for the deterministic setting and the noisy setting with sigma=0
    # single node case
    def test_noisy_1n_fhn_with_without_noise_same_output(self):
        fhn = FHNModel()

        duration = 3.0
        a = 10.0
        test_iterations = 30

        zero_input = ZeroInput().generate_input(duration=duration + fhn.params.dt, dt=fhn.params.dt)
        input_x = np.copy(zero_input)
        input_y = np.copy(input_x)

        rs = RandomState(MT19937(SeedSequence(0)))  # work with fixed seed for reproducibility

        for t in range(1, input_x.shape[1] - 2):
            input_x[0, t] = rs.uniform(-a, a)
            input_y[0, t] = rs.uniform(-a, a)
        fhn.params["x_ext"] = input_x
        fhn.params["y_ext"] = input_y

        fhn.params["duration"] = duration
        fhn.params["xs_init"] = np.array([[0.0]])
        fhn.params["ys_init"] = np.array([[0.0]])

        fhn.run()
        target = np.concatenate(
            (
                np.concatenate((fhn.params["xs_init"], fhn.params["ys_init"]), axis=1)[:, :, np.newaxis],
                np.stack((fhn.x, fhn.y), axis=1),
            ),
            axis=2,
        )

        fhn.params["y_ext"] = zero_input
        fhn.params["x_ext"] = zero_input
        fhn_controlled_noisy = oc_fhn.OcFhn(fhn, target, w_p=1, w_2=0, M=2, M_validation=1000, method="3")

        fhn_controlled_noisy.optimize(test_iterations)
        control_noisy = fhn_controlled_noisy.control

        fhn.params["y_ext"] = zero_input
        fhn.params["x_ext"] = zero_input
        fhn_controlled_det = oc_fhn.OcFhn(fhn, target, w_p=1, w_2=0)
        fhn_controlled_det.optimize(test_iterations)
        control_det = fhn_controlled_det.control

        maxdiff = np.amax(np.abs(control_noisy - control_det))
        self.assertLess(maxdiff, limit_diff)

    # tests if we get the same output from running the optimization for the deterministic setting and the noisy setting with sigma=0
    # network case
    def test_noisy_3n_fhn_with_without_noise_same_output(self):

        N = 3
        dmat = np.zeros((N, N))  # no delay
        cmat = np.array([[0.0, 1.0, 2.0], [3.0, 0.0, 1.0], [0.0, 1.0, 0.0]])

        fhn = FHNModel(Cmat=cmat, Dmat=dmat)

        duration = 3.0
        a = 10.0
        test_iterations = 30

        fhn.params["duration"] = duration
        fhn.params["xs_init"] = np.vstack([0.0, 0.0, 0.0])
        fhn.params["ys_init"] = np.vstack([0.0, 0.0, 0.0])

        zero_input = ZeroInput().generate_input(duration=duration + fhn.params.dt, dt=fhn.params.dt)
        zero_input = np.vstack((zero_input, zero_input, zero_input))
        input_x = np.copy(zero_input)
        input_y = np.copy(input_x)

        rs = RandomState(MT19937(SeedSequence(0)))  # work with fixed seed for reproducibility

        for t in range(1, input_x.shape[1] - 2):
            for n in range(N):
                input_x[n, t] = rs.uniform(-a, a)
                input_y[n, t] = rs.uniform(-a, a)

        fhn.params["x_ext"] = input_x
        fhn.params["y_ext"] = input_y

        fhn.run()
        target = np.concatenate(
            (
                np.concatenate((fhn.params["xs_init"], fhn.params["ys_init"]), axis=1)[:, :, np.newaxis],
                np.stack((fhn.x, fhn.y), axis=1),
            ),
            axis=2,
        )

        fhn.params["y_ext"] = zero_input
        fhn.params["x_ext"] = zero_input
        fhn_controlled_noisy = oc_fhn.OcFhn(fhn, target, w_p=1, w_2=0, M=2, M_validation=1000, method="3")
        fhn_controlled_noisy.optimize(test_iterations)
        control_noisy = fhn_controlled_noisy.control

        fhn.params["y_ext"] = zero_input
        fhn.params["x_ext"] = zero_input
        fhn_controlled_det = oc_fhn.OcFhn(fhn, target, w_p=1, w_2=0)
        fhn_controlled_det.optimize(test_iterations)
        control_det = fhn_controlled_det.control

        maxdiff = np.amax(np.abs(control_noisy - control_det))
        self.assertLess(maxdiff, limit_diff)

    # tests if the OC computation returns zero control when w_p = 0
    # single-node case
    def test_onenode_wp0(self):
        print("Test OC for w_p = 0 in single-node model")
        fhn = FHNModel()

        duration = 3.0
        a = 10.0

        fhn.params["duration"] = duration
        fhn.params["xs_init"] = np.array([[0.0]])
        fhn.params["ys_init"] = np.array([[0.0]])

        rs = RandomState(MT19937(SeedSequence(0)))  # work with fixed seed for reproducibility
        input_x = ZeroInput().generate_input(duration=duration + fhn.params.dt, dt=fhn.params.dt)
        input_y = np.copy(input_x)

        for t in range(1, input_x.shape[1] - 2):
            input_x[0, :] = rs.uniform(-a, a)
            input_y[0, :] = rs.uniform(-a, a)
        fhn.params["x_ext"] = input_x
        fhn.params["y_ext"] = input_y

        fhn.run()
        target = np.concatenate(
            (
                np.concatenate((fhn.params["xs_init"], fhn.params["ys_init"]), axis=1)[:, :, np.newaxis],
                np.stack((fhn.x, fhn.y), axis=1),
            ),
            axis=2,
        )

        fhn.params.sigma_ou = 1.0

        fhn_controlled = oc_fhn.OcFhn(fhn, target, w_p=0, w_2=1, M=2, M_validation=1000, method="3")
        control_is_zero = False

        for i in range(100):
            fhn_controlled.optimize(1000)
            control = fhn_controlled.control

            c_max = np.amax(np.abs(control))
            if c_max < limit_diff:
                control_is_zero = True
                break

        self.assertTrue(control_is_zero)

    # tests if the OC computation returns zero control when w_p = 0
    # 3-node network case
    def test_3n_wp0(self):
        print("Test OC for w_p = 0 in single-node model")

        N = 3
        dmat = np.zeros((N, N))  # no delay
        cmat = np.array([[0.0, 1.0, 2.0], [3.0, 0.0, 1.0], [0.0, 1.0, 0.0]])

        fhn = FHNModel(Cmat=cmat, Dmat=dmat)

        duration = 3.0
        a = 1.0  # smaller amplitude to prevent numerical issues

        fhn.params["duration"] = duration
        fhn.params["xs_init"] = np.vstack([0.0, 0.0, 0.0])
        fhn.params["ys_init"] = np.vstack([0.0, 0.0, 0.0])

        rs = RandomState(MT19937(SeedSequence(0)))  # work with fixed seed for reproducibility
        input_x = ZeroInput().generate_input(duration=duration + fhn.params.dt, dt=fhn.params.dt)
        input_x = np.vstack((input_x, input_x, input_x))
        input_y = np.copy(input_x)

        for t in range(1, input_x.shape[1] - 2):
            for n in range(N):
                input_x[n, :] = rs.uniform(-a, a)
                input_y[n, :] = rs.uniform(-a, a)
        fhn.params["x_ext"] = input_x
        fhn.params["y_ext"] = input_y

        fhn.run()
        target = np.concatenate(
            (
                np.concatenate((fhn.params["xs_init"], fhn.params["ys_init"]), axis=1)[:, :, np.newaxis],
                np.stack((fhn.x, fhn.y), axis=1),
            ),
            axis=2,
        )

        fhn.params.sigma_ou = 1.0

        fhn_controlled = oc_fhn.OcFhn(fhn, target, w_p=0, w_2=1, M=2, M_validation=1000, method="3")
        control_is_zero = False

        for i in range(100):
            fhn_controlled.optimize(1000)
            control = fhn_controlled.control

            c_max = np.amax(np.abs(control))
            if c_max < limit_diff:
                control_is_zero = True
                break

        self.assertTrue(control_is_zero)


if __name__ == "__main__":
    unittest.main()
