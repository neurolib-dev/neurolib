import unittest

import numpy as np

from neurolib.models.fhn import FHNModel
from neurolib.utils.stimulus import ZeroInput
from neurolib.optimal_control import oc_fhn
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence

global limit_diff
limit_diff = 1e-4


class TestFHN(unittest.TestCase):
    """
    Test fhn in neurolib/optimal_control/
    """

    def test_noisy_fhn_with_noise_approaches_input(self):
        fhn = FHNModel()

        duration = 3.0
        a = 10.0

        zero_input = ZeroInput().generate_input(
            duration=duration + fhn.params.dt, dt=fhn.params.dt
        )
        input_x = np.copy(zero_input)
        input_y = np.copy(input_x)

        rs = RandomState(
            MT19937(SeedSequence(0))
        )  # work with fixed seed for reproducibility

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
        x_target = np.vstack([0.0, fhn.x.T])
        y_target = np.vstack([0.0, fhn.y.T])

        fhn.params["y_ext"] = zero_input
        fhn.params["x_ext"] = zero_input

        target = np.column_stack(([x_target, y_target])).T
        fhn_controlled_noisy = oc_fhn.OcFhn(fhn, target, w_p=1, w_2=0, M=2, M_validation=1000, method="3")

        control_coincide = False

        for i in range(100):
            fhn_controlled_noisy.optimize(1000)
            control = fhn_controlled_noisy.control

            c_diff = np.vstack(
                [np.abs(control[0, :] - input_x), np.abs(control[1, :] - input_y)]
            )
            if np.amax(c_diff) < limit_diff:
                control_coincide = True
                break

        self.assertTrue(control_coincide)

    def test_noisy_fhn_with_without_noise_same_output(self):
        fhn = FHNModel()

        duration = 3.0
        a = 10.0
        test_iterations = 30

        zero_input = ZeroInput().generate_input(
            duration=duration + fhn.params.dt, dt=fhn.params.dt
        )
        input_x = np.copy(zero_input)
        input_y = np.copy(input_x)

        rs = RandomState(
            MT19937(SeedSequence(0))
        )  # work with fixed seed for reproducibility

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
        x_target = np.vstack([0.0, fhn.x.T])
        y_target = np.vstack([0.0, fhn.y.T])

        fhn.params["y_ext"] = zero_input
        fhn.params["x_ext"] = zero_input

        target = np.column_stack(([x_target, y_target])).T

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


if __name__ == "__main__":
    unittest.main()
