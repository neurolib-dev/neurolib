import unittest

import numpy as np

from neurolib.models.hopf import HopfModel
from neurolib.utils.stimulus import ZeroInput
from neurolib.optimal_control import oc_hopf
from numpy.random import RandomState, SeedSequence, MT19937

limit_diff = 1e-4


class TestHopf(unittest.TestCase):
    """
    Test hopf in neurolib/optimal_control/
    """

    # tests if the control from OC computation coincides with a random input used for target forward-simulation
    # single-node case
    def test_onenode_oc(self):
        print("Test OC in single-node system")
        hopf = HopfModel()

        duration = 3.0
        a = 10.0

        zero_input = ZeroInput().generate_input(duration=duration + hopf.params.dt, dt=hopf.params.dt)
        input = np.copy(zero_input)

        rs = RandomState(MT19937(SeedSequence(0)))  # work with fixed seed for reproducibility

        for t in range(1, input.shape[1] - 2):
            input[0, t] = rs.uniform(-a, a)

        for input_channel in [0, 1]:

            prec_mat = np.zeros((hopf.params.N, len(hopf.output_vars)))
            control_mat = np.zeros((hopf.params.N, len(hopf.state_vars)))
            if input_channel == 0:
                print("Input to x channel, measure in y channel")
                prec_mat[0, 1] = 1.0  # only measure in y-channel in one channel
                control_mat[0, 0] = 1.0  # only allow inputs to other channel
                hopf.params["x_ext"] = input
                hopf.params["y_ext"] = zero_input
            elif input_channel == 1:
                print("Input to y channel, measure in x channel")
                prec_mat[0, 0] = 1.0  # only measure in y-channel in one channel
                control_mat[0, 1] = 1.0  # only allow inputs to other channel
                hopf.params["x_ext"] = zero_input
                hopf.params["y_ext"] = input

            hopf.params["duration"] = duration
            hopf.params["xs_init"] = np.array([[0.0]])
            hopf.params["ys_init"] = np.array([[0.0]])

            hopf.run()
            target = np.concatenate(
                (
                    np.concatenate((hopf.params["xs_init"], hopf.params["ys_init"]), axis=1)[:, :, np.newaxis],
                    np.stack((hopf.x, hopf.y), axis=1),
                ),
                axis=2,
            )

            hopf.params["y_ext"] = zero_input
            hopf.params["x_ext"] = zero_input

            hopf_controlled = oc_hopf.OcHopf(hopf, target, w_p=1, w_2=0)

            control_coincide = False

            for i in range(100):
                hopf_controlled.optimize(1000)
                control = hopf_controlled.control

                if input_channel == 0:
                    c_diff = [
                        np.abs(control[0, 0, :] - input[0, :]),
                        np.abs(control[0, 1, :]),
                    ]
                elif input_channel == 1:
                    c_diff = [
                        np.abs(control[0, 0, :]),
                        np.abs(control[0, 1, :] - input[0, :]),
                    ]

                if np.amax(c_diff) < limit_diff:
                    control_coincide = True
                    break

            self.assertTrue(control_coincide)


if __name__ == "__main__":
    unittest.main()
