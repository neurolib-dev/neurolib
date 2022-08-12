import unittest

import numpy as np

from neurolib.models.fhn import FHNModel
from neurolib.utils.stimulus import ZeroInput
from neurolib.optimal_control import oc_fhn
from numpy.random import RandomState, SeedSequence, MT19937

global limit_diff
limit_diff = 1e-4


class TestFHN(unittest.TestCase):
    """
    Test fhn in neurolib/optimal_control/
    """

    def test_onenode_oc(self):
        print("Test OC in single-node network")
        fhn = FHNModel()

        duration = 3.0
        a = 10.0

        zero_input = ZeroInput().generate_input(
            duration=duration + fhn.params.dt, dt=fhn.params.dt
        )
        input = np.copy(zero_input)

        rs = RandomState(
            MT19937(SeedSequence(0))
        )  # work with fixed seed for reproducibility

        for t in range(1, input.shape[1] - 2):
            input[0, t] = rs.uniform(-a, a)

        for input_channel in [0, 1]:

            prec_mat = np.zeros((fhn.params.N, len(fhn.output_vars)))
            control_mat = np.zeros((fhn.params.N, len(fhn.state_vars)))
            if input_channel == 0:
                print("Input to x channel, measure in y channel")
                prec_mat[0, 1] = 1.0  # only measure in y-channel in one channel
                control_mat[0, 0] = 1.0  # only allow inputs to other channel
                fhn.params["x_ext"] = input
                fhn.params["y_ext"] = zero_input
            elif input_channel == 1:
                print("Input to y channel, measure in x channel")
                prec_mat[0, 0] = 1.0  # only measure in y-channel in one channel
                control_mat[0, 1] = 1.0  # only allow inputs to other channel
                fhn.params["x_ext"] = zero_input
                fhn.params["y_ext"] = input

            fhn.params["duration"] = duration
            fhn.params["xs_init"] = np.array([[0.0]])
            fhn.params["ys_init"] = np.array([[0.0]])

            fhn.run()
            target = np.concatenate(
                (
                    np.concatenate(
                        (fhn.params["xs_init"], fhn.params["ys_init"]), axis=1
                    )[:, :, np.newaxis],
                    np.stack((fhn.x, fhn.y), axis=1),
                ),
                axis=2,
            )

            fhn.params["y_ext"] = zero_input
            fhn.params["x_ext"] = zero_input

            fhn_controlled = oc_fhn.OcFhn(fhn, target, w_p=1, w_2=0)

            control_coincide = False

            for i in range(100):
                fhn_controlled.optimize(1000)
                control = fhn_controlled.control

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

    def test_twonode_oc(self):
        print("Test OC in 2-node network")

        dmat = np.array([[0.0, 0.0], [0.0, 0.0]])  # no delay
        duration = 1.0
        a = 1.0

        for coupling in ["additive", "diffusive"]:

            if coupling == "additive":
                continue

            for c_node in [0, 1]:
                p_node = np.abs(c_node - 1).astype(int)

                for c_channel in [0, 1]:

                    for p_channel in [0, 1]:

                        if p_channel == 1:
                            print("precision channel 1 too slow to converge, continue")
                            continue

                        for bi_dir_connectivity in [0, 1]:
                            print(coupling, " coupling")
                            print("control node = ", c_node)
                            print("control channel = ", c_channel)
                            print("precision channel = ", p_channel)
                            print("bidirectional connectivity = ", bi_dir_connectivity)

                            if bi_dir_connectivity == 0:
                                if c_node == 0:
                                    cmat = np.array([[0.0, 0.0], [1.0, 0.0]])
                                else:
                                    cmat = np.array([[0.0, 1.0], [0.0, 0.0]])
                            else:
                                cmat = np.array([[0.0, 1.0], [1.0, 0.0]])

                            fhn = FHNModel(Cmat=cmat, Dmat=dmat)

                            prec_mat = np.zeros((fhn.params.N, len(fhn.output_vars)))
                            control_mat = np.zeros((fhn.params.N, len(fhn.state_vars)))

                            control_mat[c_node, c_channel] = 1.0
                            prec_mat[p_node, p_channel] = 1.0

                            fhn.params.duration = duration
                            fhn.params.coupling = coupling
                            fhn.params.K_gl = 5.0  # for faster convergence
                            fhn.params.tau = (
                                1.0  # for stronger (faster) impact of x on y
                            )

                            zero_input = ZeroInput().generate_input(
                                duration=duration + fhn.params.dt, dt=fhn.params.dt
                            )
                            input = np.copy(zero_input)

                            rs = RandomState(
                                MT19937(SeedSequence(0))
                            )  # work with fixed seed for reproducibility

                            for t in range(2, input.shape[1] - 3):
                                input[0, t] = rs.uniform(-a, a)

                            fhn.params["y_ext"] = np.vstack([zero_input, zero_input])
                            fhn.params["x_ext"] = np.vstack([zero_input, zero_input])

                            if c_channel == 0:
                                if c_node == 0:
                                    fhn.params["x_ext"] = np.vstack([input, zero_input])
                                else:
                                    fhn.params["x_ext"] = np.vstack([zero_input, input])
                            else:
                                if c_node == 0:
                                    fhn.params["y_ext"] = np.vstack([input, zero_input])
                                else:
                                    fhn.params["y_ext"] = np.vstack([zero_input, input])

                            fhn.params["xs_init"] = np.vstack([0.0, 0.0])
                            fhn.params["ys_init"] = np.vstack([0.0, 0.0])

                            fhn.run()

                            target = np.concatenate(
                                (
                                    np.concatenate(
                                        (fhn.params["xs_init"], fhn.params["ys_init"]),
                                        axis=1,
                                    )[:, :, np.newaxis],
                                    np.stack((fhn.x, fhn.y), axis=1),
                                ),
                                axis=2,
                            )

                            fhn.params["y_ext"] = np.vstack([zero_input, zero_input])
                            fhn.params["x_ext"] = np.vstack([zero_input, zero_input])

                            fhn_controlled = oc_fhn.OcFhn(
                                fhn,
                                target,
                                w_p=1,
                                w_2=0,
                                control_matrix=control_mat,
                                precision_matrix=prec_mat,
                            )
                            control_coincide = False

                            lim = limit_diff
                            if c_channel == 1:
                                lim *= 100  # slow convergence

                            for i in range(100):
                                fhn_controlled.optimize(2000)
                                control = fhn_controlled.control

                                diff_abs = np.abs(
                                    control[c_node, c_channel, :] - input[0, :]
                                )
                                c_diff_max = np.amax(diff_abs)
                                if c_diff_max < lim:
                                    control_coincide = True
                                    break

                            self.assertTrue(control_coincide)

    def test_control_matrix(self):
        print("Test control matrix in 2-node network")

        dmat = np.array([[0.0, 0.0], [0.0, 0.0]])  # no delay
        N = 2
        duration = 1.0
        a = 1.0

        for c_node in [0, 1]:

            for c_channel in [0, 1]:
                cmat = np.array([[0.0, 1.0], [1.0, 0.0]])

                fhn = FHNModel(Cmat=cmat, Dmat=dmat)

                control_mat = np.zeros((fhn.params.N, len(fhn.state_vars)))
                control_mat[c_node, c_channel] = 1.0

                fhn.params.duration = duration

                zero_input = ZeroInput().generate_input(
                    duration=duration + fhn.params.dt, dt=fhn.params.dt
                )
                input = np.copy(zero_input)

                rs = RandomState(
                    MT19937(SeedSequence(0))
                )  # work with fixed seed for reproducibility

                for t in range(2, input.shape[1] - 3):
                    input[0, t] = rs.uniform(-a, a)

                fhn.params["y_ext"] = np.vstack([input, 2.0 * input])
                fhn.params["x_ext"] = np.vstack([-input, 3.0 * input])

                fhn.params["xs_init"] = np.vstack([0.0, 0.0])
                fhn.params["ys_init"] = np.vstack([0.0, 0.0])

                fhn.run()

                target = np.concatenate(
                    (
                        np.concatenate(
                            (fhn.params["xs_init"], fhn.params["ys_init"]),
                            axis=1,
                        )[:, :, np.newaxis],
                        np.stack((fhn.x, fhn.y), axis=1),
                    ),
                    axis=2,
                )

                fhn.params["y_ext"] = np.vstack([zero_input, zero_input])
                fhn.params["x_ext"] = np.vstack([zero_input, zero_input])

                fhn_controlled = oc_fhn.OcFhn(
                    fhn,
                    target,
                    w_p=1,
                    w_2=0,
                    control_matrix=control_mat,
                )

                fhn_controlled.optimize(1)
                control = fhn_controlled.control

                for n in range(N):
                    for v in range(len(fhn.output_vars)):
                        if n == c_node and v == c_channel:
                            continue
                        self.assertTrue(np.all(control[n, v, :] == 0))


if __name__ == "__main__":
    unittest.main()
