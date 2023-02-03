import unittest

import numpy as np

from neurolib.models.wc import WCModel
from neurolib.utils.stimulus import ZeroInput
from neurolib.optimal_control import oc_wc
from numpy.random import RandomState, SeedSequence, MT19937

global LIMIT_DIFF
LIMIT_DIFF = 1e-4


class TestWC(unittest.TestCase):
    """
    Test wc in neurolib/optimal_control/
    """

    # tests if the control from OC computation coincides with a random input used for target forward-simulation
    # single-node case
    def test_onenode_oc(self):
        print("Test OC in single-node system")
        wc = WCModel()

        duration = 3.0
        a = 1.0

        zero_input = ZeroInput().generate_input(duration=duration + wc.params.dt, dt=wc.params.dt)
        input = np.copy(zero_input)

        rs = RandomState(MT19937(SeedSequence(0)))  # work with fixed seed for reproducibility

        for t in range(1, input.shape[1] - 2):
            input[0, t] = rs.uniform(-a, a)

        for input_channel in [0, 1]:

            prec_mat = np.zeros((wc.params.N, len(wc.output_vars)))
            control_mat = np.zeros((wc.params.N, len(wc.state_vars)))
            if input_channel == 0:
                print("Input to E channel, measure in I channel")
                prec_mat[0, 1] = 1.0  # only measure in I-channel in one channel
                control_mat[0, 0] = 1.0  # only allow inputs to other channel
                wc.params["exc_ext"] = input
                wc.params["inh_ext"] = zero_input
            elif input_channel == 1:
                print("Input to I channel, measure in E channel")
                prec_mat[0, 0] = 1.0  # only measure in E-channel in one channel
                control_mat[0, 1] = 1.0  # only allow inputs to other channel
                wc.params["exc_ext"] = zero_input
                wc.params["inh_ext"] = input

            wc.params["duration"] = duration
            wc.params["exc_init"] = np.array([[0.0]])
            wc.params["inh_init"] = np.array([[0.0]])

            wc.run()
            target = np.concatenate(
                (
                    np.concatenate((wc.params["exc_init"], wc.params["inh_init"]), axis=1)[:, :, np.newaxis],
                    np.stack((wc.exc, wc.inh), axis=1),
                ),
                axis=2,
            )

            wc.params["inh_ext"] = zero_input
            wc.params["exc_ext"] = zero_input

            wc_controlled = oc_wc.OcWc(wc, target, w_p=1, w_2=0)

            control_coincide = False
            iterations = 4000

            for i in range(100):
                wc_controlled.optimize(iterations)
                control = wc_controlled.control

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

                if np.amax(c_diff) < LIMIT_DIFF:
                    control_coincide = True
                    break

            self.assertTrue(control_coincide)

    def test_twonode_oc(self):
        print("Test OC in 2-node network")

        dmat = np.array([[0.0, 0.0], [0.0, 0.0]])  # no delay
        duration = 1.0
        a = 1.0

        for coupling in ["additive", "diffusive"]:

            for c_node in [0, 1]:

                p_node = np.abs(c_node - 1).astype(int)

                for c_channel in [0, 1]:

                    for p_channel in [0, 1]:

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

                            wc = WCModel(Cmat=cmat, Dmat=dmat)

                            prec_mat = np.zeros((wc.params.N, len(wc.output_vars)))
                            control_mat = np.zeros((wc.params.N, len(wc.state_vars)))

                            control_mat[c_node, c_channel] = 1.0
                            prec_mat[p_node, p_channel] = 1.0

                            wc.params.duration = duration
                            wc.params.coupling = coupling

                            # change parameters for faster convergence
                            wc.params.K_gl = 1.0

                            # high parameter value will cause numerical problems for certain settings
                            if p_channel == 1:
                                wc.params.K_gl = 10.0
                                if coupling == "additive":
                                    if bi_dir_connectivity == 0:
                                        wc.params.K_gl = 20.0
                                elif coupling == "diffusive":
                                    wc.params.K_gl = 5.0
                            if c_channel == 1:
                                if bi_dir_connectivity == 0 and coupling == "additive":
                                    wc.params.K_gl = 20.0
                                if p_channel == 1:
                                    wc.params.duration = 0.5

                            zero_input = ZeroInput().generate_input(
                                duration=wc.params.duration + wc.params.dt, dt=wc.params.dt
                            )
                            input = np.copy(zero_input)

                            rs = RandomState(MT19937(SeedSequence(0)))  # work with fixed seed for reproducibility

                            for t in range(1, input.shape[1] - 4):
                                input[0, t] = rs.uniform(-a, a)

                            wc.params["inh_ext"] = np.vstack([zero_input, zero_input])
                            wc.params["exc_ext"] = np.vstack([zero_input, zero_input])

                            if c_channel == 0:
                                if c_node == 0:
                                    wc.params["exc_ext"] = np.vstack([input, zero_input])
                                else:
                                    wc.params["exc_ext"] = np.vstack([zero_input, input])
                            else:
                                if c_node == 0:
                                    wc.params["inh_ext"] = np.vstack([input, zero_input])
                                else:
                                    wc.params["inh_ext"] = np.vstack([zero_input, input])

                            wc.params["exc_init"] = np.vstack([0.0, 0.0])
                            wc.params["inh_init"] = np.vstack([0.0, 0.0])

                            wc.run()

                            target = np.concatenate(
                                (
                                    np.concatenate(
                                        (wc.params["exc_init"], wc.params["inh_init"]),
                                        axis=1,
                                    )[:, :, np.newaxis],
                                    np.stack((wc.exc, wc.inh), axis=1),
                                ),
                                axis=2,
                            )

                            wc.params["inh_ext"] = np.vstack([zero_input, zero_input])
                            wc.params["exc_ext"] = np.vstack([zero_input, zero_input])

                            wc_controlled = oc_wc.OcWc(
                                wc,
                                target,
                                w_p=1,
                                w_2=0,
                                control_matrix=control_mat,
                                precision_matrix=prec_mat,
                            )

                            control_coincide = False
                            lim = LIMIT_DIFF
                            if p_channel == 1 or c_channel == 1:
                                lim *= 4000  # for internal purposes: 1000, for simple testing: 4000

                            iterations = 4000
                            for i in range(100):
                                wc_controlled.optimize(iterations)
                                control = wc_controlled.control

                                c_diff_max = np.amax(np.abs(control[c_node, c_channel, :] - input[0, :]))
                                if c_diff_max < lim:
                                    control_coincide = True
                                    break

                                print(c_diff_max)

                            self.assertTrue(control_coincide)

    # tests if the control from OC computation coincides with a random input used for target forward-simulation
    # delayed network case
    def test_twonode_delay_oc(self):
        print("Test OC in delayed 2-node network")

        rs = RandomState(MT19937(SeedSequence(0)))  # work with fixed seed for reproducibility

        duration = 0.7
        a = 5.0

        delay = rs.uniform(0.1, 0.4)

        cmat = np.array([[0.0, 0.0], [1.0, 0.0]])
        dmat = np.array([[0.0, 0.0], [delay, 0.0]])

        wc = WCModel(Cmat=cmat, Dmat=dmat)

        prec_mat = np.zeros((wc.params.N, len(wc.output_vars)))
        control_mat = np.zeros((wc.params.N, len(wc.state_vars)))

        control_mat[0, 0] = 1.0
        prec_mat[1, 0] = 1.0

        wc.params.duration = duration

        # change parameters for faster convergence
        wc.params.K_gl = 10.0
        # change parameters for shorter test simulation time
        wc.params.signalV = 1.0

        zero_input = ZeroInput().generate_input(duration=wc.params.duration + wc.params.dt, dt=wc.params.dt)
        input = np.copy(zero_input)

        for t in range(1, input.shape[1] - 5):  # leave last inputs zero so signal can be reproduced despite delay
            input[0, t] = rs.uniform(-a, a)

        wc.params["exc_ext"] = np.vstack([input, zero_input])
        wc.params["inh_ext"] = np.vstack([zero_input, zero_input])

        zeroinit = np.zeros((5))

        wc.params["exc_init"] = np.vstack([zeroinit, zeroinit])
        wc.params["inh_init"] = np.vstack([zeroinit, zeroinit])

        wc.run()

        self.assertTrue(np.amax(wc.params.Dmat_ndt) >= 1)  # Relates to the given "delay" and time-discretization.

        target = np.concatenate(
            (
                np.stack(
                    (wc.params["exc_init"][:, -1], wc.params["inh_init"][:, -1]),
                    axis=1,
                )[:, :, np.newaxis],
                np.stack((wc.exc, wc.inh), axis=1),
            ),
            axis=2,
        )

        wc.params["exc_ext"] = np.vstack([zero_input, zero_input])
        wc.params["inh_ext"] = np.vstack([zero_input, zero_input])

        wc_controlled = oc_wc.OcWc(
            wc,
            target,
            w_p=1,
            w_2=0,
            control_matrix=control_mat,
            precision_matrix=prec_mat,
        )

        self.assertTrue((wc.params.Dmat_ndt == wc_controlled.Dmat_ndt).all())

        control_coincide = False

        iterations = 4000
        for i in range(100):
            wc_controlled.optimize(iterations)
            control = wc_controlled.control

            # last few entries of adjoint_state[0,0,:] are zero
            self.assertTrue(
                np.amax(
                    np.abs(wc_controlled.adjoint_state[0, 0, -np.around(np.amax(wc.params.Dmat_ndt)).astype(int) :])
                )
                == 0.0
            )

            c_diff_max = np.amax(np.abs(control[0, 0, :] - input[0, :]))
            print(c_diff_max)
            print(control[0, 0, :])
            print(input[0, :])
            print(".............")
            if c_diff_max < LIMIT_DIFF:
                control_coincide = True
                break

        self.assertTrue(control_coincide)


if __name__ == "__main__":
    unittest.main()
