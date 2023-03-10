import unittest

import numpy as np

from neurolib.models.hopf import HopfModel
from neurolib.utils.stimulus import ZeroInput
from neurolib.control.optimal_control import oc_hopf
from numpy.random import RandomState, SeedSequence, MT19937

global LIMIT_DIFF
LIMIT_DIFF = 1e-4


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

                            hopf = HopfModel(Cmat=cmat, Dmat=dmat)

                            prec_mat = np.zeros((hopf.params.N, len(hopf.output_vars)))
                            control_mat = np.zeros((hopf.params.N, len(hopf.state_vars)))

                            control_mat[c_node, c_channel] = 1.0
                            prec_mat[p_node, p_channel] = 1.0

                            hopf.params.duration = duration
                            hopf.params.coupling = coupling

                            # change parameters for faster convergence
                            hopf.params.K_gl = 1.0

                            # high parameter value will cause numerical problems for certain settings
                            if p_channel == 1:
                                hopf.params.K_gl = 10.0
                                if coupling == "additive":
                                    if bi_dir_connectivity == 0:
                                        hopf.params.K_gl = 20.0
                                elif coupling == "diffusive":
                                    hopf.params.K_gl = 5.0
                            if c_channel == 1:
                                if bi_dir_connectivity == 0 and coupling == "additive":
                                    hopf.params.K_gl = 20.0
                                if p_channel == 1:
                                    hopf.params.duration = 0.5

                            zero_input = ZeroInput().generate_input(
                                duration=hopf.params.duration + hopf.params.dt, dt=hopf.params.dt
                            )
                            input = np.copy(zero_input)

                            rs = RandomState(MT19937(SeedSequence(0)))  # work with fixed seed for reproducibility

                            for t in range(1, input.shape[1] - 4):
                                input[0, t] = rs.uniform(-a, a)

                            hopf.params["y_ext"] = np.vstack([zero_input, zero_input])
                            hopf.params["x_ext"] = np.vstack([zero_input, zero_input])

                            if c_channel == 0:
                                if c_node == 0:
                                    hopf.params["x_ext"] = np.vstack([input, zero_input])
                                else:
                                    hopf.params["x_ext"] = np.vstack([zero_input, input])
                            else:
                                if c_node == 0:
                                    hopf.params["y_ext"] = np.vstack([input, zero_input])
                                else:
                                    hopf.params["y_ext"] = np.vstack([zero_input, input])

                            hopf.params["xs_init"] = np.vstack([0.0, 0.0])
                            hopf.params["ys_init"] = np.vstack([0.0, 0.0])

                            hopf.run()

                            target = np.concatenate(
                                (
                                    np.concatenate(
                                        (hopf.params["xs_init"], hopf.params["ys_init"]),
                                        axis=1,
                                    )[:, :, np.newaxis],
                                    np.stack((hopf.x, hopf.y), axis=1),
                                ),
                                axis=2,
                            )

                            hopf.params["y_ext"] = np.vstack([zero_input, zero_input])
                            hopf.params["x_ext"] = np.vstack([zero_input, zero_input])

                            hopf_controlled = oc_hopf.OcHopf(
                                hopf,
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
                                hopf_controlled.optimize(iterations)
                                control = hopf_controlled.control

                                c_diff_max = np.amax(np.abs(control[c_node, c_channel, :] - input[0, :]))
                                if c_diff_max < lim:
                                    control_coincide = True
                                    break

                            self.assertTrue(control_coincide)

    # tests if the control from OC computation coincides with a random input used for target forward-simulation
    # delayed network case
    def test_twonode_delay_oc(self):
        print("Test OC in delayed 2-node network")

        rs = RandomState(MT19937(SeedSequence(0)))  # work with fixed seed for reproducibility

        duration = 1.0
        a = 5.0

        delay = rs.choice([0.1, 0.2, 0.3, 0.4])

        cmat = np.array([[0.0, 0.0], [1.0, 0.0]])
        dmat = np.array([[0.0, 0.0], [delay, 0.0]])

        hopf = HopfModel(Cmat=cmat, Dmat=dmat)

        prec_mat = np.zeros((hopf.params.N, len(hopf.output_vars)))
        control_mat = np.zeros((hopf.params.N, len(hopf.state_vars)))

        control_mat[0, 0] = 1.0
        prec_mat[1, 0] = 1.0

        hopf.params.duration = duration

        # change parameters for faster convergence
        hopf.params.K_gl = 1.0
        # change parameters for shorter test simulation time
        hopf.params.signalV = 1.0

        zero_input = ZeroInput().generate_input(duration=hopf.params.duration + hopf.params.dt, dt=hopf.params.dt)
        input = np.copy(zero_input)

        for t in range(1, input.shape[1] - 7):  # leave last inputs zero so signal can be reproduced despite delay
            input[0, t] = rs.uniform(-a, a)

        hopf.params["x_ext"] = np.vstack([input, zero_input])
        hopf.params["y_ext"] = np.vstack([zero_input, zero_input])

        zeroinit = np.zeros((5))

        hopf.params["xs_init"] = np.vstack([zeroinit, zeroinit])
        hopf.params["ys_init"] = np.vstack([zeroinit, zeroinit])

        hopf.run()

        self.assertTrue(np.amax(hopf.params.Dmat_ndt) >= 1)  # Relates to the given "delay" and time-discretization.

        target = np.concatenate(
            (
                np.stack(
                    (hopf.params["xs_init"][:, -1], hopf.params["ys_init"][:, -1]),
                    axis=1,
                )[:, :, np.newaxis],
                np.stack((hopf.x, hopf.y), axis=1),
            ),
            axis=2,
        )

        hopf.params["y_ext"] = np.vstack([zero_input, zero_input])
        hopf.params["x_ext"] = np.vstack([zero_input, zero_input])

        hopf_controlled = oc_hopf.OcHopf(
            hopf,
            target,
            w_p=1,
            w_2=0,
            control_matrix=control_mat,
            precision_matrix=prec_mat,
        )

        self.assertTrue((hopf.params.Dmat_ndt == hopf_controlled.Dmat_ndt).all())

        control_coincide = False

        iterations = 4000
        for i in range(100):
            hopf_controlled.optimize(iterations)
            control = hopf_controlled.control

            # last few entries of adjoint_state[0,0,:] are zero
            self.assertTrue(
                np.amax(
                    np.abs(hopf_controlled.adjoint_state[0, 0, -np.around(np.amax(hopf.params.Dmat_ndt)).astype(int) :])
                )
                == 0.0
            )

            c_diff_max = np.amax(np.abs(control[0, 0, :] - input[0, :]))
            if c_diff_max < LIMIT_DIFF:
                control_coincide = True
                break

        self.assertTrue(control_coincide)

    # Arbitrary network and control setting, get_xs() returns correct array shape (despite initial values array longer than 1)
    def test_get_xs(self):
        print("Test state shape agrees with target shape")

        cmat = np.array([[0.0, 1.0], [1.0, 0.0]])
        dmat = np.array([[0.0, 0.0], [0.0, 0.0]])  # no delay
        hopf = HopfModel(Cmat=cmat, Dmat=dmat)
        duration = 1.0
        hopf.params.duration = duration

        zero_input = ZeroInput().generate_input(duration=duration + hopf.params.dt, dt=hopf.params.dt)
        input = np.copy(zero_input)

        for t in range(input.shape[1]):
            input[0, t] = np.sin(t)

        hopf.params["y_ext"] = np.vstack([input, input])
        hopf.params["x_ext"] = np.vstack([-input, 1.1 * input])

        initind = 5

        zeroinit = np.zeros((initind))

        hopf.params["xs_init"] = np.vstack([zeroinit, zeroinit])
        hopf.params["ys_init"] = np.vstack([zeroinit, zeroinit])

        target = np.ones((2, 2, input.shape[1]))

        hopf_controlled = oc_hopf.OcHopf(
            hopf,
            target,
            w_p=1,
            w_2=1,
        )

        hopf_controlled.optimize(1)
        xs = hopf_controlled.get_xs()
        self.assertTrue(xs.shape == target.shape)


if __name__ == "__main__":
    unittest.main()
