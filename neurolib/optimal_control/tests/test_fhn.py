import unittest

import numpy as np

from neurolib.models.fhn import FHNModel
from neurolib.utils.stimulus import ZeroInput
from neurolib.optimal_control import oc_fhn
from numpy.random import RandomState, SeedSequence, MT19937

global LIMIT_DIFF
LIMIT_DIFF = 1e-4


class TestFHN(unittest.TestCase):
    """
    Test fhn in neurolib/optimal_control/
    """

    # tests if the control from OC computation coincides with a random input used for target forward-simulation
    # single-node case
    def test_onenode_oc(self):
        print("Test OC in single-node system")
        fhn = FHNModel()

        duration = 3.0
        a = 10.0

        zero_input = ZeroInput().generate_input(duration=duration + fhn.params.dt, dt=fhn.params.dt)
        input = np.copy(zero_input)

        rs = RandomState(MT19937(SeedSequence(0)))  # work with fixed seed for reproducibility

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
                    np.concatenate((fhn.params["xs_init"], fhn.params["ys_init"]), axis=1)[:, :, np.newaxis],
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

                if np.amax(c_diff) < LIMIT_DIFF:
                    control_coincide = True
                    break

            self.assertTrue(control_coincide)

    # tests if the control from OC computation coincides with a random input used for target forward-simulation
    # network case
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

                            fhn = FHNModel(Cmat=cmat, Dmat=dmat)

                            prec_mat = np.zeros((fhn.params.N, len(fhn.output_vars)))
                            control_mat = np.zeros((fhn.params.N, len(fhn.state_vars)))

                            control_mat[c_node, c_channel] = 1.0
                            prec_mat[p_node, p_channel] = 1.0

                            fhn.params.duration = duration
                            fhn.params.coupling = coupling

                            # change parameters for faster convergence
                            fhn.params.K_gl = 1.0

                            # high parameter value will cause numerical problems for certain settings
                            if p_channel == 1:
                                fhn.params.K_gl = 10.0
                                if coupling == "additive":
                                    if bi_dir_connectivity == 0:
                                        fhn.params.K_gl = 20.0
                                elif coupling == "diffusive":
                                    fhn.params.K_gl = 5.0
                            if c_channel == 1:
                                if bi_dir_connectivity == 0 and coupling == "additive":
                                    fhn.params.K_gl = 20.0
                                if p_channel == 1:
                                    fhn.params.duration = 0.5

                            zero_input = ZeroInput().generate_input(
                                duration=fhn.params.duration + fhn.params.dt, dt=fhn.params.dt
                            )
                            input = np.copy(zero_input)

                            rs = RandomState(MT19937(SeedSequence(0)))  # work with fixed seed for reproducibility

                            for t in range(1, input.shape[1] - 4):
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
                            lim = LIMIT_DIFF
                            if p_channel == 1 or c_channel == 1:
                                lim *= 4000  # for internal purposes: 1000, for simple testing: 4000

                            iterations = 4000
                            for i in range(100):
                                fhn_controlled.optimize(iterations)
                                control = fhn_controlled.control

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

        fhn = FHNModel(Cmat=cmat, Dmat=dmat)

        prec_mat = np.zeros((fhn.params.N, len(fhn.output_vars)))
        control_mat = np.zeros((fhn.params.N, len(fhn.state_vars)))

        control_mat[0, 0] = 1.0
        prec_mat[1, 0] = 1.0

        fhn.params.duration = duration

        # change parameters for faster convergence
        fhn.params.K_gl = 1.0
        # change parameters for shorter test simulation time
        fhn.params.signalV = 1.0

        zero_input = ZeroInput().generate_input(duration=fhn.params.duration + fhn.params.dt, dt=fhn.params.dt)
        input = np.copy(zero_input)

        for t in range(1, input.shape[1] - 7):  # leave last inputs zero so signal can be reproduced despite delay
            input[0, t] = rs.uniform(-a, a)

        fhn.params["x_ext"] = np.vstack([input, zero_input])
        fhn.params["y_ext"] = np.vstack([zero_input, zero_input])

        zeroinit = np.zeros((5))

        fhn.params["xs_init"] = np.vstack([zeroinit, zeroinit])
        fhn.params["ys_init"] = np.vstack([zeroinit, zeroinit])

        fhn.run()

        self.assertTrue(np.amax(fhn.params.Dmat_ndt) >= 1)  # Relates to the given "delay" and time-discretization.

        target = np.concatenate(
            (
                np.stack(
                    (fhn.params["xs_init"][:, -1], fhn.params["ys_init"][:, -1]),
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

        self.assertTrue((fhn.params.Dmat_ndt == fhn_controlled.Dmat_ndt).all())

        control_coincide = False

        iterations = 4000
        for i in range(100):
            fhn_controlled.optimize(iterations)
            control = fhn_controlled.control

            # last few entries of adjoint_state[0,0,:] are zero
            self.assertTrue(
                np.amax(
                    np.abs(fhn_controlled.adjoint_state[0, 0, -np.around(np.amax(fhn.params.Dmat_ndt)).astype(int) :])
                )
                == 0.0
            )

            c_diff_max = np.amax(np.abs(control[0, 0, :] - input[0, :]))
            if c_diff_max < LIMIT_DIFF:
                control_coincide = True
                break

        self.assertTrue(control_coincide)

    # test whether the control matrix restricts the computed control output
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

                zero_input = ZeroInput().generate_input(duration=duration + fhn.params.dt, dt=fhn.params.dt)
                input = np.copy(zero_input)

                rs = RandomState(MT19937(SeedSequence(0)))  # work with fixed seed for reproducibility

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

        fhn_controlled = oc_fhn.OcFhn(fhn, target, w_p=0, w_2=1)
        control_is_zero = False

        for i in range(100):
            fhn_controlled.optimize(1000)
            control = fhn_controlled.control

            c_max = np.amax(np.abs(control))
            if c_max < LIMIT_DIFF:
                control_is_zero = True
                break

        self.assertTrue(control_is_zero)

    # tests if the OC computation returns zero control when w_p = 0
    # 3-node network case
    def test_3n_wp0(self):
        print("Test OC for w_p = 0 in single-node model")

        N = 3
        dmat = np.zeros((N, N))  # no delay
        cmat = np.array([[0.0, 1.0, 0.9], [0.8, 0.0, 1.0], [0.0, 1.0, 0.0]])

        fhn = FHNModel(Cmat=cmat, Dmat=dmat)

        duration = 3.0
        a = 10.0

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

        fhn_controlled = oc_fhn.OcFhn(fhn, target, w_p=0, w_2=1)
        control_is_zero = False

        for i in range(100):
            fhn_controlled.optimize(1000)
            control = fhn_controlled.control

            c_max = np.amax(np.abs(control))
            if c_max < LIMIT_DIFF:
                control_is_zero = True
                break

        self.assertTrue(control_is_zero)

    def test_u_max_no_optimizations(self):
        # Arbitrary network and control setting, initial control violates the maximum absolute criterion.
        cmat = np.array([[0.0, 1.0], [1.0, 0.0]])
        dmat = np.array([[0.0, 0.0], [0.0, 0.0]])  # no delay
        fhn = FHNModel(Cmat=cmat, Dmat=dmat)
        duration = 1.0
        fhn.params.duration = duration

        zero_input = ZeroInput().generate_input(duration=duration + fhn.params.dt, dt=fhn.params.dt)
        input = np.copy(zero_input)

        for t in range(input.shape[1]):
            input[0, t] = np.sin(t)

        fhn.params["y_ext"] = np.vstack([input, input])
        fhn.params["x_ext"] = np.vstack([-input, 1.1 * input])

        fhn.params["xs_init"] = np.vstack([0.0, 0.0])
        fhn.params["ys_init"] = np.vstack([0.0, 0.0])

        precision_mat = np.ones((fhn.params.N, len(fhn.state_vars)))
        control_mat = np.ones((fhn.params.N, len(fhn.state_vars)))
        target = np.ones((2, 2, input.shape[1]))

        maximum_control_strength = 0.5

        fhn_controlled = oc_fhn.OcFhn(
            fhn,
            target,
            w_p=1,
            w_2=1,
            maximum_control_strength=maximum_control_strength,
            precision_matrix=precision_mat,
            control_matrix=control_mat,
        )

        self.assertTrue(np.max(np.abs(fhn_controlled.control) <= maximum_control_strength))

    def test_u_max_after_optimizations(self):
        # Arbitrary network and control setting, initial control violates the maximum absolute criterion.
        # Do one optimization step.
        cmat = np.array([[0.0, 1.0], [1.0, 0.0]])
        dmat = np.array([[0.0, 0.0], [0.0, 0.0]])  # no delay
        fhn = FHNModel(Cmat=cmat, Dmat=dmat)
        duration = 1.0
        fhn.params.duration = duration

        zero_input = ZeroInput().generate_input(duration=duration + fhn.params.dt, dt=fhn.params.dt)
        input = np.copy(zero_input)

        for t in range(input.shape[1]):
            input[0, t] = np.sin(t)

        fhn.params["y_ext"] = np.vstack([input, input])
        fhn.params["x_ext"] = np.vstack([-input, 1.1 * input])

        fhn.params["xs_init"] = np.vstack([0.0, 0.0])
        fhn.params["ys_init"] = np.vstack([0.0, 0.0])

        precision_mat = np.ones((fhn.params.N, len(fhn.state_vars)))
        control_mat = np.ones((fhn.params.N, len(fhn.state_vars)))
        target = np.ones((2, 2, input.shape[1]))

        maximum_control_strength = 0.5

        fhn_controlled = oc_fhn.OcFhn(
            fhn,
            target,
            w_p=1,
            w_2=1,
            maximum_control_strength=maximum_control_strength,
            precision_matrix=precision_mat,
            control_matrix=control_mat,
        )

        fhn_controlled.optimize(1)
        self.assertTrue(np.max(np.abs(fhn_controlled.control) <= maximum_control_strength))

    def test_get_xs(self):
        # Arbitrary network and control setting, get_xs() returns correct array shape (despite initial values array longer than 1)
        # Do one optimization step.

        cmat = np.array([[0.0, 1.0], [1.0, 0.0]])
        dmat = np.array([[0.0, 0.0], [0.0, 0.0]])  # no delay
        fhn = FHNModel(Cmat=cmat, Dmat=dmat)
        duration = 1.0
        fhn.params.duration = duration

        zero_input = ZeroInput().generate_input(duration=duration + fhn.params.dt, dt=fhn.params.dt)
        input = np.copy(zero_input)

        for t in range(input.shape[1]):
            input[0, t] = np.sin(t)

        fhn.params["y_ext"] = np.vstack([input, input])
        fhn.params["x_ext"] = np.vstack([-input, 1.1 * input])

        initind = 5

        zeroinit = np.zeros((initind))

        fhn.params["xs_init"] = np.vstack([zeroinit, zeroinit])
        fhn.params["ys_init"] = np.vstack([zeroinit, zeroinit])

        target = np.ones((2, 2, input.shape[1]))

        fhn_controlled = oc_fhn.OcFhn(
            fhn,
            target,
            w_p=1,
            w_2=1,
        )

        fhn_controlled.optimize(1)
        xs = fhn_controlled.get_xs()
        self.assertTrue(xs.shape == target.shape)


if __name__ == "__main__":
    unittest.main()
