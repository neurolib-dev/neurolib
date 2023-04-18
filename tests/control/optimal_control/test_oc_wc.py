import unittest

import numpy as np

from neurolib.models.wc import WCModel
from neurolib.utils.stimulus import ZeroInput
from neurolib.control.optimal_control import oc_wc
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
        model = WCModel()

        duration = 2.0
        a = 1.0

        zero_input = ZeroInput().generate_input(duration=duration + model.params.dt, dt=model.params.dt)
        input = np.copy(zero_input)
        inp_init = np.copy(zero_input)

        rs = RandomState(MT19937(SeedSequence(0)))  # work with fixed seed for reproducibility

        for t in range(1, input.shape[1] - 2):
            input[0, t] = rs.uniform(-a, a)
            inp_init[0, t] = input[0, t] + 1e-2 * rs.uniform(-a, a)

        for input_channel in [0, 1]:

            cost_mat = np.zeros((model.params.N, len(model.output_vars)))
            control_mat = np.zeros((model.params.N, len(model.state_vars)))
            if input_channel == 0:
                print("Input to E channel, measure in I channel")
                cost_mat[0, 1] = 1.0  # only measure in I-channel in one channel
                control_mat[0, 0] = 1.0  # only allow inputs to other channel
                model.params["exc_ext"] = input
                model.params["inh_ext"] = zero_input
            elif input_channel == 1:
                print("Input to I channel, measure in E channel")
                cost_mat[0, 0] = 1.0  # only measure in E-channel in one channel
                control_mat[0, 1] = 1.0  # only allow inputs to other channel
                model.params["exc_ext"] = zero_input
                model.params["inh_ext"] = input

            model.params["duration"] = duration
            model.params["exc_init"] = np.array([[0.0]])
            model.params["inh_init"] = np.array([[0.0]])

            model.run()
            target = np.concatenate(
                (
                    np.concatenate((model.params["exc_init"], model.params["inh_init"]), axis=1)[:, :, np.newaxis],
                    np.stack((model.exc, model.inh), axis=1),
                ),
                axis=2,
            )
            control_init = np.zeros((target.shape))
            control_init[0, input_channel, :] = inp_init[0, :]

            model.params["inh_ext"] = zero_input
            model.params["exc_ext"] = zero_input

            model_controlled = oc_wc.OcWc(model, target)

            control_coincide = False
            iterations = 5000

            model_controlled.control = control_init.copy()

            for i in range(100):
                model_controlled.optimize(iterations)
                control = model_controlled.control

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

                            model = WCModel(Cmat=cmat, Dmat=dmat)

                            model.params.K_gl = 10.0

                            model.duration = 1000.0
                            model.run()
                            e0, e1 = model.exc[0, -1], model.exc[1, -1]
                            i0, i1 = model.inh[0, -1], model.inh[1, -1]

                            cost_mat = np.zeros((model.params.N, len(model.output_vars)))
                            control_mat = np.zeros((model.params.N, len(model.state_vars)))

                            control_mat[c_node, c_channel] = 1.0
                            cost_mat[p_node, p_channel] = 1.0

                            model.params.duration = duration
                            model.params.coupling = coupling

                            zero_input = ZeroInput().generate_input(
                                duration=model.params.duration + model.params.dt, dt=model.params.dt
                            )
                            input = np.copy(zero_input)
                            input_optimization_start = np.copy(zero_input)

                            rs = RandomState(MT19937(SeedSequence(0)))  # work with fixed seed for reproducibility

                            for t in range(0, input.shape[1] - 4):
                                input[0, t] = rs.uniform(-a, a)
                                input_optimization_start[0, t] = input[0, t] + 1e-3 * rs.uniform(-a, a)

                            model.params["inh_ext"] = np.vstack([zero_input, zero_input])
                            model.params["exc_ext"] = np.vstack([zero_input, zero_input])

                            if c_channel == 0:
                                if c_node == 0:
                                    model.params["exc_ext"] = np.vstack([input, zero_input])
                                else:
                                    model.params["exc_ext"] = np.vstack([zero_input, input])
                            else:
                                if c_node == 0:
                                    model.params["inh_ext"] = np.vstack([input, zero_input])
                                else:
                                    model.params["inh_ext"] = np.vstack([zero_input, input])

                            model.params["exc_init"] = np.vstack([e0, e1])
                            model.params["inh_init"] = np.vstack([i0, i1])

                            model.run()

                            target = np.concatenate(
                                (
                                    np.concatenate(
                                        (model.params["exc_init"], model.params["inh_init"]),
                                        axis=1,
                                    )[:, :, np.newaxis],
                                    np.stack((model.exc, model.inh), axis=1),
                                ),
                                axis=2,
                            )
                            control_init = np.zeros((target.shape))
                            control_init[c_node, c_channel, :] = input_optimization_start[0, :]

                            model.params["exc_ext"] = np.vstack([zero_input, zero_input])
                            model.params["inh_ext"] = np.vstack([zero_input, zero_input])

                            model_controlled = oc_wc.OcWc(
                                model,
                                target,
                                control_matrix=control_mat,
                                cost_matrix=cost_mat,
                            )
                            model_controlled.maximum_control_strength = 2.0 * a

                            model_controlled.control = control_init.copy()

                            control_coincide = False
                            lim = LIMIT_DIFF
                            if p_channel == 1 or c_channel == 1:
                                lim *= 100  # for internal purposes: 1000, for simple testing: 4000
                                model_controlled.step = 1e10
                            if p_channel == 1 and c_channel == 1:
                                lim *= 1000
                                model_controlled.step = 1e20

                            iterations = 5000
                            for i in range(100):
                                model_controlled.optimize(iterations)
                                control = model_controlled.control

                                c_diff_max = np.amax(np.abs(control[c_node, c_channel, :] - input[0, :]))
                                if c_diff_max < lim:
                                    control_coincide = True
                                    break

                                if model_controlled.zero_step_encountered:
                                    break

                            self.assertTrue(control_coincide)

    # tests if the control from OC computation coincides with a random input used for target forward-simulation
    # delayed network case
    def test_twonode_delay_oc(self):
        print("Test OC in delayed 2-node network")

        rs = RandomState(MT19937(SeedSequence(0)))  # work with fixed seed for reproducibility

        duration = 0.8
        a = 5.0

        delay = rs.choice([0.1, 0.2, 0.3, 0.4])

        cmat = np.array([[0.0, 0.0], [1.0, 0.0]])
        dmat = np.array([[0.0, 0.0], [delay, 0.0]])

        model = WCModel(Cmat=cmat, Dmat=dmat)

        cost_mat = np.zeros((model.params.N, len(model.output_vars)))
        control_mat = np.zeros((model.params.N, len(model.state_vars)))

        control_mat[0, 0] = 1.0
        cost_mat[1, 0] = 1.0

        model.params.duration = duration

        # change parameters for faster convergence
        model.params.K_gl = 10.0
        # change parameters for shorter test simulation time
        model.params.signalV = 1.0

        zero_input = ZeroInput().generate_input(duration=model.params.duration + model.params.dt, dt=model.params.dt)
        input = np.copy(zero_input)

        for t in range(1, input.shape[1] - 6):  # leave last inputs zero so signal can be reproduced despite delay
            input[0, t] = rs.uniform(-a, a)

        model.params["exc_ext"] = np.vstack([input, zero_input])
        model.params["inh_ext"] = np.vstack([zero_input, zero_input])

        zeroinit = np.zeros((5))

        model.params["exc_init"] = np.vstack([zeroinit, zeroinit])
        model.params["inh_init"] = np.vstack([zeroinit, zeroinit])

        model.run()

        self.assertTrue(np.amax(model.params.Dmat_ndt) >= 1)  # Relates to the given "delay" and time-discretization.

        target = np.concatenate(
            (
                np.stack(
                    (model.params["exc_init"][:, -1], model.params["inh_init"][:, -1]),
                    axis=1,
                )[:, :, np.newaxis],
                np.stack((model.exc, model.inh), axis=1),
            ),
            axis=2,
        )

        model.params["exc_ext"] = np.vstack([zero_input, zero_input])
        model.params["inh_ext"] = np.vstack([zero_input, zero_input])

        model_controlled = oc_wc.OcWc(
            model,
            target,
            control_matrix=control_mat,
            cost_matrix=cost_mat,
        )

        self.assertTrue((model.params.Dmat_ndt == model_controlled.Dmat_ndt).all())

        control_coincide = False

        iterations = 4000
        for i in range(100):
            model_controlled.optimize(iterations)
            control = model_controlled.control

            # last few entries of adjoint_state[0,0,:] are zero
            self.assertTrue(
                np.amax(
                    np.abs(
                        model_controlled.adjoint_state[0, 0, -np.around(np.amax(model.params.Dmat_ndt)).astype(int) :]
                    )
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
        model = WCModel(Cmat=cmat, Dmat=dmat)
        duration = 1.0
        model.params.duration = duration

        zero_input = ZeroInput().generate_input(duration=duration + model.params.dt, dt=model.params.dt)
        input = np.copy(zero_input)

        for t in range(input.shape[1]):
            input[0, t] = np.sin(t)

        model.params["exc_ext"] = np.vstack([input, input])
        model.params["inh_ext"] = np.vstack([-input, 1.1 * input])

        initind = 5

        zeroinit = np.zeros((initind))

        model.params["exc_init"] = np.vstack([zeroinit, zeroinit])
        model.params["inh_init"] = np.vstack([zeroinit, zeroinit])

        target = np.ones((2, 2, input.shape[1]))

        model_controlled = oc_wc.OcWc(
            model,
            target,
        )

        model_controlled.optimize(1)
        xs = model_controlled.get_xs()
        self.assertTrue(xs.shape == target.shape)


if __name__ == "__main__":
    unittest.main()
