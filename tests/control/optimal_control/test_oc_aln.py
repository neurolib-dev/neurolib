import unittest
import numpy as np

from neurolib.models.aln import ALNModel
from neurolib.control.optimal_control import oc_aln

import test_oc_utils as test_oc_utils

p = test_oc_utils.params


def set_param_init(model, a=15.0, b=40.0):
    # intermediate external input to membrane voltage to not reach the boundaries of the transfer function
    model.params.mue_ext_mean = 2.0
    model.params.mui_ext_mean = 0.5

    # no noise
    model.params.sigma_ou = 0.0

    # adaptation parameters
    model.params.a = a
    model.params.b = b
    model.params.tauA = 1.0

    for iv in model.init_vars:
        if "rates" in iv or "IA" in iv:
            model.params[iv] = np.zeros((model.params.N, model.getMaxDelay() + 1))
        else:
            model.params[iv] = np.zeros((model.params.N,))

    model.params.mue_ou = model.params.mue_ext_mean * np.ones((model.params.N,))
    model.params.mui_ou = model.params.mui_ext_mean * np.ones((model.params.N,))

    model.params["duration"] = max(1000, 2 * model.getMaxDelay())
    model.run()

    # initial state must not be random because delay computation requires history
    setinitstate(model, getfinalstate(model))
    if (
        model.params.rates_exc_init[0, -1] < 5
        or model.params.rates_inh_init[0, -1] < 5
        or model.params.rates_exc_init[0, -1] > 150
        or model.params.rates_inh_init[0, -1] > 150
    ):
        print("WARNING------------------------")
        print("Rates might be out of table range")
        print(model.params.rates_exc_init[0, -1], model.params.rates_inh_init[0, -1])

    return


def getfinalstate(model):
    N = len(model.params.Cmat)
    V = len(model.state_vars)
    T = model.getMaxDelay() + 1
    state = np.zeros((N, V, T))
    for v in range(V):
        if "rates" in model.state_vars[v] or "IA" in model.state_vars[v]:
            for n in range(N):
                state[n, v, :] = model.state[model.state_vars[v]][n, -T:]
        else:
            for n in range(N):
                state[n, v, :] = model.state[model.state_vars[v]][n]
    return state


def setinitstate(model, state):
    N = len(model.params.Cmat)
    V = len(model.init_vars)
    T = model.getMaxDelay() + 1

    for n in range(N):
        for v in range(V):
            if "rates" in model.init_vars[v] or "IA" in model.init_vars[v]:
                model.params[model.init_vars[v]] = state[:, v, -T:]
            else:
                model.params[model.init_vars[v]] = state[:, v, -1]

    return


def getstate(model):
    return np.concatenate(
        (
            np.concatenate(
                (
                    model.params["rates_exc_init"][:, np.newaxis, -1],
                    model.params["rates_inh_init"][:, np.newaxis, -1],
                    model.params["IA_init"][:, np.newaxis, -1],
                ),
                axis=1,
            )[:, :, np.newaxis],
            np.stack((model.rates_exc, model.rates_inh, model.IA), axis=1),
        ),
        axis=2,
    )


class TestALN(unittest.TestCase):
    """
    Test wc in neurolib/optimal_control/
    """

    # single-node case with delay and adaptation
    def test_1n(self):
        print("Test OC in single-node system with delay")
        model = ALNModel()

        model.params.de = p.TEST_DELAY
        model.params.di = 1.5 * p.TEST_DELAY

        set_param_init(model)

        model.params.duration = p.TEST_DURATION_12

        for input_channel in range(len(model.input_vars)):
            for measure_channel in [0, 1]:
                if input_channel == 2 and measure_channel == 1:
                    continue
                if input_channel == 3 and measure_channel == 0:
                    continue

                print("----------------- input channel, measure channel = ", input_channel, measure_channel)

                cost_mat = np.zeros((model.params.N, len(model.output_vars)))
                control_mat = np.zeros((model.params.N, len(model.input_vars)))
                cost_mat[0, measure_channel] = 1.0
                control_mat[0, input_channel] = 1.0

                factor = 1.0
                if input_channel in [2, 3]:
                    factor = 1e-1

                test_oc_utils.set_input(model, p.ZERO_INPUT_1N_12)
                model.params[model.input_vars[input_channel]] = p.TEST_INPUT_1N_12 * factor
                model.run()
                target = getstate(model)

                test_oc_utils.set_input(model, p.ZERO_INPUT_1N_12)
                model.run()

                model_controlled = oc_aln.OcAln(model, target, control_matrix=control_mat, cost_matrix=cost_mat)

                control_init = np.zeros((model.params.N, len(model.input_vars), target.shape[2]))
                control_init[0, input_channel, :] = p.INIT_INPUT_1N_12[0, :] * factor
                model_controlled.control = control_init.copy()
                model_controlled.update_input()

                control_coincide = False

                for i in range(p.LOOPS):
                    model_controlled.optimize(p.ITERATIONS)
                    control = model_controlled.control

                    c_diff = np.abs(control[0, input_channel, :] - p.TEST_INPUT_1N_12[0, :] * factor)

                    if np.amax(c_diff) < p.LIMIT_DIFF:
                        control_coincide = True
                        break

                    if model_controlled.zero_step_encountered:
                        break

                self.assertTrue(control_coincide)

    # tests if the control from OC computation coincides with a random input used for target forward-simulation
    # network case with delay and adaptation
    def test_2n(self):
        print("Test OC in 2-node network")

        dmat = np.array([[0.0, p.TEST_DELAY], [p.TEST_DELAY, 0.0]])
        cmat = np.array([[0.0, 1.0], [1.0, 0.0]])

        model = ALNModel(Cmat=cmat, Dmat=dmat)
        model.params.de = 0.0
        model.params.di = 0.0

        set_param_init(model)
        model.params.duration = p.TEST_DURATION_12

        for input_channel in [0, 2]:
            print("----------------- input channel = ", input_channel)

            cost_mat = np.zeros((model.params.N, len(model.output_vars)))
            control_mat = np.zeros((model.params.N, len(model.input_vars)))
            cost_mat[1, 0] = 1.0
            control_mat[0, input_channel] = 1.0

            factor = 1.0
            if input_channel in [2, 3]:
                factor = 1e-1

            test_oc_utils.set_input(model, p.ZERO_INPUT_2N_12)
            model.params[model.input_vars[input_channel]] = p.TEST_INPUT_2N_12 * factor
            model.run()
            target = getstate(model)

            test_oc_utils.set_input(model, p.ZERO_INPUT_2N_12)
            model.run()

            model_controlled = oc_aln.OcAln(
                model,
                target,
                control_matrix=control_mat,
                cost_matrix=cost_mat,
            )

            control_init = np.zeros((model.params.N, len(model.input_vars), target.shape[2]))
            control_init[0, input_channel, :] = p.INIT_INPUT_2N_12[0, :] * factor
            model_controlled.control = control_init.copy()
            model_controlled.update_input()

            control_coincide = False

            for i in range(p.LOOPS):
                model_controlled.optimize(p.ITERATIONS)
                control = model_controlled.control
                c_diff = np.abs(control[0, input_channel, :] - p.TEST_INPUT_2N_12[0, :] * factor)

                if np.amax(c_diff) < p.LIMIT_DIFF:
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
        model = ALNModel(Cmat=cmat, Dmat=dmat)
        model.params.duration = p.TEST_DURATION_6

        test_oc_utils.set_input(model, p.TEST_INPUT_2N_6)

        target = np.ones((model.params.N, len(model.output_vars), p.TEST_INPUT_2N_6.shape[1]))

        model_controlled = oc_aln.OcAln(
            model,
            target,
        )

        model_controlled.optimize(1)
        xs = model_controlled.get_xs()
        self.assertTrue(xs.shape == target.shape)

    def test_adjust_init(self):
        print("Test adjust_init function of OC class")

        cmat = np.array([[0.0, 1.0], [1.0, 0.0]])
        dmat = np.array([[0.0, 10.0], [10.0, 0.0]])  # large delay
        model = ALNModel(Cmat=cmat, Dmat=dmat)
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
                if init_var[:-5] not in model.output_vars:
                    continue
                model.params[init_var] = test_init
                model_controlled = oc_aln.OcAln(
                    model,
                    target,
                )

                for init_var0 in model.init_vars:
                    if "ou" in init_var0:
                        continue
                    if init_var0[:-5] not in model.output_vars:
                        continue
                    self.assertTrue(model_controlled.model.params[init_var0].shape == targetinitshape)

    def test_adjust_input(self):
        print("Test test_adjust_input function of OC class")

        cmat = np.array([[0.0, 1.0], [1.0, 0.0]])
        dmat = np.array([[0.0, 0.0], [0.0, 0.0]])  # no delay
        model = ALNModel(Cmat=cmat, Dmat=dmat)
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
                model_controlled = oc_aln.OcAln(
                    model,
                    target,
                )

                for input_var0 in model.input_vars:
                    self.assertTrue(model_controlled.model.params[input_var0].shape == targetinputshape)


if __name__ == "__main__":
    unittest.main()

    # # tests if the control from OC computation coincides with input used for target forward-simulation
    # # single-node case
    # def test_1n(self):
    #     print("Test OC in single-node system")
    #     model = ALNModel()

    #     # no delay
    #     model.params.de = 0.0
    #     model.params.di = 0.0

    #     # no adaptation
    #     set_param_init(model, 0.0, 0.0)
    #     model.params.duration = p.TEST_DURATION_10

    #     for input_channel in range(len(model.input_vars)):
    #         for measure_channel in [0, 1]:
    #             if input_channel == 2 and measure_channel == 1:
    #                 continue
    #             if input_channel == 3 and measure_channel == 0:
    #                 continue
    #             if input_channel == 0 and measure_channel == 0:
    #                 continue  # is tested in adaptation scenario
    #             print("----------------- input channel, measure channel = ", input_channel, measure_channel)

    #             cost_mat = np.zeros((model.params.N, len(model.output_vars)))
    #             control_mat = np.zeros((model.params.N, len(model.input_vars)))
    #             cost_mat[0, measure_channel] = 1.0
    #             control_mat[0, input_channel] = 1.0

    #             factor = 1.0
    #             if input_channel in [2, 3]:
    #                 factor = 1e-1

    #             test_oc_utils.set_input(model, p.ZERO_INPUT_1N_10)
    #             model.params[model.input_vars[input_channel]] = p.TEST_INPUT_1N_10 * factor

    #             model.run()
    #             target = getstate(model)

    #             test_oc_utils.set_input(model, p.ZERO_INPUT_1N_10)
    #             model.run()

    #             model_controlled = oc_aln.OcAln(model, target, control_matrix=control_mat, cost_matrix=cost_mat)

    #             control_init = np.zeros((model.params.N, len(model.input_vars), target.shape[2]))
    #             control_init[0, input_channel, :] = p.INIT_INPUT_1N_10[0, :] * factor
    #             model_controlled.control = control_init.copy()
    #             model_controlled.update_input()

    #             control_coincide = False

    #             for i in range(p.LOOPS):
    #                 model_controlled.optimize(p.ITERATIONS)
    #                 control = model_controlled.control

    #                 c_diff = np.abs(control[0, input_channel, :] - p.TEST_INPUT_1N_10[0, :] * factor)

    #                 if np.amax(c_diff) < p.LIMIT_DIFF:
    #                     control_coincide = True
    #                     break

    #                 if model_controlled.zero_step_encountered:
    #                     break

    #             self.assertTrue(control_coincide)

    # # tests if the control from OC computation coincides with input used for target forward-simulation
    # # single-node case with adaptation
    # def test_1n_adaptation(self):
    #     print("Test OC in single-node system")
    #     model = ALNModel()

    #     model.params.de = 0.0
    #     model.params.di = 0.0

    #     # adaptation directly only affects exc population, testing is sufficient for E => E control
    #     cost_mat = np.zeros((model.params.N, len(model.output_vars)))
    #     control_mat = np.zeros((model.params.N, len(model.input_vars)))
    #     cost_mat[0, 0] = 1.0
    #     control_mat[0, 0] = 1.0

    #     test_oc_utils.set_input(model, p.ZERO_INPUT_1N_10)

    #     for [a, b] in [10.0, 10.0]:
    #         print("adaptation parameters a, b = ", a, b)
    #         set_param_init(model, a, b)
    #         model.params.duration = p.TEST_DURATION_10

    #         model.params["ext_exc_current"] = p.TEST_INPUT_1N_10
    #         model.run()
    #         target = getstate(model)
    #         test_oc_utils.set_input(model, p.ZERO_INPUT_1N_10)
    #         model.run()

    #         model_controlled = oc_aln.OcAln(model, target, control_matrix=control_mat, cost_matrix=cost_mat)

    #         control_init = np.zeros((model.params.N, len(model.input_vars), target.shape[2]))
    #         control_init[0, 0, :] = p.INIT_INPUT_1N_10[0, :]
    #         model_controlled.control = control_init.copy()
    #         model_controlled.update_input()

    #         control_coincide = False

    #         for i in range(p.LOOPS):
    #             model_controlled.optimize(p.ITERATIONS)
    #             control = model_controlled.control

    #             c_diff = np.abs(control[0, 0, :] - p.TEST_INPUT_1N_10[0, :])

    #             if np.amax(c_diff) < p.LIMIT_DIFF:
    #                 control_coincide = True
    #                 break

    #             if model_controlled.zero_step_encountered:
    #                 break

    #         self.assertTrue(control_coincide)
