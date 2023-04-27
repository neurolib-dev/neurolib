import unittest
import numpy as np

from neurolib.models.aln import ALNModel
from neurolib.utils.stimulus import ZeroInput
from neurolib.control.optimal_control import oc_aln
from numpy.random import RandomState, SeedSequence, MT19937

global LIMIT_DIFF
LIMIT_DIFF = 1e-9


def getfinalstate(model):
    N = len(model.params.Cmat)
    V = len(model.state_vars)
    T = model.getMaxDelay() + 1
    state = np.zeros((N, V, T))
    for v in range(V):
        if "rates" in model.state_vars[v] or "IA" in model.state_vars[v]:
            state[:, v, :] = model.state[model.state_vars[v]][:, -T:]
        else:
            state[:, v, :] = model.state[model.state_vars[v]][:]
    return state


def setinitstate(model, state):
    N = len(model.params.Cmat)
    V = len(model.init_vars)
    T = model.getMaxDelay() + 1

    for n in range(N):
        for v in range(V):
            if "rates" in model.init_vars[v]:
                model.params[model.init_vars[v]] = state[:, v, -T:]
            elif "IA" in model.init_vars[v]:
                model.params[model.init_vars[v]] = state[:, v, -T:]
            else:
                model.params[model.init_vars[v]] = state[:, v, -1]

    return


class TestALN(unittest.TestCase):
    """
    Test wc in neurolib/optimal_control/
    """

    # tests if the control from OC computation coincides with a random input used for target forward-simulation
    # single-node case
    def test_onenode_oc(self):
        print("Test OC in single-node system")
        model = ALNModel()

        model.params.de = 0.2
        model.params.di = 0.1
        ndt_de = np.around(model.params.de / model.params.dt).astype(int)
        ndt_di = np.around(model.params.di / model.params.dt).astype(int)

        model.params.a = 0.0
        model.params.b = 0.0
        # intermediate external input to membrane voltage to not reach the boundaries of the transfer function
        model.params.mue_ext_mean = 2.0
        model.params.mui_ext_mean = 1.0
        model.params.sigma_ou = 0.0

        model.params.cei = 1.0
        model.params.cie = 1.0

        model.params.mue_ou = model.params.mue_ext_mean * np.ones((1,))
        model.params.mui_ou = model.params.mui_ext_mean * np.ones((1,))
        model.params.mufe_init = model.params.mue_ext_mean * np.ones((1,))
        model.params.mufi_init = model.params.mui_ext_mean * np.ones((1,))

        model.params["duration"] = max(10, 2 * model.getMaxDelay())
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

        # Test duration
        duration = 1.2 + max(model.params.de, model.params.di)
        a = 0.8  # amplitude

        zero_input = ZeroInput().generate_input(duration=duration + model.params.dt, dt=model.params.dt)
        input = np.copy(zero_input)
        inp_init = np.copy(zero_input)

        rs = RandomState(MT19937(SeedSequence(0)))  # work with fixed seed for reproducibility

        intinit, intend = 1, input.shape[1] - 5 - max(ndt_de, ndt_di)

        for t in range(intinit, intend):
            input[0, t] = rs.uniform(-a, a)
            inp_init[0, t] = input[0, t] + 1e-2 * a * rs.uniform(-a, a)

        for input_channel in [0, 1]:

            for measure_channel in [0, 1]:

                cost_mat = np.zeros((model.params.N, 2))
                control_mat = np.zeros((model.params.N, 2))
                if input_channel == 0:
                    model.params["ext_exc_current"] = input
                    model.params["ext_inh_current"] = zero_input
                    control_mat[0, 0] = 1.0  # only allow inputs to input channel
                    if measure_channel == 1:
                        print("------------ Input to E channel, measure in I channel")
                        cost_mat[0, 1] = 1.0  # only measure in I-channel in measure channel
                    elif measure_channel == 0:
                        print("------------ Input to E channel, measure in E channel")
                        cost_mat[0, 0] = 1.0

                elif input_channel == 1:
                    model.params["ext_exc_current"] = zero_input
                    model.params["ext_inh_current"] = input
                    control_mat[0, 1] = 1.0
                    if measure_channel == 1:
                        print("------------ Input to I channel, measure in I channel")
                        cost_mat[0, 1] = 1.0
                    elif measure_channel == 0:
                        print("------------ Input to I channel, measure in E channel")
                        cost_mat[0, 0] = 1.0

                model.params["duration"] = duration
                model.run()

                target = np.concatenate(
                    (
                        np.concatenate(
                            (
                                model.params["rates_exc_init"][:, np.newaxis, -1],
                                model.params["rates_inh_init"][:, np.newaxis, -1],
                            ),
                            axis=1,
                        )[:, :, np.newaxis],
                        np.stack((model.rates_exc, model.rates_inh), axis=1),
                    ),
                    axis=2,
                )
                control_init = np.zeros((target.shape))
                control_init[0, input_channel, :] = inp_init[0, :]

                model.params["ext_exc_current"] = zero_input
                model.params["ext_inh_current"] = zero_input

                model_controlled = oc_aln.OcAln(model, target, control_matrix=control_mat, cost_matrix=cost_mat)

                control_coincide = False
                iterations = 10000

                model_controlled.control = control_init.copy()

                for i in range(100):
                    model_controlled.optimize(iterations)
                    control = model_controlled.control

                    c_diff = np.abs(control[0, input_channel, intinit:intend] - input[0, intinit:intend])

                    if np.amax(c_diff) < LIMIT_DIFF:
                        control_coincide = True
                        break

                    if input_channel != measure_channel:
                        if np.amax(c_diff) < 100 * LIMIT_DIFF:
                            control_coincide = True
                            break

                    if model_controlled.zero_step_encountered:
                        print(np.amax(c_diff), c_diff)
                        break

                self.assertTrue(control_coincide)

    # Arbitrary network and control setting, get_xs() returns correct array shape (despite initial values array longer than 1)
    def test_get_xs(self):
        print("Test state shape agrees with target shape")

        cmat = np.array([[0.0, 1.0], [1.0, 0.0]])
        dmat = np.array([[0.0, 0.0], [0.0, 0.0]])  # no delay
        model = ALNModel(Cmat=cmat, Dmat=dmat)
        duration = 1.0
        model.params.duration = duration

        zero_input = ZeroInput().generate_input(duration=duration + model.params.dt, dt=model.params.dt)
        input = np.copy(zero_input)

        for t in range(input.shape[1]):
            input[0, t] = np.sin(t)

        model.params["ext_exc_current"] = np.vstack([input, input])
        model.params["ext_inh_current"] = np.vstack([-input, 1.1 * input])

        initind = 50

        zeroinit = np.zeros((initind))

        model.params["rates_exc_init"] = np.vstack([zeroinit, zeroinit])
        model.params["rates_inh_init"] = np.vstack([zeroinit, zeroinit])

        target = np.ones((2, 2, input.shape[1]))

        model_controlled = oc_aln.OcAln(
            model,
            target,
        )

        model_controlled.optimize(1)
        xs = model_controlled.get_xs()
        self.assertTrue(xs.shape == target.shape)


if __name__ == "__main__":
    unittest.main()
