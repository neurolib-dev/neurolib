import unittest
import numpy as np

from neurolib.models.aln import ALNModel
from neurolib.utils.stimulus import ZeroInput
from neurolib.control.optimal_control import oc_aln
from numpy.random import RandomState, SeedSequence, MT19937

global LIMIT_DIFF, ADAP_PARAM_LIST, ITERATIONS, LOOPS
LIMIT_DIFF = 1e-5
ADAP_PARAM_LIST = [[10.0, 0.0], [0.0, 10.0], [10.0, 10.0]]
ITERATIONS = 5000
LOOPS = 100


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

    # tests if the control from OC computation coincides with a random input used for target forward-simulation
    # single-node case
    def test_onenode_oc(self):
        print("Test OC in single-node system")
        model = ALNModel()

        model.params.de = 0.0
        model.params.di = 0.0

        # Test duration
        duration = 1.0 + max(model.params.de, model.params.di)
        amplitude = 1.0  # amplitude

        zero_input = ZeroInput().generate_input(duration=duration + model.params.dt, dt=model.params.dt)
        input = np.copy(zero_input)
        inp_init = np.copy(zero_input)

        intinit, intend = 1, input.shape[1] - 5

        rs = RandomState(MT19937(SeedSequence(0)))  # work with fixed seed for reproducibility

        for [a, b] in ADAP_PARAM_LIST:
            print("adaptation parameters a, b = ", a, b)
            set_param_init(model, a, b)

            for t in range(intinit, intend):
                input[0, t] = rs.uniform(-amplitude, amplitude)
                inp_init[0, t] = input[0, t] + 1e-2 * amplitude * rs.uniform(-amplitude, amplitude)

            for input_channel in [0, 1]:

                for measure_channel in [0, 1]:

                    print("----------------- input channel, measure channel = ", input_channel, measure_channel)

                    cost_mat = np.zeros((model.params.N, len(model.output_vars)))
                    control_mat = np.zeros((model.params.N, len(model.input_vars)))
                    if input_channel == 0:
                        model.params["ext_exc_current"] = input
                        model.params["ext_inh_current"] = zero_input

                    elif input_channel == 1:
                        model.params["ext_exc_current"] = zero_input
                        model.params["ext_inh_current"] = input

                    cost_mat[0, measure_channel] = 1.0
                    control_mat[0, input_channel] = 1.0

                    model.params["duration"] = duration
                    model.run()
                    target = getstate(model)

                    control_init = np.zeros((target.shape))
                    control_init[0, input_channel, :] = inp_init[0, :]

                    model.params["ext_exc_current"] = zero_input
                    model.params["ext_inh_current"] = zero_input
                    model.run()

                    model_controlled = oc_aln.OcAln(model, target, control_matrix=control_mat, cost_matrix=cost_mat)

                    control_coincide = False

                    model_controlled.control = control_init.copy()
                    model_controlled.update_input()

                    for i in range(LOOPS):
                        model_controlled.optimize(ITERATIONS)
                        control = model_controlled.control

                        c_diff = np.abs(control[0, input_channel, intinit:intend] - input[0, intinit:intend])

                        if np.amax(c_diff) < LIMIT_DIFF:
                            control_coincide = True
                            break

                        if input_channel != measure_channel:
                            if np.amax(c_diff) < 1e2 * LIMIT_DIFF:
                                control_coincide = True
                                break

                        if model_controlled.zero_step_encountered:
                            break

                    self.assertTrue(control_coincide)

    # single-node case with delay
    def test_onenode_oc_delay(self):
        print("Test OC in single-node system with delay")
        model = ALNModel()

        rs = RandomState(MT19937(SeedSequence(0)))  # work with fixed seed for reproducibility

        model.params.de = rs.choice([0.1, 0.2, 0.3, 0.4])
        model.params.di = rs.choice([0.1, 0.2, 0.3, 0.4])
        ndt_de = np.around(model.params.de / model.params.dt).astype(int)
        ndt_di = np.around(model.params.di / model.params.dt).astype(int)

        # Test duration
        duration = 1.0 + max(model.params.de, model.params.di)
        amplitude = 1.0  # amplitude

        zero_input = ZeroInput().generate_input(duration=duration + model.params.dt, dt=model.params.dt)
        input = np.copy(zero_input)
        inp_init = np.copy(zero_input)

        intinit, intend = 1, input.shape[1] - 5 - max(ndt_de, ndt_di)

        set_param_init(model)

        for t in range(intinit, intend):
            input[0, t] = rs.uniform(-amplitude, amplitude)
            inp_init[0, t] = input[0, t] + 1e-2 * amplitude * rs.uniform(-amplitude, amplitude)

        for input_channel in [0, 1]:

            for measure_channel in [0, 1]:

                print("----------------- input channel, measure channel = ", input_channel, measure_channel)

                cost_mat = np.zeros((model.params.N, len(model.output_vars)))
                control_mat = np.zeros((model.params.N, len(model.input_vars)))
                if input_channel == 0:
                    model.params["ext_exc_current"] = input
                    model.params["ext_inh_current"] = zero_input

                elif input_channel == 1:
                    model.params["ext_exc_current"] = zero_input
                    model.params["ext_inh_current"] = input

                cost_mat[0, measure_channel] = 1.0
                control_mat[0, input_channel] = 1.0

                model.params["duration"] = duration
                model.run()
                target = getstate(model)

                control_init = np.zeros((target.shape))
                control_init[0, input_channel, :] = inp_init[0, :]

                model.params["ext_exc_current"] = zero_input
                model.params["ext_inh_current"] = zero_input
                model.run()

                model_controlled = oc_aln.OcAln(model, target, control_matrix=control_mat, cost_matrix=cost_mat)

                control_coincide = False

                model_controlled.control = control_init.copy()
                model_controlled.update_input()

                for i in range(LOOPS):
                    model_controlled.optimize(ITERATIONS)
                    control = model_controlled.control

                    c_diff = np.abs(control[0, input_channel, intinit:intend] - input[0, intinit:intend])

                    if np.amax(c_diff) < LIMIT_DIFF:
                        control_coincide = True
                        break

                    if input_channel != measure_channel:
                        if np.amax(c_diff) < 1e2 * LIMIT_DIFF:
                            control_coincide = True
                            break

                    if model_controlled.zero_step_encountered:
                        break

                self.assertTrue(control_coincide)

    # tests if the control from OC computation coincides with a random input used for target forward-simulation
    # network case
    def test_twonode_oc(self):
        print("Test OC in 2-node network")

        dmat = np.array([[0.0, 0.0], [0.0, 0.0]])  # no delay
        duration = 1.0
        amplitude = 1.0

        c_node = 0
        p_node = np.abs(c_node - 1).astype(int)

        # numerical values too small to reasonably test if control_channel = 1 or measure_channel = 1
        control_channel = 0
        measure_channel = 0

        for bi_dir_connectivity in [0, 1]:
            print("bidirectional connectivity = ", bi_dir_connectivity)

            if bi_dir_connectivity == 0:
                if c_node == 0:
                    cmat = np.array([[0.0, 0.0], [1.0, 0.0]])
                else:
                    cmat = np.array([[0.0, 1.0], [0.0, 0.0]])
            else:
                cmat = np.array([[0.0, 1.0], [1.0, 0.0]])

            for [a, b] in ADAP_PARAM_LIST:
                print("adaptation parameters a, b = ", a, b)

                model = ALNModel(Cmat=cmat, Dmat=dmat)

                model.params.de = 0.0
                model.params.di = 0.0

                set_param_init(model, a, b)

                cost_mat = np.zeros((model.params.N, len(model.output_vars)))
                control_mat = np.zeros((model.params.N, len(model.state_vars)))

                control_mat[c_node, control_channel] = 1.0
                cost_mat[p_node, measure_channel] = 1.0

                model.params.duration = duration

                zero_input = ZeroInput().generate_input(
                    duration=model.params.duration + model.params.dt, dt=model.params.dt
                )
                input = np.copy(zero_input)
                input_optimization_start = np.copy(zero_input)

                rs = RandomState(MT19937(SeedSequence(0)))  # work with fixed seed for reproducibility

                intinit, intend = 1, input.shape[1] - 6

                for t in range(intinit, intend):
                    input[0, t] = rs.uniform(-amplitude, amplitude)
                    input_optimization_start[0, t] = input[0, t] + 1e-2 * rs.uniform(-amplitude, amplitude)

                model.params["ext_inh_current"] = np.vstack([zero_input, zero_input])
                model.params["ext_exc_current"] = np.vstack([zero_input, zero_input])

                if control_channel == 0:
                    if c_node == 0:
                        model.params["ext_exc_current"] = np.vstack([input, zero_input])
                    else:
                        model.params["ext_exc_current"] = np.vstack([zero_input, input])
                else:
                    if c_node == 0:
                        model.params["ext_inh_current"] = np.vstack([input, zero_input])
                    else:
                        model.params["ext_inh_current"] = np.vstack([zero_input, input])

                model.run()

                target = getstate(model)
                control_init = np.zeros((target.shape))
                control_init[c_node, control_channel, :] = input_optimization_start[0, :]

                model.params["ext_inh_current"] = np.vstack([zero_input, zero_input])
                model.params["ext_exc_current"] = np.vstack([zero_input, zero_input])

                model_controlled = oc_aln.OcAln(
                    model,
                    target,
                    control_matrix=control_mat,
                    cost_matrix=cost_mat,
                )

                model_controlled.control = control_init.copy()
                model_controlled.update_input()

                control_coincide = False

                for i in range(LOOPS):
                    model_controlled.optimize(ITERATIONS)
                    control = model_controlled.control
                    c_diff = np.abs(control[c_node, control_channel, intinit:intend] - input[0, intinit:intend])
                    print(c_diff)

                    if np.amax(c_diff) < LIMIT_DIFF:
                        control_coincide = True
                        break

                    if control_channel != measure_channel:
                        if np.amax(c_diff) < 1e3 * LIMIT_DIFF:
                            control_coincide = True
                            break

                    if model_controlled.zero_step_encountered:
                        break

                self.assertTrue(control_coincide)

    # tests if the control from OC computation coincides with a random input used for target forward-simulation
    # network case with delay
    def test_twonode_delay_oc(self):
        print("Test OC in 2-node network with delay")

        duration = 1.0
        amplitude = 1.0

        rs = RandomState(MT19937(SeedSequence(0)))  # work with fixed seed for reproducibility
        delay = rs.choice([0.1, 0.2, 0.3, 0.4])

        cmat = np.array([[0.0, 0.0], [1.0, 0.0]])
        dmat = np.array([[0.0, 0.0], [delay, 0.0]])

        model = ALNModel(Cmat=cmat, Dmat=dmat)

        model.params.de = 0.0
        model.params.di = 0.0

        set_param_init(model)

        model.params.duration = duration

        control_mat = np.zeros((model.params.N, len(model.state_vars)))
        control_mat[0, 0] = 1.0

        measure_channel = 0

        cost_mat = np.zeros((model.params.N, len(model.output_vars)))
        cost_mat[1, measure_channel] = 1.0

        zero_input = ZeroInput().generate_input(duration=model.params.duration + model.params.dt, dt=model.params.dt)
        input = np.copy(zero_input)
        input_optimization_start = np.copy(zero_input)

        intinit, intend = 1, input.shape[1] - 6 - model.getMaxDelay()

        for t in range(intinit, intend):
            input[0, t] = rs.uniform(-amplitude, amplitude)
            input_optimization_start[0, t] = input[0, t] + 1e-2 * rs.uniform(-amplitude, amplitude)

        model.params["ext_inh_current"] = np.vstack([zero_input, zero_input])
        model.params["ext_exc_current"] = np.vstack([input, zero_input])
        model.run()

        target = getstate(model)
        control_init = np.zeros((target.shape))
        control_init[0, 0, :] = input_optimization_start[0, :]

        model.params["ext_inh_current"] = np.vstack([zero_input, zero_input])
        model.params["ext_exc_current"] = np.vstack([zero_input, zero_input])

        model_controlled = oc_aln.OcAln(
            model,
            target,
            control_matrix=control_mat,
            cost_matrix=cost_mat,
        )

        model_controlled.control = control_init.copy()
        model_controlled.update_input()

        control_coincide = False

        for i in range(LOOPS):
            model_controlled.optimize(ITERATIONS)
            control = model_controlled.control

            c_diff = np.abs(control[0, 0, intinit:intend] - input[0, intinit:intend])

            if np.amax(c_diff) < LIMIT_DIFF:
                control_coincide = True
                break

            if 0 != measure_channel:
                if np.amax(c_diff) < 1e2 * LIMIT_DIFF:
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

        target = np.ones((model.params.N, len(model.output_vars), input.shape[1]))

        model_controlled = oc_aln.OcAln(
            model,
            target,
        )

        model_controlled.optimize(1)
        xs = model_controlled.get_xs()
        self.assertTrue(xs.shape == target.shape)


if __name__ == "__main__":
    unittest.main()
