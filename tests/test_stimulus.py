"""
Tests of noise input.
"""

import unittest

import numpy as np
from chspy import CubicHermiteSpline
from neurolib.models.aln import ALNModel
from neurolib.utils.stimulus import (
    ConcatenatedInput,
    ExponentialInput,
    LinearRampInput,
    OrnsteinUhlenbeckProcess,
    SinusoidalInput,
    SquareInput,
    StepInput,
    WienerProcess,
    ZeroInput,
    construct_stimulus,
)

TESTING_TIME = 5.3
DURATION = 10
DT = 0.1
STIM_START = 2
STIM_END = 8
SHAPE = (int(DURATION / DT), 2)


class TestCubicSplines(unittest.TestCase):
    RESULT_SPLINES = np.array([-0.05100302, 0.1277721])
    RESULT_ARRAY = np.array([0.59646435, 0.05520635])

    def test_splines(self):
        dW = WienerProcess(num_iid=2, seed=42).as_cubic_splines(duration=DURATION, dt=DT)
        self.assertTrue(isinstance(dW, CubicHermiteSpline))
        np.testing.assert_allclose(self.RESULT_SPLINES, dW.get_state(TESTING_TIME))

    def test_arrays(self):
        dW = WienerProcess(num_iid=2, seed=42).as_array(duration=DURATION, dt=DT)
        self.assertTrue(isinstance(dW, np.ndarray))
        time_idx = np.around(TESTING_TIME / DT).astype(int)
        np.testing.assert_allclose(self.RESULT_ARRAY, dW[time_idx, :])


class TestZeroInput(unittest.TestCase):
    def test_generate_input(self):
        nn = ZeroInput(num_iid=2, seed=42).generate_input(duration=DURATION, dt=DT)
        self.assertTrue(isinstance(nn, np.ndarray))
        self.assertTupleEqual(nn.shape, SHAPE)
        np.testing.assert_allclose(nn, np.zeros(SHAPE))

    def test_get_params(self):
        nn = ZeroInput(num_iid=2, seed=42)
        params = nn.get_params()
        params.pop("type")
        self.assertDictEqual(params, {"num_iid": 2, "seed": 42})

    def test_set_params(self):
        nn = ZeroInput(num_iid=2, seed=42)
        UPDATE = {"seed": 635}
        nn.update_params(UPDATE)
        params = nn.get_params()
        params.pop("type")
        self.assertDictEqual(params, {"num_iid": 2, "seed": 42, **UPDATE})


class TestWienerProcess(unittest.TestCase):
    def test_generate_input(self):
        dW = WienerProcess(num_iid=2, seed=42).generate_input(duration=DURATION, dt=DT)
        self.assertTrue(isinstance(dW, np.ndarray))
        self.assertTupleEqual(dW.shape, SHAPE)

    def test_get_params(self):
        dW = WienerProcess(num_iid=2, seed=42)
        params = dW.get_params()
        params.pop("type")
        self.assertDictEqual(params, {"num_iid": 2, "seed": 42})

    def test_set_params(self):
        dW = WienerProcess(num_iid=2, seed=42)
        UPDATE = {"seed": 6152, "num_iid": 5}
        dW.update_params(UPDATE)
        params = dW.get_params()
        params.pop("type")
        self.assertDictEqual(params, {"num_iid": 2, "seed": 42, **UPDATE})


class TestOrnsteinUhlenbeckProcess(unittest.TestCase):
    def test_generate_input(self):
        ou = OrnsteinUhlenbeckProcess(
            mu=3.0,
            sigma=0.1,
            tau=2 * DT,
            num_iid=2,
            seed=42,
        ).generate_input(duration=DURATION, dt=DT)
        self.assertTrue(isinstance(ou, np.ndarray))
        self.assertTupleEqual(ou.shape, SHAPE)

    def test_get_params(self):
        ou = OrnsteinUhlenbeckProcess(
            mu=3.0,
            sigma=0.1,
            tau=2 * DT,
            num_iid=2,
            seed=42,
        )
        params = ou.get_params()
        params.pop("type")
        self.assertDictEqual(params, {"num_iid": 2, "seed": 42, "mu": 3.0, "sigma": 0.1, "tau": 2 * DT})

    def test_set_params(self):
        ou = OrnsteinUhlenbeckProcess(
            mu=3.0,
            sigma=0.1,
            tau=2 * DT,
            num_iid=2,
            seed=42,
        )
        UPDATE = {"mu": 2.3, "seed": 12}
        ou.update_params(UPDATE)
        params = ou.get_params()
        params.pop("type")
        self.assertDictEqual(params, {"num_iid": 2, "seed": 42, "mu": 3.0, "sigma": 0.1, "tau": 2 * DT, **UPDATE})


class TestStepInput(unittest.TestCase):
    STEP_SIZE = 2.3

    def test_generate_input(self):
        step = StepInput(
            step_size=self.STEP_SIZE,
            num_iid=2,
            seed=42,
        ).generate_input(duration=DURATION, dt=DT)
        self.assertTrue(isinstance(step, np.ndarray))
        self.assertTupleEqual(step.shape, SHAPE)
        np.testing.assert_allclose(step, self.STEP_SIZE)

    def test_start_end_input(self):
        step = StepInput(
            stim_start=STIM_START,
            stim_end=STIM_END,
            step_size=self.STEP_SIZE,
            num_iid=2,
            seed=42,
        ).as_array(duration=DURATION, dt=DT)
        np.testing.assert_allclose(step[: int(STIM_START / DT), :], 0.0)
        np.testing.assert_allclose(step[int(STIM_END / DT) :, :], 0.0)


class TestSinusoidalInput(unittest.TestCase):
    AMPLITUDE = 2.3
    PERIOD = 2.0

    def test_generate_input(self):
        sin = SinusoidalInput(
            amplitude=self.AMPLITUDE,
            period=self.PERIOD,
            num_iid=2,
            seed=42,
        ).generate_input(duration=DURATION, dt=DT)
        self.assertTrue(isinstance(sin, np.ndarray))
        self.assertTupleEqual(sin.shape, SHAPE)
        np.testing.assert_equal(np.mean(sin, axis=0), np.array(2 * [self.AMPLITUDE]))

    def test_start_end_input(self):
        sin = SinusoidalInput(
            stim_start=STIM_START,
            stim_end=STIM_END,
            amplitude=self.AMPLITUDE,
            period=self.PERIOD,
            num_iid=2,
            seed=42,
        ).as_array(duration=DURATION, dt=DT)
        np.testing.assert_allclose(sin[: int(STIM_START / DT), :], 0.0)
        np.testing.assert_allclose(sin[int(STIM_END / DT) :, :], 0.0)

    def test_get_params(self):
        sin = SinusoidalInput(
            stim_start=STIM_START,
            stim_end=STIM_END,
            amplitude=self.AMPLITUDE,
            period=self.PERIOD,
            num_iid=2,
            seed=42,
        )
        params = sin.get_params()
        params.pop("type")
        self.assertDictEqual(
            params,
            {
                "num_iid": 2,
                "seed": 42,
                "period": self.PERIOD,
                "amplitude": self.AMPLITUDE,
                "stim_start": STIM_START,
                "nonnegative": True,
                "stim_end": STIM_END,
            },
        )

    def test_set_params(self):
        sin = SinusoidalInput(
            stim_start=STIM_START,
            stim_end=STIM_END,
            amplitude=self.AMPLITUDE,
            period=self.PERIOD,
            num_iid=2,
            seed=42,
        )
        UPDATE = {"amplitude": 43.0, "seed": 12}
        sin.update_params(UPDATE)
        params = sin.get_params()
        params.pop("type")
        self.assertDictEqual(
            params,
            {
                "num_iid": 2,
                "seed": 42,
                "period": self.PERIOD,
                "amplitude": self.AMPLITUDE,
                "stim_start": STIM_START,
                "nonnegative": True,
                "stim_end": STIM_END,
                **UPDATE,
            },
        )


class TestSquareInput(unittest.TestCase):
    AMPLITUDE = 2.3
    PERIOD = 2.0

    def test_generate_input(self):

        sq = SquareInput(
            amplitude=self.AMPLITUDE,
            period=self.PERIOD,
            num_iid=2,
            seed=42,
        ).generate_input(duration=DURATION, dt=DT)
        self.assertTrue(isinstance(sq, np.ndarray))
        self.assertTupleEqual(sq.shape, SHAPE)
        np.testing.assert_equal(np.mean(sq, axis=0), np.array(2 * [self.AMPLITUDE]))

    def test_start_end_input(self):
        sq = SquareInput(
            stim_start=STIM_START,
            stim_end=STIM_END,
            amplitude=self.AMPLITUDE,
            period=self.PERIOD,
            num_iid=2,
            seed=42,
        ).as_array(duration=DURATION, dt=DT)
        np.testing.assert_allclose(sq[: int(STIM_START / DT), :], 0.0)
        np.testing.assert_allclose(sq[int(STIM_END / DT) :, :], 0.0)

    def test_get_params(self):
        sq = SquareInput(
            stim_start=STIM_START,
            stim_end=STIM_END,
            amplitude=self.AMPLITUDE,
            period=self.PERIOD,
            num_iid=2,
            seed=42,
        )
        params = sq.get_params()
        params.pop("type")
        self.assertDictEqual(
            params,
            {
                "num_iid": 2,
                "seed": 42,
                "period": self.PERIOD,
                "amplitude": self.AMPLITUDE,
                "stim_start": STIM_START,
                "stim_end": STIM_END,
                "nonnegative": True,
            },
        )

    def test_set_params(self):
        sq = SquareInput(
            stim_start=STIM_START,
            stim_end=STIM_END,
            amplitude=self.AMPLITUDE,
            period=self.PERIOD,
            num_iid=2,
            seed=42,
        )
        UPDATE = {"amplitude": 43.0, "seed": 12}
        sq.update_params(UPDATE)
        params = sq.get_params()
        params.pop("type")
        self.assertDictEqual(
            params,
            {
                "num_iid": 2,
                "seed": 42,
                "period": self.PERIOD,
                "amplitude": self.AMPLITUDE,
                "stim_start": STIM_START,
                "stim_end": STIM_END,
                "nonnegative": True,
                **UPDATE,
            },
        )


class TestLinearRampInput(unittest.TestCase):
    INP_MAX = 5.0
    RAMP_LENGTH = 2.0

    def test_generate_input(self):

        ramp = LinearRampInput(
            inp_max=self.INP_MAX,
            ramp_length=self.RAMP_LENGTH,
            num_iid=2,
            seed=42,
        ).generate_input(duration=DURATION, dt=DT)
        self.assertTrue(isinstance(ramp, np.ndarray))
        self.assertTupleEqual(ramp.shape, SHAPE)
        np.testing.assert_equal(np.max(ramp, axis=0), np.array(2 * [self.INP_MAX]))
        np.testing.assert_equal(np.min(ramp, axis=0), np.array(2 * [0.25]))

    def test_start_end_input(self):
        ramp = LinearRampInput(
            stim_start=STIM_START,
            stim_end=STIM_END,
            inp_max=self.INP_MAX,
            ramp_length=self.RAMP_LENGTH,
            num_iid=2,
            seed=42,
        ).as_array(duration=DURATION, dt=DT)
        np.testing.assert_allclose(ramp[: int(STIM_START / DT), :], 0.0)
        np.testing.assert_allclose(ramp[int(STIM_END / DT) :, :], 0.0)

    def test_get_params(self):
        ramp = LinearRampInput(
            stim_start=STIM_START,
            stim_end=STIM_END,
            inp_max=self.INP_MAX,
            ramp_length=self.RAMP_LENGTH,
            num_iid=2,
            seed=42,
        )
        params = ramp.get_params()
        params.pop("type")
        self.assertDictEqual(
            params,
            {
                "num_iid": 2,
                "seed": 42,
                "inp_max": self.INP_MAX,
                "ramp_length": self.RAMP_LENGTH,
                "stim_start": STIM_START,
                "stim_end": STIM_END,
            },
        )

    def test_set_params(self):
        ramp = LinearRampInput(
            stim_start=STIM_START,
            stim_end=STIM_END,
            inp_max=self.INP_MAX,
            ramp_length=self.RAMP_LENGTH,
            num_iid=2,
            seed=42,
        )
        UPDATE = {"inp_max": 41.0, "seed": 12}
        ramp.update_params(UPDATE)
        params = ramp.get_params()
        params.pop("type")
        self.assertDictEqual(
            params,
            {
                "num_iid": 2,
                "seed": 42,
                "inp_max": self.INP_MAX,
                "ramp_length": self.RAMP_LENGTH,
                "stim_start": STIM_START,
                "stim_end": STIM_END,
                **UPDATE,
            },
        )


class TestExponentialInput(unittest.TestCase):
    INP_MAX = 5.0
    EXP_COEF = 30.0
    EXP_TYPE = "rise"

    def test_generate_input_rise(self):
        exp_rise = ExponentialInput(
            inp_max=self.INP_MAX,
            exp_type="rise",
            num_iid=2,
            seed=42,
        ).generate_input(duration=DURATION, dt=DT)
        self.assertTrue(isinstance(exp_rise, np.ndarray))
        self.assertTupleEqual(exp_rise.shape, SHAPE)
        np.testing.assert_almost_equal(np.max(exp_rise, axis=0), np.array(2 * [self.INP_MAX]))
        self.assertTrue(np.all(np.diff(exp_rise) >= 0))

    def test_generate_input_decay(self):
        exp_decay = ExponentialInput(
            inp_max=self.INP_MAX,
            exp_type="decay",
            num_iid=2,
            seed=42,
        ).generate_input(duration=DURATION, dt=DT)
        self.assertTrue(isinstance(exp_decay, np.ndarray))
        self.assertTupleEqual(exp_decay.shape, SHAPE)
        self.assertTrue(np.all(np.diff(exp_decay) <= 0))

    def test_start_end_input(self):
        exp_rise = ExponentialInput(
            stim_start=STIM_START,
            stim_end=STIM_END,
            inp_max=self.INP_MAX,
            num_iid=2,
            seed=42,
        ).as_array(duration=DURATION, dt=DT)
        np.testing.assert_allclose(exp_rise[: int(STIM_START / DT), :], 0.0)
        np.testing.assert_allclose(exp_rise[int(STIM_END / DT) :, :], 0.0)

    def test_get_params(self):
        exp_rise = ExponentialInput(
            stim_start=STIM_START,
            stim_end=STIM_END,
            inp_max=self.INP_MAX,
            num_iid=2,
            seed=42,
        )
        params = exp_rise.get_params()
        params.pop("type")
        self.assertDictEqual(
            params,
            {
                "num_iid": 2,
                "seed": 42,
                "inp_max": self.INP_MAX,
                "exp_coef": self.EXP_COEF,
                "exp_type": self.EXP_TYPE,
                "stim_start": STIM_START,
                "stim_end": STIM_END,
            },
        )

    def test_set_params(self):
        exp_rise = ExponentialInput(
            stim_start=STIM_START,
            stim_end=STIM_END,
            inp_max=self.INP_MAX,
            num_iid=2,
            seed=42,
        )
        UPDATE = {"inp_max": 41.0, "seed": 12}
        exp_rise.update_params(UPDATE)
        params = exp_rise.get_params()
        params.pop("type")
        self.assertDictEqual(
            params,
            {
                "num_iid": 2,
                "seed": 42,
                "inp_max": self.INP_MAX,
                "exp_coef": self.EXP_COEF,
                "exp_type": self.EXP_TYPE,
                "stim_start": STIM_START,
                "stim_end": STIM_END,
                **UPDATE,
            },
        )


class TestConcatenatedInput(unittest.TestCase):
    def _create_input(self):
        ou = OrnsteinUhlenbeckProcess(mu=0.1, sigma=0.02, tau=2.0, num_iid=2)
        sq = SquareInput(amplitude=0.2, period=20.0, num_iid=2, stim_start=5)
        sin = SinusoidalInput(amplitude=0.1, period=10.0, num_iid=2, stim_start=2)
        step = StepInput(step_size=0.5, num_iid=2, stim_start=7)
        return sq + sin + step + ou

    def test_init(self):
        conc = self._create_input()
        self.assertTrue(isinstance(conc, ConcatenatedInput))
        self.assertEqual(conc.num_iid, 2)
        self.assertEqual(len(conc.noise_processes), 4)

    def test_generate_input(self):
        conc = self._create_input()
        ts = conc.as_array(duration=DURATION, dt=DT)
        self.assertTrue(isinstance(ts, np.ndarray))
        self.assertTupleEqual(ts.shape, SHAPE)

        ts = conc.as_cubic_splines(duration=DURATION, dt=DT)
        self.assertTrue(isinstance(ts, CubicHermiteSpline))

    def test_get_params(self):
        conc = self._create_input()
        params = conc.get_params()
        self.assertTrue(isinstance(params, dict))
        self.assertEqual(len(params), 1 + len(conc.noise_processes))
        for i, process in enumerate(conc.noise_processes):
            self.assertDictEqual(process.get_params(), params[f"noise_{i}"])

    def test_update_params(self):
        conc = self._create_input()
        UPDATE_DICT = {f"noise_{i}": {"num_iid": 3} for i in range(len(conc.noise_processes))}
        conc.update_params(UPDATE_DICT)
        self.assertEqual(conc.num_iid, 3)


class TestConstructStimulus(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.single_node = ALNModel()

    def test_construct_stimulus_ac(self):
        self.single_node.params["duration"] = 2000
        stimulus = construct_stimulus(
            "ac", duration=self.single_node.params.duration, dt=self.single_node.params.dt, stim_amp=1.0, stim_freq=1
        )
        self.single_node.params["ext_exc_current"] = stimulus
        self.single_node.run()

    def test_construct_stimulus_dc(self):
        self.single_node.params["duration"] = 2000
        stimulus = construct_stimulus(
            "dc", duration=self.single_node.params.duration, dt=self.single_node.params.dt, stim_amp=1.0, stim_freq=1
        )
        self.single_node.params["ext_exc_current"] = stimulus
        self.single_node.run()

    def test_construct_stimulus_rect(self):
        self.single_node.params["duration"] = 2000
        stimulus = construct_stimulus(
            "rect", duration=self.single_node.params.duration, dt=self.single_node.params.dt, stim_amp=1.0, stim_freq=1
        )
        self.single_node.params["ext_exc_current"] = stimulus
        self.single_node.run()


if __name__ == "__main__":
    unittest.main()
