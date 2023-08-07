"""
Tests of noise input.
"""

import unittest

import numpy as np
from chspy import CubicHermiteSpline
from neurolib.models.aln import ALNModel
from neurolib.models.fhn import FHNModel
from neurolib.models.hopf import HopfModel
from neurolib.models.wc import WCModel
from neurolib.models.kuramoto import KuramotoModel
from neurolib.utils.stimulus import (
    ConcatenatedStimulus,
    ExponentialInput,
    LinearRampInput,
    OrnsteinUhlenbeckProcess,
    RectifiedInput,
    SinusoidalInput,
    SquareInput,
    StepInput,
    SummedStimulus,
    WienerProcess,
    ZeroInput,
)

TESTING_TIME = 5.3
DURATION = 10
DT = 0.1
STIM_START = 2
STIM_END = 8
SHAPE = (2, int(DURATION / DT))


class TestCubicSplines(unittest.TestCase):
    RESULT_SPLINES = np.array([-0.214062, -0.215043])
    RESULT_ARRAY = np.array([0.193429, 0.073445])

    def test_splines(self):
        dW = WienerProcess(n=2, seed=42).as_cubic_splines(duration=DURATION, dt=DT)
        self.assertTrue(isinstance(dW, CubicHermiteSpline))
        np.testing.assert_allclose(self.RESULT_SPLINES, dW.get_state(TESTING_TIME), atol=1e-05)

    def test_arrays(self):
        dW = WienerProcess(n=2, seed=42).as_array(duration=DURATION, dt=DT)
        self.assertTrue(isinstance(dW, np.ndarray))
        time_idx = np.around(TESTING_TIME / DT).astype(int)
        np.testing.assert_allclose(self.RESULT_ARRAY, dW[:, time_idx], atol=1e-05)

    def test_shift_start_time(self):
        SHIFT = 5.0
        dW = WienerProcess(n=2, seed=42).as_cubic_splines(duration=DURATION, dt=DT, shift_start_time=SHIFT)
        self.assertTrue(isinstance(dW, CubicHermiteSpline))
        self.assertEqual(dW[0].time, SHIFT + DT)
        np.testing.assert_allclose(self.RESULT_SPLINES, dW.get_state(TESTING_TIME + SHIFT), atol=1e-05)


class TestToModel(unittest.TestCase):
    def test_single_node(self):
        model = ALNModel()
        model.params["duration"] = 2 * 1000
        stim = SinusoidalInput(amplitude=1.0, frequency=1.0)
        model_stim = stim.to_model(model)
        model.params["ext_exc_current"] = model_stim
        model.run()
        self.assertTrue(isinstance(model_stim, np.ndarray))
        self.assertTupleEqual(model_stim.shape, (1, int(model.params["duration"] / model.params["dt"])))

    def test_multi_node_multi_stim(self):
        model = ALNModel(Cmat=np.random.rand(5, 5), Dmat=np.zeros((5, 5)))
        model.params["duration"] = 2 * 1000
        stim = SinusoidalInput(amplitude=1.0, frequency=1.0)
        model_stim = stim.to_model(model)
        model.params["ext_exc_current"] = model_stim
        model.run()
        self.assertTrue(isinstance(model_stim, np.ndarray))
        self.assertTupleEqual(model_stim.shape, (5, int(model.params["duration"] / model.params["dt"])))


class TestToFHNModel(unittest.TestCase):
    def test_single_node(self):
        model = FHNModel()
        model.params["duration"] = 2 * 1000
        stim = SinusoidalInput(amplitude=1.0, frequency=1.0)
        model_stim = stim.to_model(model)
        model.params["x_ext"] = model_stim
        model.run()
        self.assertTrue(isinstance(model_stim, np.ndarray))
        self.assertTupleEqual(model_stim.shape, (1, int(model.params["duration"] / model.params["dt"])))

    def test_multi_node_multi_stim(self):
        model = FHNModel(Cmat=np.random.rand(5, 5), Dmat=np.zeros((5, 5)))
        model.params["duration"] = 2 * 1000
        stim = SinusoidalInput(amplitude=1.0, frequency=1.0)
        model_stim = stim.to_model(model)
        model.params["x_ext"] = model_stim
        model.run()
        self.assertTrue(isinstance(model_stim, np.ndarray))
        self.assertTupleEqual(model_stim.shape, (5, int(model.params["duration"] / model.params["dt"])))


class TestToHopfModel(unittest.TestCase):
    def test_single_node(self):
        model = HopfModel()
        model.params["duration"] = 2 * 1000
        stim = SinusoidalInput(amplitude=1.0, frequency=1.0)
        model_stim = stim.to_model(model)
        model.params["x_ext"] = model_stim
        model.run()
        self.assertTrue(isinstance(model_stim, np.ndarray))
        self.assertTupleEqual(model_stim.shape, (1, int(model.params["duration"] / model.params["dt"])))

    def test_multi_node_multi_stim(self):
        model = HopfModel(Cmat=np.random.rand(5, 5), Dmat=np.zeros((5, 5)))
        model.params["duration"] = 2 * 1000
        stim = SinusoidalInput(amplitude=1.0, frequency=1.0)
        model_stim = stim.to_model(model)
        model.params["x_ext"] = model_stim
        model.run()
        self.assertTrue(isinstance(model_stim, np.ndarray))
        self.assertTupleEqual(model_stim.shape, (5, int(model.params["duration"] / model.params["dt"])))


class TestToWCModel(unittest.TestCase):
    def test_single_node(self):
        model = WCModel()
        model.params["duration"] = 2 * 1000
        stim = SinusoidalInput(amplitude=1.0, frequency=1.0)
        model_stim = stim.to_model(model)
        model.params["exc_ext"] = model_stim
        model.run()
        self.assertTrue(isinstance(model_stim, np.ndarray))
        self.assertTupleEqual(model_stim.shape, (1, int(model.params["duration"] / model.params["dt"])))

    def test_multi_node_multi_stim(self):
        model = WCModel(Cmat=np.random.rand(5, 5), Dmat=np.zeros((5, 5)))
        model.params["duration"] = 2 * 1000
        stim = SinusoidalInput(amplitude=1.0, frequency=1.0)
        model_stim = stim.to_model(model)
        model.params["exc_ext"] = model_stim
        model.run()
        self.assertTrue(isinstance(model_stim, np.ndarray))
        self.assertTupleEqual(model_stim.shape, (5, int(model.params["duration"] / model.params["dt"])))


class TestToKuramotoModel(unittest.TestCase):
    def test_single_node(self):
        model = KuramotoModel()
        model.params["duration"] = 2 * 1000
        stim = SinusoidalInput(amplitude=1.0, frequency=1.0)
        model_stim = stim.to_model(model)
        model.params["theta_ext"] = model_stim
        model.run()
        self.assertTrue(isinstance(model_stim, np.ndarray))
        self.assertTupleEqual(model_stim.shape, (1, int(model.params["duration"] / model.params["dt"])))

    def test_multi_node_multi_stim(self):
        model = KuramotoModel(Cmat=np.random.rand(5, 5), Dmat=np.zeros((5, 5)))
        model.params["duration"] = 2 * 1000
        stim = SinusoidalInput(amplitude=1.0, frequency=1.0)
        model_stim = stim.to_model(model)
        model.params["theta_ext"] = model_stim
        model.run()
        self.assertTrue(isinstance(model_stim, np.ndarray))
        self.assertTupleEqual(model_stim.shape, (5, int(model.params["duration"] / model.params["dt"])))


class TestZeroInput(unittest.TestCase):
    def test_generate_input(self):
        nn = ZeroInput(n=2, seed=42).generate_input(duration=DURATION, dt=DT)
        self.assertTrue(isinstance(nn, np.ndarray))
        self.assertTupleEqual(nn.shape, SHAPE)
        np.testing.assert_allclose(nn, np.zeros(SHAPE))

    def test_get_params(self):
        nn = ZeroInput(n=2, seed=42)
        params = nn.get_params()
        params.pop("type")
        self.assertDictEqual(params, {"n": 2, "seed": 42})

    def test_set_params(self):
        nn = ZeroInput(n=2, seed=42)
        UPDATE = {"seed": 635}
        nn.update_params(UPDATE)
        params = nn.get_params()
        params.pop("type")
        self.assertDictEqual(params, {"n": 2, "seed": 42, **UPDATE})


class TestWienerProcess(unittest.TestCase):
    def test_generate_input(self):
        dW = WienerProcess(n=2, seed=42).generate_input(duration=DURATION, dt=DT)
        self.assertTrue(isinstance(dW, np.ndarray))
        self.assertTupleEqual(dW.shape, SHAPE)

    def test_get_params(self):
        dW = WienerProcess(n=2, seed=42)
        params = dW.get_params()
        params.pop("type")
        self.assertDictEqual(params, {"n": 2, "seed": 42})

    def test_set_params(self):
        dW = WienerProcess(n=2, seed=42)
        UPDATE = {"seed": 6152, "n": 5}
        dW.update_params(UPDATE)
        params = dW.get_params()
        params.pop("type")
        self.assertDictEqual(params, {"n": 2, "seed": 42, **UPDATE})


class TestOrnsteinUhlenbeckProcess(unittest.TestCase):
    def test_generate_input(self):
        ou = OrnsteinUhlenbeckProcess(
            mu=3.0,
            sigma=0.1,
            tau=2 * DT,
            n=2,
            seed=42,
        ).generate_input(duration=DURATION, dt=DT)
        self.assertTrue(isinstance(ou, np.ndarray))
        self.assertTupleEqual(ou.shape, SHAPE)

    def test_get_params(self):
        ou = OrnsteinUhlenbeckProcess(
            mu=3.0,
            sigma=0.1,
            tau=2 * DT,
            n=2,
            seed=42,
        )
        params = ou.get_params()
        params.pop("type")
        self.assertDictEqual(params, {"n": 2, "seed": 42, "mu": 3.0, "sigma": 0.1, "tau": 2 * DT})

    def test_set_params(self):
        ou = OrnsteinUhlenbeckProcess(
            mu=3.0,
            sigma=0.1,
            tau=2 * DT,
            n=2,
            seed=42,
        )
        UPDATE = {"mu": 2.3, "seed": 12}
        ou.update_params(UPDATE)
        params = ou.get_params()
        params.pop("type")
        self.assertDictEqual(params, {"n": 2, "seed": 42, "mu": 3.0, "sigma": 0.1, "tau": 2 * DT, **UPDATE})


class TestStepInput(unittest.TestCase):
    STEP_SIZE = 2.3

    def test_generate_input(self):
        step = StepInput(
            step_size=self.STEP_SIZE,
            n=2,
            seed=42,
        ).generate_input(duration=DURATION, dt=DT)
        self.assertTrue(isinstance(step, np.ndarray))
        self.assertTupleEqual(step.shape, SHAPE)
        np.testing.assert_allclose(step, self.STEP_SIZE)

    def test_start_end_input(self):
        step = StepInput(
            start=STIM_START,
            end=STIM_END,
            step_size=self.STEP_SIZE,
            n=2,
            seed=42,
        ).as_array(duration=DURATION, dt=DT)
        np.testing.assert_allclose(step[:, : int(STIM_START / DT)], 0.0)
        np.testing.assert_allclose(step[:, int(STIM_END / DT) :], 0.0)


class TestSinusoidalInput(unittest.TestCase):
    AMPLITUDE = 2.3
    FREQUENCY = 1000.0

    def test_generate_input(self):
        sin = SinusoidalInput(
            amplitude=self.AMPLITUDE, frequency=self.FREQUENCY, n=2, seed=42, dc_bias=True
        ).generate_input(duration=DURATION, dt=DT)
        self.assertTrue(isinstance(sin, np.ndarray))
        self.assertTupleEqual(sin.shape, SHAPE)
        np.testing.assert_almost_equal(np.mean(sin, axis=1), np.array(2 * [self.AMPLITUDE]))

    def test_start_end_input(self):
        sin = SinusoidalInput(
            start=STIM_START,
            end=STIM_END,
            amplitude=self.AMPLITUDE,
            frequency=self.FREQUENCY,
            n=2,
            seed=42,
        ).as_array(duration=DURATION, dt=DT)
        np.testing.assert_allclose(sin[:, : int(STIM_START / DT)], 0.0)
        np.testing.assert_allclose(sin[:, int(STIM_END / DT) :], 0.0)

    def test_get_params(self):
        sin = SinusoidalInput(
            start=STIM_START,
            end=STIM_END,
            amplitude=self.AMPLITUDE,
            frequency=self.FREQUENCY,
            n=2,
            seed=42,
        )
        params = sin.get_params()
        params.pop("type")
        self.assertDictEqual(
            params,
            {
                "n": 2,
                "seed": 42,
                "frequency": self.FREQUENCY,
                "amplitude": self.AMPLITUDE,
                "start": STIM_START,
                "dc_bias": False,
                "end": STIM_END,
            },
        )

    def test_set_params(self):
        sin = SinusoidalInput(
            start=STIM_START,
            end=STIM_END,
            amplitude=self.AMPLITUDE,
            frequency=self.FREQUENCY,
            n=2,
            seed=42,
        )
        UPDATE = {"amplitude": 43.0, "seed": 12, "start": "None"}
        sin.update_params(UPDATE)
        params = sin.get_params()
        params.pop("type")
        self.assertDictEqual(
            params,
            {
                "n": 2,
                "seed": 42,
                "frequency": self.FREQUENCY,
                "amplitude": self.AMPLITUDE,
                "dc_bias": False,
                "end": STIM_END,
                **UPDATE,
                "start": None,
            },
        )


class TestSquareInput(unittest.TestCase):
    AMPLITUDE = 2.3
    FREQUENCY = 20.0

    def test_generate_input(self):

        sq = SquareInput(
            amplitude=self.AMPLITUDE,
            frequency=self.FREQUENCY,
            n=2,
            seed=42,
        ).generate_input(duration=DURATION, dt=DT)
        self.assertTrue(isinstance(sq, np.ndarray))
        self.assertTupleEqual(sq.shape, SHAPE)
        np.testing.assert_almost_equal(np.mean(sq, axis=1), np.array(2 * [self.AMPLITUDE]))

    def test_start_end_input(self):
        sq = SquareInput(
            start=STIM_START,
            end=STIM_END,
            amplitude=self.AMPLITUDE,
            frequency=self.FREQUENCY,
            n=2,
            seed=42,
        ).as_array(duration=DURATION, dt=DT)
        np.testing.assert_allclose(sq[:, : int(STIM_START / DT)], 0.0)
        np.testing.assert_allclose(sq[:, int(STIM_END / DT) :], 0.0)

    def test_get_params(self):
        sq = SquareInput(
            start=STIM_START,
            end=STIM_END,
            amplitude=self.AMPLITUDE,
            frequency=self.FREQUENCY,
            n=2,
            seed=42,
        )
        params = sq.get_params()
        params.pop("type")
        self.assertDictEqual(
            params,
            {
                "n": 2,
                "seed": 42,
                "frequency": self.FREQUENCY,
                "amplitude": self.AMPLITUDE,
                "start": STIM_START,
                "end": STIM_END,
                "dc_bias": False,
            },
        )

    def test_set_params(self):
        sq = SquareInput(
            start=STIM_START,
            end=STIM_END,
            amplitude=self.AMPLITUDE,
            frequency=self.FREQUENCY,
            n=2,
            seed=42,
        )
        UPDATE = {"amplitude": 43.0, "seed": 12, "start": "None"}
        sq.update_params(UPDATE)
        params = sq.get_params()
        params.pop("type")
        self.assertDictEqual(
            params,
            {
                "n": 2,
                "seed": 42,
                "frequency": self.FREQUENCY,
                "amplitude": self.AMPLITUDE,
                "end": STIM_END,
                "dc_bias": False,
                **UPDATE,
                "start": None,
            },
        )


class TestLinearRampInput(unittest.TestCase):
    INP_MAX = 5.0
    RAMP_LENGTH = 2.0

    def test_generate_input(self):

        ramp = LinearRampInput(
            inp_max=self.INP_MAX,
            ramp_length=self.RAMP_LENGTH,
            n=2,
            seed=42,
        ).generate_input(duration=DURATION, dt=DT)
        self.assertTrue(isinstance(ramp, np.ndarray))
        self.assertTupleEqual(ramp.shape, SHAPE)
        np.testing.assert_equal(np.max(ramp, axis=1), np.array(2 * [self.INP_MAX]))
        np.testing.assert_equal(np.min(ramp, axis=1), np.array(2 * [0.25]))

    def test_start_end_input(self):
        ramp = LinearRampInput(
            start=STIM_START,
            end=STIM_END,
            inp_max=self.INP_MAX,
            ramp_length=self.RAMP_LENGTH,
            n=2,
            seed=42,
        ).as_array(duration=DURATION, dt=DT)
        np.testing.assert_allclose(ramp[:, : int(STIM_START / DT)], 0.0)
        np.testing.assert_allclose(ramp[:, int(STIM_END / DT) :], 0.0)

    def test_get_params(self):
        ramp = LinearRampInput(
            start=STIM_START,
            end=STIM_END,
            inp_max=self.INP_MAX,
            ramp_length=self.RAMP_LENGTH,
            n=2,
            seed=42,
        )
        params = ramp.get_params()
        params.pop("type")
        self.assertDictEqual(
            params,
            {
                "n": 2,
                "seed": 42,
                "inp_max": self.INP_MAX,
                "ramp_length": self.RAMP_LENGTH,
                "start": STIM_START,
                "end": STIM_END,
            },
        )

    def test_set_params(self):
        ramp = LinearRampInput(
            start=STIM_START,
            end=STIM_END,
            inp_max=self.INP_MAX,
            ramp_length=self.RAMP_LENGTH,
            n=2,
            seed=42,
        )
        UPDATE = {"inp_max": 41.0, "seed": 12}
        ramp.update_params(UPDATE)
        params = ramp.get_params()
        params.pop("type")
        self.assertDictEqual(
            params,
            {
                "n": 2,
                "seed": 42,
                "inp_max": self.INP_MAX,
                "ramp_length": self.RAMP_LENGTH,
                "start": STIM_START,
                "end": STIM_END,
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
            n=2,
            seed=42,
        ).generate_input(duration=DURATION, dt=DT)
        self.assertTrue(isinstance(exp_rise, np.ndarray))
        self.assertTupleEqual(exp_rise.shape, SHAPE)
        np.testing.assert_almost_equal(np.max(exp_rise, axis=1), np.array(2 * [self.INP_MAX]))
        self.assertTrue(np.all(np.diff(exp_rise) >= 0))

    def test_generate_input_decay(self):
        exp_decay = ExponentialInput(
            inp_max=self.INP_MAX,
            exp_type="decay",
            n=2,
            seed=42,
        ).generate_input(duration=DURATION, dt=DT)
        self.assertTrue(isinstance(exp_decay, np.ndarray))
        self.assertTupleEqual(exp_decay.shape, SHAPE)
        self.assertTrue(np.all(np.diff(exp_decay) <= 0))

    def test_start_end_input(self):
        exp_rise = ExponentialInput(
            start=STIM_START,
            end=STIM_END,
            inp_max=self.INP_MAX,
            n=2,
            seed=42,
        ).as_array(duration=DURATION, dt=DT)
        np.testing.assert_allclose(exp_rise[:, : int(STIM_START / DT)], 0.0)
        np.testing.assert_allclose(exp_rise[:, int(STIM_END / DT) :], 0.0)

    def test_get_params(self):
        exp_rise = ExponentialInput(
            start=STIM_START,
            end=STIM_END,
            inp_max=self.INP_MAX,
            n=2,
            seed=42,
        )
        params = exp_rise.get_params()
        params.pop("type")
        self.assertDictEqual(
            params,
            {
                "n": 2,
                "seed": 42,
                "inp_max": self.INP_MAX,
                "exp_coef": self.EXP_COEF,
                "exp_type": self.EXP_TYPE,
                "start": STIM_START,
                "end": STIM_END,
            },
        )

    def test_set_params(self):
        exp_rise = ExponentialInput(
            start=STIM_START,
            end=STIM_END,
            inp_max=self.INP_MAX,
            n=2,
            seed=42,
        )
        UPDATE = {"inp_max": 41.0, "seed": 12}
        exp_rise.update_params(UPDATE)
        params = exp_rise.get_params()
        params.pop("type")
        self.assertDictEqual(
            params,
            {
                "n": 2,
                "seed": 42,
                "inp_max": self.INP_MAX,
                "exp_coef": self.EXP_COEF,
                "exp_type": self.EXP_TYPE,
                "start": STIM_START,
                "end": STIM_END,
                **UPDATE,
            },
        )


class TestSummedStimulus(unittest.TestCase):
    def _create_input(self):
        ou = OrnsteinUhlenbeckProcess(mu=0.1, sigma=0.02, tau=2.0, n=2)
        sq = SquareInput(amplitude=0.2, frequency=50, n=2, start=5)
        sin = SinusoidalInput(amplitude=0.1, frequency=100, n=2, start=2)
        step = StepInput(step_size=0.5, n=2, start=7)
        return sq + (sin + step + ou)

    def test_init(self):
        summed = self._create_input()
        self.assertEqual(len(summed), 4)
        self.assertTrue(isinstance(summed, SummedStimulus))
        self.assertEqual(summed.n, 2)
        self.assertEqual(len(summed.inputs), 4)

    def test_set_n(self):
        summed = self._create_input()
        self.assertEqual(summed.n, 2)
        ts = summed.as_array(duration=DURATION, dt=DT)
        self.assertEqual(ts.shape[0], 2)
        summed.n = 5
        self.assertEqual(summed.n, 5)
        ts = summed.as_array(duration=DURATION, dt=DT)
        self.assertEqual(ts.shape[0], 5)

    def test_generate_input(self):
        summed = self._create_input()
        ts = summed.as_array(duration=DURATION, dt=DT)
        self.assertTrue(isinstance(ts, np.ndarray))
        self.assertTupleEqual(ts.shape, SHAPE)

        ts = summed.as_cubic_splines(duration=DURATION, dt=DT)
        self.assertTrue(isinstance(ts, CubicHermiteSpline))

    def test_get_params(self):
        summed = self._create_input()
        params = summed.get_params()
        self.assertTrue(isinstance(params, dict))
        self.assertEqual(len(params), 1 + len(summed.inputs))
        for i, process in enumerate(summed):
            self.assertDictEqual(process.get_params(), params[f"input_{i}"])

    def test_update_params(self):
        summed = self._create_input()
        UPDATE_DICT = {f"input_{i}": {"n": 3} for i in range(len(summed))}
        summed.update_params(UPDATE_DICT)
        self.assertEqual(summed.n, 3)


class TestConcatenatedStimulus(unittest.TestCase):
    def _create_input(self):
        ou = OrnsteinUhlenbeckProcess(mu=0.1, sigma=0.02, tau=2.0, n=2)
        sq = SquareInput(amplitude=0.2, frequency=20.0, n=2)
        sin = SinusoidalInput(amplitude=0.1, frequency=10.0, n=2)
        step = StepInput(step_size=0.5, n=2)
        return ou & (sq & sin & step)

    def test_init(self):
        conc = self._create_input()
        self.assertEqual(len(conc), 4)
        self.assertTrue(isinstance(conc, ConcatenatedStimulus))
        self.assertEqual(conc.n, 2)
        self.assertEqual(len(conc.inputs), 4)

    def test_set_n(self):
        conc = self._create_input()
        self.assertEqual(conc.n, 2)
        ts = conc.as_array(duration=DURATION, dt=DT)
        self.assertEqual(ts.shape[0], 2)
        conc.n = 5
        self.assertEqual(conc.n, 5)
        ts = conc.as_array(duration=DURATION, dt=DT)
        self.assertEqual(ts.shape[0], 5)

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
        self.assertEqual(len(params), 1 + len(conc.inputs))
        for i, process in enumerate(conc):
            self.assertDictEqual(process.get_params(), params[f"input_{i}"])

    def test_update_params(self):
        conc = self._create_input()
        UPDATE_DICT = {f"input_{i}": {"n": 3} for i in range(len(conc))}
        conc.update_params(UPDATE_DICT)
        self.assertEqual(conc.n, 3)


class TestBeastInput(unittest.TestCase):
    def _create_input(self):
        ou = OrnsteinUhlenbeckProcess(mu=0.1, sigma=0.02, tau=2.0, n=2)
        sq = SquareInput(amplitude=0.2, frequency=20.0, n=2)
        sin = SinusoidalInput(amplitude=0.1, frequency=10.0, n=2)
        step = StepInput(step_size=0.5, n=2)
        return (sq + sin) & (step + ou)

    def test_init(self):
        beast = self._create_input()
        self.assertEqual(len(beast), 2)
        self.assertTrue(isinstance(beast, ConcatenatedStimulus))
        for process in beast:
            self.assertTrue(isinstance(process, SummedStimulus))
        self.assertEqual(beast.n, 2)

    def test_set_n(self):
        beast = self._create_input()
        self.assertEqual(beast.n, 2)
        ts = beast.as_array(duration=DURATION, dt=DT)
        self.assertEqual(ts.shape[0], 2)
        beast.n = 5
        self.assertEqual(beast.n, 5)
        ts = beast.as_array(duration=DURATION, dt=DT)
        self.assertEqual(ts.shape[0], 5)

    def test_generate_input(self):
        beast = self._create_input()
        ts = beast.as_array(duration=DURATION, dt=DT)
        self.assertTrue(isinstance(ts, np.ndarray))
        self.assertTupleEqual(ts.shape, SHAPE)

        ts = beast.as_cubic_splines(duration=DURATION, dt=DT)
        self.assertTrue(isinstance(ts, CubicHermiteSpline))

    def test_get_params(self):
        beast = self._create_input()
        params = beast.get_params()
        self.assertTrue(isinstance(params, dict))
        self.assertEqual(len(params), 1 + len(beast.inputs))
        for i, process in enumerate(beast):
            self.assertDictEqual(process.get_params(), params[f"input_{i}"])


class TestRectifiedInput(unittest.TestCase):
    def test_init(self):
        rect = RectifiedInput(0.2, n=2)
        self.assertTrue(isinstance(rect, ConcatenatedStimulus))
        self.assertEqual(len(rect), 5)
        self.assertEqual(rect.n, 2)

    def test_generate(self):
        rect = RectifiedInput(0.2, n=2)
        ts = rect.as_array(DURATION, DT)
        self.assertTrue(isinstance(ts, np.ndarray))
        self.assertTupleEqual(ts.shape, SHAPE)

        ts = rect.as_cubic_splines(duration=DURATION, dt=DT)
        self.assertTrue(isinstance(ts, CubicHermiteSpline))


if __name__ == "__main__":
    unittest.main()
