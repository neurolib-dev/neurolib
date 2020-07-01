"""
Tests of noise input.
"""

import unittest

import numpy as np
from chspy import CubicHermiteSpline

from neurolib.models.multimodel.builder.model_input import (
    LinearRampInput,
    OrnsteinUhlenbeckProcess,
    SinusoidalInput,
    SquareInput,
    StepInput,
    WienerProcess,
    ZeroInput,
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
        dW = WienerProcess(duration=DURATION, dt=DT, independent_realisations=2, seed=42).as_cubic_splines()
        self.assertTrue(isinstance(dW, CubicHermiteSpline))
        np.testing.assert_allclose(self.RESULT_SPLINES, dW.get_state(TESTING_TIME))

    def test_arrays(self):
        dW = WienerProcess(duration=DURATION, dt=DT, independent_realisations=2, seed=42).as_array()
        self.assertTrue(isinstance(dW, np.ndarray))
        time_idx = np.around(TESTING_TIME / DT).astype(int)
        np.testing.assert_allclose(self.RESULT_ARRAY, dW[time_idx, :])


class TestZeroInput(unittest.TestCase):
    def test_generate_input(self):
        nn = ZeroInput(duration=DURATION, dt=DT, independent_realisations=2, seed=42).generate_input()
        self.assertTrue(isinstance(nn, np.ndarray))
        self.assertTupleEqual(nn.shape, SHAPE)
        np.testing.assert_allclose(nn, np.zeros(SHAPE))


class TestWienerProcess(unittest.TestCase):
    def test_generate_input(self):
        dW = WienerProcess(duration=DURATION, dt=DT, independent_realisations=2, seed=42).generate_input()
        self.assertTrue(isinstance(dW, np.ndarray))
        self.assertTupleEqual(dW.shape, SHAPE)


class TestOrnsteinUhlenbeckProcess(unittest.TestCase):
    def test_generate_input(self):
        ou = OrnsteinUhlenbeckProcess(
            duration=DURATION, dt=DT, mu=3.0, sigma=0.1, tau=2 * DT, independent_realisations=2, seed=42,
        ).generate_input()
        self.assertTrue(isinstance(ou, np.ndarray))
        self.assertTupleEqual(ou.shape, SHAPE)


class TestStepInput(unittest.TestCase):
    STEP_SIZE = 2.3

    def test_generate_input(self):
        step = StepInput(
            duration=DURATION, dt=DT, step_size=self.STEP_SIZE, independent_realisations=2, seed=42,
        ).generate_input()
        self.assertTrue(isinstance(step, np.ndarray))
        self.assertTupleEqual(step.shape, SHAPE)
        np.testing.assert_allclose(step, self.STEP_SIZE)

    def test_start_end_input(self):
        step = StepInput(
            duration=DURATION,
            dt=DT,
            stim_start=STIM_START,
            stim_end=STIM_END,
            step_size=self.STEP_SIZE,
            independent_realisations=2,
            seed=42,
        ).as_array()
        np.testing.assert_allclose(step[: int(STIM_START / DT) - 1, :], 0.0)
        np.testing.assert_allclose(step[int(STIM_END / DT) :, :], 0.0)


class TestSinusoidalInput(unittest.TestCase):
    AMPLITUDE = 2.3
    PERIOD = 2.0

    def test_generate_input(self):

        sin = SinusoidalInput(
            duration=DURATION, dt=DT, amplitude=self.AMPLITUDE, period=self.PERIOD, independent_realisations=2, seed=42,
        ).generate_input()
        self.assertTrue(isinstance(sin, np.ndarray))
        self.assertTupleEqual(sin.shape, SHAPE)
        np.testing.assert_equal(np.mean(sin, axis=0), np.array(2 * [self.AMPLITUDE]))

    def test_start_end_input(self):
        sin = SinusoidalInput(
            duration=DURATION,
            dt=DT,
            stim_start=STIM_START,
            stim_end=STIM_END,
            amplitude=self.AMPLITUDE,
            period=self.PERIOD,
            independent_realisations=2,
            seed=42,
        ).as_array()
        np.testing.assert_allclose(sin[: int(STIM_START / DT) - 1, :], 0.0)
        np.testing.assert_allclose(sin[int(STIM_END / DT) :, :], 0.0)


class TestSquareInput(unittest.TestCase):
    AMPLITUDE = 2.3
    PERIOD = 2.0

    def test_generate_input(self):

        sq = SquareInput(
            duration=DURATION, dt=DT, amplitude=self.AMPLITUDE, period=self.PERIOD, independent_realisations=2, seed=42,
        ).generate_input()
        self.assertTrue(isinstance(sq, np.ndarray))
        self.assertTupleEqual(sq.shape, SHAPE)
        np.testing.assert_equal(np.mean(sq, axis=0), np.array(2 * [self.AMPLITUDE]))

    def test_start_end_input(self):
        sq = SquareInput(
            duration=DURATION,
            dt=DT,
            stim_start=STIM_START,
            stim_end=STIM_END,
            amplitude=self.AMPLITUDE,
            period=self.PERIOD,
            independent_realisations=2,
            seed=42,
        ).as_array()
        np.testing.assert_allclose(sq[: int(STIM_START / DT) - 1, :], 0.0)
        np.testing.assert_allclose(sq[int(STIM_END / DT) :, :], 0.0)


class TestLinearRampInput(unittest.TestCase):
    INP_MAX = 5.0
    RAMP_LENGTH = 2.0

    def test_generate_input(self):

        ramp = LinearRampInput(
            duration=DURATION,
            dt=DT,
            input_max=self.INP_MAX,
            ramp_length=self.RAMP_LENGTH,
            independent_realisations=2,
            seed=42,
        ).generate_input()
        self.assertTrue(isinstance(ramp, np.ndarray))
        self.assertTupleEqual(ramp.shape, SHAPE)
        np.testing.assert_equal(np.max(ramp, axis=0), np.array(2 * [self.INP_MAX]))
        np.testing.assert_equal(np.min(ramp, axis=0), np.array(2 * [0.25]))

    def test_start_end_input(self):
        ramp = LinearRampInput(
            duration=DURATION,
            dt=DT,
            stim_start=STIM_START,
            stim_end=STIM_END,
            input_max=self.INP_MAX,
            ramp_length=self.RAMP_LENGTH,
            independent_realisations=2,
            seed=42,
        ).as_array()
        np.testing.assert_allclose(ramp[: int(STIM_START / DT) - 1, :], 0.0)
        np.testing.assert_allclose(ramp[int(STIM_END / DT) :, :], 0.0)


if __name__ == "__main__":
    unittest.main()
