"""
Handles input to neural models. Constructs both noisy and stimulation-like input
and supports both CubicHermiteSplines for jitcdde backend and np.array for numba
backend.
"""

import numba
import numpy as np
from chspy import CubicHermiteSpline
from scipy.signal import square


class ModelInput:
    """
    Generates input to neural model.
    """

    def __init__(self, duration, dt, independent_realisations=1, seed=None):
        """
        :param duration: duration of the input, in miliseconds
        :type duration: float
        :param dt: some reasonable "speed" of input, in miliseconds
        :type dt: float
        :param independent_realisations: how many independent realisation of
            the input we want - for constant inputs the array is just copied,
            for noise this means independent realisation
        :type independent_realisations: int
        :param seed: optional seed for noise generator
        :type seed: int|None
        """
        self.times = np.arange(dt, duration + dt, dt)
        self.dt = dt
        self.num_iid = independent_realisations
        # seed the generator
        np.random.seed(seed)

    def generate_input(self):
        """
        Function to generate input.
        """
        raise NotImplementedError

    def as_array(self):
        """
        Return input as numpy array.
        """
        return self.generate_input()

    def as_cubic_splines(self):
        """
        Return as cubic Hermite splines.
        """
        return CubicHermiteSpline.from_data(self.times, self.generate_input())


class StimulusInput(ModelInput):
    """
    Generates stimulus input with optional start and end times.
    """

    def __init__(
        self, duration, dt, stim_start=None, stim_end=None, independent_realisations=1, seed=None,
    ):
        """
        :param stim_start: start of the stimulus, in miliseconds
        :type stim_start: float
        :param stim_end: end of the stimulus, in miliseconds
        :type stim_end: float
        """
        self.stim_start = stim_start or 0.0
        self.stim_end = stim_end or duration
        assert self.stim_start < duration
        assert self.stim_end <= duration
        super().__init__(
            duration=duration, dt=dt, independent_realisations=independent_realisations, seed=seed,
        )

    def as_array(self):
        """
        Return input as numpy array after checking stimulus bounds.
        """
        stim_input = self.generate_input()
        stim_input[self.times < self.stim_start] = 0.0
        stim_input[self.times > self.stim_end] = 0.0
        return stim_input

    def as_cubic_splines(self):
        """
        Return as cubic Hermite splines after checking stimulus bounds.
        """
        stim_input = self.generate_input()
        stim_input[self.times < self.stim_start] = 0.0
        stim_input[self.times > self.stim_end] = 0.0
        return CubicHermiteSpline.from_data(self.times, stim_input)


class ZeroInput(ModelInput):
    """
    No noise input, i.e. all zeros. For convenience.
    """

    def generate_input(self):
        return np.zeros((self.times.shape[0], self.num_iid))


class WienerProcess(ModelInput):
    """
    Basic Wiener process, dW, i.e. drawn from standard normal N(0, sqrt(dt)).
    """

    def generate_input(self):
        return np.random.normal(0.0, np.sqrt(self.dt), (self.times.shape[0], self.num_iid))


class OrnsteinUhlenbeckProcess(ModelInput):
    """
    Ornstein–Uhlenbeck process, i.e.
        dX = (mu - X)/tau * dt + sigma*dW
    """

    def __init__(
        self, duration, dt, mu, sigma, tau, independent_realisations=1, seed=None,
    ):
        """
        :param mu: drift of the O-U process
        :type mu: float
        :param sigma: scale of the Wiener process
        :type sigma: float
        :param tau: O-U process timescale, same unit as time
        :type tau: float
        """
        self.mu = mu
        self.sigma = sigma
        self.tau = tau
        super().__init__(
            duration=duration, dt=dt, independent_realisations=independent_realisations, seed=seed,
        )

    def generate_input(self):
        x = np.random.rand(self.times.shape[0], self.num_iid) * self.mu
        return self.numba_ou(x, self.times, self.dt, self.mu, self.sigma, self.tau, self.num_iid)

    @staticmethod
    @numba.njit()
    def numba_ou(x, times, dt, mu, sigma, tau, num_iid):
        """
        Generation of Ornstein-Uhlenback process - wrapped in numba's jit for
        speed.
        """
        for i in range(times.shape[0] - 1):
            x[i + 1, :] = x[i, :] + dt * ((mu - x[i, :]) / tau) + sigma * np.sqrt(dt) * np.random.randn(num_iid)
        return x


class StepInput(StimulusInput):
    """
    Basic step process.
    """

    def __init__(
        self, duration, dt, step_size, stim_start=None, stim_end=None, independent_realisations=1, seed=None,
    ):
        """
        :param step_size: size of the stimulus
        :type step_size: float
        """
        self.step_size = step_size
        super().__init__(
            duration=duration,
            dt=dt,
            stim_start=stim_start,
            stim_end=stim_end,
            independent_realisations=independent_realisations,
            seed=seed,
        )

    def generate_input(self):
        return np.ones((self.times.shape[0], self.num_iid)) * self.step_size


class SinusoidalInput(StimulusInput):
    """
    Sinusoidal input.
    """

    def __init__(
        self,
        duration,
        dt,
        amplitude,
        period,
        nonnegative=True,
        stim_start=None,
        stim_end=None,
        independent_realisations=1,
        seed=None,
    ):
        """
        :param amplitude: amplitude of the sinusoid
        :type amplitude: float
        :param period: period of the sinusoid, in miliseconds
        :type period: float
        :param nonnegative: whether the sinusoid oscillates around 0 point
            (False), or around its amplitude, thus is nonnegative (True)
        :type nonnegative: bool
        """
        self.amplitude = amplitude
        self.period = period
        self.nonnegative = nonnegative
        super().__init__(
            duration=duration,
            dt=dt,
            stim_start=stim_start,
            stim_end=stim_end,
            independent_realisations=independent_realisations,
            seed=seed,
        )

    def generate_input(self):
        sinusoid = self.amplitude * np.sin(2 * np.pi * self.times * (1.0 / self.period))
        if self.nonnegative:
            sinusoid += self.amplitude
        return np.vstack([sinusoid] * self.num_iid).T


class SquareInput(StimulusInput):
    """
    Square input.
    """

    def __init__(
        self,
        duration,
        dt,
        amplitude,
        period,
        nonnegative=True,
        stim_start=None,
        stim_end=None,
        independent_realisations=1,
        seed=None,
    ):
        """
        :param amplitude: amplitude of the square
        :type amplitude: float
        :param period: period of the square, in miliseconds
        :type period: float
        :param nonnegative: whether the square oscillates around 0 point
            (False), or around its amplitude, thus is nonnegative (True)
        :type nonnegative: bool
        """
        self.amplitude = amplitude
        self.period = period
        self.nonnegative = nonnegative
        super().__init__(
            duration=duration,
            dt=dt,
            stim_start=stim_start,
            stim_end=stim_end,
            independent_realisations=independent_realisations,
            seed=seed,
        )

    def generate_input(self):
        square_inp = self.amplitude * square(2 * np.pi * self.times * (1.0 / self.period))
        if self.nonnegative:
            square_inp += self.amplitude
        return np.vstack([square_inp] * self.num_iid).T


class LinearRampInput(StimulusInput):
    """
    Linear ramp input.
    """

    def __init__(
        self,
        duration,
        dt,
        input_max,
        ramp_length,
        stim_start=None,
        stim_end=None,
        independent_realisations=1,
        seed=None,
    ):
        """
        :param input_max: maximum of stimulus
        :type input_max: float
        :param ramp_length: length of linear ramp, in miliseconds
        :type ramp_length: float
        """
        self.inp_max = input_max
        self.ramp_length = ramp_length
        super().__init__(
            duration=duration,
            dt=dt,
            stim_start=stim_start,
            stim_end=stim_end,
            independent_realisations=independent_realisations,
            seed=seed,
        )

    def generate_input(self):
        # need to adjust times for stimulus start
        times = self.times - self.stim_start
        linear_inp = (self.inp_max / self.ramp_length) * times * (times < self.ramp_length) + self.inp_max * (
            times >= self.ramp_length
        )
        return np.vstack([linear_inp] * self.num_iid).T
