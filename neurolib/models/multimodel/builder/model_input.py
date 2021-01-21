"""
Handles input to model. Constructs both noisy and stimulation-like input
and supports both CubicHermiteSplines for jitcdde backend and np.array for numba
backend.
"""
import inspect

import numba
import numpy as np
from chspy import CubicHermiteSpline
from scipy.signal import square


class ModelInput:
    """
    Generates input to model.
    """

    def __init__(self, num_iid=1, seed=None):
        """
        :param num_iid: how many independent realisation of
            the input we want - for constant inputs the array is just copied,
            for noise this means independent realisation
        :type num_iid: int
        :param seed: optional seed for noise generator
        :type seed: int|None
        """
        self.num_iid = num_iid
        self.seed = seed
        # seed the generator
        np.random.seed(seed)
        # get parameter names
        self.param_names = inspect.getfullargspec(self.__init__).args
        self.param_names.remove("self")

    def __add__(self, other):
        """
        Concatenate two processes.
        """
        assert isinstance(other, ModelInput)
        assert self.num_iid == other.num_iid
        if isinstance(other, ConcatenatedInput):
            return ConcatenatedInput(noise_processes=[self] + other.noise_processes)
        else:
            return ConcatenatedInput(noise_processes=[self, other])

    def get_params(self):
        """
        Return model input parameters as dict.
        """
        assert all(hasattr(self, name) for name in self.param_names), self.param_names
        params = {name: getattr(self, name) for name in self.param_names}
        return {"type": self.__class__.__name__, **params}

    def update_params(self, params_dict):
        """
        Update model input parameters.

        :param params_dict: new parameters for this model input
        :type params_dict: dict
        """
        for param, value in params_dict.items():
            if hasattr(self, param):
                setattr(self, param, value)

    def _get_times(self, duration, dt):
        """
        Generate time vector.

        :param duration: duration of the input, in miliseconds
        :type duration: float
        :param dt: dt of input, in miliseconds
        :type dt: float
        """
        self.times = np.arange(dt, duration + dt, dt)

    def generate_input(self, duration, dt):
        """
        Function to generate input.

        :param duration: duration of the input, in miliseconds
        :type duration: float
        :param dt: dt of input, in miliseconds
        :type dt: float
        """
        raise NotImplementedError

    def as_array(self, duration, dt):
        """
        Return input as numpy array.

        :param duration: duration of the input, in miliseconds
        :type duration: float
        :param dt: some reasonable "speed" of input, in miliseconds
        :type dt: float
        """
        return self.generate_input(duration, dt)

    def as_cubic_splines(self, duration, dt):
        """
        Return as cubic Hermite splines.

        :param duration: duration of the input, in miliseconds
        :type duration: float
        :param dt: some reasonable "speed" of input, in miliseconds
        :type dt: float
        """
        self._get_times(duration, dt)
        return CubicHermiteSpline.from_data(self.times, self.generate_input(duration, dt))


class StimulusInput(ModelInput):
    """
    Generates stimulus input with optional start and end times.
    """

    def __init__(
        self,
        stim_start=None,
        stim_end=None,
        num_iid=1,
        seed=None,
    ):
        """
        :param stim_start: start of the stimulus, in miliseconds
        :type stim_start: float
        :param stim_end: end of the stimulus, in miliseconds
        :type stim_end: float
        """
        self.stim_start = stim_start
        self.stim_end = stim_end
        super().__init__(
            num_iid=num_iid,
            seed=seed,
        )

    def _get_times(self, duration, dt):
        super()._get_times(duration=duration, dt=dt)
        self.stim_start = self.stim_start or 0.0
        self.stim_end = self.stim_end or duration + dt
        assert self.stim_start < duration
        assert self.stim_end <= duration + dt

    def _trim_stim_input(self, stim_input):
        """
        Trim stimulation input. Translate the start of the stimulation by
        padding with zeros and just nullify end of the stimulation.
        """
        # trim start
        how_much = int(np.sum(self.times < self.stim_start))
        # translate start of the stim by padding the beginning with zeros
        stim_input = np.pad(stim_input, ((how_much, 0), (0, 0)), mode="constant")
        if how_much > 0:
            stim_input = stim_input[:-how_much, :]
        # trim end
        stim_input[self.times > self.stim_end] = 0.0
        return stim_input


class ConcatenatedInput(StimulusInput):
    """
    Represents concatenation of inputs - typically for stimulus plus noise.
    Supports concatenation of arbitrary many objects.
    """

    def __init__(self, noise_processes):
        """
        :param noise_processes: list of noise/stimulation processes to concatinate
        :type noise_processes: list[`ModelInput`]
        """
        assert all(isinstance(process, ModelInput) for process in noise_processes)
        self.noise_processes = noise_processes

    @property
    def num_iid(self):
        num_iid = set([process.num_iid for process in self.noise_processes])
        assert len(num_iid) == 1
        return next(iter(num_iid))

    def __add__(self, other):
        assert isinstance(other, ModelInput)
        assert self.num_iid == other.num_iid
        if isinstance(other, ConcatenatedInput):
            return ConcatenatedInput(noise_processes=self.noise_processes + other.noise_processes)
        else:
            return ConcatenatedInput(noise_processes=self.noise_processes + [other])

    def get_params(self):
        """
        Get parameters recursively from all input processes.
        """
        return {
            "type": self.__class__.__name__,
            **{f"noise_{i}": process.get_params() for i, process in enumerate(self.noise_processes)},
        }

    def update_params(self, params_dict):
        """
        Update all parameters recursively.
        """
        for i, process in enumerate(self.noise_processes):
            process.update_params(params_dict.get(f"noise_{i}", {}))

    def as_array(self, duration, dt):
        """
        Return sum of all processes as numpy array.
        """
        return np.sum(np.stack([process.as_array(duration, dt) for process in self.noise_processes]), axis=0)

    def as_cubic_splines(self, duration, dt):
        """
        Return sum of all processes as cubic Hermite splines.
        """
        result = self.noise_processes[0].as_cubic_splines(duration, dt)
        for process in self.noise_processes[1:]:
            result.plus(process.as_cubic_splines(duration, dt))
        return result


class ZeroInput(ModelInput):
    """
    No noise input, i.e. all zeros. For convenience.
    """

    def generate_input(self, duration, dt):
        self._get_times(duration=duration, dt=dt)
        return np.zeros((self.times.shape[0], self.num_iid))


class WienerProcess(ModelInput):
    """
    Basic Wiener process, dW, i.e. drawn from standard normal N(0, sqrt(dt)).
    """

    def generate_input(self, duration, dt):
        self._get_times(duration=duration, dt=dt)
        return np.random.normal(0.0, np.sqrt(dt), (self.times.shape[0], self.num_iid))


class OrnsteinUhlenbeckProcess(ModelInput):
    """
    Ornstein–Uhlenbeck process, i.e.
        dX = (mu - X)/tau * dt + sigma*dW
    """

    def __init__(
        self,
        mu,
        sigma,
        tau,
        num_iid=1,
        seed=None,
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
            num_iid=num_iid,
            seed=seed,
        )

    def generate_input(self, duration, dt):
        self._get_times(duration=duration, dt=dt)
        x = np.random.rand(self.times.shape[0], self.num_iid) * self.mu
        return self.numba_ou(x, self.times, dt, self.mu, self.sigma, self.tau, self.num_iid)

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
        self,
        step_size,
        stim_start=None,
        stim_end=None,
        num_iid=1,
        seed=None,
    ):
        """
        :param step_size: size of the stimulus
        :type step_size: float
        """
        self.step_size = step_size
        super().__init__(
            stim_start=stim_start,
            stim_end=stim_end,
            num_iid=num_iid,
            seed=seed,
        )

    def generate_input(self, duration, dt):
        self._get_times(duration=duration, dt=dt)
        return self._trim_stim_input(np.ones((self.times.shape[0], self.num_iid)) * self.step_size)


class SinusoidalInput(StimulusInput):
    """
    Sinusoidal input.
    """

    def __init__(
        self,
        amplitude,
        period,
        nonnegative=True,
        stim_start=None,
        stim_end=None,
        num_iid=1,
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
            stim_start=stim_start,
            stim_end=stim_end,
            num_iid=num_iid,
            seed=seed,
        )

    def generate_input(self, duration, dt):
        self._get_times(duration=duration, dt=dt)
        sinusoid = self.amplitude * np.sin(2 * np.pi * self.times * (1.0 / self.period))
        if self.nonnegative:
            sinusoid += self.amplitude
        return self._trim_stim_input(np.vstack([sinusoid] * self.num_iid).T)


class SquareInput(StimulusInput):
    """
    Square input.
    """

    def __init__(
        self,
        amplitude,
        period,
        nonnegative=True,
        stim_start=None,
        stim_end=None,
        num_iid=1,
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
            stim_start=stim_start,
            stim_end=stim_end,
            num_iid=num_iid,
            seed=seed,
        )

    def generate_input(self, duration, dt):
        self._get_times(duration=duration, dt=dt)
        square_inp = self.amplitude * square(2 * np.pi * self.times * (1.0 / self.period))
        if self.nonnegative:
            square_inp += self.amplitude
        return self._trim_stim_input(np.vstack([square_inp] * self.num_iid).T)


class LinearRampInput(StimulusInput):
    """
    Linear ramp input.
    """

    def __init__(
        self,
        inp_max,
        ramp_length,
        stim_start=None,
        stim_end=None,
        num_iid=1,
        seed=None,
    ):
        """
        :param inp_max: maximum of stimulus
        :type inp_max: float
        :param ramp_length: length of linear ramp, in miliseconds
        :type ramp_length: float
        """
        self.inp_max = inp_max
        self.ramp_length = ramp_length
        super().__init__(
            stim_start=stim_start,
            stim_end=stim_end,
            num_iid=num_iid,
            seed=seed,
        )

    def generate_input(self, duration, dt):
        self._get_times(duration=duration, dt=dt)
        # need to adjust times for stimulus start
        times = self.times - self.stim_start
        linear_inp = (self.inp_max / self.ramp_length) * times * (times < self.ramp_length) + self.inp_max * (
            times >= self.ramp_length
        )
        return self._trim_stim_input(np.vstack([linear_inp] * self.num_iid).T)
