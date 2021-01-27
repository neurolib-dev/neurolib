"""
Functions for creating stimulus and noise inputs for the models.
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
        how_much = int(np.sum(self.times <= self.stim_start))
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
        return np.sum(
            np.stack([process.as_array(duration, dt) for process in self.noise_processes]),
            axis=0,
        )

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


class ExponentialInput(StimulusInput):
    """
    Exponential rise or decay input.
    """

    def __init__(
        self,
        inp_max,
        exp_coef=30.0,
        exp_type="rise",
        stim_start=None,
        stim_end=None,
        num_iid=1,
        seed=None,
    ):
        """
        :param inp_max: maximum of stimulus
        :type inp_max: float
        :param exp_coeficient: coeffiecent for exponential, the higher the
            coefficient is, the faster it rises or decays
        :type exp_coeficient: float
        :param exp_type: whether its rise or decay
        :type exp_type: str
        """
        self.inp_max = inp_max
        self.exp_coef = exp_coef
        assert exp_type in ["rise", "decay"]
        self.exp_type = exp_type
        super().__init__(
            stim_start=stim_start,
            stim_end=stim_end,
            num_iid=num_iid,
            seed=seed,
        )

    def generate_input(self, duration, dt):
        self._get_times(duration=duration, dt=dt)
        exponential = np.exp(-(self.exp_coef / self.times[-1]) * self.times) * self.inp_max
        if self.exp_type == "rise":
            exponential = -exponential + self.inp_max
        return self._trim_stim_input(np.vstack([exponential] * self.num_iid).T)


def construct_stimulus(
    stim="dc",
    duration=6000,
    dt=0.1,
    stim_amp=0.2,
    stim_freq=1,
    stim_bias=0,
    n_periods=None,
    nostim_before=0,
    nostim_after=0,
):
    """Constructs a stimulus that can be applied to a model

    :param stim: Stimulation type: 'ac':oscillatory stimulus, 'dc': stimple step current,
                'rect': step current in negative then positive direction with slowly
                decaying amplitude, used for bistability detection, defaults to 'dc'
    :type stim: str, optional
    :param duration: Duration of stimulus in ms, defaults to 6000
    :type duration: int, optional
    :param dt: Integration time step in ms, defaults to 0.1
    :type dt: float, optional
    :param stim_amp: Amplitude of stimulus (for AdEx: in mV/ms, multiply by conductance C to get current in pA), defaults to 0.2
    :type stim_amp: float, optional
    :param stim_freq: Stimulation frequency, defaults to 1
    :type stim_freq: int, optional
    :param stim_bias: Stimulation offset (bias), defaults to 0
    :type stim_bias: int, optional
    :param n_periods: Numer of periods of stimulus, defaults to None
    :type n_periods: [type], optional
    :param nostim_before: Time before stimulation, defaults to 0
    :type nostim_before: int, optional
    :param nostim_after: Time after stimulation, defaults to 0
    :type nostim_after: int, optional
    :raises ValueError: Raises error if unsupported stimulus type is chosen.
    :return: Stimulus timeseries
    :rtype: numpy.ndarray
    """
    """Constructs a sitmulus that can be applied as input to a model

    TODO: rewrite

    stim:       Stimulus type: 'ac':oscillatory stimulus, 'dc': stimple step current, 
                'rect': step current in negative then positive direction with slowly
                decaying amplitude, used for bistability detection
    stim_amp:   Amplitude of stimulus (for AdEx: in mV/ms, multiply by conductance C to get current in pA)
    """

    def sinus_stim(f=1, amplitude=0.2, positive=0, phase=0, cycles=1, t_pause=0):
        x = np.linspace(np.pi, -np.pi, int(1000 / dt / f))
        sinus_function = np.hstack(((np.sin(x + phase) + positive), np.tile(0, t_pause)))
        sinus_function *= amplitude
        return np.tile(sinus_function, cycles)

    if stim == "ac":
        """Oscillatory stimulus"""
        n_periods = n_periods or int(stim_freq)

        stimulus = np.hstack(
            (
                [stim_bias] * int(nostim_before / dt),
                np.tile(sinus_stim(stim_freq, stim_amp) + stim_bias, n_periods),
            )
        )
        stimulus = np.hstack((stimulus, [stim_bias] * int(nostim_after / dt)))
    elif stim == "dc":
        """Simple DC input and return to baseline"""
        stimulus = np.hstack(([stim_bias] * int(nostim_before / dt), [stim_bias + stim_amp] * int(1000 / dt)))
        stimulus = np.hstack((stimulus, [stim_bias] * int(nostim_after / dt)))
        stimulus[stimulus < 0] = 0
    elif stim == "rect":
        """Rectified step current with slow decay"""
        # construct input
        stimulus = np.zeros(int(duration / dt))
        tot_len = int(duration / dt)
        stim_epoch = tot_len / 6

        stim_increase_counter = 0
        stim_decrease_counter = 0
        stim_step_increase = 5.0 / stim_epoch

        for i, m in enumerate(stimulus):
            if 0 * stim_epoch <= i < 0.5 * stim_epoch:
                stimulus[i] -= stim_amp
            elif 0.5 * stim_epoch <= i < 3.0 * stim_epoch:
                stimulus[i] = -np.exp(-stim_increase_counter) * stim_amp
                stim_increase_counter += stim_step_increase
            elif 3.0 * stim_epoch <= i < 3.5 * stim_epoch:
                stimulus[i] += stim_amp
            elif 3.5 * stim_epoch <= i < 5 * stim_epoch:
                stimulus[i] = np.exp(-stim_decrease_counter) * stim_amp
                stim_decrease_counter += stim_step_increase
    else:
        raise ValueError(f'Stimulus {stim} not found. Use "ac", "dc" or "rect".')

    # repeat stimulus until full length
    steps = int(duration / dt)
    stimlength = int(len(stimulus))
    stimulus = np.tile(stimulus, int(steps / stimlength + 2))
    stimulus = stimulus[:steps]

    return stimulus
