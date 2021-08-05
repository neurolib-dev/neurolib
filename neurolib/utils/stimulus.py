"""
Functions for creating stimuli and noise inputs for models.
"""

import inspect
import logging

import numba
import numpy as np
from chspy import CubicHermiteSpline
from ..models.model import Model
from scipy.signal import square


class Input:
    """
    Generates input to model.

    Base class for other input types.
    """

    def __init__(self, n=1, seed=None):
        """
        :param n: Number of spatial dimensions / independent realizations of the input.
            For determinstic inputs, the array is just copied,
            for stociastic / noisy inputs, this means independent realizations.
        :type n: int
        :param seed: Seed for the random number generator.
        :type seed: int|None
        """
        self.n = n
        self.seed = seed
        # seed the generator
        np.random.seed(seed)
        # get parameter names
        self.param_names = inspect.getfullargspec(self.__init__).args
        self.param_names.remove("self")

    def __add__(self, other):
        """
        Sum two inputs into one SummedStimulus.
        """
        assert isinstance(other, Input)
        assert self.n == other.n
        if isinstance(other, SummedStimulus):
            return SummedStimulus(inputs=[self] + other.inputs)
        else:
            return SummedStimulus(inputs=[self, other])

    def __and__(self, other):
        """
        Concatenate two inputs into ConcatenatedStimulus.
        """
        assert isinstance(other, Input)
        assert self.n == other.n
        if isinstance(other, ConcatenatedStimulus):
            return ConcatenatedStimulus(inputs=[self] + other.inputs, length_ratios=[1] + other.length_ratios)
        else:
            return ConcatenatedStimulus(inputs=[self, other])

    def _reset(self):
        """
        Reset is called after generating an input. Can be used to reset
        intrinsic properties.
        """
        pass

    def get_params(self):
        """
        Return the parameters of the input as dict.
        """
        assert all(hasattr(self, name) for name in self.param_names), self.param_names
        params = {name: getattr(self, name) for name in self.param_names}
        return {"type": self.__class__.__name__, **params}

    def update_params(self, params_dict):
        """
        Update model input parameters.

        :param params_dict: New parameters for this input
        :type params_dict: dict
        """

        def _sanitize(value):
            """
            Change string `None` to actual None - can happen with Exploration or
            Evolution, since `pypet` does None -> "None".
            """
            if value == "None":
                return None
            else:
                return value

        for param, value in params_dict.items():
            if hasattr(self, param):
                setattr(self, param, _sanitize(value))

    def _get_times(self, duration, dt):
        """
        Generate time vector.

        :param duration: Duration of the input, in milliseconds
        :type duration: float
        :param dt: dt of input, in milliseconds
        :type dt: float
        """
        self.times = np.arange(dt, duration + dt, dt)

    def generate_input(self, duration, dt):
        """
        Function to generate input.

        :param duration: Duration of the input, in milliseconds
        :type duration: float
        :param dt: dt of input, in milliseconds
        :type dt: float
        """
        raise NotImplementedError

    def as_array(self, duration, dt):
        """
        Return input as numpy array.

        :param duration: Duration of the input, in milliseconds
        :type duration: float
        :param dt: dt of input, in milliseconds
        :type dt: float
        """
        array = self.generate_input(duration, dt)
        self._reset()
        return array

    def as_cubic_splines(self, duration, dt, shift_start_time=0.0):
        """
        Return as cubic Hermite splines.

        :param duration: Duration of the input, in milliseconds
        :type duration: float
        :param dt: dt of input, in milliseconds
        :type dt: float
        :param shift_start_time: By how much to shift the stimulus start time
        :type shift_start_time: float
        """
        self._get_times(duration, dt)
        splines = CubicHermiteSpline.from_data(self.times + shift_start_time, self.generate_input(duration, dt).T)
        self._reset()
        return splines

    def to_model(self, model):
        """
        Return numpy array of stimuli based on model parameters.

        Example:
        ```
        model.params["ext_exc_input"] = SinusoidalInput(...).to_model(model)
        ```

        :param model: neurolib's model
        :type model: `neurolib.models.Model`
        """
        assert isinstance(model, Model)
        # set number of spatial dimensions as the number of nodes in the brian network
        self.n = model.params["N"]
        return self.as_array(duration=model.params["duration"], dt=model.params["dt"])


class Stimulus(Input):
    """
    Generates a stimulus with optional start and end times.
    """

    def __init__(
        self,
        start=None,
        end=None,
        n=1,
        seed=None,
    ):
        """
        :param start: start of the stimulus, in milliseconds
        :type start: float
        :param end: end of the stimulus, in milliseconds
        :type end: float
        """
        self.start = start
        self.end = end
        self._default_start = start
        self._default_end = end
        super().__init__(
            n=n,
            seed=seed,
        )

    def _reset(self):
        self.start = self._default_start
        self.end = self._default_end

    def _get_times(self, duration, dt):
        super()._get_times(duration=duration, dt=dt)
        self.start = self.start or 0.0
        self.end = self.end or duration + dt
        assert self.start < duration
        assert self.end <= duration + dt

    def _trim_stim(self, stim_input):
        """
        Trim stimulus. Translate the start of the stimulus by
        padding the beginning and replace the end with zeros.
        """
        # trim start
        how_much = int(np.sum(self.times <= self.start))
        # translate start of the stim by padding the beginning with zeros
        stim_input = np.pad(stim_input, ((0, 0), (how_much, 0)), mode="constant")
        if how_much > 0:
            stim_input = stim_input[:, :-how_much]
        # trim end
        stim_input[:, self.times > self.end] = 0.0
        return stim_input


class BaseMultipleInputs(Stimulus):
    """
    Base class for stimuli consisting of multiple time series, such as summed inputs or concatenated inputs.
    """

    def __init__(self, inputs):
        """
        :param inputs: List of Inputs to combine
        :type inputs: list[`Input`]
        """
        assert all(isinstance(input, Input) for input in inputs)
        self.inputs = inputs

    def __len__(self):
        """
        Return number of inputs.
        """
        return len(self.inputs)

    def __getitem__(self, index):
        """
        Return inputs by index. This also allows iteration.
        """
        return self.inputs[index]

    @property
    def n(self):
        n = set([input.n for input in self])
        assert len(n) == 1
        return next(iter(n))

    @n.setter
    def n(self, n):
        for input in self:
            input.n = n

    def get_params(self):
        """
        Get all parameters recursively for all inputs.
        """
        return {
            "type": self.__class__.__name__,
            **{f"input_{i}": input.get_params() for i, input in enumerate(self)},
        }

    def update_params(self, params_dict):
        """
        Update all parameters recursively.
        """
        for i, input in enumerate(self):
            input.update_params(params_dict.get(f"input_{i}", {}))


class SummedStimulus(BaseMultipleInputs):
    """
    Represents the summation of arbitrary many stimuli.

    Example:
    ```
        summed_stimulus = SinusoidalInput(...) + OrnsteinUhlenbeckProcess(...)
    ```
    """

    def __add__(self, other):
        assert isinstance(other, Input)
        assert self.n == other.n
        if isinstance(other, SummedStimulus):
            return SummedStimulus(inputs=self.inputs + other.inputs)
        else:
            return SummedStimulus(inputs=self.inputs + [other])

    def as_array(self, duration, dt):
        """
        Return sum of all inputes as numpy array.
        """
        return np.sum(
            np.stack([input.as_array(duration, dt) for input in self.inputs]),
            axis=0,
        )

    def as_cubic_splines(self, duration, dt, shift_start_time=0.0):
        """
        Return sum of all inputes as cubic Hermite splines.
        """
        result = self.inputs[0].as_cubic_splines(duration, dt, shift_start_time)
        for input in self.inputs[1:]:
            result.plus(input.as_cubic_splines(duration, dt, shift_start_time))
        return result


class ConcatenatedStimulus(BaseMultipleInputs):
    """
    Represents temporal concatenation of of arbitrary many stimuli.

    Example:
    ```
        summed_stimulus = SinusoidalInput(...) & OrnsteinUhlenbeckProcess(...)
    ```
    """

    def __init__(self, inputs, length_ratios=None):
        """
        :param length_ratios: Ratios of lengths of concatenated stimuli
        :type length_ratios: list[int|float]
        """
        if length_ratios is None:
            length_ratios = [1] * len(inputs)
        assert len(inputs) == len(length_ratios)
        assert all(length > 0 for length in length_ratios)
        self.length_ratios = length_ratios
        super().__init__(inputs)

    def __and__(self, other):
        assert isinstance(other, Input)
        assert self.n == other.n
        if isinstance(other, ConcatenatedStimulus):
            return ConcatenatedStimulus(
                inputs=self.inputs + other.inputs,
                length_ratios=self.length_ratios + other.length_ratios,
            )
        else:
            return ConcatenatedStimulus(inputs=self.inputs + [other], length_ratios=self.length_ratios + [1])

    def as_array(self, duration, dt):
        """
        Return concatenation of all stimuli as numpy array.
        """
        # normalize ratios to sum = 1
        ratios = [i / sum(self.length_ratios) for i in self.length_ratios]
        concat = np.concatenate(
            [input.as_array(duration * ratio, dt) for input, ratio in zip(self.inputs, ratios)],
            axis=1,
        )
        length = int(duration / dt)
        # due to rounding errors, the overall length might be longer by a few dt
        return concat[:, :length]

    def as_cubic_splines(self, duration, dt, shift_start_time=0.0):
        # normalize ratios to sum = 1
        ratios = [i / sum(self.length_ratios) for i in self.length_ratios]
        result = self.inputs[0].as_cubic_splines(duration * ratios[0], dt, shift_start_time)
        for input, ratio in zip(self.inputs[1:], ratios[1:]):
            last_time = result[-1].time
            temp = input.as_cubic_splines(duration * ratio, dt, shift_start_time=last_time)
            # `extend` adds an iteratable (whole `CubicHermiteSpline` is an
            # iterable of `Anchors`) to the current spline
            result.extend(temp)
        return result


class ZeroInput(Input):
    """
    No stimulus, i.e. all zeros. Can be used to add a delay between two stimuli.
    """

    def generate_input(self, duration, dt):
        self._get_times(duration=duration, dt=dt)
        return np.zeros((self.n, self.times.shape[0]))


class WienerProcess(Input):
    """
    Stimulus sampled from a Wiener process, i.e. drawn from standard normal distribution N(0, sqrt(dt)).
    """

    def generate_input(self, duration, dt):
        self._get_times(duration=duration, dt=dt)
        return np.random.normal(0.0, np.sqrt(dt), (self.n, self.times.shape[0]))


class OrnsteinUhlenbeckProcess(Input):
    """
    Ornstein–Uhlenbeck input, i.e.
        dX = (mu - X)/tau * dt + sigma*dW
    """

    def __init__(
        self,
        mu,
        sigma,
        tau,
        n=1,
        seed=None,
    ):
        """
        :param mu: Drift of the OU process
        :type mu: float
        :param sigma: Standard deviation of the Wiener process, i.e. strength of the noise
        :type sigma: float
        :param tau: Timescale of the OU process, in ms
        :type tau: float
        """
        self.mu = mu
        self.sigma = sigma
        self.tau = tau
        super().__init__(
            n=n,
            seed=seed,
        )

    def generate_input(self, duration, dt):
        self._get_times(duration=duration, dt=dt)
        x = np.random.rand(self.n, self.times.shape[0]) * self.mu
        return self.numba_ou(x, self.times, dt, self.mu, self.sigma, self.tau, self.n)

    @staticmethod
    @numba.njit()
    def numba_ou(x, times, dt, mu, sigma, tau, n):
        """
        Generation of Ornstein-Uhlenback input - wrapped in numba's jit for
        speed.
        """
        for i in range(times.shape[0] - 1):
            x[:, i + 1] = x[:, i] + dt * ((mu - x[:, i]) / tau) + sigma * np.sqrt(dt) * np.random.randn(n)
        return x


class StepInput(Stimulus):
    """
    Step input.
    """

    def __init__(
        self,
        step_size,
        start=None,
        end=None,
        n=1,
        seed=None,
    ):
        """
        :param step_size: Size of the step, i.e., the amplitude.
        :type step_size: float
        """
        self.step_size = step_size
        super().__init__(
            start=start,
            end=end,
            n=n,
            seed=seed,
        )

    def generate_input(self, duration, dt):
        self._get_times(duration=duration, dt=dt)
        return self._trim_stim(np.ones((self.n, self.times.shape[0])) * self.step_size)


class SinusoidalInput(Stimulus):
    """
    Sinusoidal input.
    """

    def __init__(
        self,
        amplitude,
        frequency,
        dc_bias=False,
        start=None,
        end=None,
        n=1,
        seed=None,
    ):
        """
        :param amplitude: Amplitude of the sinusoid.
        :type amplitude: float
        :param frequency: Frequency of the sinus oscillation, in Hz
        :type frequency: float
        :param dc_bias: Whether the sinusoid oscillates around 0
            (False), or has a positive DC bias, thus non-negative (True).
        :type dc_bias: bool
        """
        self.amplitude = amplitude
        self.frequency = frequency
        self.dc_bias = dc_bias
        super().__init__(
            start=start,
            end=end,
            n=n,
            seed=seed,
        )

    def generate_input(self, duration, dt):
        self._get_times(duration=duration, dt=dt)
        sinusoid = self.amplitude * np.sin(2 * np.pi * self.times * (self.frequency / 1000.0))
        if self.dc_bias:
            sinusoid += self.amplitude
        return self._trim_stim(np.vstack([sinusoid] * self.n))


class SquareInput(Stimulus):
    """
    Oscillatory square input.
    """

    def __init__(
        self,
        amplitude,
        frequency,
        dc_bias=False,
        start=None,
        end=None,
        n=1,
        seed=None,
    ):
        """
        :param amplitude: Amplitude of the square
        :type amplitude: float
        :param frequency: Frequency of the square oscillation, in Hz
        :type frequency: float
        :param dc_bias: Whether the square oscillates around 0
            (False), or has a positive DC bias, thus non-negative (True).
        :type dc_bias: bool
        """
        self.amplitude = amplitude
        self.frequency = frequency
        self.dc_bias = dc_bias
        super().__init__(
            start=start,
            end=end,
            n=n,
            seed=seed,
        )

    def generate_input(self, duration, dt):
        self._get_times(duration=duration, dt=dt)
        square_inp = self.amplitude * square(2 * np.pi * self.times * (self.frequency / 1000.0))
        if self.dc_bias:
            square_inp += self.amplitude
        return self._trim_stim(np.vstack([square_inp] * self.n))


class LinearRampInput(Stimulus):
    """
    Linear ramp input.
    """

    def __init__(
        self,
        inp_max,
        ramp_length,
        start=None,
        end=None,
        n=1,
        seed=None,
    ):
        """
        :param inp_max: Maximum of stimulus.
        :type inp_max: float
        :param ramp_length: Duration of linear ramp, in milliseconds
        :type ramp_length: float
        """
        self.inp_max = inp_max
        self.ramp_length = ramp_length
        super().__init__(
            start=start,
            end=end,
            n=n,
            seed=seed,
        )

    def generate_input(self, duration, dt):
        self._get_times(duration=duration, dt=dt)
        linear_inp = (self.inp_max / self.ramp_length) * self.times * (self.times < self.ramp_length) + self.inp_max * (
            self.times >= self.ramp_length
        )
        return self._trim_stim(np.vstack([linear_inp] * self.n))


class ExponentialInput(Stimulus):
    """
    Exponential rise or decay input.
    """

    def __init__(
        self,
        inp_max,
        exp_coef=30.0,
        exp_type="rise",
        start=None,
        end=None,
        n=1,
        seed=None,
    ):
        """
        :param inp_max: Maximum of stimulus.
        :type inp_max: float
        :param exp_coeficient: Coeffiecent for the exponential (the higher the
            coefficient, the faster it rises or decays).
        :type exp_coeficient: float
        :param exp_type: Whether to "rise" or to "decay".
        :type exp_type: str
        """
        self.inp_max = inp_max
        self.exp_coef = exp_coef
        assert exp_type in ["rise", "decay"]
        self.exp_type = exp_type
        super().__init__(
            start=start,
            end=end,
            n=n,
            seed=seed,
        )

    def generate_input(self, duration, dt):
        self._get_times(duration=duration, dt=dt)
        exponential = np.exp(-(self.exp_coef / self.times[-1]) * self.times) * self.inp_max
        if self.exp_type == "rise":
            exponential = -exponential + self.inp_max
        return self._trim_stim(np.vstack([exponential] * self.n))


def RectifiedInput(amplitude, n=1):
    """
    Return rectified input with exponential decay, i.e. a negative step followed by a
    slow decay to zero, followed by a positive step and again a slow decay to zero.
    Can be used for bistablity detection.

    :param amplitude: Amplitude (both negative and positive) for the step
    :type amplitude: float
    :param n: Number of realizations (spatial dimension)
    :type n: int
    :return: Concatenated input which represents the rectified stimulus with exponential decay
    :rtype: `ConctatenatedInput`
    """

    return ConcatenatedStimulus(
        [
            StepInput(step_size=-amplitude, n=n),
            ExponentialInput(inp_max=amplitude, exp_type="rise", exp_coef=12.5, n=n)
            + StepInput(step_size=-amplitude, n=n),
            StepInput(step_size=amplitude, n=n),
            ExponentialInput(amplitude, exp_type="decay", exp_coef=7.5, n=n),
            StepInput(step_size=0.0, n=n),
        ],
        length_ratios=[0.5, 2.5, 0.5, 1.5, 1.0],
    )
