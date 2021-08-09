"""
Backend integrator and backends definitions. Currently supported are following
backends:
 - `jitcdde`: (just-in-time compilation into C) which uses `jitcdde`
    which translates symbolic derivatives into C code and then calls C functions
    from Python interface:
    - very reliable
    - uses adaptive `dt` hence very useful for stiff problems and when you are
        not sure how stiff your model is
    - reasonable speed

 - `numba`: compilation of python code through `numba.njit()` - symbolic
    derivatives are converted to strings and prepared and compiled to
    numba-compatible python code; Equations are integrated using the Euler
    integration scheme.
    - Euler scheme could be less reliable
    - fixed integration time step `dt`
    - very fast
"""

import logging
import os
import re
import time
from copy import deepcopy
from functools import wraps
from sys import platform
from types import FunctionType

import numba
import numpy as np
import symengine as se
import sympy as sp
import xarray as xr
from chspy import CubicHermiteSpline
from jitcdde import jitcdde_input
from numpy import *  # noqa: F403,F401
from tqdm import tqdm

from .....utils.collections import flatten_nested_dict

DEFAULT_BACKEND = "jitcdde"


def timer(method):
    """
    Decorator for timing functions. Writes the time to logger.
    """

    @wraps(method)
    def decorator(*args, **kwargs):
        time_start = time.time()
        result = method(*args, **kwargs)
        logging.info(f"`{method.__name__}` call took {time.time() - time_start:.2f} s")
        return result

    return decorator


class BaseBackend:
    """
    Base class for backends.
    """

    _derivatives = None
    _sync = None
    _callbacks = None
    initial_state = None
    num_state_variables = None
    max_delay = None
    state_variable_names = None
    label = None

    backend_name = ""

    def run(self):
        raise NotImplementedError

    def clean(self):
        pass


class NumbaBackend(BaseBackend):
    """
    Numba integration backend using Euler scheme with delays. The symbolic code for
    derivatives is rendered into a prepared string template using numba's njit.
    """

    backend_name = "numba"

    DEFAULT_DT = 0.1  # in ms

    CURRENT_Y_REGEX = r"current_y\([0-9]*\)"
    CURRENT_Y_NUMBA = "y[{idx}, max_delay + i - 1]"

    PAST_Y_REGEX = r"past_y\((.*?)\)"
    PAST_Y_NUMBA = "y[{idx}, max_delay + i - 1 - {dt_ndt}]"
    _convert_to_dt = []

    SYSTEM_INPUT_REGEX = r"past_y\(-external_input \+ t, ([0-9]* \+ )?input_base_n, anchors\(-external_input \+ t\)\)"
    SYSTEM_INPUT_NUMBA = "input_y[{idx}, i]"

    compiled_function = None

    NUMBA_EULER_TEMPLATE = """
def integrate(dt, system_size, max_delay, t_max, y0, input_y, {params}):
    y = np.empty((system_size, t_max + max_delay + 1))
    y[:] = np.nan
    y[:, :max_delay + 1] = y0
    for i in range(1, t_max + 1):
        dy = np.array({dy_eqs})
        y[:, max_delay + i] = y[:, max_delay + i - 1] + dt*dy

    return y[:, max_delay + 1:]
"""

    def _replace_current_ys(self, expression):
        """
        Replace `current_y` symbolic representation of current state of the
        state vector with numpy array. Assume output as `y[time, space]`.

        :param expression: string to search for in symbolic expressions
        :type expression: str
        :return: expression with replaced symbolic values to numpy array
        :rtype: str
        """
        matches = re.findall(self.CURRENT_Y_REGEX, expression)
        for match in matches:
            idx = match[10:-1]
            replace_with = self.CURRENT_Y_NUMBA.format(idx=idx)
            expression = expression.replace(match, replace_with)
        return expression

    def _replace_past_ys(self, expression, dt):
        """
        Replace `past_y` symbolic representation of past state
        vector with numpy array. Assume output as `y[time, space]`.

        :param expression: string to search for in symbolic expressions
        :type expression: str
        :param dt: dt for the integration
        :type dt: float
        :return: expression with replaced symbolic values to numpy array
        :rtype: str
        """
        matches = re.findall(self.PAST_Y_REGEX, expression)
        for match in matches:
            time_past, idx, _ = match.split(",")
            idx = int(idx)
            if time_past[0] == "t":
                assert time_past[2] == "-"
                t_past = time_past[4:]
            else:
                assert time_past[-4:] == " + t"
                t_past = time_past[:-3]
            # if the delay is float, convert to ndt right now
            try:
                t_past_ndx = np.around(np.abs(float(t_past)) / dt).astype(int)
            # if it is not, just write Symbol and append to list - will be converted upon run
            except ValueError:
                t_past_ndx = t_past.strip()
                t_past_ndx = t_past_ndx[1:] if t_past_ndx[0] == "-" else t_past_ndx
                self._convert_to_dt.append(t_past_ndx)
            replace_with = self.PAST_Y_NUMBA.format(dt_ndt=t_past_ndx, idx=idx)
            expression = expression.replace(f"past_y({match}))", replace_with)
        return expression

    def _replace_inputs(self, expression):
        """
        Replace `system_input` (usually noise and/or external stimulus) symbolic
        representation with numpy array of inputs. Assume input as
        `input[time,index]`.

        :param expression: string to search for in symbolic expressions
        :type expression: str
        :return: expression with replaced symbolic values to numpy array
        :rtype: str
        """
        matches = re.findall(self.SYSTEM_INPUT_REGEX, expression)
        matches = [match if len(match) > 0 else None for match in matches]
        splits = re.split(self.SYSTEM_INPUT_REGEX, expression)
        return_string = []
        for split in splits:
            if split in matches:
                if split is None:
                    return_string.append(self.SYSTEM_INPUT_NUMBA.format(idx=0))
                else:
                    return_string.append(self.SYSTEM_INPUT_NUMBA.format(idx=split[:-3]))
            else:
                return_string.append(split)
        return "".join(return_string)

    @staticmethod
    def _substitute_helpers(derivatives, helpers):
        """
        Substitute helpers (usually used for coupling) to derivatives.

        :param derivatives: list of symbolic expressions for derivatives
        :type derivatives: list
        :param helpers: list of tuples as (helper name, symbolic expression) for
            helpers
        :type helpers: list[tuple]
        """
        sympified_helpers = [(se.sympify(helper[0]), se.sympify(helper[1])) for helper in helpers]
        sympified_derivatives = [se.sympify(derivative) for derivative in derivatives]
        substitutions = {helper[0]: helper[1] for helper in sympified_helpers}
        return [derivative.subs(substitutions) for derivative in sympified_derivatives]

    @staticmethod
    def _get_numba_function_params(symbol_params):
        """
        Get all symbolic parameters as a list for numba function string template.

        :param symbol_params: symbolic parameters as a flat dict
        :type symbol_params: dict
        :return: list of all parameters as symbols
        :rtype: list[sp.Symbol]
        """
        all_params = set()
        for k, v in symbol_params.items():
            if "input" in k:
                continue
            if isinstance(v, sp.Symbol):
                # extract symbols from sympy expressions
                all_params |= v.free_symbols
            elif isinstance(v, np.ndarray):
                # extract symbols from sympy expressions and do the union of all sets
                all_params |= set().union(*[item.free_symbols for item in v.flatten().tolist()])
        return list(all_params)

    @staticmethod
    def _create_symbol_to_float_dict(symbol_params, float_params):
        """
        Create parameters dictionary as {param_symbol: float value}.

        :param symbol_params: symbolic parameters as a flat dict
        :type symbol_params: dict
        :param float_params: float parameters as a flat dict
        :type float_params: dict
        :return: symbol: float parameters as an input to compiled numba function
        :rtype: dict
        """
        dct = {}
        for (k_sym, v_sym), (k_fl, v_fl) in zip(symbol_params.items(), float_params.items()):
            assert k_sym == k_fl
            if "input" in k_sym:
                continue
            if isinstance(v_sym, sp.Symbol):
                dct[str(v_sym)] = v_fl
            elif isinstance(v_sym, np.ndarray):
                for v_mat_sym, v_mat_fl in zip(v_sym.flatten(), v_fl.flatten()):
                    dct[str(v_mat_sym)] = v_mat_fl
        return dct

    def compile_to_numba(self, symbol_params, dt, system_size):
        """
        Compile system into numba jitted nopython function.

        :param symbol_params: symbolic parameters as a flat dict
        :type symbol_params: dict
        :param dt: dt in ms
        :type dt: float
        :param system_size: number of equations in the system
        :type system_size: int
        :return: numba compiled function
        :rtype: callable
        """
        # substitute helpers to symbolic derivatives
        logging.info("Compiling model definition to numba function...")
        logging.info("Substituting helpers...")
        substituted = self._substitute_helpers(derivatives=self._derivatives(), helpers=self._sync())
        assert len(substituted) == system_size

        # replace symbolic expressions with numpy ones
        logging.info("Replacing symbolic expressions with arrays...")
        derivatives_str = self._replace_inputs(str(substituted))
        derivatives_str = self._replace_current_ys(derivatives_str)
        self._convert_to_dt = []
        derivatives_str = self._replace_past_ys(derivatives_str, dt=dt)
        assert isinstance(derivatives_str, str)
        logging.info("Compiling python code using numba's njit...")
        function_params = self._get_numba_function_params(flatten_nested_dict(symbol_params))
        numba_func = self.NUMBA_EULER_TEMPLATE.format(
            dy_eqs=derivatives_str, params=", ".join([str(param) for param in function_params])
        )
        # compile numba function and load into namespace
        compiled_code = compile(numba_func, "<string>", "exec")
        integrate = numba.njit(fastmath=True)(
            FunctionType(
                compiled_code.co_consts[-3],
                {
                    **globals(),
                    # add numba callbacks to the mix
                    **{name: definition for name, definition in self._numba_callbacks()},
                },
                name="integrate",
                argdefs=(),
            )
        )
        self.compiled_function = integrate

    def run(self, duration, dt, noise_input, symbol_params, float_params, **kwargs):
        """
        Run integration.

        :kwargs: actually none - for compatiblity
        """
        assert isinstance(noise_input, np.ndarray)

        system_size = len(self._derivatives())
        logging.info("Loading system from cache...")
        integrate = self.compiled_function

        assert callable(integrate)
        # run the numba-jitted function
        times = np.arange(dt, duration + dt, dt)
        assert times.shape[0] == noise_input.shape[1]
        logging.info(f"Integrating for {times.shape[0]} time steps...")
        max_delay_dt = np.around(self.max_delay / dt).astype(int)
        init_state = self.initial_state
        if init_state.ndim == 1:
            init_state = np.tile(init_state[:, np.newaxis], reps=(1, max_delay_dt + 1))
        model_params = self._create_symbol_to_float_dict(
            flatten_nested_dict(symbol_params), flatten_nested_dict(float_params)
        )
        for delay_param in set(self._convert_to_dt):
            assert delay_param in model_params, delay_param
            model_params[delay_param] = np.around(float(model_params[delay_param]) / dt).astype(int)
        result = integrate(
            dt=dt,
            system_size=system_size,
            max_delay=max_delay_dt,
            t_max=np.around(duration / dt).astype(int),
            y0=init_state,
            input_y=noise_input,
            **model_params,
        )
        return times, result


class JitcddeBackend(BaseBackend):
    """
    Backend using jitcdde integrator. Uses just-in-time compilation for delay
    differential equations with integration method proposed by Shampine and
    Thompson.

    Reference (package):
        Ansmann, G. (2018). Efficiently and easily integrating differential
        equations with JiTCODE, JiTCDDE, and JiTCSDE. Chaos: An
        Interdisciplinary Journal of Nonlinear Science, 28(4), 043116.

    Reference (method):
        Shampine, L. F., & Thompson, S. (2001). Solving DDEs in matlab. Applied
        Numerical Mathematics, 37(4), 441-458.
    """

    backend_name = "jitcdde"
    extra_compile_args = []
    dde_system = None

    def _init_and_compile_C(
        self,
        derivatives,
        helpers=None,
        inputs=None,
        max_delay=0.0,
        callbacks=None,
        chunksize=1,
    ):
        """
        Initialise DDE system and compile to C.
        """
        logging.info("Setting up the DDE system...")
        if platform == "darwin":
            # assert clang is used on macOS
            os.environ["CC"] = "clang"
        callbacks = callbacks or ()
        self.dde_system = jitcdde_input(
            f_sym=derivatives,
            input=inputs,
            helpers=helpers,
            n=len(derivatives),
            max_delay=max_delay,
            callback_functions=callbacks,
        )

        logging.info("Compiling to C...")
        self.dde_system.compile_C(
            simplify=False,
            do_cse=False,
            extra_compile_args=self.extra_compile_args,
            omp=False,
            chunk_size=chunksize,
            verbose=False,
        )

    def _set_constant_past(self, past_state, squeeze=False):
        """
        Sets past of the delayed system with a constant vector. This usually
        means that `past_state` is 1D vector of length `num_state_variables`.
        `jitcdde` automatically determines how long into the past it need to
        extrapolate based on `max_delay`.

        :param past_state: vector of past states of length `num_state_variables`
        :type past_state: np.ndarray
        """
        if squeeze:
            self.dde_system.constant_past(past_state.squeeze())
        else:
            self.dde_system.constant_past(past_state)
        self.dde_system.adjust_diff()

    def _set_past_from_vector(self, past_state, dt):
        """
        Sets past of the delayed system with temporal dependance. This means
        that `past_state` is 2D array as (`num_state_variables` x `time`) and
        each time vector is added as so-called `Anchor`. `jitcdde` then
        automatically interpolate in between `dt`s when needed.

        :param past_state: vector of past states as (`num_state_variables` x
            `time`)
        :type past_state: np.ndarray
        :param dt: dt of the system, to infer how much in to the past the vector
            is (in `jitcdde` this is actually sampling dt)
        :type dt: float
        """
        derivatives = np.hstack([np.zeros((past_state.shape[0], 1)), np.diff(past_state, axis=1)])
        assert derivatives.shape == past_state.shape
        for t in range(past_state.shape[1]):
            self.dde_system.add_past_point(-t * dt, past_state[:, -t], derivatives[:, -t])
        self.dde_system.adjust_diff()

    def _integrate_blindly(self, max_delay):
        """
        Deals with initial discontinuities using `integrate_blindly` method,
        where the adaptive integrator just integrates until 1.5 * `max_delay`
        which smooths the initial and past discontinuities. Currently not used,
        but something is telling me this would be necessary for autochunk
        feature to work with `jitcdde` with adaptive time step.

        :param max_delay: maximum delay in the system, in ms
        :type max_delay: float
        """
        self.dde_system.integrate_blindly(max_delay * 1.5)

    def _check(self):
        """
        Check the delay system.
        """
        if self.dde_system is not None:
            self.dde_system.check()

    def run(self, duration, dt, noise_input, **kwargs):
        """
        Run integration.
        """
        assert isinstance(noise_input, CubicHermiteSpline)
        # compile model in C
        self._init_and_compile_C(
            derivatives=self._derivatives(),
            helpers=self._sync(),
            inputs=noise_input,
            max_delay=self.max_delay,
            callbacks=self._callbacks(),
            chunksize=kwargs.pop("chunksize", 1),
        )
        assert self.dde_system is not None
        self.dde_system.purge_past()
        self.dde_system.reset_integrator()
        logging.info("Setting past of the state vector...")
        if self.initial_state.ndim == 1:
            self._set_constant_past(self.initial_state)
        elif self.initial_state.shape[1] == 1:
            self._set_constant_past(self.initial_state, squeeze=True)
        else:
            self._set_past_from_vector(self.initial_state, dt)
        # integrate
        times = np.arange(dt, duration + dt, dt)
        logging.info(f"Integrating for {times.shape[0]} time steps...")
        result = np.vstack([self.dde_system.integrate(time) for time in tqdm(times)]).T
        return times, result

    def clean(self):
        """
        Clean - i.e. remove temp directory from C compilation.
        """
        self.dde_system.__del__()


class BackendIntegrator:
    """
    Backend integrator mixin - implements integration using various backends,
    stores results in xarray and is able to save xr.Datasets to pickle or
    netCDF.
    """

    backend_instance = None
    # parameters handling for numba
    symbol_params = None
    float_params = None

    # these attributes are need for successful integrator run
    NEEDED_ATTRIBUTES = [
        "_derivatives",
        "_sync",
        "_callbacks",
        "_numba_callbacks",
        "initial_state",
        "num_state_variables",
        "max_delay",
        "state_variable_names",
        "label",
        "initialised",
    ]

    def _init_jitcdde_backend(self):
        self.backend_instance = JitcddeBackend()
        for attr in self.NEEDED_ATTRIBUTES:
            setattr(self.backend_instance, attr, getattr(self, attr))

    def _init_numba_backend(self, dt):
        self.backend_instance = NumbaBackend()
        for attr in self.NEEDED_ATTRIBUTES:
            setattr(self.backend_instance, attr, getattr(self, attr))
        # compile
        self.make_params_symbolic(vector=False)
        assert not self.are_params_floats
        self.symbol_params = deepcopy(self.get_nested_params())
        self.backend_instance.compile_to_numba(
            symbol_params=self.symbol_params, dt=dt, system_size=len(self._derivatives())
        )
        self.make_params_floats()

    @timer
    def run(self, duration, dt, noise_input, backend=DEFAULT_BACKEND, return_xarray=True, **kwargs):
        """
        Run the integration.

        :param duration: duration of the run, in ms
        :type duration: float
        :param dt: sampling dt for `jitcdde` backend - which actually uses
            adaptive dt, or integration dt for numba backend; in ms
        :type dt: float
        :param noise_input: noise input to the network or node
        :type noise_input: `chspy.CubicHermiteSpline`|np.ndarray
        :param backend: which backend to use
        :type backend: str
        :param return_xarray: whether to return xarray's Dataset, or simply time
            and result as a matrix
        :type return_xarray: bool
        :*kwargs: optional keyword arguments, will be passed to backend instance
        """
        assert all(hasattr(self, attr) for attr in self.NEEDED_ATTRIBUTES)
        assert self.initialised, "Model must be initialised"
        assert isinstance(noise_input, (CubicHermiteSpline, np.ndarray))

        def _check_backend_init():
            # if backend was never initialized -> init and compile
            if self.backend_instance is None:
                return True
            # if user wants different backend -> init other backend
            if self.backend_instance.backend_name != backend:
                return True
            # if symbol_params exists (i.e. we used numba backend before)
            if self.symbol_params is not None:
                # set on dict gets the set of dict's keys
                existing_params = set(flatten_nested_dict(self.symbol_params))
                new_params = set(flatten_nested_dict(self.get_nested_params()))
                # if the keys differ (i.e. parameter keys changed) -> recompile
                # set - set produces an intersection and if empty, then bool(<empty set>) = False
                # if not empty then bool(<set>) = True
                if bool(existing_params - new_params):
                    return True

            return False

        if _check_backend_init():
            logging.info(f"Initialising {backend} backend...")
            if backend == "jitcdde":
                self._init_jitcdde_backend()
            elif backend == "numba":
                self._init_numba_backend(dt=dt)
            else:
                raise ValueError(f"Unknown backend {backend}")
        assert isinstance(self.backend_instance, BaseBackend)
        if isinstance(self.backend_instance, NumbaBackend):
            assert self.are_params_floats
            self.float_params = deepcopy(self.get_nested_params())

        # update initial state
        self.backend_instance.initial_state = self.initial_state.copy()
        times, result = self.backend_instance.run(
            duration=duration,
            dt=dt,
            noise_input=noise_input,
            symbol_params=self.symbol_params,
            float_params=self.float_params,
            **kwargs,
        )
        logging.info("Integration done.")
        assert times.shape[0] == result.shape[1]
        assert result.shape[0] == self.num_state_variables
        if return_xarray:
            return self._init_xarray(times=times, results=result.T)
        else:
            # return result as nodes x time for compatibility with neurolib
            return times, result

    def _init_xarray(self, times, results):
        """
        Initialise results array.

        :param times: time for the result, in ms
        :type times: np.ndarray
        :param results: results as times x state variable
        :type results: np.ndarray
        """
        assert times.shape[0] == results.shape[0]
        assert results.shape[1] == sum([len(state_vars) for state_vars in self.state_variable_names])
        # get union of state variables
        names_union = set.union(*[set(state_vars) for state_vars in self.state_variable_names])
        num_nodes = len(self.state_variable_names)

        # init empty DataArrays per state variable
        xr_results = {
            state_var: xr.DataArray(
                None,
                dims=["time", "node"],
                coords={"time": np.around(times / 1000.0, 5), "node": np.arange(num_nodes)},
            )
            for state_var in names_union
        }

        # track index in results
        var_idx = 0
        for node_idx, node_results in enumerate(self.state_variable_names):
            for result_idx, result in enumerate(node_results):
                xr_results[result][:, node_idx] = results[:, var_idx + result_idx].astype(np.floating)
            var_idx += len(node_results)

        for state_var, array in xr_results.items():
            xr_results[state_var] = array.astype(np.floating)

        dataset = xr.Dataset(xr_results)

        return dataset

    def clean(self):
        """
        Clean after myself, if needed.
        """
        self.backend_instance.clean()
