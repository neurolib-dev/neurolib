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
from functools import wraps
from sys import platform
from types import FunctionType

import numba
import numpy as np
import symengine as se
import xarray as xr
from chspy import CubicHermiteSpline
from jitcdde import jitcdde_input
from numpy import *  # noqa: F403,F401
from tqdm import tqdm

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

    def run(self):
        raise NotImplementedError

    def clean(self):
        pass


class NumbaBackend(BaseBackend):
    """
    Numba integration backend using Euler scheme with delays. The symbolic code for
    derivatives is rendered into a prepared string template using numba's njit.
    """

    DEFAULT_DT = 0.1  # in ms

    CURRENT_Y_REGEX = r"current_y\([0-9]*\)"
    CURRENT_Y_NUMBA = "y[max_delay + i - 1, {idx}]"

    PAST_Y_REGEX = r"past_y\([-+]?[0-9]*\.?[0-9]+ \+ t, [0-9]*, anchors\([-+]" r"?[0-9]*\.?[0-9]+ \+ t\)\)"
    PAST_Y_NUMBA = "y[max_delay + i - 1 - {dt_ndt}, {idx}]"

    SYSTEM_INPUT_REGEX = (
        r"past_y\(-external_input \+ t, ([0-9]* \+ )?input_base_n, anchors" r"\(-external_input \+ t\)\)"
    )
    SYSTEM_INPUT_NUMBA = "input_y[i, {idx}]"

    NUMBA_EULER_TEMPLATE = """
def integrate(dt, n, max_delay, t_max, y0, input_y):
    y = np.empty((t_max + max_delay + 1, n))
    y[:] = np.nan
    y[:max_delay + 1] = y0
    for i in range(1, t_max + 1):
        dy = np.array({dy_eqs})
        y[max_delay + i, :] = y[max_delay + i - 1, :] + dt*dy

    return y[max_delay + 1:, :]
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
            t_past = float(time_past.split("+")[0].strip()[7:])
            t_past_ndx = np.around(t_past / dt).astype(int)
            replace_with = self.PAST_Y_NUMBA.format(dt_ndt=np.abs(t_past_ndx), idx=idx)
            expression = expression.replace(match, replace_with)
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

    def _substitute_helpers(self, derivatives, helpers):
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

    def run(self, duration, dt, noise_input, **kwargs):
        """
        Run integration.

        :kwargs: actually none - for compatiblity
        """
        assert isinstance(noise_input, np.ndarray)
        system_size = len(self._derivatives())

        # substitute helpers to symbolic derivatives
        logging.info("Substituting helpers...")
        substituted = self._substitute_helpers(derivatives=self._derivatives(), helpers=self._sync())
        assert len(substituted) == system_size

        # replace symbolic expressions with numpy ones
        logging.info("Replacing symbolic expressions with arrays...")
        derivatives_str = self._replace_inputs(str(substituted))
        derivatives_str = self._replace_current_ys(derivatives_str)
        derivatives_str = self._replace_past_ys(derivatives_str, dt=dt)
        assert isinstance(derivatives_str, str)
        logging.info("Compiling python code using numba's njit...")
        numba_func = self.NUMBA_EULER_TEMPLATE.format(dy_eqs=derivatives_str)
        # compile numba function and load into namespace
        numba_code = compile(numba_func, "<string>", "exec")
        integrate = numba.njit(
            FunctionType(
                numba_code.co_consts[-3],
                {
                    **globals(),
                    # add numba callbacks to the mix
                    **{name: definition for name, definition in self._numba_callbacks()},
                },
                name="integrate",
                argdefs=(),
            )
        )
        assert callable(integrate)
        # run the numba-jitted function
        times = np.arange(dt, duration + dt, dt)
        logging.info(f"Integrating for {times.shape[0]} time steps...")
        result = integrate(
            dt=dt,
            n=system_size,
            max_delay=np.around(self.max_delay / dt).astype(int),
            t_max=np.around(duration / dt).astype(int),
            y0=self.initial_state,
            input_y=noise_input,
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

    extra_compile_args = []
    dde_system = None

    def _init_load_compiled(
        self, compiled_dir, derivatives, helpers=None, inputs=None, max_delay=0.0, callbacks=None,
    ):
        """
        Initialise DDE system and load from compiled.
        """
        logging.info("Setting up the DDE system...")
        compiled_filename = os.path.join(compiled_dir, f"{self.label}.so")
        assert os.path.exists(compiled_filename)
        self.dde_system = jitcdde_input(
            f_sym=derivatives,
            input=inputs,
            helpers=helpers,
            n=len(derivatives),
            max_delay=max_delay,
            callback_functions=callbacks,
            module_location=compiled_filename,
        )

    def _init_and_compile_C(
        self, derivatives, helpers=None, inputs=None, max_delay=0.0, callbacks=None, chunksize=1, use_open_mp=False,
    ):
        """
        Initialise DDE system and compiled to C.
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
        if use_open_mp:
            logging.info("Using OpenMP parallelisation...")
            compiler_args = ["-fopenmp"]
            if platform == "darwin":
                logging.warning(
                    "MacOS detected. For openMP parallelisation the llvm and "
                    "libomp must be installed. If not, install them please "
                    "with `brew install llvm libomp` and follow steps for "
                    "prepending system path with llvm path."
                )
                linker_args = ["-lomp", "-L/usr/local/opt/llvm/lib"]
            else:
                linker_args = ["-lgomp"]
            omp_flags = (compiler_args, linker_args)
        else:
            omp_flags = False

        logging.info("Compiling to C...")
        self.dde_system.compile_C(
            simplify=False,
            do_cse=False,
            extra_compile_args=self.extra_compile_args,
            omp=omp_flags,
            chunk_size=chunksize,
            verbose=False,
        )

    def _set_constant_past(self, past_state):
        self.dde_system.constant_past(past_state)
        self.dde_system.adjust_diff()

    def _integrate_blindly(self, max_delay):
        self.dde_system.integrate_blindly(max_delay * 1.5)

    def _check(self):
        if self.dde_system is not None:
            self.dde_system.check()

    def run(self, duration, dt, noise_input, **kwargs):
        """
        Run integration.

        :kwargs:
            - use_open_mp: OpenMP parallelization - significant overhead, only
                useful (probably) for very large systems
            - save_compiled_to: path to which save compiled C code -
                derivatives and helpers; for large networks this saves the
                compilation time given multiple runs
            - load_compiled: whether to try load compiled C code, i.e. do not
            compile before the run
        """
        chunksize = kwargs.pop("chunksize", 1)
        use_open_mp = kwargs.pop("use_open_mp", False)
        save_compiled_to = kwargs.pop("save_compiled_to", None)
        load_compiled = kwargs.pop("load_compiled", False)
        assert isinstance(noise_input, CubicHermiteSpline)

        if load_compiled:
            self._init_load_compiled(
                compiled_dir=save_compiled_to,
                derivatives=self._derivatives(),
                helpers=self._sync(),
                inputs=noise_input,
                max_delay=self.max_delay,
                callbacks=self._callbacks(),
            )
        else:
            self._init_and_compile_C(
                derivatives=self._derivatives(),
                helpers=self._sync(),
                inputs=noise_input,
                max_delay=self.max_delay,
                callbacks=self._callbacks(),
                chunksize=chunksize,
                use_open_mp=use_open_mp,
            )
        assert self.dde_system is not None
        self.dde_system.reset_integrator()
        logging.info("Setting past of the state vector...")
        self._set_constant_past(self.initial_state)
        # integrate
        times = np.arange(dt, duration + dt, dt)
        logging.info(f"Integrating for {times.shape[0]} time steps...")
        result = np.vstack([self.dde_system.integrate(time) for time in tqdm(times)])
        if save_compiled_to is not None and not load_compiled:
            os.makedirs(save_compiled_to, exist_ok=True)
            logging.info(f"Saving compiled C code to {save_compiled_to}")
            self.dde_system.save_compiled(
                os.path.join(save_compiled_to, f"{self.label}.so"), overwrite=True,
            )
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

    def _init_backend(self, backend):
        if backend == "jitcdde":
            self.backend_instance = JitcddeBackend()
        elif backend == "numba":
            self.backend_instance = NumbaBackend()
        else:
            raise ValueError(f"Unknown backend: {backend}")
        # copy necessary attributes (i.e. symbolic derivatives, helpers, etc) to
        # backend instance
        for attr in self.NEEDED_ATTRIBUTES:
            setattr(self.backend_instance, attr, getattr(self, attr))

    @timer
    def run(
        self, duration, dt, noise_input, backend=DEFAULT_BACKEND, return_xarray=True, **kwargs,
    ):
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

        logging.info(f"Initialising {backend} backend...")
        self._init_backend(backend)
        assert isinstance(self.backend_instance, BaseBackend)
        times, result = self.backend_instance.run(duration=duration, dt=dt, noise_input=noise_input, **kwargs)
        logging.info("Integration done.")
        assert times.shape[0] == result.shape[0]
        assert result.shape[1] == self.num_state_variables
        if return_xarray:
            return self._init_xarray(times=times, results=result)
        else:
            # return result as nodes x time for compatibility with neurolib
            return times, result.T

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
