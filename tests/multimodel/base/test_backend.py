"""
Test for the backend integrator.
"""
import json
import os
import pickle
import unittest
from shutil import rmtree

import numba
import numpy as np
import pytest
import symengine as se
import xarray as xr
from jitcdde import t as time_vector
from jitcdde import y as state_vector
from neurolib.models.multimodel.builder.base.backend import BackendIntegrator, BaseBackend, JitcddeBackend, NumbaBackend
from neurolib.utils.saver import save_to_netcdf, save_to_pickle
from neurolib.utils.stimulus import ZeroInput


class TestBaseBackend(unittest.TestCase):
    def test_init(self):
        base = BaseBackend()
        self.assertTrue(isinstance(base, BaseBackend))
        self.assertTrue(hasattr(base, "run"))
        self.assertTrue(hasattr(base, "clean"))
        self.assertEqual(base._derivatives, None)
        self.assertEqual(base._sync, None)
        self.assertEqual(base._callbacks, None)
        self.assertEqual(base.initial_state, None)
        self.assertEqual(base.num_state_variables, None)
        self.assertEqual(base.max_delay, None)
        self.assertEqual(base.state_variable_names, None)
        self.assertEqual(base.label, None)

    def test_methods(self):
        base = BaseBackend()
        base.clean()
        self.assertRaises(NotImplementedError, base.run)


class TestJitcddeBackend(unittest.TestCase):
    def test_init(self):
        backend = JitcddeBackend()
        self.assertTrue(isinstance(backend, JitcddeBackend))
        self.assertTrue(isinstance(backend, BaseBackend))
        self.assertTrue(hasattr(backend, "_init_and_compile_C"))
        self.assertTrue(hasattr(backend, "_set_constant_past"))
        self.assertTrue(hasattr(backend, "_set_past_from_vector"))
        self.assertTrue(hasattr(backend, "_integrate_blindly"))
        self.assertTrue(hasattr(backend, "_check"))


class TestNumbaBackend(unittest.TestCase):
    def test_init(self):
        backend = NumbaBackend()
        self.assertTrue(isinstance(backend, NumbaBackend))
        self.assertTrue(isinstance(backend, BaseBackend))
        self.assertTrue(hasattr(backend, "_replace_current_ys"))
        self.assertTrue(hasattr(backend, "_replace_past_ys"))
        self.assertTrue(hasattr(backend, "_replace_inputs"))
        self.assertTrue(hasattr(backend, "_substitute_helpers"))

    def test_replace_current_ys(self):
        backend = NumbaBackend()
        STRING_IN = "0.4*(1.0*(1.0 - current_y(0))"
        EXPECTED = "0.4*(1.0*(1.0 - y[0, max_delay + i - 1])"
        result = backend._replace_current_ys(STRING_IN)
        self.assertEqual(result, EXPECTED)

    def test_replace_past_ys(self):
        backend = NumbaBackend()
        STRING_IN = "0.06*past_y(-3.21 + t, 2, anchors(-3.21 + t)) + 0.23*past_y(-0.5 + t, 0, anchors(-0.5 + t))"
        EXPECTED = "0.06*y[2, max_delay + i - 1 - 32] + 0.23*y[0, max_delay + i - 1 - 5]"
        result = backend._replace_past_ys(STRING_IN, dt=0.1)
        self.assertEqual(result, EXPECTED)

    def test_replace_inputs(self):
        backend = NumbaBackend()
        STRING_IN = "12.4*past_y(-external_input + t, 3 + input_base_n, " "anchors(-external_input + t))"
        EXPECTED = "12.4*input_y[3, i]"
        result = backend._replace_inputs(STRING_IN)
        self.assertEqual(result, EXPECTED)

    def test_substitute_helpers(self):
        backend = NumbaBackend()
        a = se.Symbol("a")
        b = se.Symbol("b")
        y = se.Symbol("y")
        HELPERS = [(a, se.exp(-12 * y))]
        DERIVATIVES = [-b * a + y, y ** 2]
        result = backend._substitute_helpers(DERIVATIVES, HELPERS)
        self.assertListEqual(result, [-b * se.exp(-12 * y) + y, y ** 2])


class BackendTestingHelper(BackendIntegrator):
    """
    Testing class that mimics the structure of brain models. Implements the 1D
    Mackey-Glass delay differential equation.
    """

    label = "MackeyGlass"
    params = {"tau": 15.0, "n": 10.0, "beta": 0.25, "gamma": 0.1}
    initial_state = np.array([1.0])
    num_state_variables = 1
    max_delay = params["tau"]
    state_variable_names = [["y"]]
    # override initialisation
    initialised = True

    def _derivatives(self):
        """
        Define system's derivatives as a list.
        """
        return [
            self.params["beta"]
            * state_vector(0, time_vector - self.params["tau"])
            / (1.0 + state_vector(0, time_vector - self.params["tau"]) ** self.params["n"])
            - self.params["gamma"] * state_vector(0)
        ]

    def _sync(self):
        """
        Defines helpers - usually coupling between nodes and masses - here
        empty.
        """
        return []

    def _callbacks(self):
        """
        Defines optional python callbacks within symbolic derivatives - here
        empty.
        """
        return []

    def _numba_callbacks(self):
        return []


class TestBackendIntegrator(unittest.TestCase):

    DURATION = 10000
    DT = 10
    here = os.path.dirname(os.path.realpath(__file__))
    TEST_DIR = os.path.join(here, "test_temp")
    EXTRA_ATTRS = {
        "a": "b",
        "c": 0.1,
        "d": [1, 2, 3],
        "e": np.random.rand(1, 1),
        "f": {"aa": [np.random.rand(1, 1), np.random.rand(1, 1)], "bb": "3"},
    }

    @classmethod
    def setUpClass(cls):
        os.makedirs(cls.TEST_DIR, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        rmtree(cls.TEST_DIR, ignore_errors=True)

    def test_init_system(self):
        system = BackendTestingHelper()
        self.assertTrue(hasattr(system, "_init_xarray"))
        self.assertTrue(hasattr(system, "_init_backend"))
        self.assertTrue(hasattr(system, "run"))

    def test_run_jitcdde(self):
        system = BackendTestingHelper()
        results = system.run(
            self.DURATION,
            self.DT,
            ZeroInput().as_cubic_splines(self.DURATION, self.DT),
            backend="jitcdde",
        )
        results.attrs = self.EXTRA_ATTRS
        # assert type, length and shape of results
        self.assertTrue(isinstance(results, xr.Dataset))
        self.assertEqual(len(results), 1)
        self.assertTupleEqual(
            results[system.state_variable_names[0][0]].shape,
            (int(self.DURATION / self.DT), 1),
        )
        self.assertTrue(all(dim in results.dims for dim in ["time", "node"]))
        self.assertDictEqual(results.attrs, self.EXTRA_ATTRS)

    def test_run_jitcdde_vector_past(self):
        system = BackendTestingHelper()
        system.initial_state = np.random.rand(1, 4)
        results = system.run(
            self.DURATION,
            self.DT,
            ZeroInput().as_cubic_splines(self.DURATION, self.DT),
            backend="jitcdde",
        )
        self.assertTrue(isinstance(results, xr.Dataset))
        self.assertEqual(len(results), 1)
        self.assertTupleEqual(
            results[system.state_variable_names[0][0]].shape,
            (int(self.DURATION / self.DT), 1),
        )
        self.assertTrue(all(dim in results.dims for dim in ["time", "node"]))

    def test_jitcdde_other_features(self):
        system = BackendTestingHelper()
        _ = system.run(self.DURATION, self.DT, ZeroInput().as_cubic_splines(self.DURATION, self.DT), backend="jitcdde")
        system.backend_instance._check()
        system.backend_instance.dde_system.reset_integrator()
        system.backend_instance._integrate_blindly(system.max_delay)
        system.backend_instance.dde_system.purge_past()
        # past state as nodes x time
        system.backend_instance._set_past_from_vector(np.random.rand(1, 4), dt=self.DT)
        system.clean()

    def test_backend_value_error(self):
        system = BackendTestingHelper()
        with pytest.raises(ValueError):
            _ = system.run(
                self.DURATION, self.DT, ZeroInput().as_cubic_splines(self.DURATION, self.DT), backend="wrong"
            )

    def test_return_raw_and_xarray(self):
        system = BackendTestingHelper()
        results_xr = system.run(
            self.DURATION,
            self.DT,
            ZeroInput().as_cubic_splines(self.DURATION, self.DT),
            backend="jitcdde",
            return_xarray=True,
        )
        self.assertTrue(isinstance(results_xr, xr.Dataset))
        times, results_raw = system.run(
            self.DURATION,
            self.DT,
            ZeroInput().as_cubic_splines(self.DURATION, self.DT),
            backend="jitcdde",
            return_xarray=False,
        )
        self.assertTrue(isinstance(times, np.ndarray))
        self.assertTrue(isinstance(results_raw, np.ndarray))
        np.testing.assert_equal(times / 1000.0, results_xr.time)
        np.testing.assert_equal(results_raw.squeeze(), results_xr["y"].values.squeeze())

    def test_run_numba(self):
        system = BackendTestingHelper()
        results = system.run(
            self.DURATION,
            self.DT,
            ZeroInput().as_array(self.DURATION, self.DT),
            backend="numba",
        )
        results.attrs = self.EXTRA_ATTRS
        # assert type, length and shape of results
        self.assertTrue(isinstance(results, xr.Dataset))
        self.assertEqual(len(results), 1)
        self.assertTupleEqual(
            results[system.state_variable_names[0][0]].shape,
            (int(self.DURATION / self.DT), 1),
        )
        self.assertTrue(all(dim in results.dims for dim in ["time", "node"]))
        self.assertDictEqual(results.attrs, self.EXTRA_ATTRS)

    def test_save_pickle(self):
        """
        Testing for saver done here since we have a model to integrate so it's
        easy.
        """
        system = BackendTestingHelper()
        # add attributes to test saving them
        results = system.run(self.DURATION, self.DT, ZeroInput().as_cubic_splines(self.DURATION, self.DT))
        results.attrs = self.EXTRA_ATTRS
        # save to pickle
        pickle_name = os.path.join(self.TEST_DIR, "pickle_test")
        save_to_pickle(results, pickle_name)
        pickle_name += ".pkl"
        self.assertTrue(os.path.exists(pickle_name))
        # load and check
        with open(pickle_name, "rb") as f:
            loaded = pickle.load(f)
        xr.testing.assert_equal(results, loaded)
        self.assertDictEqual(loaded.attrs, self.EXTRA_ATTRS)

    def test_save_netcdf(self):
        """
        Testing for saver done here since we have a model to integrate so it's
        easy.
        """
        system = BackendTestingHelper()
        results = system.run(self.DURATION, self.DT, ZeroInput().as_cubic_splines(self.DURATION, self.DT))
        results.attrs = self.EXTRA_ATTRS
        # save to pickle
        nc_name = os.path.join(self.TEST_DIR, "netcdf_test")
        save_to_netcdf(results, nc_name)
        # actual data
        self.assertTrue(os.path.exists(nc_name + ".nc"))
        # metadata
        self.assertTrue(os.path.exists(nc_name + ".json"))
        # load and check
        loaded = xr.load_dataset(nc_name + ".nc")
        with open(nc_name + ".json", "r") as f:
            attrs = json.load(f)
        loaded.attrs = attrs
        xr.testing.assert_equal(results, loaded)
        self.assertDictEqual(loaded.attrs, self.EXTRA_ATTRS)


if __name__ == "__main__":
    unittest.main()
