"""
Test for the backend integrator.
"""

import json
import os
import pickle
import unittest
from shutil import rmtree

# import numba
import numpy as np
import pytest
import symengine as se
import xarray as xr
from jitcdde import t as time_vector
from jitcdde import y as state_vector
from neurolib.models.multimodel.builder.base.backend import BackendIntegrator, BaseBackend, JitcddeBackend, NumbaBackend
from neurolib.models.multimodel.builder.model_input import ZeroInput


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


class TestJitcddeBackend(unittest.TestCase):
    def test_init(self):
        backend = JitcddeBackend()
        self.assertTrue(isinstance(backend, JitcddeBackend))
        self.assertTrue(isinstance(backend, BaseBackend))
        self.assertTrue(hasattr(backend, "_init_and_compile_C"))
        self.assertTrue(hasattr(backend, "_set_constant_past"))
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
        EXPECTED = "0.4*(1.0*(1.0 - y[max_delay + i - 1, 0])"
        result = backend._replace_current_ys(STRING_IN)
        self.assertEqual(result, EXPECTED)

    def test_replace_past_ys(self):
        backend = NumbaBackend()
        STRING_IN = "0.06*past_y(-3.21 + t, 2, anchors(-3.21 + t)) + 0.23*" "past_y(-0.5 + t, 0, anchors(-0.5 + t))"
        EXPECTED = "0.06*y[max_delay + i - 1 - 32, 2] + 0.23*y[max_delay + i" " - 1 - 5, 0]"
        result = backend._replace_past_ys(STRING_IN, dt=0.1)
        self.assertEqual(result, EXPECTED)

    def test_replace_inputs(self):
        backend = NumbaBackend()
        STRING_IN = "12.4*past_y(-external_input + t, 3 + input_base_n, " "anchors(-external_input + t))"
        EXPECTED = "12.4*input_y[i, 3]"
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

    @pytest.mark.skip("currently does nothing")
    def test_prepare_callbacks(self):
        pass


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
            ZeroInput(self.DURATION, self.DT).as_cubic_splines(),
            metadata=self.EXTRA_ATTRS,
            backend="jitcdde",
        )
        # assert type, length and shape of results
        self.assertTrue(isinstance(results, xr.Dataset))
        self.assertEqual(len(results), 1)
        self.assertTupleEqual(
            results[system.state_variable_names[0][0]].shape, (int(self.DURATION / self.DT), 1),
        )
        self.assertTrue(all(dim in results.dims for dim in ["time", "node"]))
        self.assertDictEqual(results.attrs, self.EXTRA_ATTRS)

    def test_run_numba(self):
        system = BackendTestingHelper()
        results = system.run(
            self.DURATION,
            self.DT,
            ZeroInput(self.DURATION, self.DT).as_array(),
            metadata=self.EXTRA_ATTRS,
            backend="numba",
        )
        # assert type, length and shape of results
        self.assertTrue(isinstance(results, xr.Dataset))
        self.assertEqual(len(results), 1)
        self.assertTupleEqual(
            results[system.state_variable_names[0][0]].shape, (int(self.DURATION / self.DT), 1),
        )
        self.assertTrue(all(dim in results.dims for dim in ["time", "node"]))
        self.assertDictEqual(results.attrs, self.EXTRA_ATTRS)

    def test_run_save_compiled(self):
        system = BackendTestingHelper()
        results = system.run(
            self.DURATION,
            self.DT,
            ZeroInput(self.DURATION, self.DT).as_cubic_splines(),
            metadata=self.EXTRA_ATTRS,
            save_compiled_to=self.TEST_DIR,
            backend="jitcdde",
        )
        # check the so file exists
        self.assertTrue(os.path.exists(os.path.join(self.TEST_DIR, f"{system.label}.so")))
        # run again but load from compiled
        results_from_loaded = system.run(
            self.DURATION,
            self.DT,
            ZeroInput(self.DURATION, self.DT).as_cubic_splines(),
            metadata=self.EXTRA_ATTRS,
            save_compiled_to=self.TEST_DIR,
            load_compiled=True,
        )
        # check results are the same
        self.assertEqual(results.dims, results_from_loaded.dims)
        [
            np.testing.assert_equal(coord1.values, coord2.values)
            for coord1, coord2 in zip(results.coords.values(), results_from_loaded.coords.values())
        ]
        for data_var in results:
            np.testing.assert_allclose(
                results[data_var].values.astype(float), results_from_loaded[data_var].values.astype(float),
            )

    def test_run_openmp(self):
        system = BackendTestingHelper()
        results = system.run(
            self.DURATION,
            self.DT,
            ZeroInput(self.DURATION, self.DT).as_cubic_splines(),
            metadata=self.EXTRA_ATTRS,
            chunksize=5,
            use_open_mp=True,
            backend="jitcdde",
        )
        # assert type, length and shape of results
        self.assertTrue(isinstance(results, xr.Dataset))
        self.assertEqual(len(results), 1)
        self.assertTupleEqual(
            results[system.state_variable_names[0][0]].shape, (int(self.DURATION / self.DT), 1),
        )
        self.assertTrue(all(dim in results.dims for dim in ["time", "node"]))
        self.assertDictEqual(results.attrs, self.EXTRA_ATTRS)

    def test_save_pickle(self):
        system = BackendTestingHelper()
        # add attributes to test saving them
        results = system.run(
            self.DURATION, self.DT, ZeroInput(self.DURATION, self.DT).as_cubic_splines(), metadata=self.EXTRA_ATTRS,
        )
        # save to pickle
        pickle_name = os.path.join(self.TEST_DIR, "pickle_test")
        system.save_to_pickle(results, pickle_name)
        pickle_name += ".pkl"
        self.assertTrue(os.path.exists(pickle_name))
        # load and check
        with open(pickle_name, "rb") as f:
            loaded = pickle.load(f)
        xr.testing.assert_equal(results, loaded)
        self.assertDictEqual(loaded.attrs, self.EXTRA_ATTRS)

    def test_save_netcdf(self):
        system = BackendTestingHelper()
        results = system.run(
            self.DURATION, self.DT, ZeroInput(self.DURATION, self.DT).as_cubic_splines(), metadata=self.EXTRA_ATTRS,
        )
        # save to pickle
        nc_name = os.path.join(self.TEST_DIR, "netcdf_test")
        system.save_to_netcdf(results, nc_name)
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
