"""
Set of tests for Wong-Wang model.
"""
import unittest
import pytest

import numpy as np
import xarray as xr
from jitcdde import jitcdde_input
from neurolib.models.multimodel.builder.base.constants import EXC
from neurolib.models.multimodel.builder.wong_wang import (
    WW_EXC_DEFAULT_PARAMS,
    WW_INH_DEFAULT_PARAMS,
    WW_REDUCED_DEFAULT_PARAMS,
    ExcitatoryWongWangMass,
    InhibitoryWongWangMass,
    ReducedWongWangMass,
    ReducedWongWangNetwork,
    ReducedWongWangNode,
    WongWangNetwork,
    WongWangNode,
)
from neurolib.utils.stimulus import ZeroInput

SEED = 42
DURATION = 100.0
DT = 0.1
CORR_THRESHOLD = 0.9

# dictionary as backend name: format in which the noise is passed
BACKENDS_TO_TEST = {
    "jitcdde": lambda x, d, dt: x.as_cubic_splines(d, dt),
    "numba": lambda x, d, dt: x.as_array(d, dt).T,
}


class MassTestCase(unittest.TestCase):
    def _run_mass(self, node, duration, dt):
        coupling_variables = {k: 0.0 for k in node.required_couplings}
        noise = ZeroInput(num_iid=node.num_noise_variables).as_cubic_splines(duration, dt)
        system = jitcdde_input(node._derivatives(coupling_variables), input=noise)
        system.constant_past(np.array(node.initial_state))
        system.adjust_diff()
        times = np.arange(dt, duration + dt, dt)
        return np.vstack([system.integrate(time) for time in times])


class TestWongWangMass(MassTestCase):
    def _create_exc_mass(self):
        exc = ExcitatoryWongWangMass()
        exc.index = 0
        exc.idx_state_var = 0
        exc.init_mass()
        return exc

    def _create_inh_mass(self):
        inh = InhibitoryWongWangMass()
        inh.index = 0
        inh.idx_state_var = 0
        inh.init_mass()
        return inh

    def test_init(self):
        ww_exc = self._create_exc_mass()
        ww_inh = self._create_inh_mass()
        self.assertTrue(isinstance(ww_exc, ExcitatoryWongWangMass))
        self.assertTrue(isinstance(ww_inh, InhibitoryWongWangMass))
        self.assertDictEqual({k: v for k, v in ww_exc.params.items() if "noise" not in k}, WW_EXC_DEFAULT_PARAMS)
        self.assertDictEqual({k: v for k, v in ww_inh.params.items() if "noise" not in k}, WW_INH_DEFAULT_PARAMS)
        for ww in [ww_exc, ww_inh]:
            coupling_variables = {k: 0.0 for k in ww.required_couplings}
            self.assertEqual(len(ww._derivatives(coupling_variables)), ww.num_state_variables)
            self.assertEqual(len(ww.initial_state), ww.num_state_variables)
            self.assertEqual(len(ww.noise_input_idx), ww.num_noise_variables)

    def test_run(self):
        ww_exc = self._create_exc_mass()
        ww_inh = self._create_inh_mass()
        for ww in [ww_exc, ww_inh]:
            result = self._run_mass(ww, DURATION, DT)
            self.assertTrue(isinstance(result, np.ndarray))
            self.assertTupleEqual(result.shape, (int(DURATION / DT), ww.num_state_variables))


class TestReducedWongWangMass(MassTestCase):
    def _create_mass(self):
        rww = ReducedWongWangMass()
        rww.index = 0
        rww.idx_state_var = 0
        rww.init_mass()
        return rww

    def test_init(self):
        rww = self._create_mass()
        self.assertTrue(isinstance(rww, ReducedWongWangMass))
        self.assertDictEqual({k: v for k, v in rww.params.items() if "noise" not in k}, WW_REDUCED_DEFAULT_PARAMS)
        coupling_variables = {k: 0.0 for k in rww.required_couplings}
        self.assertEqual(len(rww._derivatives(coupling_variables)), rww.num_state_variables)
        self.assertEqual(len(rww.initial_state), rww.num_state_variables)
        self.assertEqual(len(rww.noise_input_idx), rww.num_noise_variables)

    def test_run(self):
        rww = self._create_mass()
        result = self._run_mass(rww, DURATION, DT)
        self.assertTrue(isinstance(result, np.ndarray))
        self.assertTupleEqual(result.shape, (int(DURATION / DT), rww.num_state_variables))


class TestWongWangNode(unittest.TestCase):
    def _create_node(self):
        node = WongWangNode(exc_seed=SEED, inh_seed=SEED)
        node.index = 0
        node.idx_state_var = 0
        node.init_node()
        return node

    def test_init(self):
        ww = self._create_node()
        self.assertTrue(isinstance(ww, WongWangNode))
        self.assertEqual(len(ww), 2)
        self.assertDictEqual({k: v for k, v in ww[0].params.items() if "noise" not in k}, WW_EXC_DEFAULT_PARAMS)
        self.assertDictEqual({k: v for k, v in ww[1].params.items() if "noise" not in k}, WW_INH_DEFAULT_PARAMS)
        self.assertEqual(len(ww.default_network_coupling), 2)
        np.testing.assert_equal(
            np.array(sum([wwm.initial_state for wwm in ww], [])),
            ww.initial_state,
        )

    def test_run(self):
        ww = self._create_node()
        all_results = []
        for backend, noise_func in BACKENDS_TO_TEST.items():
            result = ww.run(
                DURATION,
                DT,
                noise_func(ZeroInput(ww.num_noise_variables), DURATION, DT),
                backend=backend,
            )
            self.assertTrue(isinstance(result, xr.Dataset))
            self.assertEqual(len(result), ww.num_state_variables)
            self.assertTrue(all(state_var in result for state_var in ww.state_variable_names[0]))
            self.assertTrue(
                all(result[state_var].shape == (int(DURATION / DT), 1) for state_var in ww.state_variable_names[0])
            )
            all_results.append(result)
        # test results are the same from different backends
        for state_var in all_results[0]:
            corr_mat = np.corrcoef(
                np.vstack([result[state_var].values.flatten().astype(float) for result in all_results])
            )
            self.assertTrue(np.greater(corr_mat, CORR_THRESHOLD).all())


class TestReducedWongWangNode(unittest.TestCase):
    def _create_node(self):
        node = ReducedWongWangNode(seed=SEED)
        node.index = 0
        node.idx_state_var = 0
        node.init_node()
        return node

    def test_init(self):
        rww = self._create_node()
        self.assertTrue(isinstance(rww, ReducedWongWangNode))
        self.assertEqual(len(rww), 1)
        self.assertDictEqual({k: v for k, v in rww[0].params.items() if "noise" not in k}, WW_REDUCED_DEFAULT_PARAMS)
        self.assertEqual(len(rww.default_network_coupling), 1)
        np.testing.assert_equal(np.array(rww[0].initial_state), rww.initial_state)

    def test_run(self):
        rww = self._create_node()
        all_results = []
        for backend, noise_func in BACKENDS_TO_TEST.items():
            result = rww.run(
                DURATION, DT, noise_func(ZeroInput(rww.num_noise_variables), DURATION, DT), backend=backend
            )
            self.assertTrue(isinstance(result, xr.Dataset))
            self.assertEqual(len(result), rww.num_state_variables)
            self.assertTrue(all(state_var in result for state_var in rww.state_variable_names[0]))
            self.assertTrue(
                all(result[state_var].shape == (int(DURATION / DT), 1) for state_var in rww.state_variable_names[0])
            )
            all_results.append(result)
        # test results are the same from different backends
        for state_var in all_results[0]:
            corr_mat = np.corrcoef(
                np.vstack([result[state_var].values.flatten().astype(float) for result in all_results])
            )
            self.assertTrue(np.greater(corr_mat, CORR_THRESHOLD).all())


class TestWongWangNetwork(unittest.TestCase):
    SC = np.random.rand(2, 2)
    DELAYS = np.array([[0.0, 0.0], [0.0, 0.0]])

    def test_init(self):
        ww = WongWangNetwork(self.SC, self.DELAYS)
        self.assertTrue(isinstance(ww, WongWangNetwork))
        self.assertEqual(len(ww), self.SC.shape[0])
        self.assertEqual(ww.initial_state.shape[0], ww.num_state_variables)
        self.assertEqual(ww.default_output, f"S_{EXC}")

    @pytest.mark.xfail
    def test_run(self):
        ww = WongWangNetwork(self.SC, self.DELAYS, exc_seed=SEED, inh_seed=SEED)
        all_results = []
        for backend, noise_func in BACKENDS_TO_TEST.items():
            result = ww.run(
                DURATION,
                DT,
                noise_func(ZeroInput(ww.num_noise_variables), DURATION, DT),
                backend=backend,
            )
            self.assertTrue(isinstance(result, xr.Dataset))
            self.assertEqual(len(result), ww.num_state_variables / ww.num_nodes)
            self.assertTrue(all(result[result_].shape == (int(DURATION / DT), ww.num_nodes) for result_ in result))
            all_results.append(result)
        # test results are the same from different backends
        for state_var in all_results[0]:
            corr_mat = np.corrcoef(
                np.vstack([result[state_var].values.flatten().astype(float) for result in all_results])
            )
            self.assertTrue(np.greater(corr_mat, CORR_THRESHOLD).all())


class TestReducedWongWangNetwork(unittest.TestCase):
    SC = np.random.rand(2, 2)
    DELAYS = np.array([[0.0, 0.0], [0.0, 0.0]])

    def test_init(self):
        rww = ReducedWongWangNetwork(self.SC, self.DELAYS)
        self.assertTrue(isinstance(rww, ReducedWongWangNetwork))
        self.assertEqual(len(rww), self.SC.shape[0])
        self.assertEqual(rww.initial_state.shape[0], rww.num_state_variables)
        self.assertEqual(rww.default_output, "S")

    def test_run(self):
        rww = ReducedWongWangNetwork(self.SC, self.DELAYS, seed=SEED)
        all_results = []
        for backend, noise_func in BACKENDS_TO_TEST.items():
            result = rww.run(
                DURATION,
                DT,
                noise_func(ZeroInput(rww.num_noise_variables), DURATION, DT),
                backend=backend,
            )
            self.assertTrue(isinstance(result, xr.Dataset))
            self.assertEqual(len(result), rww.num_state_variables / rww.num_nodes)
            self.assertTrue(all(result[result_].shape == (int(DURATION / DT), rww.num_nodes) for result_ in result))
            all_results.append(result)
        # test results are the same from different backends
        for state_var in all_results[0]:
            corr_mat = np.corrcoef(
                np.vstack([result[state_var].values.flatten().astype(float) for result in all_results])
            )
            self.assertTrue(np.greater(corr_mat, CORR_THRESHOLD).all())


if __name__ == "__main__":
    unittest.main()
