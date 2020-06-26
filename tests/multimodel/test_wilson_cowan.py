"""
Set of tests for Wilson-Cowan model.
"""

import unittest

import numpy as np
import xarray as xr
from jitcdde import jitcdde_input
from neurolib.models.multimodel.builder.base.constants import EXC
from neurolib.models.multimodel.builder.model_input import ZeroInput
from neurolib.models.multimodel.builder.wilson_cowan import (
    DEFAULT_PARAMS_EXC,
    DEFAULT_PARAMS_INH,
    ExcitatoryWilsonCowanMass,
    InhibitoryWilsonCowanMass,
    WilsonCowanNetwork,
    WilsonCowanNetworkNode,
)

DURATION = 100.0
DT = 0.1
CORR_THRESHOLD = 0.99

# dictionary as backend name: format in which the noise is passed
BACKENDS_TO_TEST = {
    "jitcdde": lambda x: x.as_cubic_splines(),
    "numba": lambda x: x.as_array(),
}


class MassTestCase(unittest.TestCase):
    def _run_mass(self, node, duration, dt):
        coupling_variables = {k: 0.0 for k in node.required_couplings}
        noise = ZeroInput(duration, dt, independent_realisations=node.num_noise_variables).as_cubic_splines()
        system = jitcdde_input(node._derivatives(coupling_variables), input=noise)
        system.constant_past(np.array(node.initial_state))
        system.adjust_diff()
        times = np.arange(dt, duration + dt, dt)
        return np.vstack([system.integrate(time) for time in times])


class TestWilsonCowanMass(MassTestCase):
    def _create_exc_mass(self):
        exc = ExcitatoryWilsonCowanMass()
        exc.index = 0
        exc.idx_state_var = 0
        exc.init_mass()
        return exc

    def _create_inh_mass(self):
        inh = InhibitoryWilsonCowanMass()
        inh.index = 0
        inh.idx_state_var = 0
        inh.init_mass()
        return inh

    def test_init(self):
        wc_exc = self._create_exc_mass()
        wc_inh = self._create_inh_mass()
        self.assertTrue(isinstance(wc_exc, ExcitatoryWilsonCowanMass))
        self.assertTrue(isinstance(wc_inh, InhibitoryWilsonCowanMass))
        self.assertDictEqual(wc_exc.params, DEFAULT_PARAMS_EXC)
        self.assertDictEqual(wc_inh.params, DEFAULT_PARAMS_INH)
        for wc in [wc_exc, wc_inh]:
            coupling_variables = {k: 0.0 for k in wc.required_couplings}
            self.assertEqual(len(wc._derivatives(coupling_variables)), wc.num_state_variables)
            self.assertEqual(len(wc.initial_state), wc.num_state_variables)
            self.assertEqual(len(wc.noise_input_idx), wc.num_noise_variables)

    def test_run(self):
        wc_exc = self._create_exc_mass()
        wc_inh = self._create_inh_mass()
        for wc in [wc_exc, wc_inh]:
            result = self._run_mass(wc, DURATION, DT)
            self.assertTrue(isinstance(result, np.ndarray))
            self.assertTupleEqual(result.shape, (int(DURATION / DT), wc.num_state_variables))


class TestWilsonCowanNetworkNode(unittest.TestCase):
    def _create_node(self):
        node = WilsonCowanNetworkNode()
        node.index = 0
        node.idx_state_var = 0
        node.init_node()
        return node

    def test_init(self):
        wc = self._create_node()
        self.assertTrue(isinstance(wc, WilsonCowanNetworkNode))
        self.assertEqual(len(wc), 2)
        self.assertDictEqual(wc[0].params, DEFAULT_PARAMS_EXC)
        self.assertDictEqual(wc[1].params, DEFAULT_PARAMS_INH)
        self.assertEqual(len(wc.default_network_coupling), 1)
        np.testing.assert_equal(
            np.array(sum([wcm.initial_state for wcm in wc], [])), wc.initial_state,
        )

    def test_run(self):
        wc = self._create_node()
        all_results = []
        for backend, noise_func in BACKENDS_TO_TEST.items():
            result = wc.run(DURATION, DT, noise_func(ZeroInput(DURATION, DT, wc.num_noise_variables)), backend=backend,)
            self.assertTrue(isinstance(result, xr.Dataset))
            self.assertEqual(len(result), wc.num_state_variables)
            self.assertTrue(all(state_var in result for state_var in wc.state_variable_names[0]))
            self.assertTrue(
                all(result[state_var].shape == (int(DURATION / DT), 1) for state_var in wc.state_variable_names[0])
            )
            all_results.append(result)
        # test results are the same from different backends
        for state_var in all_results[0]:
            corr_mat = np.corrcoef(
                np.vstack([result[state_var].values.flatten().astype(float) for result in all_results])
            )
            self.assertTrue(np.greater(corr_mat, CORR_THRESHOLD).all())


class TestWilsonCowanNetwork(unittest.TestCase):
    SC = np.random.rand(2, 2)
    DELAYS = np.zeros((2, 2))

    def test_init(self):
        wc = WilsonCowanNetwork(self.SC, self.DELAYS)
        self.assertTrue(isinstance(wc, WilsonCowanNetwork))
        self.assertEqual(len(wc), self.SC.shape[0])
        self.assertEqual(wc.initial_state.shape[0], wc.num_state_variables)
        self.assertEqual(wc.default_output, f"q_mean_{EXC}")

    def test_run(self):
        wc = WilsonCowanNetwork(self.SC, self.DELAYS)
        all_results = []
        for backend, noise_func in BACKENDS_TO_TEST.items():
            result = wc.run(DURATION, DT, noise_func(ZeroInput(DURATION, DT, wc.num_noise_variables)), backend=backend,)
            self.assertTrue(isinstance(result, xr.Dataset))
            self.assertEqual(len(result), wc.num_state_variables / wc.num_nodes)
            self.assertTrue(all(result[result_].shape == (int(DURATION / DT), wc.num_nodes) for result_ in result))
            all_results.append(result)
        # test results are the same from different backends
        for state_var in all_results[0]:
            corr_mat = np.corrcoef(
                np.vstack([result[state_var].values.flatten().astype(float) for result in all_results])
            )
            self.assertTrue(np.greater(corr_mat, CORR_THRESHOLD).all())


if __name__ == "__main__":
    unittest.main()
