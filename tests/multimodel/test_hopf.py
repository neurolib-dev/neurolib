"""
Set of tests for Hopf normal form model.
"""

import unittest

import numpy as np
import xarray as xr
from jitcdde import jitcdde_input
from neurolib.models.multimodel.builder.hopf import DEFAULT_PARAMS, HopfMass, HopfNetwork, HopfNetworkNode
from neurolib.models.multimodel.builder.model_input import ZeroInput

DURATION = 100.0
DT = 0.1
CORR_THRESHOLD = 0.99

# dictionary as backend name: format in which the noise is passed
BACKENDS_TO_TEST = {
    "jitcdde": lambda x: x.as_cubic_splines(),
    "numba": lambda x: x.as_array(),
}


class MassTestCase(unittest.TestCase):
    def _run_node(self, node, duration, dt):
        coupling_variables = {k: 0.0 for k in node.required_couplings}
        noise = ZeroInput(duration, dt, independent_realisations=node.num_noise_variables).as_cubic_splines()
        system = jitcdde_input(node._derivatives(coupling_variables), input=noise)
        system.constant_past(np.array(node.initial_state))
        system.adjust_diff()
        times = np.arange(dt, duration + dt, dt)
        return np.vstack([system.integrate(time) for time in times])


class TestHopfMass(MassTestCase):
    def _create_mass(self):
        hopf = HopfMass()
        hopf.index = 0
        hopf.idx_state_var = 0
        hopf.init_mass()
        return hopf

    def test_init(self):
        hopf = self._create_mass()
        self.assertTrue(isinstance(hopf, HopfMass))
        self.assertDictEqual(hopf.parameters, DEFAULT_PARAMS)
        coupling_variables = {k: 0.0 for k in hopf.required_couplings}
        self.assertEqual(len(hopf._derivatives(coupling_variables)), hopf.num_state_variables)
        self.assertEqual(len(hopf.initial_state), hopf.num_state_variables)
        self.assertEqual(len(hopf.noise_input_idx), hopf.num_noise_variables)

    def test_run(self):
        hopf = self._create_mass()
        result = self._run_node(hopf, DURATION, DT)
        self.assertTrue(isinstance(result, np.ndarray))
        self.assertTupleEqual(result.shape, (int(DURATION / DT), hopf.num_state_variables))


class TestHopfNetworkNode(unittest.TestCase):
    def _create_node(self):
        node = HopfNetworkNode()
        node.index = 0
        node.idx_state_var = 0
        node.init_node()
        return node

    def test_init(self):
        hopf = self._create_node()
        self.assertTrue(isinstance(hopf, HopfNetworkNode))
        self.assertEqual(len(hopf), 1)
        self.assertDictEqual(hopf[0].parameters, DEFAULT_PARAMS)
        self.assertEqual(len(hopf.default_network_coupling), 2)
        np.testing.assert_equal(np.array(hopf[0].initial_state), hopf.initial_state)

    def test_run(self):
        hopf = self._create_node()
        all_results = []
        for backend, noise_func in BACKENDS_TO_TEST.items():
            result = hopf.run(
                DURATION, DT, noise_func(ZeroInput(DURATION, DT, hopf.num_noise_variables)), backend=backend,
            )
            self.assertTrue(isinstance(result, xr.Dataset))
            self.assertEqual(len(result), hopf.num_state_variables)
            self.assertTrue(all(state_var in result for state_var in hopf.state_variable_names[0]))
            self.assertTrue(
                all(result[state_var].shape == (int(DURATION / DT), 1) for state_var in hopf.state_variable_names[0])
            )
            all_results.append(result)
        # test results are the same from different backends
        for state_var in all_results[0]:
            corr_mat = np.corrcoef(
                np.vstack([result[state_var].values.flatten().astype(float) for result in all_results])
            )
            self.assertTrue(np.greater(corr_mat, CORR_THRESHOLD).all())


class TestHopfNetwork(unittest.TestCase):
    SC = np.random.rand(2, 2)
    DELAYS = np.array([[1.0, 2.0], [2.0, 1.0]])

    def test_init(self):
        hopf = HopfNetwork(self.SC, self.DELAYS)
        self.assertTrue(isinstance(hopf, HopfNetwork))
        self.assertEqual(len(hopf), self.SC.shape[0])
        self.assertEqual(hopf.initial_state.shape[0], hopf.num_state_variables)
        self.assertEqual(hopf.default_output, "x")

    def test_run(self):
        hopf = HopfNetwork(self.SC, self.DELAYS)
        all_results = []
        for backend, noise_func in BACKENDS_TO_TEST.items():
            result = hopf.run(
                DURATION, DT, noise_func(ZeroInput(DURATION, DT, hopf.num_noise_variables)), backend=backend,
            )
            self.assertTrue(isinstance(result, xr.Dataset))
            self.assertEqual(len(result), hopf.num_state_variables / hopf.num_nodes)
            self.assertTrue(all(result[result_].shape == (int(DURATION / DT), hopf.num_nodes) for result_ in result))
        all_results.append(result)
        # test results are the same from different backends
        for state_var in all_results[0]:
            corr_mat = np.corrcoef(
                np.vstack([result[state_var].values.flatten().astype(float) for result in all_results])
            )
            self.assertTrue(np.greater(corr_mat, CORR_THRESHOLD).all())


if __name__ == "__main__":
    unittest.main()
