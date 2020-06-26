"""
Set of tests for FitzHugh-Nagumo model.
"""

import unittest

import numpy as np
import xarray as xr
from jitcdde import jitcdde_input
from neurolib.models.multimodel.builder.fitzhugh_nagumo import (
    DEFAULT_PARAMS,
    FitzHughNagumoMass,
    FitzHughNagumoNetwork,
    FitzHughNagumoNetworkNode,
)
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


class TestFitzHughNagumoMass(MassTestCase):
    def _create_mass(self):
        fhn = FitzHughNagumoMass()
        fhn.index = 0
        fhn.idx_state_var = 0
        fhn.init_mass()
        return fhn

    def test_init(self):
        fhn = self._create_mass()
        self.assertTrue(isinstance(fhn, FitzHughNagumoMass))
        self.assertDictEqual(fhn.parameters, DEFAULT_PARAMS)
        coupling_variables = {k: 0.0 for k in fhn.required_couplings}
        self.assertEqual(len(fhn._derivatives(coupling_variables)), fhn.num_state_variables)
        self.assertEqual(len(fhn.initial_state), fhn.num_state_variables)
        self.assertEqual(len(fhn.noise_input_idx), fhn.num_noise_variables)

    def test_run(self):
        fhn = self._create_mass()
        result = self._run_node(fhn, DURATION, DT)
        self.assertTrue(isinstance(result, np.ndarray))
        self.assertTupleEqual(result.shape, (int(DURATION / DT), fhn.num_state_variables))


class TestFitzHughNagumoNetworkNode(unittest.TestCase):
    def _create_node(self):
        node = FitzHughNagumoNetworkNode()
        node.index = 0
        node.idx_state_var = 0
        node.init_node()
        return node

    def test_init(self):
        fhn = self._create_node()
        self.assertTrue(isinstance(fhn, FitzHughNagumoNetworkNode))
        self.assertEqual(len(fhn), 1)
        self.assertDictEqual(fhn[0].parameters, DEFAULT_PARAMS)
        self.assertEqual(len(fhn.default_network_coupling), 2)
        np.testing.assert_equal(np.array(fhn[0].initial_state), fhn.initial_state)

    def test_run(self):
        fhn = self._create_node()
        all_results = []
        for backend, noise_func in BACKENDS_TO_TEST.items():
            result = fhn.run(
                DURATION, DT, noise_func(ZeroInput(DURATION, DT, fhn.num_noise_variables)), backend=backend, dt=DT,
            )
            self.assertTrue(isinstance(result, xr.Dataset))
            self.assertEqual(len(result), fhn.num_state_variables)
            self.assertTrue(all(state_var in result for state_var in fhn.state_variable_names[0]))
            self.assertTrue(
                all(result[state_var].shape == (int(DURATION / DT), 1) for state_var in fhn.state_variable_names[0])
            )
            all_results.append(result)
        # test results are the same from different backends
        for state_var in all_results[0]:
            corr_mat = np.corrcoef(
                np.vstack([result[state_var].values.flatten().astype(float) for result in all_results])
            )
            self.assertTrue(np.greater(corr_mat, CORR_THRESHOLD).all())


class TestFitzHughNagumoNetwork(unittest.TestCase):
    SC = np.random.rand(2, 2)
    DELAYS = np.array([[1.0, 2.0], [2.0, 1.0]])

    def test_init(self):
        fhn = FitzHughNagumoNetwork(self.SC, self.DELAYS)
        self.assertTrue(isinstance(fhn, FitzHughNagumoNetwork))
        self.assertEqual(len(fhn), self.SC.shape[0])
        self.assertEqual(fhn.initial_state.shape[0], fhn.num_state_variables)
        self.assertEqual(fhn.default_output, "x")

    def test_run(self):
        fhn = FitzHughNagumoNetwork(self.SC, self.DELAYS)
        all_results = []
        for backend, noise_func in BACKENDS_TO_TEST.items():
            result = fhn.run(
                DURATION, DT, noise_func(ZeroInput(DURATION, DT, fhn.num_noise_variables)), backend=backend,
            )
            self.assertTrue(isinstance(result, xr.Dataset))
            self.assertEqual(len(result), fhn.num_state_variables / fhn.num_nodes)
            self.assertTrue(all(result[result_].shape == (int(DURATION / DT), fhn.num_nodes) for result_ in result))
            all_results.append(result)
        # test results are the same from different backends
        for state_var in all_results[0]:
            corr_mat = np.corrcoef(
                np.vstack([result[state_var].values.flatten().astype(float) for result in all_results])
            )
            self.assertTrue(np.greater(corr_mat, CORR_THRESHOLD).all())


if __name__ == "__main__":
    unittest.main()
