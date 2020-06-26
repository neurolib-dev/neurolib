"""
Set of tests for thalamus mass models.
"""

import unittest

import numpy as np
import xarray as xr
from jitcdde import jitcdde_input
from neurolib.models.multimodel.builder.model_input import ZeroInput
from neurolib.models.multimodel.builder.thalamus import (
    DEFAULT_PARAMS_TCR,
    DEFAULT_PARAMS_TRN,
    ThalamicNetworkNode,
    ThalamicReticularPopulation,
    ThalamocorticalPopulation,
)

DURATION = 100.0
SPIN_UP = 1.0
DT = 0.01
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


class TestThalamicMass(MassTestCase):
    def _create_tcr_mass(self):
        tcr = ThalamocorticalPopulation()
        tcr.index = 0
        tcr.idx_state_var = 0
        tcr.init_mass()
        return tcr

    def _create_trn_mass(self):
        trn = ThalamicReticularPopulation()
        trn.index = 0
        trn.idx_state_var = 0
        trn.init_mass()
        return trn

    def test_init(self):
        tcr = self._create_tcr_mass()
        trn = self._create_trn_mass()
        self.assertTrue(isinstance(tcr, ThalamocorticalPopulation))
        self.assertTrue(isinstance(trn, ThalamicReticularPopulation))
        self.assertDictEqual(tcr.params, DEFAULT_PARAMS_TCR)
        self.assertDictEqual(trn.params, DEFAULT_PARAMS_TRN)
        for thlm in [tcr, trn]:
            coupling_variables = {k: 0.0 for k in thlm.required_couplings}
            self.assertEqual(
                len(thlm._derivatives(coupling_variables)), thlm.num_state_variables,
            )
            self.assertEqual(len(thlm.initial_state), thlm.num_state_variables)
            self.assertEqual(len(thlm.noise_input_idx), thlm.num_noise_variables)

    def test_run(self):
        tcr = self._create_tcr_mass()
        trn = self._create_trn_mass()
        for thlm in [tcr, trn]:
            result = self._run_node(thlm, DURATION, DT)
            self.assertTrue(isinstance(result, np.ndarray))
            self.assertTupleEqual(result.shape, (int(DURATION / DT), thlm.num_state_variables))


class TestThalamicNetworkNode(unittest.TestCase):
    def _create_node(self):
        node = ThalamicNetworkNode()
        node.index = 0
        node.idx_state_var = 0
        node.init_node()
        return node

    def test_init(self):
        thlm = self._create_node()
        self.assertTrue(isinstance(thlm, ThalamicNetworkNode))
        self.assertEqual(len(thlm), 2)
        self.assertDictEqual(thlm[0].params, DEFAULT_PARAMS_TCR)
        self.assertDictEqual(thlm[1].params, DEFAULT_PARAMS_TRN)
        self.assertEqual(len(thlm.default_network_coupling), 2)
        np.testing.assert_equal(
            np.array(sum([thlmm.initial_state for thlmm in thlm], [])), thlm.initial_state,
        )

    def test_run(self):
        thlm = self._create_node()
        all_results = []
        for backend, noise_func in BACKENDS_TO_TEST.items():
            result = thlm.run(
                DURATION,
                DT,
                noise_func(ZeroInput(DURATION + SPIN_UP, DT, thlm.num_noise_variables)),
                time_spin_up=SPIN_UP,
                backend=backend,
                dt=DT,
            )
            self.assertTrue(isinstance(result, xr.Dataset))
            self.assertEqual(len(result), thlm.num_state_variables)
            self.assertTrue(all(state_var in result for state_var in thlm.state_variable_names[0]))
            self.assertTrue(
                all(result[state_var].shape == (int(DURATION / DT), 1) for state_var in thlm.state_variable_names[0])
            )
            all_results.append(result)
        # test results are the same from different backends
        for state_var in all_results[0]:
            corr_mat = np.corrcoef(
                np.vstack([result[state_var].values.flatten().astype(float) for result in all_results])
            )
            if not np.any(np.isnan(corr_mat)):
                # some variables have zero variance (i.e. excitatory synaptic
                # activity to the TCR - it does not have any in isolated mode
                # without noise)
                self.assertTrue(np.greater(corr_mat, CORR_THRESHOLD).all())


if __name__ == "__main__":
    unittest.main()
