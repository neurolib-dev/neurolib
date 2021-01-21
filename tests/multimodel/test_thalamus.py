"""
Set of tests for thalamus mass models.
"""

import unittest

import numpy as np
import xarray as xr
from jitcdde import jitcdde_input
from neurolib.models.multimodel.builder.thalamus import (
    TCR_DEFAULT_PARAMS,
    TRN_DEFAULT_PARAMS,
    ThalamicNode,
    ThalamicReticularMass,
    ThalamocorticalMass,
)
from neurolib.models.thalamus import ThalamicMassModel
from neurolib.utils.stimulus import ZeroInput

DURATION = 100.0
DT = 0.01
CORR_THRESHOLD = 0.9
NEUROLIB_VARIABLES_TO_TEST = [
    ("r_mean_EXC", "Q_t"),
    ("r_mean_INH", "Q_r"),
    ("V_EXC", "V_t"),
    ("V_INH", "V_r"),
]

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


class TestThalamicMass(MassTestCase):
    def _create_tcr_mass(self):
        tcr = ThalamocorticalMass()
        tcr.index = 0
        tcr.idx_state_var = 0
        tcr.init_mass()
        return tcr

    def _create_trn_mass(self):
        trn = ThalamicReticularMass()
        trn.index = 0
        trn.idx_state_var = 0
        trn.init_mass()
        return trn

    def test_init(self):
        tcr = self._create_tcr_mass()
        trn = self._create_trn_mass()
        self.assertTrue(isinstance(tcr, ThalamocorticalMass))
        self.assertTrue(isinstance(trn, ThalamicReticularMass))
        self.assertDictEqual({k: v for k, v in tcr.params.items() if "noise" not in k}, TCR_DEFAULT_PARAMS)
        self.assertDictEqual({k: v for k, v in trn.params.items() if "noise" not in k}, TRN_DEFAULT_PARAMS)
        for thlm in [tcr, trn]:
            coupling_variables = {k: 0.0 for k in thlm.required_couplings}
            self.assertEqual(
                len(thlm._derivatives(coupling_variables)),
                thlm.num_state_variables,
            )
            self.assertEqual(len(thlm.initial_state), thlm.num_state_variables)
            self.assertEqual(len(thlm.noise_input_idx), thlm.num_noise_variables)

    def test_run(self):
        tcr = self._create_tcr_mass()
        trn = self._create_trn_mass()
        for thlm in [tcr, trn]:
            result = self._run_mass(thlm, DURATION, DT)
            self.assertTrue(isinstance(result, np.ndarray))
            self.assertTupleEqual(result.shape, (int(DURATION / DT), thlm.num_state_variables))


class TestThalamicNode(unittest.TestCase):
    def _create_node(self):
        node = ThalamicNode()
        node.index = 0
        node.idx_state_var = 0
        node.init_node()
        return node

    def test_init(self):
        thlm = self._create_node()
        self.assertTrue(isinstance(thlm, ThalamicNode))
        self.assertEqual(len(thlm), 2)
        self.assertDictEqual({k: v for k, v in thlm[0].params.items() if "noise" not in k}, TCR_DEFAULT_PARAMS)
        self.assertDictEqual({k: v for k, v in thlm[1].params.items() if "noise" not in k}, TRN_DEFAULT_PARAMS)
        self.assertEqual(len(thlm.default_network_coupling), 2)
        np.testing.assert_equal(
            np.array(sum([thlmm.initial_state for thlmm in thlm], [])),
            thlm.initial_state,
        )

    def test_run(self):
        thlm = self._create_node()
        all_results = []
        for backend, noise_func in BACKENDS_TO_TEST.items():
            result = thlm.run(
                DURATION,
                DT,
                noise_func(ZeroInput(thlm.num_noise_variables), DURATION, DT),
                backend=backend,
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

    def test_compare_w_neurolib_native_model(self):
        """
        Compare with neurolib's native thalamic model.
        """
        # run this model
        thalamus_multi = self._create_node()
        multi_result = thalamus_multi.run(
            DURATION, DT, ZeroInput(thalamus_multi.num_noise_variables).as_array(DURATION, DT), backend="numba"
        )
        # run neurolib's model
        thlm_neurolib = ThalamicMassModel()
        thlm_neurolib.params["duration"] = DURATION
        thlm_neurolib.params["dt"] = DT
        thlm_neurolib.params["V_t_init"] = np.array([-70])
        thlm_neurolib.params["V_r_init"] = np.array([-70])
        thlm_neurolib.run()
        for (var_multi, var_neurolib) in NEUROLIB_VARIABLES_TO_TEST:
            corr_mat = np.corrcoef(thlm_neurolib[var_neurolib], multi_result[var_multi].values.T)
            self.assertTrue(np.greater(corr_mat, CORR_THRESHOLD).all())


if __name__ == "__main__":
    unittest.main()
