"""
Set of tests for FitzHugh-Nagumo model.
"""

import unittest

import numpy as np
import xarray as xr
from jitcdde import jitcdde_input
from neurolib.models.fhn import FHNModel
from neurolib.models.multimodel.builder.fitzhugh_nagumo import (
    FHN_DEFAULT_PARAMS,
    FitzHughNagumoMass,
    FitzHughNagumoNetwork,
    FitzHughNagumoNode,
)
from neurolib.utils.stimulus import ZeroInput

SEED = 42
DURATION = 100.0
DT = 0.1
CORR_THRESHOLD = 0.95
NEUROLIB_VARIABLES_TO_TEST = ["x", "y"]

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
        self.assertDictEqual({k: v for k, v in fhn.params.items() if "noise" not in k}, FHN_DEFAULT_PARAMS)
        coupling_variables = {k: 0.0 for k in fhn.required_couplings}
        self.assertEqual(len(fhn._derivatives(coupling_variables)), fhn.num_state_variables)
        self.assertEqual(len(fhn.initial_state), fhn.num_state_variables)
        self.assertEqual(len(fhn.noise_input_idx), fhn.num_noise_variables)

    def test_run(self):
        fhn = self._create_mass()
        result = self._run_mass(fhn, DURATION, DT)
        self.assertTrue(isinstance(result, np.ndarray))
        self.assertTupleEqual(result.shape, (int(DURATION / DT), fhn.num_state_variables))


class TestFitzHughNagumoNode(unittest.TestCase):
    def _create_node(self):
        node = FitzHughNagumoNode(seed=SEED)
        node.index = 0
        node.idx_state_var = 0
        node.init_node()
        return node

    def test_init(self):
        fhn = self._create_node()
        self.assertTrue(isinstance(fhn, FitzHughNagumoNode))
        self.assertEqual(len(fhn), 1)
        self.assertDictEqual({k: v for k, v in fhn[0].params.items() if "noise" not in k}, FHN_DEFAULT_PARAMS)
        self.assertEqual(len(fhn.default_network_coupling), 2)
        np.testing.assert_equal(np.array(fhn[0].initial_state), fhn.initial_state)

    def test_run(self):
        fhn = self._create_node()
        all_results = []
        for backend, noise_func in BACKENDS_TO_TEST.items():
            result = fhn.run(
                DURATION,
                DT,
                noise_func(ZeroInput(fhn.num_noise_variables), DURATION, DT),
                backend=backend,
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

    def test_compare_w_neurolib_native_model(self):
        """
        Compare with neurolib's native FitzHugh-Nagumo model.
        """
        # run this model
        fhn_multi = self._create_node()
        multi_result = fhn_multi.run(
            DURATION, DT, ZeroInput(fhn_multi.num_noise_variables).as_array(DURATION, DT), backend="numba"
        )
        # run neurolib's model
        fhn = FHNModel(seed=SEED)
        fhn.params["duration"] = DURATION
        fhn.params["dt"] = DT
        fhn.run()
        for var in NEUROLIB_VARIABLES_TO_TEST:
            corr_mat = np.corrcoef(fhn[var], multi_result[var].values.T)
            self.assertTrue(np.greater(corr_mat, CORR_THRESHOLD).all())


class TestFitzHughNagumoNetwork(unittest.TestCase):
    SC = np.array(([[0.0, 1.0], [0.0, 0.0]]))
    DELAYS = np.array([[0.0, 0.0], [0.0, 0.0]])

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
                DURATION,
                DT,
                noise_func(ZeroInput(fhn.num_noise_variables), DURATION, DT),
                backend=backend,
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

    def test_compare_w_neurolib_native_model(self):
        """
        Compare with neurolib's native FitzHugh-Nagumo model.
        """
        # run this model - default is diffusive coupling
        fhn_multi = FitzHughNagumoNetwork(self.SC, self.DELAYS, seed=SEED)
        multi_result = fhn_multi.run(
            DURATION, DT, ZeroInput(fhn_multi.num_noise_variables).as_array(DURATION, DT), backend="numba"
        )
        # run neurolib's model
        fhn_neurolib = FHNModel(Cmat=self.SC, Dmat=self.DELAYS, seed=SEED)
        fhn_neurolib.params["duration"] = DURATION
        fhn_neurolib.params["dt"] = DT
        # there is no "global coupling" parameter in MultiModel
        fhn_neurolib.params["K_gl"] = 1.0
        # delays <-> length matrix
        fhn_neurolib.params["signalV"] = 1.0
        fhn_neurolib.params["coupling"] = "diffusive"
        fhn_neurolib.params["sigma_ou"] = 0.0
        fhn_neurolib.params["xs_init"] = fhn_multi.initial_state[::2][:, np.newaxis]
        fhn_neurolib.params["ys_init"] = fhn_multi.initial_state[1::2][:, np.newaxis]
        fhn_neurolib.run()
        for var in NEUROLIB_VARIABLES_TO_TEST:
            corr_mat = np.corrcoef(fhn_neurolib[var], multi_result[var].values.T)
            self.assertTrue(np.greater(corr_mat, CORR_THRESHOLD).all())


if __name__ == "__main__":
    unittest.main()
