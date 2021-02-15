"""
Set of tests for Wilson-Cowan model.
"""

import unittest
import pytest

import numpy as np
import xarray as xr
from jitcdde import jitcdde_input
from neurolib.models.multimodel.builder.base.constants import EXC
from neurolib.models.multimodel.builder.wilson_cowan import (
    WC_EXC_DEFAULT_PARAMS,
    WC_INH_DEFAULT_PARAMS,
    ExcitatoryWilsonCowanMass,
    InhibitoryWilsonCowanMass,
    WilsonCowanNetwork,
    WilsonCowanNode,
)
from neurolib.models.wc import WCModel
from neurolib.utils.stimulus import ZeroInput

SEED = 42
DURATION = 100.0
DT = 0.01
CORR_THRESHOLD = 0.75
NEUROLIB_VARIABLES_TO_TEST = [("q_mean_EXC", "exc"), ("q_mean_INH", "inh")]

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
        self.assertDictEqual({k: v for k, v in wc_exc.params.items() if "noise" not in k}, WC_EXC_DEFAULT_PARAMS)
        self.assertDictEqual({k: v for k, v in wc_inh.params.items() if "noise" not in k}, WC_INH_DEFAULT_PARAMS)
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


class TestWilsonCowanNode(unittest.TestCase):
    def _create_node(self):
        node = WilsonCowanNode(exc_seed=SEED, inh_seed=SEED)
        node.index = 0
        node.idx_state_var = 0
        node.init_node()
        return node

    def test_init(self):
        wc = self._create_node()
        self.assertTrue(isinstance(wc, WilsonCowanNode))
        self.assertEqual(len(wc), 2)
        self.assertDictEqual({k: v for k, v in wc[0].params.items() if "noise" not in k}, WC_EXC_DEFAULT_PARAMS)
        self.assertDictEqual({k: v for k, v in wc[1].params.items() if "noise" not in k}, WC_INH_DEFAULT_PARAMS)
        self.assertEqual(len(wc.default_network_coupling), 2)
        np.testing.assert_equal(
            np.array(sum([wcm.initial_state for wcm in wc], [])),
            wc.initial_state,
        )

    def test_run(self):
        wc = self._create_node()
        all_results = []
        for backend, noise_func in BACKENDS_TO_TEST.items():
            result = wc.run(
                DURATION,
                DT,
                noise_func(ZeroInput(wc.num_noise_variables), DURATION, DT),
                backend=backend,
            )
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

    def test_compare_w_neurolib_native_model(self):
        """
        Compare with neurolib's native Wilson-Cowan model.
        """
        # run this model
        wc_multi = self._create_node()
        multi_result = wc_multi.run(
            DURATION, DT, ZeroInput(wc_multi.num_noise_variables).as_array(DURATION, DT), backend="numba"
        )
        # run neurolib's model
        wc_neurolib = WCModel(seed=SEED)
        wc_neurolib.params["duration"] = DURATION
        wc_neurolib.params["dt"] = DT
        # match initial state
        wc_neurolib.params["exc_init"] = np.array([[wc_multi.initial_state[0]]])
        wc_neurolib.params["inh_init"] = np.array([[wc_multi.initial_state[1]]])
        wc_neurolib.run()
        for (var_multi, var_neurolib) in NEUROLIB_VARIABLES_TO_TEST:
            corr_mat = np.corrcoef(wc_neurolib[var_neurolib], multi_result[var_multi].values.T)
            self.assertTrue(np.greater(corr_mat, CORR_THRESHOLD).all())


class TestWilsonCowanNetwork(unittest.TestCase):
    SC = np.array(([[0.0, 1.0], [1.0, 0.0]]))
    DELAYS = np.array([[0.0, 10.0], [10.0, 0.0]])

    def test_init(self):
        wc = WilsonCowanNetwork(self.SC, self.DELAYS)
        self.assertTrue(isinstance(wc, WilsonCowanNetwork))
        self.assertEqual(len(wc), self.SC.shape[0])
        self.assertEqual(wc.initial_state.shape[0], wc.num_state_variables)
        self.assertEqual(wc.default_output, f"q_mean_{EXC}")

    @pytest.mark.xfail
    def test_run(self):
        wc = WilsonCowanNetwork(self.SC, self.DELAYS, exc_seed=SEED, inh_seed=SEED)
        all_results = []
        for backend, noise_func in BACKENDS_TO_TEST.items():
            result = wc.run(
                DURATION,
                DT,
                noise_func(ZeroInput(wc.num_noise_variables), DURATION, DT),
                backend=backend,
            )
            self.assertTrue(isinstance(result, xr.Dataset))
            self.assertEqual(len(result), wc.num_state_variables / wc.num_nodes)
            self.assertTrue(all(result[result_].shape == (int(DURATION / DT), wc.num_nodes) for result_ in result))
            all_results.append(result)
        # test results are the same from different backends
        for state_var in all_results[0]:
            all_ts = np.vstack([result[state_var].values.flatten().astype(float) for result in all_results])
            if np.isnan(all_ts).any():
                continue
            corr_mat = np.corrcoef(all_ts)
            print(corr_mat)
            self.assertTrue(np.greater(corr_mat, CORR_THRESHOLD).all())

    @pytest.mark.xfail
    def test_compare_w_neurolib_native_model(self):
        """
        Compare with neurolib's native Wilson-Cowan model.
        """
        wc_multi = WilsonCowanNetwork(self.SC, self.DELAYS)
        multi_result = wc_multi.run(
            DURATION, DT, ZeroInput(wc_multi.num_noise_variables).as_array(DURATION, DT), backend="numba"
        )
        # run neurolib's model
        wc_neurolib = WCModel(Cmat=self.SC, Dmat=self.DELAYS, seed=SEED)
        wc_neurolib.params["duration"] = DURATION
        wc_neurolib.params["dt"] = DT
        # there is no "global coupling" parameter in MultiModel
        wc_neurolib.params["K_gl"] = 1.0
        # delays <-> length matrix
        wc_neurolib.params["signalV"] = 1.0
        wc_neurolib.params["sigma_ou"] = 0.0
        # match initial state
        wc_neurolib.params["exc_init"] = wc_multi.initial_state[::2][:, np.newaxis]
        wc_neurolib.params["inh_init"] = wc_multi.initial_state[1::2][:, np.newaxis]
        wc_neurolib.run()
        for (var_multi, var_neurolib) in NEUROLIB_VARIABLES_TO_TEST:
            for node_idx in range(len(wc_multi)):
                neurolib_ts = wc_neurolib[var_neurolib][node_idx, :]
                multi_ts = multi_result[var_multi].values.T[node_idx, :]
                if np.isnan(neurolib_ts).any() or np.isnan(multi_ts).any():
                    continue
                corr_mat = np.corrcoef(neurolib_ts, multi_ts)
                print(var_multi, node_idx, corr_mat)
                self.assertTrue(np.greater(corr_mat, CORR_THRESHOLD).all())


if __name__ == "__main__":
    unittest.main()
