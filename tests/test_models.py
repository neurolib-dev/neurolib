import logging
import time
import unittest

import numpy as np
import pytest
import xarray as xr
from chspy import join
from neurolib.models.aln import ALNModel
from neurolib.models.fhn import FHNModel
from neurolib.models.hopf import HopfModel
from neurolib.models.multimodel import MultiModel
from neurolib.models.multimodel.builder.fitzhugh_nagumo import FitzHughNagumoNetwork, FitzHughNagumoNode
from neurolib.models.thalamus import ThalamicMassModel
from neurolib.models.wc import WCModel
from neurolib.models.ww import WWModel
from neurolib.utils.collections import star_dotdict
from neurolib.utils.loadData import Dataset
from neurolib.utils.stimulus import ZeroInput
from neurolib.models.kuramoto import KuramotoModel

class TestAln(unittest.TestCase):
    """
    Basic test for ALN model.
    """

    def test_single_node(self):

        logging.info("\t > ALN: Testing single node ...")
        start = time.time()

        aln = ALNModel()
        aln.params["duration"] = 10.0 * 1000
        aln.params["sigma_ou"] = 0.1  # add some noise
        # load new initial parameters
        aln.run(bold=True)
        # access outputs
        aln.xr()
        aln.xr("BOLD")

        end = time.time()
        logging.info("\t > Done in {:.2f} s".format(end - start))

    def test_network(self):
        logging.info("\t > ALN: Testing brain network (chunkwise integration and BOLD" " simulation) ...")
        start = time.time()

        ds = Dataset("gw")

        aln = ALNModel(Cmat=ds.Cmat, Dmat=ds.Dmat)

        aln.params["duration"] = 10 * 1000

        aln.run(chunkwise=True, bold=True, append_outputs=True)

        # access outputs
        aln.xr()
        aln.xr("BOLD")

        end = time.time()
        logging.info("\t > Done in {:.2f} s".format(end - start))


class TestHopf(unittest.TestCase):
    """
    Basic test for Hopf model.
    """

    def test_single_node(self):
        logging.info("\t > Hopf: Testing single node ...")
        start = time.time()
        hopf = HopfModel()
        hopf.params["duration"] = 2.0 * 1000
        hopf.params["sigma_ou"] = 0.03

        hopf.run()

        end = time.time()
        logging.info("\t > Done in {:.2f} s".format(end - start))

    def test_network(self):
        logging.info("\t > Hopf: Testing brain network (chunkwise integration and BOLD" " simulation) ...")
        start = time.time()
        ds = Dataset("gw")
        hopf = HopfModel(Cmat=ds.Cmat, Dmat=ds.Dmat)
        hopf.params["w"] = 1.0
        hopf.params["signalV"] = 0
        hopf.params["duration"] = 10 * 1000
        hopf.params["sigma_ou"] = 0.14
        hopf.params["K_gl"] = 0.6

        hopf.run(chunkwise=True, bold=True, append_outputs=True)

        end = time.time()
        logging.info("\t > Done in {:.2f} s".format(end - start))


class TestFHN(unittest.TestCase):
    """
    Basic test for FHN model.
    """

    def test_single_node(self):
        logging.info("\t > FHN: Testing single node ...")
        start = time.time()
        fhn = FHNModel()
        fhn.params["duration"] = 2.0 * 1000
        fhn.params["sigma_ou"] = 0.03

        fhn.run()

        end = time.time()
        logging.info("\t > Done in {:.2f} s".format(end - start))

    def test_network(self):
        logging.info("\t > FHN: Testing brain network (chunkwise integration and BOLD simulation) ...")
        start = time.time()
        ds = Dataset("gw")
        fhn = FHNModel(Cmat=ds.Cmat, Dmat=ds.Dmat)
        fhn.params["signalV"] = 4.0
        fhn.params["duration"] = 10 * 1000
        fhn.params["sigma_ou"] = 0.1
        fhn.params["K_gl"] = 0.6
        fhn.params["x_ext"] = 0.72

        fhn.run(chunkwise=True, bold=True, append_outputs=True)
        end = time.time()
        logging.info("\t > Done in {:.2f} s".format(end - start))


class TestWC(unittest.TestCase):
    """
    Basic test for WC model.
    """

    def test_single_node(self):
        logging.info("\t > WC: Testing single node ...")
        start = time.time()
        model = WCModel()
        model.params["duration"] = 2.0 * 1000
        model.params["sigma_ou"] = 0.03

        model.run()

        end = time.time()
        logging.info("\t > Done in {:.2f} s".format(end - start))

    def test_network(self):
        logging.info("\t > WC: Testing brain network (chunkwise integration and BOLD simulation) ...")
        start = time.time()
        ds = Dataset("gw")
        model = WCModel(Cmat=ds.Cmat, Dmat=ds.Dmat)
        model.params["signalV"] = 4.0
        model.params["duration"] = 10 * 1000
        model.params["sigma_ou"] = 0.1
        model.params["K_gl"] = 0.6

        # local node input parameter
        model.params["exc_ext"] = 0.72

        model.run(chunkwise=True, bold=True, append_outputs=True)
        end = time.time()
        logging.info("\t > Done in {:.2f} s".format(end - start))


class TestWW(unittest.TestCase):
    """
    Basic test for WW model.
    """

    def test_single_node(self):
        logging.info("\t > WW: Testing single node ...")
        start = time.time()
        model = WWModel()
        model.params["duration"] = 2.0 * 1000
        model.params["sigma_ou"] = 0.03

        model.run()

        end = time.time()
        logging.info("\t > Done in {:.2f} s".format(end - start))

    def test_network(self):
        logging.info("\t > WW: Testing brain network (chunkwise integration and BOLD simulation) ...")
        start = time.time()
        ds = Dataset("gw")
        model = WWModel(Cmat=ds.Cmat, Dmat=ds.Dmat)
        model.params["signalV"] = 4.0
        model.params["duration"] = 10 * 1000
        model.params["sigma_ou"] = 0.1
        model.params["K_gl"] = 0.6

        model.run(chunkwise=True, bold=True, append_outputs=True)
        end = time.time()
        logging.info("\t > Done in {:.2f} s".format(end - start))


class TestKuramoto(unittest.TestCase):
    """
    Basic test for Kuramoto model.
    """

    def test_single_node(self):
        logging.info("\t > Kuramoto: Testing single node ...")
        start = time.time()
        model = KuramotoModel()
        model.params["duration"] = 2.0 * 1000
        model.params["sigma_ou"] = 0.03

        model.run()

        end = time.time()
        logging.info("\t > Done in {:.2f} s".format(end - start))

    def test_network(self):
        logging.info("\t > Kuramoto: Testing brain network (chunkwise integration and BOLD simulation) ...")
        start = time.time()
        ds = Dataset("gw")
        model = KuramotoModel(Cmat=ds.Cmat, Dmat=ds.Dmat)
        model.params["signalV"] = 4.0
        model.params["duration"] = 10 * 1000
        model.params["sigma_ou"] = 0.1
        model.params["k"] = 0.6

        
        # local node input parameter 
        model.params["theta_ext"] = 0.72

        model.run(chunkwise=True, append_outputs=True)
        end = time.time()
        logging.info("\t > Done in {:.2f} s".format(end - start))


class TestThalamus(unittest.TestCase):
    """
    Basic test for thalamic mass model.
    """

    def test_single_node(self):
        logging.info("\t > Thalamus: Testing single node ...")
        start = time.time()
        thalamus = ThalamicMassModel()
        thalamus.params["duration"] = 2.0 * 1000

        thalamus.run()

        end = time.time()
        logging.info("\t > Done in {:.2f} s".format(end - start))


class TestMultiModel(unittest.TestCase):
    """
    Basic test for MultiModel. Test with FitzHugh-Nagumo model.
    """

    def test_init(self):
        DELAY = 13.0
        fhn_net = FitzHughNagumoNetwork(np.random.rand(2, 2), np.array([[0.0, DELAY], [DELAY, 0.0]]))
        model = MultiModel(fhn_net)
        self.assertEqual(model.model_instance, fhn_net)
        self.assertTrue(isinstance(model.params, star_dotdict))
        self.assertTrue(model.integration is None)
        self.assertEqual(model.start_t, 0.0)
        self.assertEqual(model.num_noise_variables, 4)
        self.assertEqual(model.num_state_variables, 4)
        max_delay = int(DELAY / model.params["dt"])
        self.assertEqual(model.getMaxDelay(), max_delay)

    def test_noise_input(self):
        DELAY = 13.0
        fhn_net = FitzHughNagumoNetwork(np.random.rand(2, 2), np.array([[0.0, DELAY], [DELAY, 0.0]]))
        model = MultiModel(fhn_net)
        self.assertListEqual(model.noise_input, model.model_instance.noise_input)
        model.noise_input = [ZeroInput()] * model.model_instance.num_noise_variables
        self.assertListEqual(model.noise_input, model.model_instance.noise_input)

    def test_run_numba_w_bold(self):
        DELAY = 13.0
        fhn_net = FitzHughNagumoNetwork(np.random.rand(2, 2), np.array([[0.0, DELAY], [DELAY, 0.0]]))
        model = MultiModel(fhn_net)
        model.params["backend"] = "numba"
        model.params["duration"] = 10000
        model.params["dt"] = 0.1
        model.run(bold=True)
        # access outputs
        self.assertTrue(isinstance(model.xr(), xr.DataArray))
        self.assertTrue(isinstance(model.xr("BOLD"), xr.DataArray))

    def test_run_chunkwise(self):
        DELAY = 13.0
        fhn_net = FitzHughNagumoNetwork(np.random.rand(2, 2), np.array([[0.0, DELAY], [DELAY, 0.0]]))
        model = MultiModel(fhn_net)
        # not implemented for now
        with pytest.raises((NotImplementedError, AssertionError)):
            model.run(chunkwise=True)

    def test_run_network(self):
        DELAY = 13.0
        fhn_net = FitzHughNagumoNetwork(np.random.rand(2, 2), np.array([[0.0, DELAY], [DELAY, 0.0]]))
        model = MultiModel(fhn_net)
        model.params["sampling_dt"] = 10.0
        # run MultiModel
        model.run()
        # run model instance
        inst_res = fhn_net.run(
            duration=model.params["duration"],
            dt=model.params["sampling_dt"],
            noise_input=join(
                *[
                    noise.as_cubic_splines(model.params["duration"], model.params["sampling_dt"])
                    for noise in fhn_net.noise_input
                ]
            ),
            backend=model.params["backend"],
        )
        for out_var in model.output_vars:
            np.testing.assert_equal(model[out_var], inst_res[out_var].values.T)

    def test_run_node(self):
        fhn_node = FitzHughNagumoNode()
        model = MultiModel.init_node(fhn_node)
        model.params["sampling_dt"] = 10.0
        # run MultiModel
        model.run()
        # run model instance
        inst_res = fhn_node.run(
            duration=model.params["duration"],
            dt=model.params["sampling_dt"],
            noise_input=join(
                *[
                    noise.as_cubic_splines(model.params["duration"], model.params["sampling_dt"])
                    for noise in fhn_node.noise_input
                ]
            ),
            backend=model.params["backend"],
        )
        for out_var in model.output_vars:
            np.testing.assert_equal(model[out_var], inst_res[out_var].values.T)

    def test_continue_run_node(self):
        fhn_node = FitzHughNagumoNode()
        model = MultiModel.init_node(fhn_node)
        model.params["sampling_dt"] = 10.0
        model.params["backend"] = "numba"
        # run MultiModel with continuation
        model.run(continue_run=True)
        last_t = model.t[-1]
        last_x = model.state["x"][:, -model.maxDelay - 1 :]
        last_y = model.state["y"][:, -model.maxDelay - 1 :]
        # assert last state is initial state now
        self.assertEqual(model.start_t, last_t)
        np.testing.assert_equal(last_x.squeeze(), model.model_instance.initial_state[0, :])
        np.testing.assert_equal(last_y.squeeze(), model.model_instance.initial_state[1, :])
        # change noise - just to make things more interesting
        model.noise_input = [ZeroInput()] * model.model_instance.num_noise_variables
        model.run(continue_run=True)
        # assert continuous time
        self.assertAlmostEqual(model.t[0] - last_t, model.params["dt"] / 1000.0)
        # assert start_t is reset to 0, when continue_run=False
        model.run()
        self.assertEqual(model.start_t, 0.0)

    def test_continue_run_network(self):
        DELAY = 13.0
        fhn_net = FitzHughNagumoNetwork(np.random.rand(2, 2), np.array([[0.0, DELAY], [DELAY, 0.0]]))
        model = MultiModel(fhn_net)
        model.params["sampling_dt"] = 10.0
        model.params["backend"] = "numba"
        # run MultiModel with continuation
        model.run(continue_run=True)
        last_t = model.t[-1]
        last_x = model.state["x"][:, -model.maxDelay - 1 :]
        last_y = model.state["y"][:, -model.maxDelay - 1 :]
        # assert last state is initial state now
        self.assertEqual(model.start_t, last_t)
        np.testing.assert_equal(last_x, model.model_instance.initial_state[[0, 2], :])
        np.testing.assert_equal(last_y, model.model_instance.initial_state[[1, 3], :])
        # change noise - just to make things more interesting
        model.noise_input = [ZeroInput()] * model.model_instance.num_noise_variables
        model.run(continue_run=True)
        # assert continuous time
        self.assertAlmostEqual(model.t[0] - last_t, model.params["dt"] / 1000.0)
        # assert start_t is reset to 0, when continue_run=False
        model.run()
        self.assertEqual(model.start_t, 0.0)


if __name__ == "__main__":
    unittest.main()
