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
        fhn.params["x_ext_mean"] = 0.72

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
        max_delay = int(DELAY / model.params["dt"])
        self.assertEqual(model.getMaxDelay(), max_delay)

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


if __name__ == "__main__":
    unittest.main()
