import logging
import time
import unittest

from neurolib.models.aln import ALNModel
from neurolib.models.fhn import FHNModel
from neurolib.models.hopf import HopfModel
from neurolib.models.thalamus import ThalamicMassModel
from neurolib.models.wc import WCModel
from neurolib.utils.loadData import Dataset


class TestAln(unittest.TestCase):
    """
    Basic test for ALN model.
    """

    def test_single_node(self):
        import neurolib.models.aln.loadDefaultParams as dp

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
        wc = WCModel()
        wc.params["duration"] = 2.0 * 1000
        wc.params["sigma_ou"] = 0.03

        wc.run()

        end = time.time()
        logging.info("\t > Done in {:.2f} s".format(end - start))

    def test_network(self):
        logging.info("\t > WC: Testing brain network (chunkwise integration and BOLD simulation) ...")
        start = time.time()
        ds = Dataset("gw")
        wc = WCModel(Cmat=ds.Cmat, Dmat=ds.Dmat)
        wc.params["signalV"] = 4.0
        wc.params["duration"] = 10 * 1000
        wc.params["sigma_ou"] = 0.1
        wc.params["K_gl"] = 0.6
        wc.params["x_ext_mean"] = 0.72

        wc.run(chunkwise=True, bold=True, append_outputs=True)
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


if __name__ == "__main__":
    unittest.main()
