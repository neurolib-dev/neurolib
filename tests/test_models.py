import logging
import time
import unittest

from neurolib.models import aln, hopf
from neurolib.utils.loadData import Dataset


class TestALNModel(unittest.TestCase):
    """
    Basic test for ALN model.
    """

    def test_single_node(self):
        logging.info("\t > ALN: Testing single node ...")
        start = time.time()

        alnModel = aln.ALNModel()
        alnModel.params["duration"] = 2.0 * 1000
        alnModel.params["sigma_ou"] = 0.1  # add some noise

        alnModel.run()
        end = time.time()
        logging.info("\t > Done in {:.2f} s".format(end - start))

    def test_network(self):
        logging.info(
            "\t > ALN: Testing brain network (chunkwise integration and BOLD"
            " simulation) ..."
        )
        start = time.time()

        ds = Dataset("gw")

        alnModel = aln.ALNModel(Cmat=ds.Cmat, Dmat=ds.Dmat, simulateBOLD=True)
        # in ms, simulates for 5 minutes
        alnModel.params["duration"] = 10 * 1000

        alnModel.run()
        end = time.time()
        logging.info("\t > Done in {:.2f} s".format(end - start))


class TestHopfModel(unittest.TestCase):
    """
    Basic test for Hopf model.
    """

    def test_single_node(self):
        logging.info("\t > Hopf: Testing single node ...")
        start = time.time()
        hopfModel = hopf.HopfModel()
        hopfModel.params["duration"] = 2.0 * 1000
        hopfModel.params["sigma_ou"] = 0.03

        hopfModel.run()
        end = time.time()
        logging.info("\t > Done in {:.2f} s".format(end - start))

    def test_network(self):
        logging.info(
            "\t > Hopf: Testing brain network (chunkwise integration and BOLD"
            " simulation) ..."
        )
        start = time.time()
        ds = Dataset("gw")
        hopfModel = hopf.HopfModel(
            Cmat=ds.Cmat, Dmat=ds.Dmat, simulateBOLD=True
        )
        hopfModel.params["w"] = 1.0
        hopfModel.params["signalV"] = 0
        hopfModel.params["duration"] = 10 * 1000
        hopfModel.params["sigma_ou"] = 0.14
        hopfModel.params["K_gl"] = 0.6

        hopfModel.run()
        end = time.time()
        logging.info("\t > Done in {:.2f} s".format(end - start))


if __name__ == "__main__":
    unittest.main()
