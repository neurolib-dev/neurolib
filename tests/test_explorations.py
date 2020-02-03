import logging
import time
import unittest

import numpy as np

from neurolib.models.aln import ALNModel
from neurolib.optimize.exploration import BoxSearch


class TestALNExploration(unittest.TestCase):
    """
    ALN model parameter exploration with pypet.
    """

    def test_single_node(self):
        logging.info("\t > BoxSearch: Testing ALN single node ...")
        start = time.time()

        aln = ALNModel()
        parameters = {"mue_ext_mean": np.linspace(0, 3, 2).tolist(), "mui_ext_mean": np.linspace(0, 3, 2).tolist()}
        search = BoxSearch(aln, parameters)
        search.initializeExploration()
        search.run()
        search.loadResults()

        for i in search.dfResults.index:
            search.dfResults.loc[i, "max_r"] = np.max(search.results[i]["rates_exc"][:, -int(1000 / aln.params["dt"]) :])

        end = time.time()
        logging.info("\t > Done in {:.2f} s".format(end - start))


if __name__ == "__main__":
    unittest.main()
