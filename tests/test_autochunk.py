import logging
import time
import unittest

import numpy as np

from neurolib.models.aln import ALNModel
from neurolib.models.hopf import HopfModel
from neurolib.models.fhn import FHNModel

from neurolib.utils.loadData import Dataset


class TestAutochunk(unittest.TestCase):
    """
    Tests the one-dt (one-step) integration and compares it to normal integration.
    """

    def test_onstep_chunkwise(self):
        """Tests the one-dt (one-step) integration and compares it to normal integration.
        """
        ds = Dataset("hcp")

        for model_class in [ALNModel, HopfModel, FHNModel]:
            logging.info(f"Testing {model_class}...")
            ModelClass = model_class
            duration = 200
            model = ModelClass(Cmat=ds.Cmat, Dmat=ds.Dmat)
            model.params["duration"] = duration

            # run classical integration
            model.run()
            single_integration_output = model.outputs[model.default_output]

            # run one-step autochunk integration
            model.run(chunkwise=True, chunksize=1, append_outputs=True)
            chunkwise_output = model.outputs[model.default_output]

            # check for differences
            difference = np.sum(chunkwise_output[:, : single_integration_output.shape[1]] - single_integration_output)
            self.assertEqual(difference, 0.0)
            logging.info(f"Difference of output arrays = {difference}")

