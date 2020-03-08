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

            duration = 200
            model = model_class(Cmat=ds.Cmat, Dmat=ds.Dmat)
            model.params["duration"] = duration

            # run classical integration
            model.run()
            single_integration_output = model.outputs[model.default_output]
            del(model.outputs[model.default_output])			
            # run one-step autochunk integration
            model.run(chunkwise=True, chunksize=1, append_outputs=True)
            chunkwise_output = model.outputs[model.default_output]

            # check for differences
            difference = np.sum(chunkwise_output[:, : single_integration_output.shape[1]] - single_integration_output)
            self.assertEqual(difference, 0.0)
            logging.info(f"Difference of output arrays = {difference}")

    def test_onstep_input_autochunk(self):
        """Tests passing an input array to a model.
        """
        model = ALNModel()
        model.params["duration"] = 1000
        duration_dt = int(model.params["duration"] / model.params["dt"])
        ous = np.zeros((model.params["N"], duration_dt))

        # prepare input
        inp_x = np.zeros((model.params["N"], duration_dt))
        inp_y = np.zeros((model.params["N"], duration_dt))

        for n in range(model.params["N"]):
            fr = 1
            inp_x[n, :] = np.sin(np.linspace(0, fr * 2 * np.pi, duration_dt)) * 0.1

        for i in range(duration_dt):
            inputs = [inp_x[:, i], inp_y[:, i]]
            model.autochunk(inputs=inputs, append_outputs=True)

