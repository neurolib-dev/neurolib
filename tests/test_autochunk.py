import logging
import time
import unittest

import numpy as np
import copy

from neurolib.models.aln import ALNModel
from neurolib.models.hopf import HopfModel
from neurolib.models.fhn import FHNModel
from neurolib.models.wc import WCModel

from neurolib.utils.loadData import Dataset


class TestAutochunk(unittest.TestCase):
    """
    Tests the one-dt (one-step) integration and compares it to normal integration.
    """

    def test_check_chunkwise(self):
        """Full test of chunkwise integration over all models
        """
        ds = Dataset("hcp")
        Models = [ALNModel, FHNModel, HopfModel, WCModel]
        durations = [0.1, 0.5, 10.5, 22.3]
        chunksizes = [1, 5, 7, 33, 55, 123]
        modes = ["single", "network"]
        signalVs = [0, 1, 10, 10000]

        plot = False
        for mode in modes:
            for Model in Models:
                for duration in durations:
                    for chunksize in chunksizes:
                        for signalV in signalVs:
                            if mode == "network":
                                m1 = Model(Cmat=ds.Cmat, Dmat=ds.Dmat)
                            else:
                                m1 = Model()
                            m1.params.signalV = signalV
                            m1.params["duration"] = duration
                            pars_bak = copy.deepcopy(m1.params)

                            m1.run()

                            if mode == "network":
                                m2 = Model(Cmat=ds.Cmat, Dmat=ds.Dmat)
                            else:
                                m2 = Model()
                            m2.params = pars_bak.copy()
                            m2.run(chunkwise=True, chunksize=chunksize, append=True)

                            assert (
                                m1.output.shape == m2.output.shape
                            ), "Shape of chunkwise output does not match normal output!"
                            difference = np.sum(abs(m1.output - m2.output))
                            assert (
                                difference == 0
                            ), f"difference: {difference} > Model: {Model.name}, Mode: {mode}, signalV: {signalV}, Chunksize: {chunksize}, Duration: {duration}"
                            if difference > 0:
                                logging.info(
                                    f"difference: {difference} > Model: {Model.name}, Mode: {mode}, signalV: {signalV}, Chunksize: {chunksize}, Duration: {duration}"
                                )

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
