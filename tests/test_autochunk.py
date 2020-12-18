import copy
import unittest

import numpy as np
from neurolib.models.aln import ALNModel
from neurolib.models.fhn import FHNModel
from neurolib.models.hopf import HopfModel
from neurolib.models.thalamus import ThalamicMassModel
from neurolib.models.wc import WCModel
from neurolib.utils.loadData import Dataset


class AutochunkTests(unittest.TestCase):
    """
    Base class for autochunk tests.
    """

    durations = [0.1, 0.5, 10.5, 22.3]
    chunksizes = [1, 5, 7, 33, 55, 123]
    signalVs = [0, 1, 10, 10000]

    def single_node_test(self, model):
        for duration in self.durations:
            for chunksize in self.chunksizes:
                for signalV in self.signalVs:
                    # full run
                    m1 = model()
                    m1.params.signalV = signalV
                    m1.params["duration"] = duration
                    pars_bak = copy.deepcopy(m1.params)
                    m1.run()
                    # chunkwise run
                    m2 = model()
                    m2.params = pars_bak.copy()
                    m2.run(chunkwise=True, chunksize=chunksize, append=True)
                    # check
                    self.assertTupleEqual(m1.output.shape, m2.output.shape)
                    difference = np.sum(np.abs(m1.output - m2.output))
                    self.assertAlmostEqual(difference, 0.0)

    def network_test(self, model):
        ds = Dataset("hcp")
        for duration in self.durations:
            for chunksize in self.chunksizes:
                for signalV in self.signalVs:
                    # full run
                    m1 = model(Cmat=ds.Cmat, Dmat=ds.Dmat)
                    m1.params.signalV = signalV
                    m1.params["duration"] = duration
                    pars_bak = copy.deepcopy(m1.params)
                    m1.run()
                    # chunkwise run
                    m2 = model(Cmat=ds.Cmat, Dmat=ds.Dmat)
                    m2.params = pars_bak.copy()
                    m2.run(chunkwise=True, chunksize=chunksize, append=True)
                    # check
                    self.assertTupleEqual(m1.output.shape, m2.output.shape)
                    difference = np.sum(np.abs(m1.output - m2.output))
                    self.assertAlmostEqual(difference, 0.0)


class TestALNAutochunk(AutochunkTests):
    def test_single(self):
        self.single_node_test(ALNModel)

    def test_network(self):
        self.network_test(ALNModel)

    def test_onstep_input_autochunk(self):
        """Tests passing an input array to a model."""
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


class TestFHNAutochunk(AutochunkTests):
    def test_single(self):
        self.single_node_test(FHNModel)

    def test_network(self):
        self.network_test(FHNModel)


class TestHopfAutochunk(AutochunkTests):
    def test_single(self):
        self.single_node_test(HopfModel)

    def test_network(self):
        self.network_test(HopfModel)


class TestWCAutochunk(AutochunkTests):
    def test_single(self):
        self.single_node_test(WCModel)

    def test_network(self):
        self.network_test(WCModel)


class TestThalamusAutochunk(AutochunkTests):
    def test_single(self):
        self.single_node_test(ThalamicMassModel)


if __name__ == "__main__":
    unittest.main()
