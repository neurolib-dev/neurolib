import copy
import unittest
import logging

import numpy as np
import pytest
from neurolib.models.aln import ALNModel
from neurolib.models.fhn import FHNModel
from neurolib.models.hopf import HopfModel
from neurolib.models.thalamus import ThalamicMassModel
from neurolib.models.wc import WCModel
from neurolib.models.ww import WWModel
from neurolib.utils.loadData import Dataset


class AutochunkTests(unittest.TestCase):
    """
    Base class for autochunk tests.

    Cycles through all models and runs autochunk on them and compares the
    results to a normal run. Expect the outputs to be equal.

    All individual model runs are blow this base class.
    """

    durations = [0.1, 0.5, 10.5, 22.3]
    chunksizes = [1, 5, 7, 33, 55, 123]
    signalVs = [0, 1, 10, 10000]

    def single_node_test(self, model):
        for duration in self.durations:
            for chunksize in self.chunksizes:
                # full run
                m1 = model()
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
                print(
                    f"single node model: {model.name}, duration: {duration}, chunksize: {chunksize}, difference: {difference}"
                )
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
                    print(
                        f"network model: {model.name}, duration: {duration}, chunksize: {chunksize}, difference: {difference}"
                    )
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
        inp_exc_current = np.zeros((model.params["N"], duration_dt))
        inp_inh_current = np.zeros((model.params["N"], duration_dt))
        inp_exc_rate = np.zeros((model.params["N"], duration_dt))
        inp_inh_rate = np.zeros((model.params["N"], duration_dt))

        for n in range(model.params["N"]):
            fr = 1
            inp_exc_current[n, :] = np.sin(np.linspace(0, fr * 2 * np.pi, duration_dt)) * 0.1

        for i in range(duration_dt):
            inputs = [inp_exc_current[:, i], inp_inh_current[:, i], inp_exc_rate[:, i], inp_inh_rate[:, i]]
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


class TestWWAutochunk(AutochunkTests):
    def test_single(self):
        self.single_node_test(WWModel)

    def test_network(self):
        self.network_test(WWModel)


class TestThalamusAutochunk(AutochunkTests):
    @pytest.mark.xfail
    def test_single(self):
        self.single_node_test(ThalamicMassModel)


class ChunkziseImpliesChunksize(unittest.TestCase):
    """
    Simply test whether the model runs as expected when
    model.run(chunksize=1000) is used without chunkwise=True
    """

    def test_chunksize(self):
        model = HopfModel()
        chunksize = 100
        model.run(chunksize=chunksize)
        self.assertEqual(model.output.shape[1], chunksize)


if __name__ == "__main__":
    unittest.main()
