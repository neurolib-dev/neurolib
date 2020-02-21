import logging
import time
import unittest

import neurolib.utils.functions as func
from neurolib.models.aln import ALNModel
from neurolib.utils.loadData import Dataset


class TestFunctions(unittest.TestCase):
    """
    Test functions in neurolib/utils/functions.py
    """

    @classmethod
    def setUpClass(cls):
        ds = Dataset("gw")
        aln = ALNModel(Cmat=ds.Cmat, Dmat=ds.Dmat, bold=True)

        # Resting state fits
        aln.params["mue_ext_mean"] = 1.57
        aln.params["mui_ext_mean"] = 1.6
        aln.params["sigma_ou"] = 0.09
        aln.params["b"] = 5.0
        aln.params["duration"] = 0.2 * 60 * 1000
        aln.run(simulate_bold=True, chunkwise=True)

        cls.model = aln
        cls.ds = Dataset("gw")

    def test_kuramoto(self):
        kuramoto = func.kuramoto(self.model.rates_exc[:, ::10], dt=self.model.params["dt"], smoothing=1.0)

    def test_fc(self):
        FC = func.fc(self.model.BOLD.BOLD)

    def test_fcd(self):
        rFCD = func.fcd(self.model.rates_exc, stepsize=100)

    def test_matrix_correlation(self):
        FC = func.fc(self.model.BOLD.BOLD)
        cc = func.matrix_correlation(FC, self.ds.FCs[0])

    def test_kolmogorov(self):
        func.kolmogorov(self.model.rates_exc[::20, :], self.model.rates_exc, stepsize=250, windowsize=30)

    def test_print_params(self):
        func.print_params(self.model.params)

    def test_getPowerSpectrum(self):
        fr, pw = func.getPowerSpectrum(self.model.rates_exc[0, :], dt=self.model.params["dt"])

    def test_getMeanPowerSpectrum(self):
        fr, pw = func.getMeanPowerSpectrum(self.model.rates_exc, dt=self.model.params["dt"])

