import logging
import unittest

import os
import glob

from neurolib.utils.loadData import Dataset
import neurolib.utils.functions as func


class TestDatasets(unittest.TestCase):
    def test_dataset_gw(self):
        ds = Dataset("gw", fcd=True)
        self.assertTupleEqual(ds.Cmat.shape, (80, 80))
        self.assertTupleEqual(ds.Dmat.shape, (80, 80))
        self.assertTupleEqual(ds.FCs[0].shape, (80, 80))
        self.assertTrue(len(ds.FCs) == len(ds.BOLDs))

    def test_dataset_hcp(self):
        ds = Dataset("hcp")
        ds = Dataset("hcp", normalizeCmats="waytotal")
        ds = Dataset("hcp", normalizeCmats="nvoxel")

        self.assertTupleEqual(ds.Cmat.shape, (80, 80))
        self.assertTupleEqual(ds.Dmat.shape, (80, 80))
        self.assertTupleEqual(ds.FCs[0].shape, (80, 80))
        self.assertTrue(len(ds.FCs) == len(ds.BOLDs))
