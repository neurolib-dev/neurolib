import logging
import unittest

import os
import glob

from neurolib.utils.loadData import Dataset
import neurolib.utils.functions as func


class TestDatasets(unittest.TestCase):
    def test_dataset_gw(self):
        ds = Dataset("gw")
        assert ds.Cmat.shape == (80, 80)
        assert ds.Dmat.shape == (80, 80)
        assert ds.FCs[0].shape == (80, 80)
        assert len(ds.FCs) == len(ds.BOLDs)

    def test_dataset_hcp(self):
        ds = Dataset("hcp")
        assert ds.Cmat.shape == (80, 80)
        assert ds.Dmat.shape == (80, 80)
        assert ds.FCs[0].shape == (80, 80)
        assert len(ds.FCs) == len(ds.BOLDs)

    def test_manual_loading(self):
        dsBaseDirectory = os.path.join("neurolib", "data", "datasets", "gw")
        BOLDFilenames = glob.glob(os.path.join(dsBaseDirectory, "BOLD/", "*_tc.mat"))  # BOLD timeseries
        ds = Dataset("gw")
        ds.loadData(BOLDFilenames, key="tc", filter_subcortical=False, average=True)
        ds.loadData(BOLDFilenames, key="tc", filter_subcortical=True, apply_function=func.fc)
