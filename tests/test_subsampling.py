import copy
import logging
import unittest

import numpy as np
from neurolib.models.aln import ALNModel


class TestSubsampling(unittest.TestCase):
    """
    Tests wheteher model outputs are subsampled as expected.
    """

    def test_subsample_aln(self):
        model = ALNModel()
        model.params["duration"] = 100  # 100 ms
        model.params["dt"] = 0.1  # 0.1 ms
        # default behaviour should be no subsampling
        self.assertIs(model.params.get("sampling_dt", None), None)
        model.run()
        full_output = model.output.copy()

        # subsample the same model
        model.params["sampling_dt"] = 10  # 10 ms
        model.run()
        subsample_output = model.output.copy()

        sample_every = int(model.params["sampling_dt"] / model.params["dt"])
        self.assertEqual(
            full_output[:, ::sample_every].shape[1], subsample_output.shape[1]
        ), "Subsampling returned unexpected output shape"
        self.assertTrue(
            (full_output[:, ::sample_every] == subsample_output).all()
        ), "Subsampling returned unexpected output values"

    def test_subsample_aln_chunkwise(self):
        model = ALNModel(seed=42)
        model.params["sigma_ou"] = 0.0
        model.params["duration"] = 1000  # 1000 ms
        model.params["dt"] = 0.1  # 0.1 ms

        # default behaviour should be no subsampling
        self.assertIs(model.params.get("sampling_dt", None), None)

        model.run(chunksize=3000)
        full_output = model.output.copy()

        # subsample the model (same seed)
        model = ALNModel(seed=42)
        model.params["sigma_ou"] = 0.0
        model.params["duration"] = 1000  # 1000 ms
        model.params["dt"] = 0.1  # 0.1 ms
        model.params["sampling_dt"] = 10.0  # 10 ms
        model.run(chunksize=3000)
        subsample_output = model.output.copy()
        sample_every = int(model.params["sampling_dt"] / model.params["dt"])

        self.assertEqual(
            full_output[:, ::sample_every].shape[1], subsample_output.shape[1]
        ), "Subsampling returned unexpected output shape"
        self.assertTrue(
            (full_output[:, ::sample_every] == subsample_output).all()
        ), "Subsampling returned unexpected output values"

    def test_sampling_dt_smaller_than_dt(self):
        model = ALNModel()
        model.params["dt"] = 10  # 0.1 ms
        model.params["sampling_dt"] = 9
        with self.assertRaises(AssertionError):
            model.run()

    def test_sampling_dt_lower_than_duration(self):
        model = ALNModel()
        model.params["dt"] = 0.1  # 0.1 ms
        model.params["duration"] = 10
        model.params["sampling_dt"] = 30
        with self.assertRaises(AssertionError):
            model.run()

    def test_sampling_dt_invalid(self):
        model = ALNModel()
        model.params["sampling_dt"] = -30
        with self.assertRaises(ValueError):
            model.run()

    def test_sampling_dt_divisible_chunksize(self):
        model = ALNModel()
        model.params["dt"] = 0.1  # 0.1 ms
        model.params["sampling_dt"] = 11.11
        with self.assertRaises(AssertionError):
            model.run(chunksize=3000)

    def test_sampling_dt_divisible_last_chunksize(self):
        model = ALNModel()
        model.params["dt"] = 0.1  # 0.1 ms
        model.params["sampling_dt"] = 0.21
        with self.assertRaises(AssertionError):
            model.run(chunksize=210)

    def test_sampling_dt_greater_than_chunksize(self):
        model = ALNModel()
        model.params["dt"] = 0.1  # 0.1 ms
        model.params["sampling_dt"] = 30
        with self.assertRaises(AssertionError):
            model.run(chunksize=210)


if __name__ == "__main__":
    unittest.main()
