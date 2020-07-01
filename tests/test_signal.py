"""
Set of tests for `Signal` class.
"""

import logging
import os
import unittest
from copy import deepcopy
from shutil import rmtree

import numpy as np
import pytest
import xarray as xr
from neurolib.models.aln import ALNModel
from neurolib.utils.loadData import Dataset
from neurolib.utils.signal import RatesSignal, scipy_iir_filter_data


class TestSignal(unittest.TestCase):
    """
    Tests of `Signal` class.
    """

    TEST_FOLDER = os.path.join(os.path.dirname(os.path.realpath(__file__)), "temp_folder")

    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        aln = ALNModel()
        # simulate 10 seconds
        aln.params["duration"] = 10.0 * 1000
        aln.params["sigma_ou"] = 0.1  # add some noise
        aln.run()
        # init RatesSignal
        cls.signal = RatesSignal.from_model_output(aln)
        # os.makedirs(cls.TEST_FOLDER)

    # @classmethod
    # def tearDownClass(cls):
    #     rmtree(cls.TEST_FOLDER)

    def test_load_save(self):
        # create temp folder
        os.makedirs(self.TEST_FOLDER)
        # save
        filename = os.path.join(self.TEST_FOLDER, "temp")
        # do operation so we also test saving and loading of preprocessing steps
        self.signal.normalize(std=True)
        # test basic properties
        repr(self.signal)
        str(self.signal)
        self.signal.start_time
        self.signal.end_time
        self.signal.preprocessing_steps
        # now save
        self.signal.save(filename)
        # load
        loaded = RatesSignal.from_file(filename)
        # remove folder
        rmtree(self.TEST_FOLDER)
        # compare they are equal
        self.assertEqual(self.signal, loaded)

    def test_iterate(self):
        for name, it in self.signal.iterate(return_as="signal"):
            self.assertTrue(isinstance(it, RatesSignal))
            # test it is one-dim with only time axis
            print(name, it.shape)
            self.assertTupleEqual(it.shape, (self.signal.shape[-1], 1))

        for name, it in self.signal.iterate(return_as="xr"):
            self.assertTrue(isinstance(it, xr.DataArray))
            # test it is one-dim with only time axis
            self.assertTupleEqual(it.shape, (self.signal.shape[-1], 1))

        with pytest.raises(ValueError):
            for name, it in self.signal.iterate(return_as="abcde"):
                pass

    def test_sel(self):
        selected = self.signal.sel([5.43, 7.987], inplace=False)
        # test correct indices
        self.assertEqual(selected.data.time.values[0], 5.43)
        self.assertEqual(selected.data.time.values[-1], 7.987)
        # test inplace
        sig = deepcopy(self.signal)
        sig.sel([5.43, 7.987], inplace=True)
        self.assertEqual(sig, selected)

    def test_isel(self):
        selected = self.signal.isel([12143, 16424], inplace=False)
        # test correct indices
        self.assertEqual(selected.data.time.values[0], (12143 + 1) / self.signal.sampling_frequency)
        self.assertEqual(selected.data.time.values[-1], (16424) / self.signal.sampling_frequency)
        # test inplace
        sig = deepcopy(self.signal)
        sig.isel([12143, 16424], inplace=True)
        self.assertEqual(sig, selected)

    def test_sliding_window(self):
        # 2seconds window with step 0.5 second
        window_length = 2.0  # seconds
        window_step = 0.5  # seconds
        correct_shape = tuple(list(self.signal.shape[:-1]) + [int(window_length / self.signal.dt)])
        start_time = self.signal.data.time.values[0]
        end_time = self.signal.data.time.values[correct_shape[-1] - 1]
        for win in self.signal.sliding_window(
            length=window_length, step=window_step, window_function="boxcar", lengths_in_seconds=True,
        ):
            # check correct shape and indices bounds for each window
            self.assertTrue(isinstance(win, RatesSignal))
            self.assertEqual(win.shape, correct_shape)
            np.testing.assert_almost_equal(win.data.time.values[0], start_time)
            np.testing.assert_almost_equal(win.data.time.values[-1], end_time)
            # advance testing times by window step
            start_time += window_step
            end_time += window_step

    def test_pad(self):
        # pad both sides with 2.3 constant value for 2 seconds
        pad_seconds = 2.0
        const_value = 2.3

        for side in ["before", "after", "both"]:
            padded = self.signal.pad(
                pad_seconds,
                in_seconds=True,
                padding_type="constant",
                side=side,
                constant_values=const_value,
                inplace=False,
            )
            padded_multiplier = 2 if side == "both" else 1
            correct_shape = tuple(
                list(self.signal.shape[:-1])
                + [self.signal.shape[-1] + padded_multiplier * pad_seconds * self.signal.sampling_frequency]
            )
            # assert correct shape
            self.assertEqual(padded.shape, correct_shape)
            # assert correct indices - they should increase monotically without
            # sudden jump, i.e. all timestep differences should be equal to dt
            np.testing.assert_allclose(np.diff(padded.data.time.values), padded.dt)
            # assert we actually did input `const_value`
            if side in ["before", "both"]:
                np.testing.assert_allclose(
                    padded.isel([0, int(2 * padded.sampling_frequency)], inplace=False).data, const_value,
                )
            if side in ["after", "both"]:
                np.testing.assert_allclose(
                    padded.isel([-int(2 * padded.sampling_frequency), None], inplace=False).data, const_value,
                )
            # test inplace
            sig = deepcopy(self.signal)
            sig.pad(
                pad_seconds,
                in_seconds=True,
                padding_type="constant",
                side=side,
                constant_values=const_value,
                inplace=True,
            )
            self.assertEqual(sig, padded)

        with pytest.raises(ValueError):
            padded = self.signal.pad(
                pad_seconds,
                in_seconds=True,
                padding_type="constant",
                side="abcde",
                constant_values=const_value,
                inplace=False,
            )

    def test_normalize(self):
        # demean
        demeaned = self.signal.normalize(std=False, inplace=False)
        # test mean is 0
        np.testing.assert_almost_equal(demeaned.data.mean(dim="time").values, 0.0)
        # normalise
        normalized = self.signal.normalize(std=True, inplace=False)
        # test mean is 0 and std is 1
        np.testing.assert_almost_equal(normalized.data.mean(dim="time").values, 0.0)
        np.testing.assert_almost_equal(normalized.data.std(dim="time").values, 1.0)

    def test_downsample(self):
        # downsample to 200Hz
        resample_to = 200.0
        correct_shape = tuple(
            list(self.signal.shape[:-1]) + [self.signal.shape[-1] * (resample_to / self.signal.sampling_frequency)]
        )
        resampled = self.signal.resample(to_frequency=resample_to, inplace=False)
        self.assertTupleEqual(resampled.shape, correct_shape)
        self.assertEqual(resampled.sampling_frequency, resample_to)
        self.assertEqual(resampled.dt, 1.0 / resample_to)
        # test inplace
        sig = deepcopy(self.signal)
        sig.resample(to_frequency=resample_to, inplace=True)
        self.assertEqual(sig, resampled)

    def test_upsample(self):
        # upsample to 20000Hz
        resample_to = 20000.0
        correct_shape = tuple(
            list(self.signal.shape[:-1]) + [self.signal.shape[-1] * (resample_to / self.signal.sampling_frequency)]
        )
        resampled = self.signal.resample(to_frequency=resample_to, inplace=False)
        self.assertTupleEqual(resampled.shape, correct_shape)
        self.assertEqual(resampled.sampling_frequency, resample_to)
        self.assertEqual(resampled.dt, 1.0 / resample_to)
        # test inplace
        sig = deepcopy(self.signal)
        sig.resample(to_frequency=resample_to, inplace=True)
        self.assertEqual(sig, resampled)

    def test_hilbert_transform(self):
        for hilbert_type in ["complex", "amplitude", "phase_unwrapped", "phase_wrapped"]:
            hilbert = self.signal.filter(low_freq=16, high_freq=24, inplace=False)
            hil = hilbert.hilbert_transform(return_as=hilbert_type, inplace=False)
            # just check whether it runs
            self.assertTrue(isinstance(hil, RatesSignal))
            self.assertTupleEqual(hil.shape, self.signal.shape)
            # test inplace
            sig = deepcopy(hilbert)
            sig.hilbert_transform(return_as=hilbert_type, inplace=True)
            self.assertEqual(sig, hil)

        with pytest.raises(ValueError):
            hilbert = self.signal.filter(low_freq=16, high_freq=24, inplace=False)
            hil = hilbert.hilbert_transform(return_as="abcde", inplace=False)

    def test_detrend(self):
        detrended = self.signal.detrend(segments=[5000, 10000, 15000], inplace=False)
        # just check whether it runs
        self.assertTrue(isinstance(detrended, RatesSignal))
        self.assertTupleEqual(detrended.shape, self.signal.shape)
        # test inplace
        sig = deepcopy(self.signal)
        sig.detrend(segments=[5000, 10000, 15000], inplace=True)
        self.assertEqual(sig, detrended)

    def test_filter(self):
        # test more filtering combinations
        filter_specs = [
            {"low_freq": None, "high_freq": 25},  # low-pass filter
            {"low_freq": 10, "high_freq": None},  # high-pass filter
            {"low_freq": 16, "high_freq": 24},  # band-pass
            {"low_freq": 19, "high_freq": 21},  # band-stop
        ]
        for filter_spec in filter_specs:
            filtered = self.signal.filter(**filter_spec, inplace=False)
            # just check whether it runs
            self.assertTrue(isinstance(filtered, RatesSignal))
            self.assertTupleEqual(filtered.shape, self.signal.shape)
            # test inplace
            sig = deepcopy(self.signal)
            sig.filter(**filter_spec, inplace=True)
            self.assertEqual(sig, filtered)

    def test_scipy_filter(self):
        filter_specs = [
            {"l_freq": None, "h_freq": 25},  # low-pass filter
            {"l_freq": 10, "h_freq": None},  # high-pass filter
            {"l_freq": 16, "h_freq": 24},  # band-pass
            {"l_freq": 19, "h_freq": 21},  # band-stop
        ]
        for filter_spec in filter_specs:
            filtered_array = scipy_iir_filter_data(
                self.signal.data.values, sfreq=self.signal.sampling_frequency, **filter_spec
            )
            self.assertTrue(isinstance(filtered_array, np.ndarray))
            self.assertTupleEqual(filtered_array.shape, self.signal.shape)

    def test_apply(self):
        # test function which does not change shape
        def do_operation(x):
            return np.abs(x - 4.0) / 8.0

        operation = self.signal.apply(func=do_operation, inplace=False)
        self.assertTrue(isinstance(operation, RatesSignal))
        xr.testing.assert_equal(
            operation.data,
            xr.apply_ufunc(do_operation, self.signal.data, input_core_dims=[["time"]], output_core_dims=[["time"]]),
        )
        # test inplace
        sig = deepcopy(self.signal)
        sig.apply(func=do_operation, inplace=True)
        self.assertEqual(sig, operation)

        def do_operation_2(x):
            return np.mean(x, axis=-1) - 8.0 + 19.0

        # assert log warning was issued
        root_logger = logging.getLogger()
        with self.assertLogs(root_logger, level="WARNING") as cm:
            operation = self.signal.apply(func=do_operation_2)
            self.assertEqual(
                cm.output,
                [
                    "WARNING:root:Shape changed after operation! Old shape: "
                    f"{self.signal.shape}, new shape: {operation.shape}; "
                    "Cannot cast to Signal class, returing as `xr.DataArray`"
                ],
            )
        self.assertTrue(isinstance(operation, xr.DataArray))
        xr.testing.assert_equal(
            operation, xr.apply_ufunc(do_operation_2, self.signal.data, input_core_dims=[["time"]]),
        )

    def test_functional_connectivity(self):
        # assert log warning was issued
        root_logger = logging.getLogger()
        with self.assertLogs(root_logger, level="ERROR") as cm:
            fcs = self.signal.functional_connectivity()
            self.assertEqual(
                cm.output, ["ERROR:root:Cannot compute functional connectivity from one timeseries."],
            )
        # should be None when computing on one timeseries
        self.assertEqual(None, fcs)

        # now proper example with network - 3D case
        ds = Dataset("gw")
        aln = ALNModel(Cmat=ds.Cmat, Dmat=ds.Dmat)
        # in ms, simulates for 2 minutes
        aln.params["duration"] = 2 * 1000
        aln.run()
        network_sig = RatesSignal.from_model_output(aln)
        fcs = network_sig.functional_connectivity()
        self.assertTrue(isinstance(fcs, xr.DataArray))
        correct_shape = (network_sig.shape[0], network_sig.shape[1], network_sig.shape[1])
        self.assertTupleEqual(fcs.shape, correct_shape)

        # 2D case
        fc = network_sig["rates_exc"].functional_connectivity()
        self.assertTrue(isinstance(fc, xr.DataArray))
        correct_shape = (network_sig.shape[1], network_sig.shape[1])
        self.assertTupleEqual(fc.shape, correct_shape)


if __name__ == "__main__":
    unittest.main()
