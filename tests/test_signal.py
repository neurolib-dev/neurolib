"""
Set of tests for `Signal` class.
"""

import os
import unittest
from shutil import rmtree

import numpy as np
from neurolib.models.aln import ALNModel
from neurolib.utils.signal import RatesSignal


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
        cls.signal = RatesSignal(data=aln.xr(), time_in_ms=True)
        os.makedirs(cls.TEST_FOLDER)

    @classmethod
    def tearDownClass(cls):
        rmtree(cls.TEST_FOLDER)

    def test_load_save(self):
        # save
        filename = os.path.join(self.TEST_FOLDER, "temp")
        # do operation so we also test saving and loading of preprocessing steps
        self.signal.normalize(std=True)
        self.signal.save(filename)
        # load
        loaded = RatesSignal.from_file(filename)
        # compare they are equal
        self.assertEqual(self.signal, loaded)
        # self.assertTrue(False)

    def test_sel(self):
        selected = self.signal.sel([5.43, 7.987], inplace=False)
        # test correct indices
        self.assertEqual(selected.data.time.values[0], 5.43)
        self.assertEqual(selected.data.time.values[-1], 7.987)

    def test_isel(self):
        selected = self.signal.isel([12143, 16424], inplace=False)
        # test correct indices
        self.assertEqual(selected.data.time.values[0], 12143 / self.signal.sampling_frequency)
        self.assertEqual(selected.data.time.values[-1], (16424 - 1) / self.signal.sampling_frequency)

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
        correct_shape = tuple(
            list(self.signal.shape[:-1]) + [self.signal.shape[-1] + 2 * pad_seconds * self.signal.sampling_frequency]
        )
        padded = self.signal.pad(
            pad_seconds,
            in_seconds=True,
            padding_type="constant",
            side="both",
            constant_values=const_value,
            inplace=False,
        )
        # assert correct shape
        self.assertEqual(padded.shape, correct_shape)
        # assert correct indices - they should increase monotically without
        # sudden jump, i.e. all timestep differences should be equal to dt
        np.testing.assert_allclose(np.diff(padded.data.time.values), padded.dt)
        # assert we actually did input `const_value`
        np.testing.assert_allclose(
            padded.isel([0, int(2 * padded.sampling_frequency)], inplace=False).data, const_value,
        )
        np.testing.assert_allclose(
            padded.isel([-int(2 * padded.sampling_frequency), None], inplace=False).data, const_value,
        )

    def test_normalize(self):
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

    # def test_hilbert_transform(self):
    #     for hilbert_type in ["amplitude", "phase_unwrapped", "phase_wrapped"]:
    #         filtered = self.signal.filter(low_freq=16, high_freq=24, inplace=False)
    #         hilbert = filtered.hilbert_transform(return_as=hilbert_type, inplace=False)
    #         correct_result = Signal.from_file(os.path.join(self.TEST_FOLDER, f"hilbert_{hilbert_type}_test_result.csv"))
    #         self.assertEqual(correct_result, hilbert)

    # def test_detrend(self):
    #     detrended = self.signal.detrend(segments=[5000, 10000, 15000], inplace=False)
    #     correct_result = Signal.from_file(os.path.join(self.TEST_FOLDER, "detrend_test_result.csv"))
    #     self.assertEqual(correct_result, detrended)

    # def test_filter(self):
    #     # test more filtering combinations
    #     filter_specs = [
    #         {"low_freq": None, "high_freq": 25},  # low-pass filter
    #         {"low_freq": 10, "high_freq": None},  # high-pass filter
    #         {"low_freq": 16, "high_freq": 24},  # band-pass
    #         {"low_freq": 19, "high_freq": 21},  # band-stop
    #     ]
    #     for filter_spec in filter_specs:
    #         filtered = self.signal.filter(**filter_spec, inplace=False)
    #         correct_result = Signal.from_file(
    #             os.path.join(
    #                 self.TEST_FOLDER,
    #                 f"filter_{filter_spec['low_freq']}-" f"{filter_spec['high_freq']}_test_result.csv",
    #             )
    #         )
    #         self.assertEqual(correct_result, filtered)
