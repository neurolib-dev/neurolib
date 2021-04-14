"""
Base classes for representing signals.
"""

import logging
from copy import deepcopy
from functools import partial

import numpy as np
import xarray as xr
from ..models.model import Model
from scipy.signal import butter, detrend, get_window, hilbert
from scipy.signal import resample as scipy_resample
from scipy.signal import sosfiltfilt

NC_EXT = ".nc"


def scipy_iir_filter_data(x, sfreq, l_freq, h_freq, l_trans_bandwidth=None, h_trans_bandwidth=None, **kwargs):
    """
    Custom, scipy based filtering function with basic butterworth filter.

    :param x: data to be filtered, time is the last axis
    :type x: np.ndarray
    :param sfreq: sampling frequency of the data in Hz
    :type sfreq: float
    :param l_freq: frequency below which to filter the data in Hz
    :type l_freq: float|None
    :param h_freq: frequency above which to filter the data in Hz
    :type h_freq: float|None
    :param l_trans_bandwidth: keeping for compatibility with mne
    :type l_trans_bandwidth: None
    :param h_trans_bandwidth: keeping for compatibility with mne
    :type h_trans_bandwidth: None
    :return: filtered data
    :rtype: np.ndarray
    """
    nyq = 0.5 * sfreq
    if l_freq is not None:
        low = l_freq / nyq
        if h_freq is not None:
            # so we have band filter
            high = h_freq / nyq
            if l_freq < h_freq:
                btype = "bandpass"
            elif l_freq > h_freq:
                btype = "bandstop"
            Wn = [low, high]
        elif h_freq is None:
            # so we have a high-pass filter
            Wn = low
            btype = "highpass"
    elif l_freq is None:
        # we have a low-pass
        high = h_freq / nyq
        Wn = high
        btype = "lowpass"
    # get butter coeffs
    sos = butter(N=kwargs.pop("order", 8), Wn=Wn, btype=btype, output="sos")
    return sosfiltfilt(sos, x, axis=-1)


class Signal:
    name = ""
    label = ""
    signal_type = ""
    unit = ""
    description = ""
    _copy_attributes = [
        "name",
        "label",
        "signal_type",
        "unit",
        "description",
        "process_steps",
    ]
    PROCESS_STEPS_KEY = "process_steps"

    @classmethod
    def from_model_output(cls, model, group="", time_in_ms=True):
        """
        Initial Signal from modelling output.
        """
        assert isinstance(model, Model)
        return cls(model.xr(group=group), time_in_ms=time_in_ms)

    @classmethod
    def from_file(cls, filename):
        """
        Load signal from saved file.

        :param filename: filename for the Signal
        :type filename: str
        """
        if not filename.endswith(NC_EXT):
            filename += NC_EXT
        # load NC file
        xarray = xr.load_dataarray(filename)
        # init class
        signal = cls(xarray)
        # if nc file has attributes, copy them to signal class
        if xarray.attrs:
            process_steps = []
            for k, v in xarray.attrs.items():
                if cls.PROCESS_STEPS_KEY in k:
                    idx = int(k[len(cls.PROCESS_STEPS_KEY) + 1 :])
                    process_steps.insert(idx, v)
                else:
                    setattr(signal, k, v)
        else:
            logging.warning("No metadata found, setting empty...")
            process_steps = [f"raw {signal.signal_type} signal: {signal.start_time}--" f"{signal.end_time}s"]
        setattr(signal, cls.PROCESS_STEPS_KEY, process_steps)
        return signal

    def __init__(self, data, time_in_ms=False):
        """
        :param data: data for the signal, assumes time dimension with time in seconds
        :type data: xr.DataArray
        :param time_in_ms: whether time dimension is in ms
        :type time_in_ms: bool
        """
        assert isinstance(data, xr.DataArray)
        data = deepcopy(data)
        assert "time" in data.dims, "DataArray must have time axis"
        if time_in_ms:
            data["time"] = data["time"] / 1000.0
        data["time"] = np.around(data["time"], 6)
        self.data = data
        # assert time dimension is last
        self.data = self.data.transpose(*(self.dims_not_time + ["time"]))
        # compute dt and sampling frequency
        self.dt = np.around(np.diff(data.time).mean(), 6)
        self.sampling_frequency = 1.0 / self.dt
        self.process_steps = [f"raw {self.signal_type} signal: {self.start_time}--{self.end_time}s"]

    def __str__(self):
        """
        String representation.
        """
        return (
            f"{self.name} representing {self.signal_type} signal with unit of "
            f"{self.unit} with user-provided description: `{self.description}`"
            f". Shape of the signal is {self.shape} with dimensions "
            f"{self.data.dims}. Signal starts at {self.start_time} and ends at "
            f"{self.end_time}."
        )

    def __repr__(self):
        """
        Representation.
        """
        return self.__str__()

    def __eq__(self, other):
        """
        Comparison operator.

        :param other: other `Signal` to compare with
        :type other: `Signal`
        :return: whether two `Signals` are the same
        :rtype: bool
        """
        assert isinstance(other, Signal)
        # assert data are the same
        try:
            xr.testing.assert_allclose(self.data, other.data)
            eq = True
        except AssertionError:
            eq = False
        # check attributes, but if not equal, only warn the user
        for attr in self._copy_attributes:
            if getattr(self, attr) != getattr(other, attr):
                logging.warning(f"`{attr}` not equal between signals.")
        return eq

    def __getitem__(self, pos):
        """
        Get item selects in output dimension.
        """
        add_steps = [f"select `{pos}` output"]
        return self.__constructor__(self.data.sel(output=pos)).__finalize__(self, add_steps)

    def __finalize__(self, other, add_steps=None):
        """
        Copy attributes from other to self. Used when constructing class
        instance with different data, but same metadata.

        :param other: other instance of `Signal`
        :type other: `Signal`
        :param add_steps: add steps to preprocessing
        :type add_steps: list|None
        """
        assert isinstance(other, Signal)
        for attr in self._copy_attributes:
            setattr(self, attr, deepcopy(getattr(other, attr)))
        if add_steps is not None:
            self.process_steps += add_steps
        return self

    @property
    def __constructor__(self):
        """
        Return constructor, so that each child class would initiate a new
        instance of the correct class, i.e. first in the method resolution
        order.
        """
        return self.__class__.mro()[0]

    def _write_attrs_to_xr(self):
        """
        Copy attributes to xarray before saving.
        """
        # write attributes to xarray
        for attr in self._copy_attributes:
            value = getattr(self, attr)
            # if list need to unwrap
            if isinstance(value, (list, tuple)):
                for idx, val in enumerate(value):
                    self.data.attrs[f"{attr}_{idx}"] = val
            else:
                self.data.attrs[attr] = deepcopy(value)

    def save(self, filename):
        """
        Save signal.

        :param filename: filename to save, currently saves to netCDF file, which is natively supported by xarray
        :type filename: str
        """
        self._write_attrs_to_xr()
        if not filename.endswith(NC_EXT):
            filename += NC_EXT
        self.data.to_netcdf(filename)

    def iterate(self, return_as="signal"):
        """
        Return iterator over columns, so univariate measures can be computed
        per column. Loops over tuples as (variable name, timeseries).

        :param return_as: how to return columns: `xr` as xr.DataArray, `signal` as
            instance of NeuroSignal with the same attributes as the mother signal
        :type return_as: str
        """
        try:
            stacked = self.data.stack({"all": self.dims_not_time})
        except ValueError:
            logging.warning("No dimensions along which to stack...")
            stacked = self.data.expand_dims("all")

        if return_as == "xr":
            yield from stacked.groupby("all")
        elif return_as == "signal":
            for name_coords, column in stacked.groupby("all"):
                if not isinstance(name_coords, (list, tuple)):
                    name_coords = [name_coords]
                name_dict = {k: v for k, v in zip(self.dims_not_time, name_coords)}
                yield name_dict, self.__constructor__(column).__finalize__(self, [f"select {column.name}"])
        else:
            raise ValueError(f"Data type not understood: {return_as}")

    def sel(self, sel_args, inplace=True):
        """
        Subselect part of signal using xarray's `sel`, i.e. selecting by actual
        physical index, hence time in seconds.

        :param sel_args: arguments you'd give to xr.sel(), i.e. slice of times
            you want to select, in seconds as a len=2 list or tuple
        :type sel_args: tuple|list
        :param inplace: whether to do the operation in place or return
        :type inplace: bool
        """
        assert len(sel_args) == 2, "Must provide 2 arguments"
        selected = self.data.sel(time=slice(sel_args[0], sel_args[1]))
        add_steps = [f"select {sel_args[0] or 'x'}:{sel_args[1] or 'x'}s"]
        if inplace:
            self.data = selected
            self.process_steps += add_steps
        else:
            return self.__constructor__(selected).__finalize__(self, add_steps)

    def isel(self, isel_args, inplace=True):
        """
        Subselect part of signal using xarray's `isel`, i.e. selecting by index,
        hence integers.

        :param loc_args: arguments you'd give to xr.isel(), i.e. slice of
            indices you want to select, in seconds as a len=2 list or tuple
        :type loc_args: tuple|list
        :param inplace: whether to do the operation in place or return
        :type inplace: bool
        """
        assert len(isel_args) == 2, "Must provide 2 arguments"
        selected = self.data.isel(time=slice(isel_args[0], isel_args[1]))
        start = isel_args[0] * self.dt if isel_args[0] is not None else "x"
        end = isel_args[1] * self.dt if isel_args[1] is not None else "x"
        add_steps = [f"select {start}:{end}s"]
        if inplace:
            self.data = selected
            self.process_steps += add_steps
        else:
            return self.__constructor__(selected).__finalize__(self, add_steps)

    def rolling(self, roll_over, function=np.mean, dropnans=True, inplace=True):
        """
        Return rolling reduction over signal's time dimension. The window is
        centered around the midpoint.

        :param roll_over: window to use, in seconds
        :type roll_over: float
        :param function: function to use for reduction
        :type function: callable
        :param dropnans: whether to drop NaNs - will shorten time dimension, or
            not
        :type dropnans: bool
        :param inplace: whether to do the operation in place or return
        :type inplace: bool
        """
        assert callable(function)
        rolling = self.data.rolling(time=int(roll_over * self.sampling_frequency), center=True).reduce(function)
        add_steps = [f"rolling {function.__name__} over {roll_over}s"]
        if dropnans:
            rolling = rolling.dropna("time")
            add_steps[0] += "; drop NaNs"
        if inplace:
            self.data = rolling
            self.process_steps += add_steps
        else:
            return self.__constructor__(rolling).__finalize__(self, add_steps)

    def sliding_window(self, length, step=1, window_function="boxcar", lengths_in_seconds=False):
        """
        Return iterator over sliding windows with windowing function applied.
        Each window has length `length` and each is translated by `step` steps.
        For no windowing function use "boxcar". If the last window would have
        the same length as other, it is omitted, i.e. last window does not have
        to end with the final timeseries point!

        :param length: length of the window, can be index or time in seconds,
            see `lengths_in_seconds`
        :type length: int|float
        :param step: how much to translate window in the temporal sense, can be
            index or time in seconds, see `lengths_in_seconds`
        :type step: int|float
        :param window_function: windowing function to use, this is passed to
            `get_window()`; see `scipy.signal.windows.get_window` documentation
        :type window_function: str|tuple|float
        :param lengths_in_seconds: if True, `length` and `step` are interpreted
            in seconds, if False they are indices
        :type lengths_in_seconds: bool
        :yield: generator with windowed Signals
        """
        if lengths_in_seconds:
            length = int(length / self.dt)
            step = int(step / self.dt)
        assert (
            length < self.data.time.shape[0]
        ), f"Length must be smaller than time span of the timeseries: {self.data.time.shape[0]}"
        assert step <= length, "Step cannot be larger than length, some part of timeseries would be omitted!"
        current_idx = 0
        add_steps = f"{str(window_function)} window: "
        windowing_function = get_window(window_function, Nx=length)
        while current_idx <= (self.data.time.shape[0] - length):
            yield self.__constructor__(
                self.data.isel(time=slice(current_idx, current_idx + length)) * windowing_function
            ).__finalize__(self, [add_steps + f"{current_idx}:{current_idx + length}"])
            current_idx += step

    @property
    def shape(self):
        """
        Return shape of the data. Time axis is the first one.
        """
        return self.data.shape

    @property
    def dims_not_time(self):
        """
        Return list of dimensions that are not time.
        """
        return [dim for dim in self.data.dims if dim != "time"]

    @property
    def coords_not_time(self):
        """
        Return dict with all coordinates except time.
        """
        return {k: v.values for k, v in self.data.coords.items() if k != "time"}

    @property
    def start_time(self):
        """
        Return starting time of the signal.
        """
        return self.data.time.values[0]

    @property
    def end_time(self):
        """
        Return ending time of the signal.
        """
        return self.data.time.values[-1]

    @property
    def time(self):
        """
        Return time vector.
        """
        return self.data.time.values

    @property
    def preprocessing_steps(self):
        """
        Return preprocessing steps done on the data.
        """
        return " -> ".join(self.process_steps)

    def pad(self, how_much, in_seconds=False, padding_type="constant", side="both", inplace=True, **kwargs):
        """
        Pad signal by `how_much` on given side of given type.

        :param how_much: how much we should pad, can be time points, or seconds,
            see `in_seconds`
        :type how_much: float|int
        :param in_seconds: whether `how_much` is in seconds, if False, it is
            number of time points
        :type in_seconds: bool
        :param padding_type: how to pad the signal, see `np.pad` documentation
        :type padding_type: str
        :param side: which side to pad - "before", "after", or "both"
        :type side: str
        :param inplace: whether to do the operation in place or return
        :type inplace: bool
        :kwargs: passed to `np.pad`
        """
        if in_seconds:
            how_much = int(np.around(how_much / self.dt))
        if side == "before":
            pad_width = (how_much, 0)
            pad_times = np.arange(-how_much, 0) * self.dt + self.data.time.values[0]
            new_times = np.concatenate([pad_times, self.data.time.values], axis=0)
        elif side == "after":
            pad_width = (0, how_much)
            pad_times = np.arange(1, how_much + 1) * self.dt + self.data.time.values[-1]
            new_times = np.concatenate([self.data.time.values, pad_times], axis=0)
        elif side == "both":
            pad_width = (how_much, how_much)
            pad_before = np.arange(-how_much, 0) * self.dt + self.data.time.values[0]
            pad_after = np.arange(1, how_much + 1) * self.dt + self.data.time.values[-1]
            new_times = np.concatenate([pad_before, self.data.time.values, pad_after], axis=0)
            side += " sides"
        else:
            raise ValueError(f"Unknown padding side: {side}")
        # add padding for other axes than time - zeroes
        pad_width = [(0, 0)] * len(self.dims_not_time) + [pad_width]
        padded = np.pad(self.data.values, pad_width, mode=padding_type, **kwargs)
        # to dataframe
        padded = xr.DataArray(padded, dims=self.data.dims, coords={**self.coords_not_time, "time": new_times})
        add_steps = [f"{how_much * self.dt}s {padding_type} {side} padding"]
        if inplace:
            self.data = padded
            self.process_steps += add_steps
        else:
            return self.__constructor__(padded).__finalize__(self, add_steps)

    def normalize(self, std=False, inplace=True):
        """
        De-mean the timeseries. Optionally also standardise.

        :param std: normalize by std, i.e. to unit variance
        :type std: bool
        :param inplace: whether to do the operation in place or return
        :type inplace: bool
        """

        def norm_func(x, dim):
            demeaned = x - x.mean(dim=dim)
            if std:
                return demeaned / x.std(dim=dim)
            else:
                return demeaned

        normalized = norm_func(self.data, dim="time")
        add_steps = ["normalize", "standardize"] if std else ["normalize"]
        if inplace:
            self.data = normalized
            self.process_steps += add_steps
        else:
            return self.__constructor__(normalized).__finalize__(self, add_steps)

    def resample(self, to_frequency, inplace=True):
        """
        Resample signal to target frequency.

        :param to_frequency: target frequency of the signal, in Hz
        :type to_frequency: float
        :param inplace: whether to do the operation in place or return
        :type inplace: bool
        """
        to_frequency = float(to_frequency)
        try:
            from mne.filter import resample

            resample_func = partial(
                resample, up=to_frequency, down=self.sampling_frequency, npad="auto", axis=-1, pad="edge"
            )
        except ImportError:
            logging.warning("`mne` module not found, falling back to basic scipy's function")

            def resample_func(x):
                return scipy_resample(
                    x,
                    num=int(round((to_frequency / self.sampling_frequency) * self.data.shape[-1])),
                    axis=-1,
                    window="boxcar",
                )

        resampled = resample_func(self.data.values)
        # construct new times
        new_times = (np.arange(resampled.shape[-1], dtype=np.float) / to_frequency) + self.data.time.values[0]
        # to dataframe
        resampled = xr.DataArray(resampled, dims=self.data.dims, coords={**self.coords_not_time, "time": new_times})
        add_steps = [f"resample to {to_frequency}Hz"]
        if inplace:
            self.data = resampled
            self.sampling_frequency = to_frequency
            self.dt = np.around(np.diff(resampled.time).mean(), 6)
            self.process_steps += add_steps
        else:
            return self.__constructor__(resampled).__finalize__(self, add_steps)

    def hilbert_transform(self, return_as="complex", inplace=True):
        """
        Perform hilbert transform on the signal resulting in analytic signal.

        :param return_as: what to return
            `complex` will compute only analytical signal
            `amplitude` will compute amplitude, hence abs(H(x))
            `phase_wrapped` will compute phase, hence angle(H(x)), in -pi,pi
            `phase_unwrapped` will compute phase in a continuous sense, hence
                monotonic
        :param inplace: whether to do the operation in place or return
        :type inplace: bool
        """
        analytic = hilbert(self.data, axis=-1)
        if return_as == "amplitude":
            analytic = np.abs(analytic)
            add_steps = ["Hilbert - amplitude"]
        elif return_as == "phase_unwrapped":
            analytic = np.unwrap(np.angle(analytic))
            add_steps = ["Hilbert - unwrapped phase"]
        elif return_as == "phase_wrapped":
            analytic = np.angle(analytic)
            add_steps = ["Hilbert - wrapped phase"]
        elif return_as == "complex":
            add_steps = ["Hilbert - complex"]
        else:
            raise ValueError(f"Do not know how to return: {return_as}")

        analytic = xr.DataArray(analytic, dims=self.data.dims, coords=self.data.coords)
        if inplace:
            self.data = analytic
            self.process_steps += add_steps
        else:
            return self.__constructor__(analytic).__finalize__(self, add_steps)

    def detrend(self, segments=None, inplace=True):
        """
        Linearly detrend signal. If segments are given, detrending will be
        performed in each part.

        :param segments: segments for detrending, if None will detrend whole
            signal, given as indices of the time array
        :type segments: list|None
        :param inplace: whether to do the operation in place or return
        :type inplace: bool
        """
        segments = segments or 0
        detrended = detrend(self.data, type="linear", bp=segments, axis=-1)
        detrended = xr.DataArray(detrended, dims=self.data.dims, coords=self.data.coords)
        segments_text = f" with segments: {segments}" if segments != 0 else ""
        add_steps = [f"detrend{segments_text}"]
        if inplace:
            self.data = detrended
            self.process_steps += add_steps
        else:
            return self.__constructor__(detrended).__finalize__(self, add_steps)

    def filter(self, low_freq, high_freq, l_trans_bandwidth="auto", h_trans_bandwidth="auto", inplace=True, **kwargs):
        """
        Filter data. Can be:
            low-pass (low_freq is None, high_freq is not None),
            high-pass (high_freq is None, low_freq is not None),
            band-pass (l_freq < h_freq),
            band-stop (l_freq > h_freq) filter type

        :param low_freq: frequency below which to filter the data
        :type low_freq: float|None
        :param high_freq: frequency above which to filter the data
        :type high_freq: float|None
        :param l_trans_bandwidth: transition band width for low frequency
        :type l_trans_bandwidth: float|str
        :param h_trans_bandwidth: transition band width for high frequency
        :type h_trans_bandwidth: float|str
        :param inplace: whether to do the operation in place or return
        :type inplace: bool
        :**kwargs: possible keywords to `mne.filter.create_filter`:
            `filter_length`="auto",
            `method`="fir",
            `iir_params`=None
            `phase`="zero",
            `fir_window`="hamming",
            `fir_design`="firwin"
        """
        try:
            from mne.filter import filter_data

        except ImportError:
            logging.warning("`mne` module not found, falling back to basic scipy's function")
            filter_data = scipy_iir_filter_data

        filtered = filter_data(
            self.data.values,  # times has to be the last axis
            sfreq=self.sampling_frequency,
            l_freq=low_freq,
            h_freq=high_freq,
            l_trans_bandwidth=l_trans_bandwidth,
            h_trans_bandwidth=h_trans_bandwidth,
            **kwargs,
        )
        add_steps = [f"filter: low {low_freq or 'x'}Hz - high {high_freq or 'x'}Hz"]
        # to dataframe
        filtered = xr.DataArray(filtered, dims=self.data.dims, coords=self.data.coords)
        if inplace:
            self.data = filtered
            self.process_steps += add_steps
        else:
            return self.__constructor__(filtered).__finalize__(self, add_steps)

    def functional_connectivity(self, fc_function=np.corrcoef):
        """
        Compute and return functional connectivity from the data.

        :param fc_function: function which to use for FC computation, should
            take 2D array as space x time and convert it to space x space with
            desired measure
        """
        if len(self.data["space"]) <= 1:
            logging.error("Cannot compute functional connectivity from one timeseries.")
            return None
        if self.data.ndim == 3:
            assert callable(fc_function)
            fcs = []
            for output in self.data["output"]:
                current_slice = self.data.sel({"output": output})
                assert current_slice.ndim == 2
                fcs.append(fc_function(current_slice.values))

            return xr.DataArray(
                np.array(fcs),
                dims=["output", "space", "space"],
                coords={"output": self.data.coords["output"], "space": self.data.coords["space"]},
            )
        if self.data.ndim == 2:
            return xr.DataArray(
                fc_function(self.data.values),
                dims=["space", "space"],
                coords={"space": self.data.coords["space"]},
            )

    def apply(self, func, inplace=True):
        """
        Apply func for each timeseries.

        :param func: function to be applied for each 1D timeseries
        :type func: callable
        :param inplace: whether to do the operation in place or return
        :type inplace: bool
        """
        assert callable(func)
        try:
            # this will work for element-wise function that does not reduces dimensions
            processed = xr.apply_ufunc(func, self.data, input_core_dims=[["time"]], output_core_dims=[["time"]])
            add_steps = [f"apply `{func.__name__}` function over time dim"]
            if inplace:
                self.data = processed
                self.process_steps += add_steps
            else:
                return self.__constructor__(processed).__finalize__(self, add_steps)
        except ValueError:
            # this works for functions that reduce time dimension
            processed = xr.apply_ufunc(func, self.data, input_core_dims=[["time"]])
            logging.warning(
                f"Shape changed after operation! Old shape: {self.shape}, new "
                f"shape: {processed.shape}; Cannot cast to Signal class, "
                "returing as `xr.DataArray`"
            )
            return processed


class VoltageSignal(Signal):
    name = "Population mean membrane potential"
    label = "V"
    signal_type = "voltage"
    unit = "mV"


class RatesSignal(Signal):
    name = "Population firing rate"
    label = "q"
    signal_type = "rate"
    unit = "Hz"


class BOLDSignal(Signal):
    name = "Population blood oxygen level-dependent signal"
    label = "BOLD"
    signal_type = "bold"
    unit = "%"
