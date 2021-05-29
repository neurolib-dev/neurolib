import logging
import numpy as np
import scipy.signal
import numba

"""Collection of useful functions for data processing.
"""


def kuramoto(traces, smoothing=0.0, distance=10, prominence=5):
    """
    Computes the Kuramoto order parameter of a timeseries which is a measure for synchrony.
    Can smooth timeseries if there is noise.
    Peaks are then detected using a peakfinder. From these peaks a phase is derived and then
    the amount of phase synchrony (the Kuramoto order parameter) is computed.

    :param traces: Multidimensional timeseries array
    :type traces: numpy.ndarray
    :param smoothing: Gaussian smoothing strength
    :type smoothing: float, optional
    :param distance: minimum distance between peaks in samples
    :type distance: int, optional
    :param prominence: vertical distance between the peak and its lowest contour line
    :type prominence: int, optional

    :return: Timeseries of Kuramoto order paramter
    :rtype: numpy.ndarray
    """
    @numba.njit
    def _estimate_phase(maximalist, n_times):
        lastMax = 0
        phases = np.empty((n_times), dtype=np.float64)
        n = 0
        for m in maximalist:
            for t in range(lastMax, m):
                # compute instantaneous phase
                phi = 2 * np.pi * float(t - lastMax) / float(m - lastMax)
                phases[n] = phi
                n += 1
            lastMax = m
        phases[-1] = 2 * np.pi
        return phases

    @numba.njit
    def _estimate_r(ntraces, times, phases):
        kuramoto = np.empty((times), dtype=np.float64)
        for t in range(times):
            R = 1j*0
            for n in range(ntraces):
                R += np.exp(1j * phases[n, t])
            R /= ntraces
            kuramoto[t] = np.absolute(R)
        return kuramoto

    nTraces, nTimes = traces.shape
    phases = np.empty_like(traces)
    for n in range(nTraces):
        a = traces[n]
        # find peaks
        if smoothing > 0:
            # smooth data
            a = scipy.ndimage.filters.gaussian_filter(traces[n], smoothing)
        maximalist = scipy.signal.find_peaks(a, distance=distance,
                                             prominence=prominence)[0]
        maximalist = np.append(maximalist, len(traces[n])-1).astype(int)

        if len(maximalist) > 1:
            phases[n, :] = _estimate_phase(maximalist, nTimes)
        else:
            logging.warning("Kuramoto: No peaks found, returning 0.")
            return 0
    # determine kuramoto order paramter
    kuramoto = _estimate_r(nTraces, nTimes, phases)
    return kuramoto


def matrix_correlation(M1, M2):
    """Pearson correlation of the lower triagonal of two matrices.
    The triangular matrix is offset by k = 1 in order to ignore the diagonal line

    :param M1: First matrix
    :type M1: numpy.ndarray
    :param M2: Second matrix
    :type M2: numpy.ndarray
    :return: Correlation coefficient
    :rtype: float
    """
    cc = np.corrcoef(M1[np.triu_indices_from(M1, k=1)], M2[np.triu_indices_from(M2, k=1)])[0, 1]
    return cc


def weighted_correlation(x, y, w):
    """Weighted Pearson correlation of two series.

    :param x: Timeseries 1
    :type x: list, np.array
    :param y: Timeseries 2, must have same length as x
    :type y: list, np.array
    :param w: Weight vector, must have same length as x and y
    :type w: list, np.array
    :return: Weighted correlation coefficient
    :rtype: float
    """

    def weighted_mean(x, w):
        """Weighted Mean"""
        return np.sum(x * w) / np.sum(w)

    def weighted_cov(x, y, w):
        """Weighted Covariance"""
        return np.sum(w * (x - weighted_mean(x, w)) * (y - weighted_mean(y, w))) / np.sum(w)

    return weighted_cov(x, y, w) / np.sqrt(weighted_cov(x, x, w) * weighted_cov(y, y, w))


def fc(ts):
    """Functional connectivity matrix of timeseries multidimensional `ts` (Nxt).
    Pearson correlation (from `np.corrcoef()` is used).

    :param ts: Nxt timeseries
    :type ts: numpy.ndarray
    :return: N x N functional connectivity matrix
    :rtype: numpy.ndarray
    """
    fc = np.corrcoef(ts)
    fc = np.nan_to_num(fc)  # remove NaNs
    return fc


def fcd(ts, windowsize=30, stepsize=5):
    """Computes FCD (functional connectivity dynamics) matrix, as described in Deco's whole-brain model papers.
    Default paramters are suited for computing FCS matrices of BOLD timeseries:
    A windowsize of 30 at the BOLD sampling rate of 0.5 Hz equals 60s and stepsize = 5 equals 10s.

    :param ts: Nxt timeseries
    :type ts: numpy.ndarray
    :param windowsize: Size of each rolling window in timesteps, defaults to 30
    :type windowsize: int, optional
    :param stepsize: Stepsize between each rolling window, defaults to 5
    :type stepsize: int, optional
    :return: T x T FCD matrix
    :rtype: numpy.ndarray
    """
    t_window_width = int(windowsize)  # int(windowsize * 30) # x minutes
    stepsize = stepsize  # ts.shape[1]/N
    corrFCs = []
    try:
        counter = range(0, ts.shape[1] - t_window_width, stepsize)

        for t in counter:
            ts_slice = ts[:, t : t + t_window_width]
            corrFCs.append(np.corrcoef(ts_slice))

        FCd = np.empty([len(corrFCs), len(corrFCs)])
        f1i = 0
        for f1 in corrFCs:
            f2i = 0
            for f2 in corrFCs:
                FCd[f1i, f2i] = np.corrcoef(f1.reshape((1, f1.size)), f2.reshape((1, f2.size)))[0, 1]
                f2i += 1
            f1i += 1

        return FCd
    except:
        return 0


def matrix_kolmogorov(m1, m2):
    """Computes the Kolmogorov distance between the distributions of lower-triangular entries of two matrices
    See: https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test#Two-sample_Kolmogorov%E2%80%93Smirnov_test

    :param m1: matrix 1
    :type m1: np.ndarray
    :param m2: matrix 2
    :type m2: np.ndarray
    :return: 2-sample KS statistics
    :rtype: float
    """
    # get the values of the lower triangle
    triu_ind1 = np.triu_indices(m1.shape[0], k=1)
    m1_vals = m1[triu_ind1]

    triu_ind2 = np.triu_indices(m2.shape[0], k=1)
    m2_vals = m2[triu_ind2]

    # return the distance, omit p-value
    return scipy.stats.ks_2samp(m1_vals, m2_vals)[0]


def ts_kolmogorov(ts1, ts2, **fcd_kwargs):
    """Computes kolmogorov distance between two timeseries.
    This is done by first computing two FCD matrices (one for each timeseries)
    and then measuring the Kolmogorov distance of the upper triangle of these matrices.

    :param ts1: Timeseries 1
    :type ts1: np.ndarray
    :param ts2: Timeseries 2
    :type ts2: np.ndarray
    :return: 2-sample KS statistics
    :rtype: float
    """
    fcd1 = fcd(ts1, **fcd_kwargs)
    fcd2 = fcd(ts2, **fcd_kwargs)

    return matrix_kolmogorov(fcd1, fcd2)


# def max_distance_cumulative(data1, data2):
#     """
#     From: https://github.com/scipy/scipy/issues/9389

#     Computes the maximal vertical distance between cumulative distributions
#     (this is the statistic for KS tests). Code mostly copied from
#     scipy.stats.ks_twosamp

#     Parameters
#     ----------
#     data1 : array_like
#         First data set
#     data2 : array_like
#         Second data set
#     Returns
#     -------
#     d : float
#         Max distance, i.e. value of the Kolmogorov Smirnov test. Sign is + if
#         the cumulative of data1 < the one of data2 at that location, else -.
#     x : float
#         Value of x where maximal distance d is reached.
#     """
#     from numpy import ma

#     (data1, data2) = (ma.asarray(data1), ma.asarray(data2))
#     (n1, n2) = (data1.count(), data2.count())
#     mix = ma.concatenate((data1.compressed(), data2.compressed()))
#     mixsort = mix.argsort(kind="mergesort")
#     csum = np.where(mixsort < n1, 1.0 / n1, -1.0 / n2).cumsum()

#     # Check for ties
#     if len(np.unique(mix)) < (n1 + n2):
#         ind = np.r_[np.diff(mix[mixsort]).nonzero()[0], -1]
#         csum = csum[ind]
#         mixsort = mixsort[ind]

#     csumabs = ma.abs(csum)
#     i = csumabs.argmax()

#     d = csum[i]
#     # mixsort[i] contains the index of mix with the max distance
#     x = mix[mixsort[i]]

#     return (d, x)


# def print_params(params):
#     """
#     Helpfer function for printing a subset of the paramters of the aln model.
#     Todo: This function should not be here, it is too specific for the aln model.
#     Idea: A model could register "parameters of interest" and be printed with this function.
#     However, this should be placed in the Model class in any case
#     """
#     paramsOfInterest = [
#         "dt",
#         "Ke_gl",
#         "mue_ext_mean",
#         "mui_ext_mean",
#         "sigma_ou",
#         "signalV",
#         "a",
#         "b",
#         "Jee_max",
#         "Jie_max",
#         "Jii_max",
#         "Jei_max",
#         "cee",
#         "cie",
#         "cii",
#         "cei",
#         "Ke",
#         "Ki",
#         "de",
#         "di",
#     ]
#     for p in paramsOfInterest:
#         print("params['%s'] = %0.3f" % (p, params[p]))


def getPowerSpectrum(activity, dt, maxfr=70, spectrum_windowsize=1.0, normalize=False):
    """Returns a power spectrum using Welch's method.

    :param activity: One-dimensional timeseries
    :type activity: np.ndarray
    :param dt: Simulation time step
    :type dt: float
    :param maxfr: Maximum frequency in Hz to cutoff from return, defaults to 70
    :type maxfr: int, optional
    :param spectrum_windowsize: Length of the window used in Welch's method (in seconds), defaults to 1.0
    :type spectrum_windowsize: float, optional
    :param normalize: Maximum power is normalized to 1 if True, defaults to False
    :type normalize: bool, optional

    :return: Frquencies and the power of each frequency
    :rtype: [np.ndarray, np.ndarray]
    """
    # convert to one-dimensional array if it is an (1xn)-D array
    if activity.shape[0] == 1 and activity.shape[1] > 1:
        activity = activity[0]
    assert len(activity.shape) == 1, "activity is not one-dimensional!"

    f, Pxx_spec = scipy.signal.welch(
        activity,
        1000 / dt,
        window="hanning",
        nperseg=int(spectrum_windowsize * 1000 / dt),
        scaling="spectrum",
    )
    f = f[f < maxfr]
    Pxx_spec = Pxx_spec[0 : len(f)]
    if normalize:
        Pxx_spec /= np.max(Pxx_spec)
    return f, Pxx_spec


def getMeanPowerSpectrum(activities, dt, maxfr=70, spectrum_windowsize=1.0, normalize=False):
    """Returns the mean power spectrum of multiple timeseries.

    :param activities: N-dimensional timeseries
    :type activities: np.ndarray
    :param dt: Simulation time step
    :type dt: float
    :param maxfr: Maximum frequency in Hz to cutoff from return, defaults to 70
    :type maxfr: int, optional
    :param spectrum_windowsize: Length of the window used in Welch's method (in seconds), defaults to 1.0
    :type spectrum_windowsize: float, optional
    :param normalize: Maximum power is normalized to 1 if True, defaults to False
    :type normalize: bool, optional

    :return: Frquencies and the power of each frequency
    :rtype: [np.ndarray, np.ndarray]
    """

    powers = np.zeros(getPowerSpectrum(activities[0], dt, maxfr, spectrum_windowsize)[0].shape)
    ps = []
    for rate in activities:
        f, Pxx_spec = getPowerSpectrum(rate, dt, maxfr, spectrum_windowsize)
        ps.append(Pxx_spec)
        powers += Pxx_spec
    powers /= len(ps)
    if normalize:
        powers /= np.max(powers)
    return f, powers
