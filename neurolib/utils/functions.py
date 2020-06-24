import logging
import numpy as np
import scipy.signal


def kuramoto(traces, dt=0.1, smoothing=0.0, peakrange=[0.1, 0.2]):
    """
    Computes the Kuramoto order parameter of a timeseries which is a measure for synchrony.
    Can smooth timeseries if there is noise. 
    Peaks are then detected using a peakfinder. From these peaks a phase is derived and then 
    the amount of phase synchrony (the Kuramoto order parameter) is computed.

    :param traces: Multidimensional timeseries array
    :type traces: numpy.ndarray
    :param dt: Integration time step
    :type dt: float
    :param smoothing: Gaussian smoothing strength
    :type smoothing: float, optional
    :param peakrange: Width range of peaks for peak detection with `scipy.signal.find_peaks_cwt`
    :type peakrange: list[float], length 2
            
    :return: Timeseries of Kuramoto order paramter 
    :rtype: numpy.ndarray
    """
    phases = []
    nTraces = len(traces)
    for n in range(nTraces):
        tList = np.dot(range(len(traces[n])), dt / 1000)
        a = traces[n]

        # find peaks
        if smoothing > 0:
            a = scipy.ndimage.filters.gaussian_filter(traces[n], smoothing)  # smooth data
        maximalist = scipy.signal.find_peaks_cwt(a, np.arange(peakrange[0], peakrange[1]))
        maximalist = np.append(maximalist, len(traces[n]) - 1).astype(int)

        if len(maximalist) > 1:
            phases.append([])
            lastMax = 0
            for m in maximalist:
                for t in range(lastMax, m):
                    # compute instantaneous phase
                    phi = 2 * np.pi * float(t - lastMax) / float(m - lastMax)
                    phases[n].append(phi)
                lastMax = m
            phases[n].append(2 * np.pi)
        else:
            logging.warning("Kuramoto: No peaks found, returning 0.")
            return 0

    # determine kuramoto order paramter
    kuramoto = []
    for t in range(len(tList)):
        R = 1j * 0
        for n in range(nTraces):
            R += np.exp(1j * phases[n][t])
        R /= nTraces
        kuramoto.append(np.absolute(R))

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
        activity, 1000 / dt, window="hanning", nperseg=int(spectrum_windowsize * 1000 / dt), scaling="spectrum",
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


def construct_stimulus(
    stim="dc",
    duration=6000,
    dt=0.1,
    stim_amp=0.2,
    stim_freq=1,
    stim_bias=0,
    n_periods=None,
    nostim_before=0,
    nostim_after=0,
):
    """Constructs a stimulus that can be applied to a model

    :param stim: Stimulation type: 'ac':oscillatory stimulus, 'dc': stimple step current, 
                'rect': step current in negative then positive direction with slowly
                decaying amplitude, used for bistability detection, defaults to 'dc'
    :type stim: str, optional
    :param duration: Duration of stimulus in ms, defaults to 6000
    :type duration: int, optional
    :param dt: Integration time step in ms, defaults to 0.1
    :type dt: float, optional
    :param stim_amp: Amplitude of stimulus (for AdEx: in mV/ms, multiply by conductance C to get current in pA), defaults to 0.2
    :type stim_amp: float, optional
    :param stim_freq: Stimulation frequency, defaults to 1
    :type stim_freq: int, optional
    :param stim_bias: Stimulation offset (bias), defaults to 0
    :type stim_bias: int, optional
    :param n_periods: Numer of periods of stimulus, defaults to None
    :type n_periods: [type], optional
    :param nostim_before: Time before stimulation, defaults to 0
    :type nostim_before: int, optional
    :param nostim_after: Time after stimulation, defaults to 0
    :type nostim_after: int, optional
    :raises ValueError: Raises error if unsupported stimulus type is chosen.
    :return: Stimulus timeseries
    :rtype: numpy.ndarray
    """
    """Constructs a sitmulus that can be applied as input to a model

    TODO: rewrite

    stim:       Stimulus type: 'ac':oscillatory stimulus, 'dc': stimple step current, 
                'rect': step current in negative then positive direction with slowly
                decaying amplitude, used for bistability detection
    stim_amp:   Amplitude of stimulus (for AdEx: in mV/ms, multiply by conductance C to get current in pA)
    """

    def sinus_stim(f=1, amplitude=0.2, positive=0, phase=0, cycles=1, t_pause=0):
        x = np.linspace(np.pi, -np.pi, int(1000 / dt / f))
        sinus_function = np.hstack(((np.sin(x + phase) + positive), np.tile(0, t_pause)))
        sinus_function *= amplitude
        return np.tile(sinus_function, cycles)

    if stim == "ac":
        """Oscillatory stimulus
        """
        n_periods = n_periods or int(stim_freq)

        stimulus = np.hstack(
            ([stim_bias] * int(nostim_before / dt), np.tile(sinus_stim(stim_freq, stim_amp) + stim_bias, n_periods),)
        )
        stimulus = np.hstack((stimulus, [stim_bias] * int(nostim_after / dt)))
    elif stim == "dc":
        """Simple DC input and return to baseline
        """
        stimulus = np.hstack(([stim_bias] * int(nostim_before / dt), [stim_bias + stim_amp] * int(1000 / dt),))
        stimulus = np.hstack((stimulus, [stim_bias] * int(nostim_after / dt)))
        stimulus[stimulus < 0] = 0
    elif stim == "rect":
        """Rectified step current with slow decay
        """
        # construct input
        stimulus = np.zeros(int(duration / dt))
        tot_len = int(duration / dt)
        stim_epoch = tot_len / 6

        stim_increase_counter = 0
        stim_decrease_counter = 0
        stim_step_increase = 5.0 / stim_epoch

        for i, m in enumerate(stimulus):
            if 0 * stim_epoch <= i < 0.5 * stim_epoch:
                stimulus[i] -= stim_amp
            elif 0.5 * stim_epoch <= i < 3.0 * stim_epoch:
                stimulus[i] = -np.exp(-stim_increase_counter) * stim_amp
                stim_increase_counter += stim_step_increase
            elif 3.0 * stim_epoch <= i < 3.5 * stim_epoch:
                stimulus[i] += stim_amp
            elif 3.5 * stim_epoch <= i < 5 * stim_epoch:
                stimulus[i] = np.exp(-stim_decrease_counter) * stim_amp
                stim_decrease_counter += stim_step_increase
    else:
        raise ValueError(f'Stimulus {stim} not found. Use "ac", "dc" or "rect".')

    # repeat stimulus until full length
    steps = int(duration / dt)
    stimlength = int(len(stimulus))
    stimulus = np.tile(stimulus, int(steps / stimlength + 2))
    stimulus = stimulus[:steps]

    return stimulus
