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


def kolmogorov(ts1, ts2, stepsize=5, windowsize=1.0):
    """Computes kolmogorov distance between two timeseries. 
    This is done by first computing two FCD matrices (one for each timeseries)
    and then measuring the Kolmogorov distance of the upper triangle of these matrices.
    
    :param ts1: Timeseries 1
    :type ts1: np.ndarray
    :param ts2: Timeseries 2
    :type ts2: np.ndarray
    :param stepsize: Step size for FCD matrix calculation, defaults to 5
    :type stepsize: int, optional
    :param windowsize: Window size for FCD matrix calculation, defaults to 1.0
    :type windowsize: float, optional
    :return: Kolmogorov distance
    :rtype: float
    """
    empiricalFCD = fcd(ts2[:, : len(ts1[0, :])], stepsize, windowsize)
    FCD = fcd(ts1, stepsize, windowsize)

    triUFCD = np.triu(FCD)
    triUFCD = triUFCD[(triUFCD > 0.0) & (triUFCD < 1.0)]

    emptriUFCD = np.triu(empiricalFCD)
    emptriUFCD = emptriUFCD[(emptriUFCD > 0.0) & (emptriUFCD < 1.0)]

    return scipy.stats.ks_2samp(triUFCD, emptriUFCD)[0]


def print_params(params):
    """
    Helpfer function for printing a subset of the paramters of the aln model.
    Todo: This function should not be here, it is too specific for the aln model.
    Idea: A model could register "parameters of interest" and be printed with this function. 
    However, this should be placed in the Model class in any case
    """
    paramsOfInterest = [
        "dt",
        "Ke_gl",
        "mue_ext_mean",
        "mui_ext_mean",
        "sigma_ou",
        "signalV",
        "a",
        "b",
        "Jee_max",
        "Jie_max",
        "Jii_max",
        "Jei_max",
        "cee",
        "cie",
        "cii",
        "cei",
        "Ke",
        "Ki",
        "de",
        "di",
    ]
    for p in paramsOfInterest:
        print("params['%s'] = %0.3f" % (p, params[p]))


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
