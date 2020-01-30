import numpy as np

import scipy.signal


def analyse_run(measure="domfr", result=[], dt=0.1):
    """
    Analysis routine for bifurcation diagrams in Cakan2020
    with a 'rect' stimulus that is used to detect bistability.

    Parameters
    ----------
        :param measure:    Pass the measure you want to compute in string form, such as "domfr_power_exc"
        :param result:     Timeseries of a successful simulation.
        :param dt:         Integration timestep of simulations in ms
    """

    t = result["t"]

    down_window = (2000 < t) & (
        t < 3000
    )  # time period in ms where we expect the down-state
    up_window = (5000 < t) & (t < 6000)  # and up state

    if measure.endswith("inh"):
        rate = result["rate_inh"]
    else:
        rate = result["rate_exc"]

    if measure.startswith("domfr_power"):
        # returns power of dominant frequency
        if np.any((rate > 0)):
            spectrum_windowsize = 0.5  # in seconds
            f, Pxx_spec = scipy.signal.welch(
                rate[down_window],
                1000 / dt,
                window="hanning",
                nperseg=int(spectrum_windowsize * 1000 / dt) - 1,
                scaling="spectrum",
            )
            f = f[f < 70]
            Pxx_spec = Pxx_spec[0 : len(f)]
            return np.max(Pxx_spec)
        else:
            return 0.0
    elif measure.startswith("domfr"):
        # returns dominant frequency
        if np.any((rate > 0)):
            spectrum_windowsize = 0.5  # in seconds
            f, Pxx_spec = scipy.signal.welch(
                rate[down_window],
                1000 / dt,
                window="hanning",
                nperseg=int(spectrum_windowsize * 1000 / dt) - 1,
                scaling="spectrum",
            )
            f = f[f < 70]
            Pxx_spec = Pxx_spec[0 : len(f)]
            domfr = f[Pxx_spec.argmax()] if max(Pxx_spec) > 1 else 0
            return domfr
        else:
            return 0.0

    elif measure.startswith("max"):
        return np.max(rate[up_window])

    elif measure.startswith("min"):
        return np.min(rate[up_window])

    elif measure.startswith("updowndiff"):
        up_state_rate = np.mean(rate[up_window])
        down_state_rate = np.mean(rate[down_window])
        up_down_difference = up_state_rate - down_state_rate
        return up_down_difference

    elif measure.startswith("spectrum"):
        if np.any((rate > 0)):
            spectrum_windowsize = 1.0
            f, Pxx_spec = scipy.signal.welch(
                rate[t > 1000],
                1000 / dt,
                window="hanning",
                nperseg=int(spectrum_windowsize * 1000 / dt),
                scaling="spectrum",
            )
            f = f[f < 70]
            Pxx_spec = Pxx_spec[0 : len(f)]
            Pxx_spec /= np.max(Pxx_spec)
            return f, Pxx_spec


def kuramoto(traces, dt=0.1, smoothing=0.0, peakrange=[0.1, 0.2]):
    """
    Computes the Kuramoto order parameter of a timeseries.
    Can smooth timeseries if there is noise. Peaks are then detected using a peakfinder.
    From these peaks a phase is derived and then phase synchrony (Kuramoto order parameter)
    is computed across all timeseries.

    Parameters
    ----------
        traces : numpy array
            Multidimensional timeseries array
        dt : float
            Integration timestep
        smoothing : float, optional
            Gaussian smoothing strength
        peakrange : list of floats, length 2
            Width range of peaks for peak detection

    Returns
    -------
        kuramoto : numpy array
            Timeseries of Kuramoto order paramter
    """

    phases = []
    nTraces = len(traces)
    for n in range(nTraces):
        tList = np.dot(range(len(traces[n])), dt / 1000)
        a = traces[n]

        # find peaks
        if smoothing > 0:
            a = scipy.ndimage.filters.gaussian_filter(
                traces[n], smoothing
            )  # smooth data
        maximalist = scipy.signal.find_peaks_cwt(
            a, np.arange(peakrange[0], peakrange[1])
        )
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
    """
    Pearson correlation of the lower triagonal of two matrices.
    The triangular matrix is offset by k = 1 in order to ignore the diagonal line

    Parameters
    ----------
        M1 : numpy array
            N x N matrix
        M2 : numpy array
            N x N matrix, must have same dimensions as M1

    Returns
    -------
        cc : float
            Correlation coefficient

    """

    cc = np.corrcoef(
        M1[np.triu_indices_from(M1, k=1)], M2[np.triu_indices_from(M2, k=1)]
    )[0, 1]
    return cc


def fc(ts):
    """
    Functional connectivity matrix of timeseries multidimensional `ts` (Nxt).
    Pearson correlation (from `np.corrcoef()` is used).

    Parameters
    ----------
        ts : numpy array
            Nxt timeseries

    Returns
    -------
        fc : numpy array
            N x N functional connectivity matrix
    """

    fc = np.corrcoef(ts)
    fc = np.nan_to_num(fc)  # remove NaNs
    return fc


def fcd(ts, windowsize=30, stepsize=5):
    """
    Computes FCD (functional connectivity dynamics) matrix, as described in Deco's whole-brain model papers.
    Default paramters are adjusted for BOLD timeseries: windowsize = 30 (=60s) and stepsize = 5 (=10s).

    Parameters
    ----------
        ts : numpy array
            Nxt timeseries
        windowsize : int
            Size of each rolling window in timesteps
        stepsize : int
            Stepsize between each rolling window

    Returns
    -------
        fc : numpy array
            T x T FCD matrix
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
                FCd[f1i, f2i] = np.corrcoef(
                    f1.reshape((1, f1.size)), f2.reshape((1, f2.size))
                )[0, 1]
                f2i += 1
            f1i += 1

        return FCd
    except:
        return 0


def kolmogorov(BOLD1, BOLD2, windowsize=1.0):
    """
    Computes kolmogorov distance between two FCS matrices.
    """
    empiricalFCD = fcd(BOLD2[:, : len(BOLD1[0, :])], windowsize=windowsize)
    FCD = fcd(BOLD1[:, 10:], windowsize=windowsize)

    triUFCD = np.triu(FCD)
    triUFCD = triUFCD[(triUFCD > 0.0) & (triUFCD < 1.0)]

    emptriUFCD = np.triu(empiricalFCD)
    emptriUFCD = emptriUFCD[(emptriUFCD > 0.0) & (emptriUFCD < 1.0)]

    return scipy.stats.ks_2samp(triUFCD, emptriUFCD)[0]


def print_params(params):
    """
    Helpfer function to print the current set of parameters.
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
    """
    Returns a power spectrum using Welch's method.

    Parameters
    ----------
        activity : numpy array
            One-dimensional timeseries
        dt : float
            Simulation time step
        maxfr : float, optional
            Maximum frequency to cutoff from return
        spectrum_windowsize : float
            Length of the window used in Welch's method (in seconds)
        normalize : bool
            Maximum power is normalized to 1 if True

    Returns
    -------
        f : list
            Frequencies
        pwers : list
            Powers
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


def getMeanPowerSpectrum(
    activities, dt, maxfr=70, spectrum_windowsize=1.0, normalize=False
):
    """
    Returns the mean power spectrum of multiple timeseries.
    """
    powers = np.zeros(
        getPowerSpectrum(activities[0], dt, maxfr, spectrum_windowsize)[0].shape
    )
    ps = []
    for rate in activities:
        f, Pxx_spec = getPowerSpectrum(rate, dt, maxfr, spectrum_windowsize)
        ps.append(Pxx_spec)
        powers += Pxx_spec
    powers /= len(ps)
    if normalize:
        powers /= np.max(powers)
    return f, powers
