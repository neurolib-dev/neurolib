import scipy
import numpy as np

def getPowerSpectrum(rate, dt, maxfr = 70, spectrum_windowsize = 1.0):
    f, Pxx_spec =  scipy.signal.welch(
            rate,
            1000/dt,
            window='hanning',
            nperseg=int(spectrum_windowsize * 1000 / dt),
            scaling='spectrum')
    #print(f[f < maxfr])
    f = f[f < maxfr]
    Pxx_spec = Pxx_spec[0:len(f)]
    Pxx_spec /= np.max(Pxx_spec)
    return f, Pxx_spec

def getPowerSpectra(rates, dt, maxfr = 70, spectrum_windowsize = 1.0):
    fs = []
    Ps = []
    for r in rates:
        f, P = getPowerSpectrum(r, dt, maxfr, spectrum_windowsize)
        fs.append(f)
        Ps.append(P)
    result = np.ndarray((len(rates), len(f)))
    for i, P in enumerate(Ps):
        result[i] = P
    return f, result
