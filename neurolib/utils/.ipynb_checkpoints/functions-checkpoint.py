import scipy.signal
import numpy as np


def analyse_run(measure = 'domfr', result = [], dt = 0.1):
    '''
    Analysis routine for bifurcation diagrams with a 'rect' stimulus that is used
    to detect bistability.
    
    measure:    Pass the measure you want to compute in string form, such as "domfr_power_exc"
    result:     Timeseries of a successful simulation.
    dt:         Integration timestep of simulations in ms

    '''
    t = result['t']
    
    down_window = (2000<t) & (t<3000) # time period in ms where we expect the down-state
    up_window = (5000<t) & (t<6000) # and up state
    
    if measure.endswith('inh'):
        rate = result['rate_inh']
    else:
        rate = result['rate_exc']
    
        
    if measure.startswith('domfr_power'):
        # returns power of dominant frequency
        if np.any((rate > 0)):
            spectrum_windowsize =  0.5 # in seconds 
            f, Pxx_spec = scipy.signal.welch(
            rate[down_window],
            1000/dt,
            window='hanning',
            nperseg=int(spectrum_windowsize * 1000 / dt)-1,
            scaling='spectrum')
            f = f[f < 70]
            Pxx_spec = Pxx_spec[0:len(f)]
            return np.max(Pxx_spec)
        else: 
            return 0.0
    elif measure.startswith('domfr'):
        # returns dominant frequency
        if np.any((rate > 0)):
            spectrum_windowsize =  0.5 # in seconds 
            f, Pxx_spec = scipy.signal.welch(
            rate[down_window],
            1000/dt,
            window='hanning',
            nperseg=int(spectrum_windowsize * 1000 / dt)-1,
            scaling='spectrum')
            f = f[f < 70]
            Pxx_spec = Pxx_spec[0:len(f)]
            domfr = f[Pxx_spec.argmax()] if max(Pxx_spec) > 1 else 0
            return domfr
        else: 
            return 0.0
        
    elif measure.startswith('max'):
        return np.max(rate[up_window])
    
    elif measure.startswith('min'):
        return np.min(rate[up_window])    
    
    elif measure.startswith('updowndiff'):
        up_state_rate = np.mean(rate[up_window])
        down_state_rate = np.mean(rate[down_window])
        up_down_difference = up_state_rate - down_state_rate
        return up_down_difference

    elif measure.startswith('spectrum'):
        if np.any((rate > 0)):
            spectrum_windowsize = 1.0 
            f, Pxx_spec = scipy.signal.welch(
            rate[t>1000],
            1000/dt,
            window='hanning',
            nperseg=int(spectrum_windowsize * 1000 / dt),
            scaling='spectrum')
            f = f[f < 70]
            Pxx_spec = Pxx_spec[0:len(f)]
            Pxx_spec /= np.max(Pxx_spec)
            return f, Pxx_spec

def kuramoto(traces, smoothing=0.0, dt = 0.1, debug = False, peakrange=[0.1, 0.2]):
    phases = []
    nTraces = len(traces)
    for n in range(nTraces):
        tList = np.dot(range(len(traces[n])),dt/1000)
        a = traces[n]
        
        # find peaks
        if smoothing > 0:
            a = scipy.ndimage.filters.gaussian_filter(traces[n], smoothing) # smooth data
        maximalist = scipy.signal.find_peaks_cwt(a, np.arange(peakrange[0], peakrange[1]))
        maximalist = np.append(maximalist, len(traces[n])-1).astype(int)
        
        if len(maximalist) > 1:
            phases.append([])
            lastMax = 0
            for m in maximalist:
                for t in range(lastMax, m):
                    phi = 2 * np.pi * float(t - lastMax) / float(m - lastMax)
                    phases[n].append(phi)
                lastMax = m
            phases[n].append(2 * np.pi)
        else:
            return 0
    
    # determine kuramoto order paramter
    kuramoto = []
    for t in range(len(tList)):
        R = 1j*0
        for n in range(nTraces):
            R += np.exp(1j * phases[n][t])
        R /= nTraces
        kuramoto.append(np.absolute(R))
    
       
    return kuramoto

def matrix_correlation(M1, M2):
    return np.corrcoef(M1.reshape((1,M1.size)), M2.reshape( (1,M2.size) ) )[0,1]
    
def fc(BOLD):
    simFC = np.corrcoef(BOLD)
    simFC = np.nan_to_num(simFC) # remove NaNs
    return simFC

def fcd(BOLD, windowsize = 30, stepsize = 5, N=90):
    # compute FCD matrix
    t_window_width = int(windowsize)# int(windowsize * 30) # x minutes
    stepsize = stepsize # BOLD.shape[1]/N
    corrFCs = []
    try:
        counter = range(0, BOLD.shape[1]-t_window_width, stepsize)

        for t in counter:
            BOLD_slice = BOLD[:,t:t+t_window_width]
            corrFCs.append(np.corrcoef(BOLD_slice))

        FCd = np.empty([len(corrFCs),len(corrFCs)])
        f1i = 0
        for f1 in corrFCs:
            f2i = 0
            for f2 in corrFCs:
                FCd[f1i, f2i] = np.corrcoef(f1.reshape((1,f1.size)), f2.reshape( (1,f2.size) ) )[0,1]
                f2i+=1
            f1i+=1

        return FCd
    except:
        return 0

def dom_freqs(ts, one_over_dt = 2e3, freqcut = 300, welch_param = 4*1024):
    # return dominant frequencies of a timeseries    
    dominant_freqs = []
    if len(ts.shape)>1: # multiple nodes
        for n in range(len(ts)): #range(33,45):
            thisdata = ts[n, :]
            f, Pxx_spec = scipy.signal.welch(thisdata, one_over_dt, 'flattop', welch_param, scaling='spectrum')
            # find dominant frequencies
            f = f[f<freqcut]
            y_av = np.sqrt(Pxx_spec)
            x = f
            max_y = max(y_av)  # Find the maximum y value
            max_x = x[y_av.argmax()]  # Find the x value corresponding to the maximum y value
            dominant_freqs.append(max_x)
    else: # single node
        thisdata = ts
        f, Pxx_spec = scipy.signal.welch(thisdata, one_over_dt, 'flattop', welch_param, scaling='spectrum')
        # frequency cut
        f = f[f<freqcut]
        Pxx_spec = Pxx_spec[0:len(f)]
        # find dominant frequencies
        y_av = Pxx_spec
        x = f
        max_y = max(y_av)  # Find the maximum y value
        max_x = x[y_av.argmax()]  # Find the x value corresponding to the maximum y value
        dominant_freqs = max_x
    return dominant_freqs

def kolmogorov(BOLD1, BOLD2, windowsize = 1.0):
    # return kolmogorov distance between two functional connectivities
    empiricalFCD = fcd(BOLD2[:,:len(BOLD1[0,:])], windowsize = windowsize)
    FCD = fcd(BOLD1[:, 10:], windowsize = windowsize);
    
    triUFCD = np.triu(FCD)
    triUFCD = triUFCD[(triUFCD>0.0)&(triUFCD<1.0)]

    emptriUFCD = np.triu(empiricalFCD)
    emptriUFCD = emptriUFCD[(emptriUFCD>0.0)&(emptriUFCD<1.0)]

    

    return scipy.stats.ks_2samp(triUFCD, emptriUFCD)[0]

def print_params(params):
    # helper function to print the current set of parameters
    paramsOfInterest = ['dt', 'Ke_gl', 'mue_ext_mean', 'mui_ext_mean', 'sigma_ou', 'signalV', 'a', 'b', 'Jee_max', 'Jie_max', 'Jii_max', 'Jei_max', 'cee', 'cie', 'cii', 'cei', 'Ke', 'Ki', 'de', 'di']
    for p in paramsOfInterest:
        print("params[\'%s\'] = %0.3f"%(p, params[p]))

