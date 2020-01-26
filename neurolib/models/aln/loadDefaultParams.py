import os
import scipy.io
import h5py
import numpy as np

def loadDefaultParams(Cmat = [], Dmat = [], lookupTableFileName = None, seed=None):
    '''
    Load default parameters for a network of aLN nodes. 
    if Cmat is an empty list, only a single node (consisting of E and I) will be simulated.
    A lot of the code which deals with loading structural connectivity 
    and delay matrices for an interareal whole-brain network simulation.

    Parameters:
        :param Cmat: Structural connectivity matrix
        :param Dmat: Distance matrix in mm
        :param lookUpTableFileName:     Matlab filename where to find the lookup table. Default: aln-precalc/quantities_cascade.h5'
        :param seed: Seed for the RNG
    :returns:   A dictionary of default parameters
    '''
    class struct(object):
        pass
    
    params = struct()
    
    ### runtime parameters
    params.dt           = 0.1  #ms 0.1ms is reasonable
    params.duration     = 2000 # Simulation duration (ms)
    params.seed         = np.int64(0) # seed for RNG of noise and ICs

    # recently added for easier simulation of aln and brian in pypet
    params.model        = 'aln'
    params.load_point   = 'none'
    
    ### options
    params.warn         = 0  # if out of precalc limits in interpolation, set to 0 for faster computation
    params.dosc_version = 0
    params.distr_delay  = 0 # if 1, use distributed intra-areal delay insted of fixed
    params.filter_sigma = 0 # if 1, full filter used for sigmae/sigmai, 
                            # else use dirac filter
    params.fast_interp  = 1 # if 1, Interpolate the value from the look-up table \
                           #    instead of taking the closest value
    
    
    # ------------------------------------------------------------------------
    # global whole-brain network parameters
    # ------------------------------------------------------------------------

    if len(Cmat) == 0:
        params.N = 1
        params.Cmat = np.zeros((1,1))
        params.lengthMat = np.zeros((1,1))
        
    else:
        params.Cmat = Cmat.copy() # coupling matrix
        np.fill_diagonal(Cmat,0)  # no self connections
        params.Cmat = Cmat  / np.max(Cmat) # normalize matrix
        params.N = len(params.Cmat) # number of nodes                  
        params.lengthMat = Dmat # delay matrix

    params.global_delay = 1     # if 1, use INTER-areal delay (from lengthMat) NOTE: value 0 doesn't work yet, don't change!!!
    params.signalV      = 20.0  # Signal transmission speed in mm/ms
    params.c_gl         = 0.4   # PSP current amplitude in (mV/ms) (or nA/[C]) for global coupling connections between areas
    params.Ke_gl        = 250.0 # number of incoming E connections (to E population) from each area
    
    
    # ------------------------------------------------------------------------
    # local E-I node parameters
    # ------------------------------------------------------------------------

    # external input parameters:
    params.tau_ou       = 5.0   # ms
    params.sigma_ou     = 0.0  # mV/ms/sqrt(ms)
    params.mue_ext_mean = 1.6   # mV/ms (OU process) [0-5]
    params.mui_ext_mean = 0.3   # mV/ms (OU process) [0-5]
    params.ext_exc_rate = 0.0   # external excitatory rate drive [kHz]
    params.ext_inh_rate = 0.0   # external inhibiroty rate drive [kHz]

    params.ext_exc_current = 0.0  # external excitatory field drive [mV/ms]
    params.ext_inh_current = 0.0  # external inhibiroty field drive [mV/ms]

    
    # Fokker Planck noise (for N->inf)
    params.sigmae_ext   = 1.5   # mV/sqrt(ms) (fixed, for now) [1-5] (Internal noise due to random coupling)
    params.sigmai_ext   = 1.5   # mV/sqrt(ms) (fixed, for now) [1-5]


    # recurrent coupling parameters
    params.Ke           = 800.0  # Number of excitatory inputs per neuron
    params.Ki           = 200.0  # Number of inhibitory inputs per neuron

    params.de           = 4.0    # ms local constant delay "EE = IE"
    params.di           = 2.0    # ms local constant delay "EI = II"
    
    params.tau_se       = 2.0    # ms  "EE = IE"
    params.tau_si       = 5.0    # ms  "EI = II"
    params.tau_de       = 1.0    # ms  "EE = IE"
    params.tau_di       = 1.0    # ms  "EI = II"
    
    params.cee          = 0.3    # mV/ms 
    params.cie          = 0.3    # AMPA
    params.cei          = 0.5    # GABA BrunelWang2003
    params.cii          = 0.5

    # Coupling strengths used in Cakan2020
    params.Jee_max      = 2.43    # mV/ms [all 0-10, compare to mue_ext_mean, will be added to it]
    params.Jie_max      = 2.60    # mV/ms 
    params.Jei_max      = -3.3   # mV/ms [0-(-10)] 
    params.Jii_max      = -1.64   # mV/ms  

    # neuron model parameters
    params.a            = 12.0   # nS
    params.b            = 60.0   # pA
    params.EA           = -80.0  # mV
    params.tauA         = 200.0  # ms

    # single neuron paramters - if these are changed, new transfer functions must be precomputed!
    params.C            = 200.0  # pF
    params.gL           = 10.0   # nS
    params.EL           = -65.0  # mV
    params.DeltaT       = 1.5    # mV
    params.VT           = -50.0  # mV
    params.Vr           = -70.0  # mV
    params.Vs           = -40.0  # mV    
    params.Tref         = 1.5    # ms
    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------    
    
    # Generate and set random initial conditions
    mufe_init, IA_init, mufi_init, seem_init, seim_init, seev_init, seiv_init, \
        siim_init, siem_init, siiv_init, siev_init, rates_exc_init, rates_inh_init \
            = generateRandomICs(params.N, seed)
            
    params.mufe_init      = mufe_init       # aLN linear-filtered mean input dmu_f/ dt = mu_syn - mu_f / t_eff 
    params.IA_init        = IA_init         # adaptation current
    params.mufi_init      = mufi_init       #   
    params.seem_init      = seem_init       # mean of fraction of active synapses [0-1] (post-synaptic variable), chap. 4.2
    params.seim_init      = seim_init       # 
    params.seev_init      = seev_init       # variance of fraction of active synapses [0-1]
    params.seiv_init      = seiv_init       #
    params.siim_init      = siim_init       #
    params.siem_init      = siem_init       #
    params.siiv_init      = siiv_init       #
    params.siev_init      = siev_init       #
    params.rates_exc_init = rates_exc_init  #
    params.rates_inh_init = rates_inh_init  #

    # load precomputed aLN transfer functions from hdfs
    if lookupTableFileName is None:
        lookupTableFileName = os.path.join(os.path.dirname(__file__), 'aln-precalc', 'quantities_cascade.h5')
        
    hf = h5py.File(lookupTableFileName, 'r')
    params.Irange = hf.get('mu_vals')[()]
    params.sigmarange = hf.get('sigma_vals')[()]
    params.dI = params.Irange[1] - params.Irange[0]
    params.ds = params.sigmarange[1] - params.sigmarange[0]

    params.precalc_r = hf.get('r_ss')[()][()]
    params.precalc_V = hf.get('V_mean_ss')[()]
    params.precalc_tau_mu = hf.get('tau_mu_exp')[()]
    params.precalc_tau_sigma = hf.get('tau_sigma_exp')[()]

    params_dict       = params.__dict__

    return params_dict

def computeDelayMatrix(lengthMat,signalV,segmentLength=1):
    """Compute the delay matrix from the fiber length matrix and the signal velocity
        
        :param lengthMat:       A matrix containing the connection length in segment
        :param signalV:         Signal velocity in m/s
        :param segmentLength:   Length of a single segment in mm  
      
        :returns:    A matrix of connexion delay in ms
    """
    
    normalizedLenMat = lengthMat * segmentLength    # Each segment is ~1.8mm
    if signalV > 0:
        Dmat = normalizedLenMat / signalV  # Interareal connection delays, Dmat(i,j) in ms
    else:
        Dmat = lengthMat * 0.0
    return Dmat

def generateRandomICs(N, seed = None):
    """ Generates random Initial Conditions for the interareal network

        :params N:  Number of area in the large scale network

        :returns:   A tuple of 9 N-length numpy arrays representining:
                        mufe_init, IA_init, mufi_init, sem_init, sev_init, sim_init, siv_init, rates_exc_init, rates_inh_init
    """
    if seed: np.random.seed(seed)


    mufe_init      = 3*np.random.uniform(0,1,(N,))       # mV/ms  if DOSC_version: complex variable
    IA_init        = 200.0*np.random.uniform(0,1,(N,)) # pA
    mufi_init      = 3*np.random.uniform(0,1,(N,))       # mV/ms  if DOSC_version: complex variable
    seem_init      = 0.5*np.random.uniform(0,1,(N,))
    seim_init      = 0.5*np.random.uniform(0,1,(N,))
    seev_init      = 0.001*np.random.uniform(0,1,(N,))
    seiv_init      = 0.001*np.random.uniform(0,1,(N,))
    siim_init      = 0.5*np.random.uniform(0,1,(N,))
    siem_init      = 0.5*np.random.uniform(0,1,(N,))
    siiv_init      = 0.01*np.random.uniform(0,1,(N,))
    siev_init      = 0.01*np.random.uniform(0,1,(N,))
    rates_exc_init = 0.05*np.random.uniform(0,1,(N,1))
    rates_inh_init = 0.05*np.random.uniform(0,1,(N,1))

    
    return mufe_init, IA_init, mufi_init, seem_init, seim_init, seev_init, seiv_init, \
            siim_init, siem_init, siiv_init, siev_init, rates_exc_init, rates_inh_init

def loadICs(params, N, seed = None):
    # Generate and set random initial conditions
    mufe_init, IA_init, mufi_init, seem_init, seim_init, seev_init, seiv_init, \
        siim_init, siem_init, siiv_init, siev_init, rates_exc_init, rates_inh_init \
            = generateRandomICs(N, seed)

    params['mufe_init']      = mufe_init       # aLN linear-filtered mean input dmu_f/ dt = mu_syn - mu_f / t_eff 
    params['IA_init']        = IA_init         # adaptation current
    params['mufi_init']      = mufi_init       #   
    params['seem_init']      = seem_init       # mean of fraction of active synapses [0-1] (post-synaptic variable), chap. 4.2
    params['seim_init']      = seim_init       # 
    params['seev_init']      = seev_init       # variance of fraction of active synapses [0-1]
    params['seiv_init']      = seiv_init       #
    params['siim_init']      = siim_init       #
    params['siem_init']      = siem_init       #
    params['siiv_init']      = siiv_init       #
    params['siev_init']      = siev_init       #
    params['rates_exc_init'] = rates_exc_init  #
    params['rates_inh_init'] = rates_inh_init  #

    return params
