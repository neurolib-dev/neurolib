import scipy.io
import h5py
import numpy as np

def loadDefaultParams(Cmat = [], Dmat = [], seed=None):
    '''
    Load default parameters for the for a whole network of aLN nodes. 
    A lot of the code which deals with loading structural connectivity 
    and delay matrices for an interareal whole-brain network simulation 
    can be ignored if only a single E-I module is to be simulated (set singleNode=1 then)

    Parameters:
        :param lookUpTableFileName:     Matlab filename where to find the lookup table. Default: interarea/networks/aln-python/aln-precalc/precalc05sps_all.mat'
    :returns:   A dictionary of default parameters
    '''
    class struct(object):
        pass
    
    params = struct()
    
    ### runtime parameters
    params.dt           = 0.1  #ms 0.1ms is reasonable
    params.duration     = 2000 # Simulation duration (ms)
    params.seed         = np.int64(0) # seed for RNG of noise and ICs

    
    # ------------------------------------------------------------------------
    # global whole-brain network parameters
    # ------------------------------------------------------------------------

    params.signalV      = 20.0
    params.K_gl        = 250.0   # global coupling strength

    if len(Cmat) == 0:
        params.N = 1
        params.Cmat = np.zeros((1,1))
        params.lengthMat = np.zeros((1,1))
        
    else:
        params.Cmat = Cmat.copy() # coupling matrix
        np.fill_diagonal(Cmat,0)  # no self connections
        params.Cmat = Cmat  / np.max(Cmat) # normalize matrix
        params.N = len(params.Cmat) # number of nodes                  
        params.lengthMat = Dmat
        #params.Dmat = computeDelayMatrix(Dmat,params.signalV) # delay matrix

    
    
    
    
    # ------------------------------------------------------------------------
    # local node parameters
    # ------------------------------------------------------------------------

    # external input parameters:
    params.tau_ou       = 5.0   # ms
    params.sigma_ou     = 0.05  # mV/ms/sqrt(ms)
    params.x_ext_mean = 0.0   # mV/ms (OU process) [0-5]
    params.y_ext_mean = 0.0   # mV/ms (OU process) [0-5]

    # neuron model parameters
    params.a            = 0.25  # Hopf bifurcation parameter
    params.w            = 1.0   # Oscillator frequency

    # ------------------------------------------------------------------------

    params.xs_init = 0.05*np.random.uniform(0,1,(params.N,1))
    params.ys_init = 0.05*np.random.uniform(0,1,(params.N,1))

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
