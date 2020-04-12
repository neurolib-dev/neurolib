import numpy as np

from ...utils.collections import dotdict


def loadDefaultParams(Cmat=None, Dmat=None, seed=None):
    """Load default parameters for the subtractive-divisive Wilson-Cowan model

    :param Cmat: Structural connectivity matrix (adjacency matrix) of coupling strengths, will be normalized to 1. If not given, then a single node simulation will be assumed, defaults to None
    :type Cmat: numpy.ndarray, optional
    :param Dmat: Fiber length matrix, will be used for computing the delay matrix together with the signal transmission speed parameter `signalV`, defaults to None
    :type Dmat: numpy.ndarray, optional
    :param seed: Seed for the random number generator, defaults to None
    :type seed: int, optional

    :return: A dictionary with the default parameters of the model
    :rtype: dict
    """

    params = dotdict({})

    ### runtime parameters
    params.dt = 0.1  # ms 0.1ms is reasonable
    params.duration = 2000  # Simulation duration (ms)
    np.random.seed(seed)  # seed for RNG of noise and ICs
    # set seed to 0, pypet will complain otherwise
    if seed is None:
        seed = 0
    params.seed = seed

    # ------------------------------------------------------------------------
    # global whole-brain network parameters
    # ------------------------------------------------------------------------

    # signal transmission speed between areas
    params.signalV = 20.0
    params.K_gl = 0.6  # global coupling strength

    if Cmat is None:
        params.N = 1
        params.Cmat = np.zeros((1, 1))
        params.lengthMat = np.zeros((1, 1))

    else:
        params.Cmat = Cmat.copy()  # coupling matrix
        np.fill_diagonal(params.Cmat, 0)  # no self connections
        params.N = len(params.Cmat)  # number of nodes
        params.lengthMat = Dmat

    # ------------------------------------------------------------------------
    # local node parameters
    # ------------------------------------------------------------------------

    # external input parameters:
    params.tau_ou = 5.0  # ms Timescale of the Ornstein-Uhlenbeck noise process
    params.sigma_ou = 0.0  # mV/ms/sqrt(ms) noise intensity
    params.exc_ou_mean = 0.0  # mV/ms (OU process) [0-5]
    params.inh_s_ou_mean = 0.0  # mV/ms (OU process) [0-5]
    params.inh_d_ou_mean = 0.0  # mV/ms (OU process) [0-5]

    # neural mass model parameters (following the notation in the paper)
    params.tau_exc = 2.5  # excitatory time constant
    params.tau_inh_s = 3.75  # inhibitory time constant (somatic inhibition)
    params.tau_inh_d = 3.75  # inhibitory time constant (dendritic inhibition)
    params.w1 = 24.368  # local E-E coupling
    params.w3 = 9.677  # local E-I_s coupling
    params.w2 = 0.54*params.w3  # local I_d-E coupling
    params.w4 = 27.249  # local E-I_d coupling
    params.w5 = 30.913  # local E-I_s coupling
    params.w6 = 0.33*params.w3  # local I_s-I_d coupling
    params.w7 = params.w3  # local I_s-I_s coupling
    params.alpha_exc = 1.3  #
    params.alpha_inh_s = 2.  #
    params.alpha_inh_d = 2.  #
    params.theta_exc = 4.  #
    params.theta_inh_s = 3.7  #
    params.theta_inh_d = 3.7  #
    params.q = 0.  # subtractive/divisive parameter (q=0 purely subtractive; q=1 purely divisive)

    # ------------------------------------------------------------------------

    params.exc_init = 0.05 * np.random.uniform(0, 1, (params.N, 1))
    params.inh_s_init = 0.05 * np.random.uniform(0, 1, (params.N, 1))
    params.inh_d_init = 0.05 * np.random.uniform(0, 1, (params.N, 1))

    # Ornstein-Uhlenbeck noise state variables
    params.exc_ou = np.zeros((params.N,))
    params.inh_s_ou = np.zeros((params.N,))
    params.inh_d_ou = np.zeros((params.N,))

    # values of the external inputs
    params.exc_ext = 1.428 * np.ones((params.N,))
    params.inh_s_ext = np.zeros((params.N,))
    params.inh_d_ext = np.zeros((params.N,))

    return params


def computeDelayMatrix(lengthMat, signalV, segmentLength=1):
    """Compute the delay matrix from the fiber length matrix and the signal velocity

        :param lengthMat:       A matrix containing the connection length in segment
        :param signalV:         Signal velocity in m/s
        :param segmentLength:   Length of a single segment in mm

        :returns:    A matrix of connexion delay in ms
    """

    normalizedLenMat = lengthMat * segmentLength
    # Interareal connection delays, Dmat(i,j) in ms
    if signalV > 0:
        Dmat = normalizedLenMat / signalV
    else:
        Dmat = lengthMat * 0.0
    return Dmat
