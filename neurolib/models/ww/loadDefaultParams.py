import numpy as np

from ...utils.collections import dotdict


def loadDefaultParams(Cmat=None, Dmat=None, seed=None):
    """Load default parameters for the Wong-Wang model

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
    params.seed = seed

    # ------------------------------------------------------------------------
    # global whole-brain network parameters
    # ------------------------------------------------------------------------

    # signal transmission speec between areas
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

    # # the coupling parameter determines how nodes are coupled.
    # # "original" for original wong-wang model, "reduced" for reduced wong-wang model
    # params.version = "original"

    # external noise parameters:
    params.tau_ou = 5.0  # ms Timescale of the Ornstein-Uhlenbeck noise process
    params.sigma_ou = 0.0  #  noise intensity
    params.exc_ou_mean = 0.0  # OU process mean
    params.inh_ou_mean = 0.0  # OU process mean

    # neural mass model parameters
    params.a_exc = 0.31  # nC^-1
    params.b_exc = 0.125  # kHz
    params.d_exc = 160.0  # ms
    params.tau_exc = 100.0  # ms
    params.gamma_exc = 0.641
    params.w_exc = 1.0
    params.exc_current = 0.382  # nA

    params.a_inh = 0.615  # nC^-1
    params.b_inh = 0.177  # kHz
    params.d_inh = 87.0  # ms
    params.tau_inh = 10.0  # ms
    params.w_inh = 0.7
    params.inh_current = 0.382  # nA

    params.J_NMDA = 0.15  # nA, excitatory synaptic coupling
    params.J_I = 1.0  # nA, inhibitory synaptic coupling
    params.w_ee = 1.4  # excitatory feedback coupling strength

    # ------------------------------------------------------------------------

    params.ses_init = 0.05 * np.random.uniform(0, 1, (params.N, 1))
    params.sis_init = 0.05 * np.random.uniform(0, 1, (params.N, 1))

    # Ornstein-Uhlenbeck noise state variables
    params.exc_ou = np.zeros((params.N,))
    params.inh_ou = np.zeros((params.N,))

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
