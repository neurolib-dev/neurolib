import numpy as np

from ...utils.collections import dotdict


def loadDefaultParams(Cmat=None, Dmat=None, seed=None):
    """Load default parameters for the Hopf model

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
    # set seed to 0 if None, pypet will complain otherwise
    params.seed = seed or 0

    # make sure that seed=0 remains None
    if seed == 0:
        seed = None

    # ------------------------------------------------------------------------
    # global whole-brain network parameters
    # ------------------------------------------------------------------------

    # the coupling parameter determines how nodes are coupled.
    # "diffusive" for diffusive coupling, "additive" for additive coupling
    params.coupling = "diffusive"

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
    params.x_ou_mean = 0.0  # mV/ms (OU process) [0-5]
    params.y_ou_mean = 0.0  # mV/ms (OU process) [0-5]

    # neural mass model parameters
    params.a = 0.25  # Hopf bifurcation parameter
    params.w = 0.2  # Oscillator frequency, 32 Hz at w = 0.2

    # ------------------------------------------------------------------------

    # initial values of the state variables
    params.xs_init = 0.5 * np.random.uniform(-1, 1, (params.N, 1))
    params.ys_init = 0.5 * np.random.uniform(-1, 1, (params.N, 1))

    # Ornstein-Uhlenbeck noise state variables
    params.x_ou = np.zeros((params.N,))
    params.y_ou = np.zeros((params.N,))

    # values of the external inputs
    params.x_ext = np.zeros((params.N,))
    params.y_ext = np.zeros((params.N,))

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
