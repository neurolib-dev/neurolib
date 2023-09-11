import numpy as np

from ...utils.collections import dotdict


def loadDefaultParams(Cmat=None, Dmat=None, seed=None):
    """Load default parameters for the Wilson-Cowan model

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
    params.sigma_ou = 0.0  # noise intensity
    params.exc_ou_mean = 0.0  # OU process mean
    params.inh_ou_mean = 0.0  # OU process mean

    # neural mass model parameters
    params.tau_exc = 2.5  # excitatory time constant
    params.tau_inh = 3.75  # inhibitory time constant
    params.c_excexc = 16  # local E-E coupling
    params.c_excinh = 15  # local E-I coupling
    params.c_inhexc = 12  # local I-E coupling
    params.c_inhinh = 3  # local I-I coupling
    params.a_exc = 1.5  # excitatory gain
    params.a_inh = 1.5  # inhibitory gain
    params.mu_exc = 3.0  # excitatory firing threshold
    params.mu_inh = 3.0  # inhibitory firing threshold

    # values of the external inputs
    params.exc_ext_baseline = 0  # baseline external input to E (static)
    params.inh_ext_baseline = 0  # baseline external input to I (static)
    params.exc_ext = 0  # time-dependent external input to E
    params.inh_ext = 0  # time-dependent external input to I

    # ------------------------------------------------------------------------

    params.exc_init = 0.05 * np.random.uniform(0, 1, (params.N, 1))
    params.inh_init = 0.05 * np.random.uniform(0, 1, (params.N, 1))

    # Ornstein-Uhlenbeck noise state variables
    params.exc_ou = np.zeros((params.N,))
    params.inh_ou = np.zeros((params.N,))

    return params
