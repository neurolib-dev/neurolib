import numpy as np

from neurolib.utils.collections import dotdict


def loadDefaultParams(Cmat=None, Dmat=None, seed=None):
    """Load default parameters for the Kuramoto Model model

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

    params.dt = 0.1 
    params.duration = 2000  

    np.random.seed(seed)  
    params.seed = seed

    # model parameters
    params.N = 1
    params.k = 2

    # connectivity
    if Cmat is None:
        params.N = 1
        params.Cmat = np.zeros((1, 1))
        params.lengthMat = np.zeros((1, 1))
    else:
        params.Cmat = Cmat.copy()  # coupling matrix
        np.fill_diagonal(params.Cmat, 0)  # no self connections
        params.N = len(params.Cmat)  # override number of nodes
        params.lengthMat = Dmat
            
    params.omega = np.ones((params.N,)) * np.pi
    
    params.signalV = 20.0

    # Ornstein-Uhlenbeck process
    params.tau_ou = 5.0  # ms Timescale of the Ornstein-Uhlenbeck noise process
    params.sigma_ou = 0.0  # 1/ms/sqrt(ms) noise intensity

    # init values
    params.theta_init = np.random.uniform(low=0, high=2*np.pi, size=(params.N, 1))

    # Ornstein-Uhlenbeck process
    params.theta_ou = np.zeros((params.N,))

    # external input
    params.theta_ext = np.zeros((params.N,))

    return params
    