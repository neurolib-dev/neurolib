import numpy as np

from neurolib.utils.collections import dotdict


def loadDefaultParams(Cmat=None, Dmat=None, seed=None):
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
        params.Cmat = np.ones((params.N, params.N))
        np.fill_diagonal(params.Cmat, 0)  # no self connections
        params.lengthMat = np.zeros((params.N, params.N))
    else:
        params.Cmat = Cmat.copy()  # coupling matrix
        np.fill_diagonal(params.Cmat, 0)  # no self connections
        params.N = len(params.Cmat)  # override number of nodes
        params.lengthMat = Dmat

    params.omega = np.random.normal(loc=np.pi, scale=np.pi, size=(params.N,))

    params.signalV = 20.0

    # Ornstein-Uhlenbeck process
    params.tau_ou = 5.0  # ms Timescale of the Ornstein-Uhlenbeck noise process
    params.sigma_ou = 0.0  # mV/ms/sqrt(ms) noise intensity

    # init values
    params.theta_init = np.random.uniform(low=0, high=2*np.pi, size=(params.N, 1))

    # Ornstein-Uhlenbeck process
    params.theta_ou = np.zeros((params.N,))

    # external input
    params.theta_ext = np.ones((params.N,))

    return params
    