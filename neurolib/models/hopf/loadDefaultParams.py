import numpy as np

import h5py
import scipy.io


def loadDefaultParams(Cmat=[], Dmat=[], seed=None):
    """
    Load default parameters for the for a whole network of aLN nodes.
    A lot of the code which deals with loading structural connectivity
    and delay matrices for an interareal whole-brain network simulation
    can be ignored if only a single E-I module is to be simulated (set singleNode=1 then)

    Parameters:
        :param lookUpTableFileName:     Matlab filename where to find the lookup table. Default: interarea/networks/aln-python/aln-precalc/precalc05sps_all.mat'
    :returns:   A dictionary of default parameters
    """

    class struct(object):
        pass

    params = struct()

    ### runtime parameters
    params.dt = 0.1  # ms 0.1ms is reasonable
    params.duration = 2000  # Simulation duration (ms)
    params.seed = np.int64(0)  # seed for RNG of noise and ICs

    # ------------------------------------------------------------------------
    # global whole-brain network parameters
    # ------------------------------------------------------------------------

    params.signalV = 20.0
    params.K_gl = 250.0  # global coupling strength

    if len(Cmat) == 0:
        params.N = 1
        params.Cmat = np.zeros((1, 1))
        params.lengthMat = np.zeros((1, 1))

    else:
        params.Cmat = Cmat.copy()  # coupling matrix
        np.fill_diagonal(Cmat, 0)  # no self connections
        params.Cmat = Cmat / np.max(Cmat)  # normalize matrix
        params.N = len(params.Cmat)  # number of nodes
        params.lengthMat = Dmat

    # ------------------------------------------------------------------------
    # local node parameters
    # ------------------------------------------------------------------------

    # external input parameters:
    params.tau_ou = 5.0  # ms Timescale of the Ornstein-Uhlenbeck noise process
    params.sigma_ou = 0.0  # mV/ms/sqrt(ms) noise intensity
    params.x_ext_mean = 0.0  # mV/ms (OU process) [0-5]
    params.y_ext_mean = 0.0  # mV/ms (OU process) [0-5]

    # neuron model parameters
    params.a = 0.25  # Hopf bifurcation parameter
    params.w = 0.2  # Oscillator frequency, 32 Hz at w = 0.2

    # ------------------------------------------------------------------------

    params.xs_init = 0.05 * np.random.uniform(0, 1, (params.N, 1))
    params.ys_init = 0.05 * np.random.uniform(0, 1, (params.N, 1))

    params_dict = params.__dict__

    return params_dict


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
