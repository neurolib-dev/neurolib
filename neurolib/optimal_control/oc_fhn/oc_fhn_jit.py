import numba
import numpy as np


@numba.njit
def jacobian_fhn(alpha, beta, gamma, tau, epsilon, x, V):
    """Jacobian of the FHN dynamical system.
    :param alpha:   FHN model parameter.
    :type alpha:    float

    :param beta:    FHN model parameter.
    :type beta:     float

    :param gamma:   FHN model parameter.
    :type gamma:    float

    :param tau:     FHN model parameter.
    :type tau:      float

    :param epsilon: FHN model parameter.
    :type epsilon:  float

    :param x:       Value of the x-population in the FHN model at a specific time step.
    :type x:        float

    :param V:           number of system variables
    :type V:            int

    :return:        Jacobian matrix.
    :rtype:         np.ndarray of dimensions 2x2
    """
    jacobian = np.zeros((V, V))
    jacobian[0, :2] = [3 * alpha * x**2 - 2 * beta * x - gamma, 1]
    jacobian[1, :2] = [-1 / tau, epsilon / tau]
    return jacobian


@numba.njit
def compute_hx(alpha, beta, gamma, tau, epsilon, N, V, T, xs):
    """Jacobians for each time step.

    :param alpha:   FHN model parameter.
    :type alpha:    float

    :param beta:    FHN model parameter.
    :type beta:     float

    :param gamma:   FHN model parameter.
    :type gamma:    float

    :param tau:     FHN model parameter.
    :type tau:      float

    :param epsilon: FHN model parameter.
    :type epsilon:  float

    :param N:           number of nodes in the network
    :type N:            int
    :param V:           number of system variables
    :type V:            int
    :param T:           length of simulation (time dimension)
    :type T:            int

    :param xs:  The jacobian of the FHN systems dynamics depends only on the constant parameters and the values of
                    the x-population.
    :type xs:   np.ndarray of shape 1xT

    :return: array of length T containing 2x2-matrices
    :rtype: np.ndarray of shape Tx2x2
    """
    hx = np.zeros((N, T, V, V))

    for n in range(N):
        for ind, x in enumerate(xs[n, 0, :]):
            hx[n, ind, :, :] = jacobian_fhn(alpha, beta, gamma, tau, epsilon, x, V)
    return hx


@numba.njit
def compute_hx_nw(K_gl, cmat, coupling, N, V, T):
    """Jacobian for network connectivity.

    :param K_gl:    FHN model parameter.
    :type K_gl:     float

    :param cmat:    FHN model parameter, connectivity matrix.
    :type cmat:     ndarray

    :param coupling: FHN model parameter.
    :type coupling:  string

    :param N:           number of nodes in the network
    :type N:            int
    :param V:           number of system variables
    :type V:            int
    :param T:           length of simulation (time dimension)
    :type T:            int

    :return: array of length T containing 2x2-matrices
    :rtype: np.ndarray of shape Tx2x2
    """
    hx_nw = np.zeros((N, N, T, V, V))

    for n1 in range(N):
        for n2 in range(N):
            hx_nw[n1, n2, :, 0, 0] = K_gl * cmat[n1, n2]
            if coupling == "diffusive":
                hx_nw[n1, n1, :, 0, 0] += -K_gl * cmat[n1, n2]

    return -hx_nw
