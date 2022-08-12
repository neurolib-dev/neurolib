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
def solve_adjoint(hx, hx_nw, fx, state_dim, dt, N, T):
    """Backwards integration of the adjoint state.
    :param fx: df/dx    Derivative of cost function wrt. to systems dynamics.
    :type fx:           np.ndarray
    :param hx: dh/dx    Jacobians.
    :type hx:           np.ndarray of shape Tx2x2
    :param state_dim:
    :type state_dim:    tuple
    :param dt:          Time resolution of integration.
    :type dt:           float
    :param N:           number of nodes in the network
    :type N:            int
    :param T:           length of simulation (time dimension)
    :type T:            int

    :return:            adjoint state.
    :rtype:             np.ndarray of shape `state_dim`
    """
    # ToDo: generalize, not only precision cost
    adjoint_state = np.zeros(state_dim)
    fx_fullstate = adjoint_state.copy()
    fx_fullstate[:, :2, :] = fx

    for ind in range(T - 2, 0, -1):
        for n in range(N):
            der = 0.0
            der = (
                fx_fullstate[n, :, ind + 1]
                + adjoint_state[n, :, ind + 1] @ hx[n, ind + 1]
            )
            for n2 in range(N):
                der += adjoint_state[n2, :, ind + 1] @ hx_nw[n2, n, ind + 1]
            adjoint_state[n, :, ind] = adjoint_state[n, :, ind + 1] - der * dt

    return adjoint_state


@numba.njit
def compute_hx_nw(K_gl, cmat, coupling, N, V, T):
    """Jacobian for network connectivity.

    :param K_gl:    FHN model parameter.
    :type K_gl:     float

    :param cmat:    FHN model parameter, connectivity matrx.
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
