import numba
import numpy as np


@numba.njit
def jacobian_fhn(alpha, beta, gamma, tau, epsilon, x):
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

    :return:        Jacobian matrix.
    :rtype:         np.ndarray of dimensions 2x2
    """
    jacobian = np.array(
        [[3 * alpha * x**2 - 2 * beta * x - gamma, 1], [-1 / tau, epsilon / tau]]
    )
    return jacobian


@numba.njit
def compute_hx(alpha, beta, gamma, tau, epsilon, T, xs):
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

    :param xs:  The jacobian of the FHN systems dynamics depends only on the constant parameters and the values of
                    the x-population.
    :type xs:   np.ndarray of shape 1xT

    :return: array of length T containing 2x2-matrices
    :rtype: np.ndarray of shape Tx2x2
    """
    hx = np.zeros((T, 2, 2))

    for ind, x in enumerate(xs):
        hx[ind, :, :] = jacobian_fhn(alpha, beta, gamma, tau, epsilon, x)

    return hx


@numba.njit
def solve_adjoint(hx, fx, output_dim, dt, T):
    """Backwards integration of the adjoint state.
    :param fx: df/dx    Derivative of cost function wrt. to systems dynamics.
    :type fx:           np.ndarray
    :param hx: dh/dx    Jacobians.
    :type hx:           np.ndarray of shape Tx2x2
    :param output_dim:
    :type output_dim:   tuple
    :param dt:          Time resolution of integration.
    :type dt:           float
    :param T:           Length of simulation (time dimension).
    :type T:            int

    :return:            Adjoint state.
    :rtype:             np.ndarray of shape `output_dim`
    """
    # ToDo: generalize, not only precision cost
    adjoint_state = np.zeros(output_dim)
    adjoint_state[:, -1] = 0

    for ind in range(T - 2, 0, -1):
        adjoint_state[:, ind] = (
            adjoint_state[:, ind + 1]
            - (fx[:, ind + 1] + adjoint_state[:, ind + 1] @ hx[ind + 1]) * dt
        )

    return adjoint_state
