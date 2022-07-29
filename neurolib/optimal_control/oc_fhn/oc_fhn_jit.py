import numba
import numpy as np


@numba.njit
def jacobian_fhn(alpha, beta, gamma, tau, epsilon, x):
    """ """
    jacobian = np.array(
        [[3 * alpha * x**2 - 2 * beta * x - gamma, 1], [-1 / tau, epsilon / tau]]
    )
    return jacobian


@numba.njit
def compute_hx(alpha, beta, gamma, tau, epsilon, T, xs):
    """ """
    hx = np.zeros((T, 2, 2))

    for ind, x in enumerate(xs):
        hx[ind, :, :] = jacobian_fhn(alpha, beta, gamma, tau, epsilon, x)

    return hx


@numba.njit
def solve_adjoint(hx, fx, output_dim, dt, T):
    """Backwards integration.
    :param fx: df/dx
    :param hx: dh/dx
    :param output_dim:
    :param dt:
    :param T:
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
