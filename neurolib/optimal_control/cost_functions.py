import numpy as np
import numba


def precision_cost_in_interval(x_target, x_sim, w_p, interval: tuple[int, int]=(0, None)):
    """
        :param interval: [t_start, t_end], default is full time series
    """
    return w_p * 0.5 * np.sum((x_target[:, interval[0]:interval[1]] - x_sim[:, interval[0]:interval[1]]) ** 2.)


def derivative_precision_cost_in_interval(x_target, x_sim, w_p, interval: tuple[int]=(0, None)):
    """
        :param interval: [t_start, t_end], default is full time series
    """
    derivative = np.zeros(x_target.shape)
    derivative[:, interval[0]:interval[1]] = - w_p * (x_target[:, interval[0]:interval[1]]
                                                      - x_sim[:, interval[0]:interval[1]])
    return derivative


#@numba.njit
def precision_cost(x_target, x_sim, w_p):
    """ Summed squared difference between target and simulation weighted by w_p.
    :param x_target:
    :type x_target:

    :param x_sim:
    :type x_sim:

    :param w_p:
    :type w_p: float

    :return:
    :rtype:
    """
    # ToDo: tests for multidimensional case
    # np.sum without specified axis implicitly performs
    # summation that would correspond to np.sum((x1(t)-x2(t)**2)
    # for the norm at one particular t as well as the integration over t
    # (commutative)
    return w_p*0.5 * np.sum((x_target-x_sim)**2.)

#@numba.njit
def derivative_precision_cost(x_target, x_sim, w_p):
    """ Derivative of precision cost wrt. to x_sim.
    """
    # REMARK - why not "-"?
    return - w_p * (x_target - x_sim)

#@numba.njit
def energy_cost(u, w_2):
    """
    :param u:
    :type u:

    :param w_2:
    :type w_2: float

    :return:
    :rtype:
    """
    # ToDo: tests for multidimensional case
    return w_2/2. * np.sum(u**2.)

##@numba.njit
def derivative_energy_cost(u, w_2):
    """
    :param u:
    :type u:

    :param w_2:
    :type w_2: float

    :return:
    :rtype:
    """
    # return w_2 * np.abs(u)
    return w_2 * u
