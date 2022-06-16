import numpy as np


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


def derivative_precision_cost(x_target, x_sim, w_p):
    """ Derivative of precision cost wrt. to x_sim.
    """
    # REMARK - why not "-"?
    return - w_p * (x_target - x_sim)


def energy_cost(u, w_e):
    """
    :param u:
    :type u:

    :param w_e:
    :type w_e: float

    :return:
    :rtype:
    """
    # ToDo: tests for multidimensional case
    return w_e/2. * np.sum(u**2.)


def derivative_energy_cost(u, w_e):
    """
    :param u:
    :type u:

    :param w_e:
    :type w_e: float

    :return:
    :rtype:
    """
    # return w_e * np.abs(u)
    return w_e * u
