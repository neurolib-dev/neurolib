import numpy as np
import numba


@numba.njit
def precision_cost(x_target, x_sim, w_p, precision_matrix, dt, interval=(0, None)):
    """Summed squared difference between target and simulation within specified time interval weighted by w_p.

    :param x_target:    N x V x T array that contains the target time series.
    :type x_target:     np.ndarray

    :param x_sim:       N x V x T array that contains the simulated time series.
    :type x_sim:        np.ndarray

    :param w_p:         Weight that is multiplied with the precision cost.
    :type w_p:          float

    :param precision_matrix: N x V binary matrix that defines nodes and channels of precision measurement. Defaults to
                                 None.
    :type precision_matrix:  np.ndarray

    :param dt:  Time step.
    :type dt:   float

    :param interval:    (t_start, t_end). Indices of start and end point of the slice (both inclusive) in time
                        dimension. Default is full time series, defaults to (0, None).
    :type interval:     tuple, optional

    :return:            Precision cost for time interval.
    :rtype:             float

    """

    if interval[1] is None:  # until and including the last entry along the time axis
        interval = (interval[0], x_target.shape[2])

    cost = 0.0
    for n in range(x_target.shape[0]):
        for v in range(x_target.shape[1]):
            for t in range(interval[0], interval[1]):
                cost += precision_matrix[n, v] * (x_target[n, v, t] - x_sim[n, v, t]) ** 2

    return w_p * 0.5 * cost * dt


@numba.njit
def derivative_precision_cost(x_target, x_sim, w_p, precision_matrix, interval=(0, None)):
    """Derivative of precision cost wrt. to x_sim.

    :param x_target:    N x V x T array that contains the target time series.
    :type x_target:     np.ndarray

    :param x_sim:       N x V x T array that contains the simulated time series.
    :type x_sim:        np.ndarray

    :param w_p:         Weight that is multiplied with the precision cost.
    :type w_p:          float

    :param precision_matrix: N x V binary matrix that defines nodes and channels of precision measurement, defaults to
                                 None
    :type precision_matrix:  np.ndarray

    :param interval:    (t_start, t_end). Indices of start and end point of the slice (both inclusive) in time
                        dimension. Default is full time series, defaults to (0, None).
    :type interval:     tuple, optional

    :return:            Control-dimensions x T array of precision cost gradients.
    :rtype:             np.ndarray
    """
    if interval[1] is None:  # until and including the last entry along the time axis
        interval = (interval[0], x_target.shape[2])

    derivative = np.zeros(x_target.shape)

    for n in range(x_target.shape[0]):
        for v in range(x_target.shape[1]):
            for t in range(interval[0], interval[1]):
                derivative[n, v, t] = np.multiply(-w_p * (x_target[n, v, t] - x_sim[n, v, t]), precision_matrix[n, v])

    return derivative


@numba.njit
def energy_cost(u, w_2, dt):
    """
    :param u:   Control-dimensions x T array. Control signals.
    :type u:    np.ndarray

    :param w_2: Weight that is multiplied with the W2 ("energy") cost.
    :type w_2:  float

    :param dt:  Time step.
    :type dt:   float

    :return:    W2 cost of the control.
    :rtype:     float
    """
    return w_2 * 0.5 * np.sum(u**2.0) * dt


@numba.njit
def derivative_energy_cost(u, w_2):
    """
    :param u:   Control-dimensions x T array. Control signals.
    :type u:    np.ndarray

    :param w_2: Weight that is multiplied with the W2 ("energy") cost.
    :type w_2:  float

    :return :   Control-dimensions x T array of W2-cost gradients.
    :rtype:     np.ndarray
    """
    return w_2 * u
